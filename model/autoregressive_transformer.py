import torch
from model.base_transformer import BaseTransformer
import torch.nn.functional as F
import torch.nn as nn

class AutoRegressiveTransformer(BaseTransformer):
    """
    Full Decision Transformer that predicts actions for the entire sequence.
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=1000,
            action_tanh=True,
            enable_past_actions=False,
            enable_past_states=False,
            enable_timestep_embedding=False,
            num_classes=5,
            num_params=6,
            num_params_values=1000,
            num_decoder_layers=8,
            dim_feedforward=512,
            use_pretrained_cad_model=False,
            nhead=4,
            dropout=0.1,
            normalize=False,
            device=None,
            encoder="vit",
            num_views=0,
            window_size=1,
            **kwargs
    ):
        super().__init__(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            max_length=max_length,
            max_ep_len=max_ep_len,
            action_tanh=action_tanh,
            encoder=encoder,
            use_pretrained_cad_model=use_pretrained_cad_model,
            **kwargs
        )
        # assert enable_past_actions or enable_past_states, "At least one of past actions or past states must be enabled"
        self.enable_past_actions = enable_past_actions
        self.enable_past_states = enable_past_states
        self.act_dim = act_dim
        assert window_size > 0, "Window size must be greater than 0"
        self.window_size = window_size
        self.transformer_decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers=num_decoder_layers,
        )
        self.normalize = normalize
        self.predict_action_class_0_4 = torch.nn.Linear(hidden_size, num_classes) # Class 0 to 4 (6 classes)
        self.predict_action_class_0_999 = torch.nn.Linear(hidden_size, num_params * num_params_values)
        self.num_views = num_views
        self.use_pretrained_cad_model = use_pretrained_cad_model

        self.num_inputs = 1 # CAD
        if self.enable_past_states:
            self.num_inputs += 1 # CAD and past states
        if num_views > 0:
            self.embed_multiview = torch.nn.Linear(self.state_embedding_model_size*num_views, hidden_size)
            self.num_inputs += 1 # UI, CAD, and multiview
         # 2 for UI and CAD, num_views for multiview
        self.image_projection = torch.nn.Linear(hidden_size*self.num_inputs, hidden_size)  # Projection for ResNet18 output
        self.embed_action = torch.nn.Linear(act_dim, hidden_size)  # Embedding for actions
        self.enable_timestep_embedding = enable_timestep_embedding
        if self.enable_timestep_embedding:
            self.timestep_embedding = torch.nn.Embedding(max_ep_len, hidden_size)

        # Define action masks
        self.action_mask = torch.tensor([
            [1, 1, 0, 0, 0, 0],  # Command 0
            [0, 0, 1, 1, 0, 0],  # Command 1
            [0, 0, 0, 0, 1, 0],  # Command 2
            [0, 0, 0, 0, 0, 1],  # Command 3
            [0, 0, 0, 0, 0, 0]   # Command 4
        ]).float().to(device)

    def apply_action_mask(self, cmd_pred, param_pred):
        """
        Applies the mask based on the predicted command.
        - Keeps values where mask is 1 unchanged.
        - Sets values where mask is 0 to -1.
        """
        # Get the corresponding mask for each predicted command
        mask = self.action_mask[cmd_pred] # Shape: (batch_size, seq_length, 7)
        
        # Apply the mask to parameters
        masked_params = param_pred.clone()
        masked_params[mask == 0] = -1  # Set masked values to -1
        masked_params[:, :, 3] = torch.where(
            (masked_params[:, :, 2] >= 200) & (masked_params[:, :, 2] < 250),
            masked_params[:, :, 3],
            -1
        )
        return masked_params

    def process_actions(self, actions):
        batch_size = actions.shape[0]
        action_embeddings = self.embed_action(actions.float())
        return action_embeddings

    def normalize_actions(self, actions):
        actions[:, :, 0] = actions[:, :, 0]/4.0
        actions[:, :, 1:] = actions[:, :, 1:]/1000.0
        return actions


    def forward(self, inputs, attention_mask=None):
        """
        Forward pass of the autoregressive transformer.
        
        Args:
            inputs (dict): Dictionary containing:
                - frames: Tensor of shape [batch_size, seq_length, channels, height, width]
                - actions: Tensor of shape [batch_size, seq_length, act_dim]
                - timesteps: Tensor of shape [batch_size, seq_length]
                - cad_image: Tensor of shape [batch_size, channels, height, width]
                - multiview_images: Optional tensor of shape [batch_size, num_views, channels, height, width]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted commands and parameters
        """
        # Unpack inputs
        ui_images = inputs['frames']
        actions = inputs['actions']
        cad_image = inputs['cad_image']
        multiview_images = inputs.get('multiview_images', None)
        
        batch_size, seq_length = actions.shape[0], actions.shape[1]
        timesteps = torch.arange(seq_length, device=actions.device)
        if self.enable_timestep_embedding:
            timesteps_embeddings = self.timestep_embedding(timesteps)
        else:
            timesteps_embeddings = torch.zeros(seq_length, self.hidden_size, device=actions.device)
        
        # Process sequential UI image embeddings
        images = []
        if self.enable_past_states:
            ui_images_reshaped = ui_images.reshape(-1, *ui_images.shape[2:])
            ui_image_embeddings = self.process_state(ui_images_reshaped)
            ui_image_embeddings = self.embed_state(ui_image_embeddings).reshape(batch_size, seq_length, -1)
            ui_image_embeddings = ui_image_embeddings + timesteps_embeddings
            ui_image_embeddings = nn.Tanh()(ui_image_embeddings)
            if self.enable_past_actions:
                images.append(ui_image_embeddings)
        
        # Process CAD image embeddings (static)
        cad_image_embeddings = self.process_image(cad_image)
        cad_image_embeddings = self.embed_image(cad_image_embeddings).unsqueeze(1).repeat(1, seq_length, 1)
        images.append(cad_image_embeddings)
        
        # Combine UI image history and CAD image context
        if multiview_images is not None and self.num_views > 0:
            multiview_embeddings = self.process_multiview_images(multiview_images, seq_length)
            multiview_embeddings = self.embed_multiview(multiview_embeddings)
            images.append(multiview_embeddings)
            
        combined_image_embeddings = torch.cat(images, dim=-1)
        if len(images) > 1:
            combined_image_embeddings = self.image_projection(combined_image_embeddings)
        combined_image_embeddings = nn.Tanh()(combined_image_embeddings)
        action_embeddings = self.process_actions(actions)
        action_embeddings = action_embeddings + timesteps_embeddings
        action_embeddings = nn.Tanh()(action_embeddings)
        # Apply causal mask
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length).to(cad_image.device)
        # 0 on diagonal -inf elsewhere
        time_mask = torch.ones(seq_length, seq_length) * float(-torch.inf)
        rows = torch.arange(seq_length)[:, None]
        cols = torch.arange(seq_length)
        # Create a boolean mask of shape (n, n)
        mask = (cols > (rows - self.window_size)) & (cols <= rows)
        # Apply mask to set selected elements to 0
        time_mask[mask] = 0
        
        # Use past actions as input for auto-regression
        if self.enable_past_actions:
            transformer_outputs = self.transformer_decoder(
                tgt=action_embeddings.permute(1, 0, 2),  # Target sequence (past actions)
                memory=combined_image_embeddings.permute(1, 0, 2),  # UI image embeddings until time i-1
                tgt_mask=causal_mask.to(device=cad_image.device),
                memory_mask=time_mask.to(device=cad_image.device),
            )
        elif self.enable_past_states:
            # Transformer decoder with cross-attention to combined image embeddings
            transformer_outputs = self.transformer_decoder(
                tgt=ui_image_embeddings.permute(1, 0, 2),  # Target sequence (past actions)
                memory=combined_image_embeddings.permute(1, 0, 2),  # UI image embeddings until time i-1
                tgt_mask=time_mask.to(device=cad_image.device),
                memory_mask=time_mask.to(device=cad_image.device),
            )
        else:
            # raise ValueError("No past actions or past states provided")
            transformer_outputs = self.transformer_decoder(
                tgt=combined_image_embeddings.permute(1, 0, 2),  # Target sequence (past actions)
                memory=combined_image_embeddings.permute(1, 0, 2),  # UI image embeddings until time i-1
                tgt_mask=time_mask.to(device=cad_image.device),
                memory_mask=time_mask.to(device=cad_image.device),
            )
        sequence_hidden = transformer_outputs.permute(1, 0, 2)
        
        # Predict actions at time step i
        cmds = self.predict_action_class_0_4(sequence_hidden)
        params = self.predict_action_class_0_999(sequence_hidden).reshape(batch_size, seq_length, 6, 1000)
        
        return cmds, params

    def sequential_inference(self, ui_images, cad_image, action=False):
        """
        Performs sequential inference by predicting actions step-by-step.
        
        Args:
            ui_images (torch.Tensor): Input UI images of shape (batch_size, seq_length, channels, height, width).
            cad_image (torch.Tensor): CAD image tensor of shape (batch_size, channels, height, width).
            action (bool): Whether to use past actions during inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted commands (batch_size, seq_length, 5)
                                            and parameters (batch_size, seq_length, 6, 1000).
        """
        batch_size, seq_length = ui_images.shape[:2]
        device = ui_images.device

        # Initialize storage for predicted actions
        predicted_cmds = []
        predicted_params = []

        # Initialize actions tensor
        actions = torch.zeros(batch_size, 1, self.act_dim, device=device) if action else None

        for t in range(seq_length):
            # Prepare inputs dictionary
            inputs = {
                'frames': ui_images[:, :t+1],
                'actions': actions if action else torch.zeros(batch_size, t+1, 7, device=device),
                'timesteps': torch.arange(t+1, device=device),
                'cad_image': cad_image,
            }
            
            if action:
                cmd, params = self.forward(inputs)
                cmd_pred = torch.argmax(cmd[:, -1].clone(), dim=-1)
                param_pred = torch.argmax(params[:, -1].clone(), dim=-1)

                # Apply masking
                next_params = self.apply_action_mask(cmd_pred, param_pred).unsqueeze(1).float()
                next_action = torch.cat([cmd_pred.unsqueeze(1).unsqueeze(1), next_params], dim=2)
    
                # Append new action
                actions = torch.cat([actions, self.normalize_actions(next_action.float())], dim=1)
            else:
                cmd, params = self.forward(inputs)

            predicted_cmds.append(cmd[:, -1])
            predicted_params.append(params[:, -1])

        # Stack predictions along the time axis
        predicted_cmds = torch.stack(predicted_cmds, dim=1)  # (batch_size, seq_length, 5)
        predicted_params = torch.stack(predicted_params, dim=1)  # (batch_size, seq_length, 6, 1000)

        return predicted_cmds, predicted_params