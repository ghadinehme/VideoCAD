import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, vit_b_16
from torchvision import models
import timm
from vit_pytorch import ViT

def convert_bn_to_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups=num_groups)

class TrajectoryModel(nn.Module):

    def __init__(self, 
                 state_dim, 
                 act_dim, 
                 encoder, 
                 use_pretrained_cad_model,
                 max_length=None, 
                 **kwargs):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        # ResNet for processing images. Freeze the parameters.
        if state_dim > 0:
            self.state_embedding_model, self.state_embedding_model_size = self.create_embedding_model(encoder)
            for param in self.state_embedding_model.parameters():
                param.requires_grad = True
        else:
            self.state_embedding_model = None
            self.state_embedding_model_size = 0
        cad_model_type = "gencad" if use_pretrained_cad_model else encoder
        self.cad_embedding_model, self.cad_embedding_model_size = self.create_embedding_model(cad_model_type)
        
        


        
            
        for param in self.cad_embedding_model.parameters():
            # freeze the parameters if we are using a pretrained model
            param.requires_grad =  not use_pretrained_cad_model

    def create_embedding_model(self, model_type='vit', channels=1):
        if model_type == 'vit':
            model = ViT(
                image_size=224,
                patch_size=32,
                num_classes=1000,
                dim=512,
                depth=6,
                heads=16,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1,
                channels=channels
            )
            model.mlp_head = nn.Identity()
            embedding_size = 512
        elif model_type == 'resnet':  # resnet
            model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            convert_bn_to_gn(model)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            embedding_size = 512
        else:
            raise ValueError(f"Model type {model_type} not supported")
        return model, embedding_size

    def process_multiview_images(self, multiview_images, seq_length):
        # Reshape multiview images to process each view
        batch_size, num_views = multiview_images.shape[0], multiview_images.shape[1]
        multiview_reshaped = multiview_images.reshape(-1, *multiview_images.shape[2:])
        multiview_embeddings = self.process_image(multiview_reshaped)
        multiview_embeddings = multiview_embeddings.reshape(batch_size, num_views, -1)
        # Average embeddings across views
        # for some reason this is 512 as opposed to 1536
        multiview_embeddings = multiview_embeddings.unsqueeze(1).expand(-1, seq_length, -1, -1)
        multiview_embeddings = multiview_embeddings.reshape(batch_size, seq_length, -1)
        return multiview_embeddings
    

    def process_state(self, state):
        """Process a state through ResNet and create embeddings."""
        state_embeddings = self.state_embedding_model(state)
        state_embeddings = state_embeddings.squeeze(-1).squeeze(-1)
        return state_embeddings
    
    def process_image(self, image):
        """Process an image through ResNet and create embeddings."""
        image_embeddings = self.cad_embedding_model(image).reshape(-1, self.cad_embedding_model_size)
        image_embeddings = image_embeddings.squeeze(-1).squeeze(-1)
        return image_embeddings

    def forward(self, states, actions, timesteps, cad_image, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])