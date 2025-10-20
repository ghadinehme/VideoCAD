import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import datetime
import json
import time
from enum import Enum
import torch.distributed as dist
from model.model_factory import ModelType
import copy
from torch.utils.data import DistributedSampler
import csv
from torchvision.utils import save_image
import random
import torch.profiler # Import the profiler

TOLERANCE = 3

def get_current_datetime(formatted=True):
    now = datetime.datetime.now()
    if formatted:
        return now.strftime("%Y_%m_%d_%H_%M_%S")  # Format: YYYY-MM-DD HH:MM:SS
    return now

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
    

class MetricsHandler:
    def __init__(self, experiment_name, rank=0):
        self.metrics = {}
        self.experiment_name = experiment_name
        self.rank = rank
        self.is_master = rank == 0
        self.create_log_file()
    
    def print_metrics(self, metrics, mode=""):
        """Print evaluation metrics in a formatted way.
        
        Args:
            metrics (dict): Dictionary containing evaluation metrics
            mode (str): Mode identifier (e.g., "Validation", "Ablation")
        """
        # Only print from rank 0 in distributed training
        if not self.is_master:
            return

        # Calculate overall accuracy
        if 'total_predictions' in metrics and metrics['total_predictions'] > 0:
            accuracy = metrics['correct_predictions'] / metrics['total_predictions'] * 100 if 'correct_predictions' in metrics else 0
        else:
            accuracy = 0
        
        # Print main metrics
        print(f"{mode}: "
            f"CMD accuracy: {metrics['cmd_accuracy']:.2f}%, "
            f"Params accuracy: {metrics['params_accuracy']:.2f}%, "
            f"Overall: {accuracy:.2f}%, "
            f"Top-30 CMD accuracy: {metrics['cmd_accuracy_topk']:.2f}%, "
            f"Top-30 Params accuracy: {metrics['param_accuracy_topk']:.2f}%")
        
        if 'perfect_sequences' in metrics and 'total_sequences' in metrics:
            print(f"Perfect Command Accuracy: {metrics['perfect_command_accuracy']:.2f}% ({metrics['perfect_commands']}/{metrics['total_sequences']} sequences)")
            print(f"Perfect Sequence Accuracy: {metrics['perfect_sequence_accuracy']:.2f}% ({metrics['perfect_sequences']}/{metrics['total_sequences']} sequences)")
        
        # Print per-parameter accuracies
        print(f"Per-parameter accuracies{' ('+mode.lower()+')' if mode else ''}:")
        for i in range(6):
            if f'param_{i}_accuracy' in metrics:
                print(f"  Parameter {i}: {metrics[f'param_{i}_accuracy']:.2f}%")

            
    def create_log_file(self):
        if not self.is_master:
            return
        path_name = f"logs/{self.experiment_name}"
        if not os.path.exists(path_name):
            os.makedirs(path_name, exist_ok=True)

    def save_metrics(self, metrics, ext=""):
        if not self.is_master:
            return
        if ext:
            save_path = f'logs/{self.experiment_name}/{ext}.json'
        else:
            save_path = f'logs/{self.experiment_name}/{get_current_datetime()}.json'
        tmp = f"logs/{self.experiment_name}"
        if not os.path.exists(tmp):
            os.makedirs(tmp, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)


class CheckpointHandler:
    def __init__(self, experiment_name, rank=0, dir_name="checkpoints"):
        self.experiment_name = experiment_name
        self.rank = rank
        self.is_master = rank == 0
        self.checkpoint_dir = f'{dir_name}/{self.experiment_name}'
        if self.is_master:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def build_checkpoint(self, epoch, loss, model, optimizer):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        return checkpoint

    def save_checkpoint(self, epoch, loss, model, optimizer, is_best=False):
        if not self.is_master:
            return
        checkpoint = self.build_checkpoint(epoch, loss, model, optimizer)
        if is_best:
            torch.save(checkpoint, f'{self.checkpoint_dir}/best_model.pt')
            print(f"Saved best model checkpoint for epoch {epoch+1}")
        else:
            torch.save(checkpoint, f'{self.checkpoint_dir}/epoch_{epoch+1}.pt')

        print(f"Saved model checkpoint for epoch {epoch+1}")
        return checkpoint


class Metric:

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}: {self.value}"
        

class BaseTrainer:
    def __init__(self, 
                 train_packet, 
                 val_packet, 
                 test_packet,
                 model, 
                 training_config, 
                 device,
                 rank=0):
        self.device = device
        self.rank = rank
        self.is_master = rank == 0
        print(f"Trainer rank: {self.rank}, is master: {self.is_master}")
        self.training_config = training_config
        self.checkpoint = training_config.get('checkpoint', False)

        self.image_type = training_config.get('image_type', 'cad')

        # Early stopping parameters
        self.early_stopping_enabled = training_config.get('early_stopping_enabled', False)
        self.early_stopping_patience = training_config.get('early_stopping_patience', 100)
        self.early_stopping_min_delta = training_config.get('early_stopping_min_delta', 0.0)
        self.early_stopping_metric = training_config.get('early_stopping_metric', 'accuracy')
        self.early_stopping_mode = training_config.get('early_stopping_mode', 'max')  # 'min' for loss, 'max' for accuracy
        self.frozen = training_config.get('frozen', False)
        
        if "experiment_name" not in training_config:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"default_{timestamp}"
        else:
            self.experiment_name = training_config.get('experiment_name')
        
        self.metrics_handler = MetricsHandler(self.experiment_name, self.rank)
        self.checkpoint_handler = CheckpointHandler(
            self.experiment_name, self.rank, training_config.get('checkpoint_dir', "checkpoints"))

        self.train_loader = train_packet["loader"]
        self.val_loader = val_packet["loader"]
        self.test_loader = test_packet["loader"]
        self.train_sampler = train_packet["sampler"]
        self.val_sampler = val_packet["sampler"]
        self.test_sampler = test_packet["sampler"]
        self.model = model
        lr = training_config.get('lr', 1e-3)
        if self.frozen:
            # Define learning rates for each component
            lr_cad = training_config.get('lr_cad', 1e-3)
            lr_state = training_config.get('lr_state', 1e-3)

            # Create parameter groups
            param_groups = [
                {'params': self.model.cad_embedding_model.parameters(), 'lr': lr_cad},
                {'params': self.model.state_embedding_model.parameters(), 'lr': lr_state},
                {'params': [param for name, param in self.model.named_parameters() 
                            if 'cad_embedding_model' not in name and 'state_embedding_model' not in name], 
                'lr': lr}
            ]

            self.optimizer = torch.optim.Adam(param_groups)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.use_mse = training_config.get('use_mse', False)

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

    def save_checkpoint(self, epoch, loss, is_best=False):
        ckpt = self.checkpoint_handler.save_checkpoint(
            epoch, loss, self.model, self.optimizer, is_best)
        return ckpt
        

    def prepare_batch(self, batch):
        """
        Prepares a batch of data for the model.
        
        Args:
            batch (dict): Dictionary containing:
                - frames: Tensor of shape [batch_size, seq_length, channels, height, width]
                - actions: Tensor of shape [batch_size, seq_length, act_dim]
                - cad_image: Tensor of shape [batch_size, channels, height, width]
                - multiview_images: Optional tensor of shape [batch_size, num_views, channels, height, width]
                - timesteps: Optional tensor of shape [batch_size, seq_length]
                
        Returns:
            dict: Dictionary containing processed tensors
        """
        # Move all tensors to device and convert to float
        processed_batch = {
            'frames': batch['frames'].to(self.device, dtype=torch.float),
            'actions': batch['actions'].to(self.device, dtype=torch.float),
            'cad_image': batch['cad_image'].to(self.device, dtype=torch.float),
        }
        
        # Add timesteps if not present
        if 'timesteps' not in batch:
            batch_size = processed_batch['frames'].size(0)
            processed_batch['timesteps'] = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        else:
            processed_batch['timesteps'] = batch['timesteps'].to(self.device, dtype=torch.long)
            
        # Add multiview images if present
        if 'multiview_images' in batch and batch['multiview_images'] is not None:
            processed_batch['multiview_images'] = batch['multiview_images'].to(self.device, dtype=torch.float)
            
        return processed_batch
    
    def log(self, message):
        if self.is_master:
            print(message)

    def compute_loss(self, action_preds, actions):
        raise NotImplementedError("Subclasses must implement compute_loss")
    

    def log_metrics(self, epoch, epochs, batch_idx, loader_len, loss, **kwargs):
        raise NotImplementedError("Subclasses must implement log_metrics")

    def train(self, epochs, sequential=False, noise=False):
        """Main training loop."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Early stopping variables
        best_metric_value = float('inf') if self.early_stopping_mode == 'min' else float('-inf')
        best_model_state = None
        patience_counter = 0
        start_time = time.time()
        for epoch in tqdm(range(epochs)):
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            # Training phase
            avg_loss, metrics = self._train_epoch(epoch, noise)
            self.log_epoch_metrics(epoch, epochs, avg_loss, metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.training_config['save_frequency'] == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            # Validation phase
            val_metrics = self._run_validation(epoch)
            if dist.is_initialized():
                dist.barrier()   # all ranks wait until validation is done everywhere
            
            # Early stopping
            best_metric_value, patience_counter, best_model_state, should_stop = \
                self._handle_early_stopping(epoch, avg_loss, val_metrics, 
                                         best_metric_value, patience_counter, best_model_state)
            
            if should_stop:
                self.log(f"Early stopping triggered after {epoch+1} epochs")
                if best_model_state:
                    self.model.load_state_dict(best_model_state['model_state_dict'])
                    self.log(f"Loaded best model from epoch {best_model_state['epoch']}")
                break
            end_time = time.time()
            self.log(f"Epoch {epoch+1} took {end_time - start_time:.2f} seconds")
            start_time = time.time()
        # Load best model if early stopping didn't trigger
        if self.early_stopping_enabled and best_model_state and patience_counter < self.early_stopping_patience:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            self.log(f"Loaded best model from epoch {best_model_state['epoch']}")
        
        return self.model

    def _train_epoch(self, epoch, noise=False):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        metrics = self.init_metrics()
        loss_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        
        batch_timer = time.time()

        # Profiling configuration
        enable_profiling = self.training_config.get('enable_profiling', False)
        num_profile_warmup_steps = self.training_config.get('profile_warmup_steps', 5)
        num_profile_active_steps = self.training_config.get('profile_active_steps', 15)
        total_profiler_steps = num_profile_warmup_steps + num_profile_active_steps

        # Setup profiler if enabled
        prof = None
        if enable_profiling:
            profile_log_dir_base = f'./logs/{self.experiment_name}/profile_traces'
            profile_log_dir_rank = f'{profile_log_dir_base}/epoch{epoch}/rank{self.rank}'

            # Master rank creates the base directory
            if self.is_master:
                os.makedirs(f'{profile_log_dir_base}/epoch{epoch}', exist_ok=True)
            
            # Synchronize all ranks
            if dist.is_initialized():
                dist.barrier()

            # Each rank creates its own directory
            try:
                os.makedirs(profile_log_dir_rank, exist_ok=True)
            except OSError as e:
                print(f"Warning: Rank {self.rank} could not create profiler directory: {e}")
                profile_log_dir_rank = f'./profile_traces_epoch{epoch}_rank{self.rank}'
                os.makedirs(profile_log_dir_rank, exist_ok=True)

            # Initialize profiler
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU, 
                    torch.profiler.ProfilerActivity.CUDA
                ],
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=num_profile_warmup_steps,
                    active=num_profile_active_steps,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_log_dir_rank),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.__enter__()

        # Training loop
        for batch_idx, batch in tqdm(enumerate(self.train_loader), disable=not self.is_master):
            # Data loading timing
            time_data = time.time() - batch_timer
            data_time.update(time_data)
            batch_timer = time.time()
            
            # Process batch
            loss, batch_metrics = self._process_batch(batch, noise)
            
            running_loss += loss.item()
            self.update_metrics(metrics, batch_metrics)
            
            # Loss timing and logging
            time_loss = time.time() - batch_timer 
            loss_time.update(time_loss)
            if (batch_idx + 1) % 2 == 0:
                self._log_batch_metrics(epoch, batch_idx, loss.item(), metrics, data_time, loss_time)
            
            batch_timer = time.time()

            # Step profiler if enabled
            if enable_profiling and prof and batch_idx < total_profiler_steps:
                prof.step()
        
        # Clean up profiler
        if enable_profiling and prof:
            prof.__exit__(None, None, None)
            if self.is_master:
                print(f"Profiler trace for epoch {epoch} (rank {self.rank}) saved to: {profile_log_dir_rank}")
                print(f"To view traces, start TensorBoard: tensorboard --logdir {profile_log_dir_base}")
        
        # Calculate average loss
        if len(self.train_loader) > 0:
            avg_epoch_loss = running_loss / len(self.train_loader)
        else:
            avg_epoch_loss = 0.0
        return avg_epoch_loss, metrics

    def _process_batch(self, batch, noise=False):
        """Process a single batch and compute loss."""
        self.optimizer.zero_grad()
        batch_dict = self.prepare_batch(batch)
        
        if noise:
            batch_dict['actions'] = self._add_noise_to_actions(batch_dict['actions'])
        
        model_inputs = self._prepare_model_inputs(batch_dict, noise)
        action_preds = self.model(model_inputs)
        loss, batch_metrics = self.compute_loss(action_preds, batch_dict['actions'][:, 1:])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss, batch_metrics

    def _add_noise_to_actions(self, actions):
        """Add noise to actions based on command type."""
        noise_actions = actions.clone()
        cmd_0 = (actions[:, :, 0] == 0).unsqueeze(-1)
        cmd_3 = (actions[:, :, 0] == 3).unsqueeze(-1)
        noise_actions[:, :, 1:3] += torch.randint_like(noise_actions[:, :, 1:3], -2, 3) * cmd_0
        noise_actions[:, :, -1:] += torch.randint_like(noise_actions[:, :, -1:], -2, 3) * cmd_3
        return noise_actions

    def _prepare_model_inputs(self, batch_dict, noise):
        """Prepare inputs for the model."""
        model_inputs = {
            'frames': batch_dict['frames'][:, :-1],
            'actions': self.normalize_actions(batch_dict['actions'][:, :-1]),
            'timesteps': batch_dict['timesteps'],
            'cad_image': batch_dict['cad_image'],
        }
        if 'multiview_images' in batch_dict:
            model_inputs['multiview_images'] = batch_dict['multiview_images']
        return model_inputs

    def _log_batch_metrics(self, epoch, batch_idx, loss, metrics, data_time, loss_time):
        """Log metrics for a batch."""
        self.log_metrics(epoch, self.training_config['epochs'], batch_idx, 
                        len(self.train_loader), loss, metrics=metrics)
        self.save_metrics(metrics, ext=f"epoch_{epoch+1}")
        self.log(f"Average loss time: {loss_time.avg:.4f} seconds")
        self.log(f"Average load time: {data_time.avg:.4f} seconds")

    def _run_validation(self, epoch):
        """Run validation and handle early stopping."""
        val_metrics = None
        if (epoch + 1) % self.training_config.get('seq_val_frequency', 30) == 0 and self.training_config.get('sequential', False):
            self.log("Evaluating sequential model")
            val_metrics = self.sequential_evaluate(self.model, mode="val_seq")
            self.print_metrics(val_metrics, mode="Validation Seq")
        
        if (epoch + 1) % self.training_config['val_frequency'] == 0:
            #val_metrics = self.evaluate(self.model, ablation=True)
            #self.print_metrics(val_metrics, mode="Ablation")
            val_metrics = self.evaluate(self.model, mode="val", epoch=epoch)
            self.print_metrics(val_metrics, mode="Validation")
        return val_metrics

    def _handle_early_stopping(self, epoch, avg_loss, val_metrics, best_metric_value, patience_counter, best_model_state):
        """Handle early stopping logic."""
        if not self.early_stopping_enabled:
            return best_metric_value, patience_counter, best_model_state, False

        current_metric = self._get_current_metric(avg_loss, val_metrics)
        improved = self._check_metric_improvement(current_metric, best_metric_value)
        
        if improved:
            self.log(f"Validation {self.early_stopping_metric} improved from {best_metric_value:.4f} to {current_metric:.4f}")
            best_metric_value = current_metric
            patience_counter = 0
            best_model_state = self.save_checkpoint(epoch, avg_loss, is_best=True)
            self.log(f"Saved best model checkpoint at epoch {epoch+1}")
        else:
            patience_counter += 1
            self.log(f"Validation {self.early_stopping_metric} did not improve. Patience: {patience_counter}/{self.early_stopping_patience}")
        should_stop_local = patience_counter >= self.early_stopping_patience
        if dist.is_initialized():
            flag = torch.tensor([int(should_stop_local)], device=self.device)
            dist.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            should_stop = bool(flag.item())        # every rank now has the same answer
        else:
            should_stop = should_stop_local

        return best_metric_value, patience_counter, best_model_state, should_stop 

    def _get_current_metric(self, avg_loss, val_metrics):
        """Get the current metric value for early stopping."""
        if self.early_stopping_metric == 'loss':
            return avg_loss
        elif self.early_stopping_metric == 'accuracy' and 'correct_predictions' in val_metrics:
            return val_metrics['correct_predictions'] / val_metrics['total_predictions']
        return avg_loss

    def _check_metric_improvement(self, current_metric, best_metric_value):
        """Check if the metric has improved."""
        if self.early_stopping_mode == 'min':
            return current_metric < best_metric_value - self.early_stopping_min_delta
        return current_metric > best_metric_value + self.early_stopping_min_delta

    

    def print_metrics(self, metrics, mode=""):
        self.metrics_handler.print_metrics(metrics, mode)
        
    
    def save_metrics(self, metrics, ext=""):
        self.metrics_handler.save_metrics(metrics, ext)


    def init_metrics(self):
        return {}

    def update_metrics(self, metrics, batch_metrics):
        pass

    def log_epoch_metrics(self, epoch, epochs, avg_loss, metrics):
        pass


    
    def generate_saliency_batch(self, dataloader, target_class=None):
        """
        Computes saliency maps on one batch from the given dataloader.

        Args:
            dataloader (DataLoader): PyTorch dataloader (e.g. self.val_loader).
            target_class (int or None): If set, uses this class index for saliency.
                                        Otherwise, uses model prediction.

        Returns:
            cad_images (torch.Tensor): Original CAD images, shape (B, C, H, W)
            saliency_maps (torch.Tensor): Saliency maps, shape (B, H, W)
        """
        self.model.eval()
        batch = next(iter(dataloader))
        input_batch = self.prepare_batch(batch)

        cad_image = input_batch['cad_image'].clone().detach().requires_grad_(True)
        frames = input_batch['frames'][:, :1]
        actions = self.normalize_actions(input_batch['actions'][:, :1])
        timesteps = input_batch['timesteps']

        model_inputs = {
            'frames': frames,
            'actions': actions,
            'timesteps': timesteps,
            'cad_image': cad_image
        }

        if 'multiview_images' in input_batch:
            model_inputs['multiview_images'] = input_batch['multiview_images'][:, :1]

        output = self.model(model_inputs)
        cmd_logits = output[0][:, 0]  # shape (B, num_classes)

        if target_class is None:
            target_class = torch.argmax(cmd_logits, dim=1)

        selected_logits = cmd_logits[torch.arange(len(cmd_logits)), target_class]
        selected_logits.sum().backward()

        saliency = cad_image.grad.abs()
        saliency_maps = saliency.max(dim=1)[0]  # shape (B, H, W)

        return cad_image.detach(), saliency_maps

    def compute_attention_rollout(self, input_batch, discard_ratio=0.0):
        """
        Computes Attention Rollout for self.cad_embedding_model (ViT-based encoder).

        Args:
            input_batch (dict): A batch of data prepared using self.prepare_batch.
            discard_ratio (float): Proportion of lowest attention weights to discard per layer.

        Returns:
            torch.Tensor: Attention rollout heatmaps of shape (B, H, W)
        """
        self.model.eval()
        attn_weights = []

        # Hook to capture attention output
        def hook_fn(module, input, output):
            # output shape: (B, heads, N, N)
            attn_weights.append(output.detach().cpu())

        # Register hooks on dropout layers in each Attention block
        hooks = []
        for block in self.model.cad_embedding_model.transformer.layers:
            attention_module = block[0]  # block[0] is Attention
            h = attention_module.dropout.register_forward_hook(hook_fn)
            hooks.append(h)

        # Forward pass through cad_embedding_model to trigger hooks
        cad_image = input_batch['cad_image'].clone().detach().to(self.device)
        with torch.no_grad():
            _ = self.model.cad_embedding_model(cad_image)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Stack and average attention heads
        attn_mat = torch.stack(attn_weights)  # (L, B, heads, N, N)
        attn_mat = attn_mat.mean(dim=2)       # (L, B, N, N)

        # Add residual and normalize
        N = attn_mat.size(-1)
        eye = torch.eye(N).unsqueeze(0).unsqueeze(0).expand_as(attn_mat)
        attn_mat = attn_mat + eye
        attn_mat = attn_mat / attn_mat.sum(dim=-1, keepdim=True)

        # Rollout across layers
        joint_attn = attn_mat[0]
        for i in range(1, attn_mat.size(0)):
            joint_attn = attn_mat[i].bmm(joint_attn)

        # Take attention from class token to all patches
        mask = joint_attn[:, 0, 1:]  # (B, N-1)

        # Convert to spatial attention map
        num_patches = mask.size(1)
        patch_dim = int(num_patches ** 0.5)
        mask = mask.view(-1, 1, patch_dim, patch_dim)  # (B, 1, patch_h, patch_w)
        mask = torch.nn.functional.interpolate(mask, size=(224, 224), mode='bilinear', align_corners=False)

        return mask.squeeze(1)  # (B, H, W)
    
                
    
    def evaluate(self, model, mode="test", ablation=False, epoch=-1):
            
        model.eval()
        metrics = self.init_metrics()
        if mode == "train":
            loader = self.train_loader
        elif mode == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(loader), disable=not self.is_master):
                batch_dict = self.prepare_batch(batch)
                
                # Prepare inputs for model
                model_inputs = {
                    'frames': batch_dict['frames'][:, :-1],
                    'actions': self.normalize_actions(batch_dict['actions'].clone())[:, :-1],
                    'timesteps': batch_dict['timesteps'],
                    'cad_image': batch_dict['cad_image'],
                }
                
                # Add multiview images if present
                if 'multiview_images' in batch_dict:
                    model_inputs['multiview_images'] = batch_dict['multiview_images']
                
                if ablation:
                    model_inputs['cad_image'] = torch.zeros_like(model_inputs['cad_image'])
                
                action_preds = model(model_inputs)
                loss, batch_metrics = self.compute_loss(action_preds, batch_dict['actions'][:, 1:])
                self.update_metrics(metrics, batch_metrics)

        if epoch != -1:
            mode = f"{mode}_epoch_{epoch+1}"
        if self.is_master:
            self.save_metrics(metrics, mode)
        return metrics
    
    def sequential_evaluate(self, model, mode="test", ablation=False):
        """Run sequential evaluation across all processes."""
        model.eval()
        metrics = self.init_metrics()
        if mode == "train_seq":
            loader = self.train_loader 
        elif mode == "val_seq":
            loader = self.val_loader
        else:
            loader = self.test_loader
        with torch.no_grad():
            for batch in loader:
                batch_dict = self.prepare_batch(batch)
                
                if ablation:
                    batch_dict['cad_image'] = torch.zeros_like(batch_dict['cad_image'])
                
                action_preds = self.sequential_inference(batch_dict)
                loss, batch_metrics = self.compute_loss(action_preds, batch_dict['actions'][:, 1:])
                self.update_metrics(metrics, batch_metrics)

        # Synchronize metrics across all processes
        if dist.is_initialized():
            # Convert all metrics to tensors at once
            metric_tensors = {}
            for key in metrics:
                if isinstance(metrics[key], (int, float)):
                    metric_tensors[key] = torch.tensor(metrics[key], dtype=torch.float32, device=self.device)
            
            # Perform a single all_reduce operation for all metrics
            handles = []
            for key, tensor in metric_tensors.items():
                handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
                handles.append(handle)
            
            # Wait for all operations to complete
            for handle in handles:
                handle.wait()
            
            # Update metrics with synchronized values
            for key, tensor in metric_tensors.items():
                metrics[key] = tensor.item()

        # Only save metrics on master process
        if self.is_master:
            self.save_metrics(metrics, mode)
        return metrics
    
    def normalize_actions(self, actions):
        actions = actions.clone()
        actions[:, :, 0] = actions[:, :, 0]/4.0
        actions[:, :, 1:] = actions[:, :, 1:]/1000.0
        return actions
    


class MultiClassesTrainer(BaseTrainer):
    def __init__(self, 
            train_loader, 
            val_loader, 
            test_loader, 
            model, 
            training_config, 
            device,
            rank):
        super().__init__(train_loader, val_loader, test_loader, model, training_config, device, 
        rank=rank)
        
        self.rank = rank
        # Load class weights
        with open("class_weights.json", "r") as f:
            weight_data = json.load(f)
        
        self.param_to_label = [0, 0, 1, 1, 2, 3]

        self.tolerances = [TOLERANCE-1, TOLERANCE-1, 50, 200, 500, TOLERANCE-1]

        self.above = [False, False, True, True, True, False]

        self.cmd_weights = weight_data["Label"]
        self.weights = weight_data

        self.param_names = ["Label", "x", "y", "Key Pressed", "Times Key Pressed", "Scroll Amount", "Typed Value"]

        # Setup weighted CE losses for each category
        self.loss_fns = {
            key: nn.CrossEntropyLoss(
                ignore_index=-1,
                weight=torch.tensor(weight_data[key], dtype=torch.float32).to(device),
            )
            for key in self.param_names
        }

        self.cmd_loss_fn = self.loss_fns["Label"]

        # Map parameter index to weight key
        self.param_loss_map = {i : self.param_names[i+1] for i in range(6)} 

        self.mse_loss = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    def flexible_cross_entropy(self, logits, targets, num_classes, weights = None, tolerance=2, ignore_index=-1, above=False, ignore_valid=True):
        """
        Args:
            logits: Tensor of shape (batch, num_classes) or (batch, seq_len, num_classes)
            targets: Tensor of shape (batch,) or (batch, seq_len)
            num_classes: Number of output classes
            tolerance: Allowed distance from the target class
            ignore_index: Index to ignore
            above: If True, only allow classes >= target (up to target + tolerance)
            ignore_valid: If True, only apply loss to predictions outside the tolerance window
        """
        logits = logits.reshape(-1, num_classes)
        targets = targets.reshape(-1)

        mask = targets != ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute predicted classes
        preds = torch.argmax(logits, dim=1)

        # Define allowed targets for each sample
        if above:
            allowed_targets = torch.stack([
                torch.clamp(targets + offset, 0, num_classes - 1)
                for offset in range(tolerance)
            ], dim=1)
        else:
            allowed_targets = torch.stack([
                torch.clamp(targets + offset, 0, num_classes - 1)
                for offset in range(-tolerance, tolerance + 1)
            ], dim=1)

        # Determine which predictions are invalid (outside tolerance)
        is_valid = (allowed_targets == preds.unsqueeze(1)).any(dim=1)
        if ignore_valid:
            logits = logits[~is_valid]
            targets = targets[~is_valid]

        if logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Build soft targets
        soft_targets = torch.zeros_like(logits).float()
        if above:
            for offset in range(tolerance):
                idx = torch.clamp(targets + offset, 0, num_classes - 1)
                soft_targets[torch.arange(len(idx)), idx] = 1.0
        else:
            for offset in range(-tolerance, tolerance + 1):
                idx = torch.clamp(targets + offset, 0, num_classes - 1)
                soft_targets[torch.arange(len(idx)), idx] = 1.0

        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        if weights is not None and weights.size(0) == num_classes:
            # weight for each class
            weighted_log_probs = log_probs * weights[targets].unsqueeze(1)
            loss = -(soft_targets * weighted_log_probs).sum(dim=1).mean()
        else:
            loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss
    
    def _count_correct_params(self, params_predicted, actions_params, params_mask, i):
        if self.above[i]:
            diff = params_predicted[..., i:i+1][params_mask[..., i:i+1]] - actions_params[..., i:i+1][params_mask[..., i:i+1]]
            params_correct = ((diff >= 0) & (diff < self.tolerances[i])).sum().item()
        else:
            diff = torch.abs(params_predicted[..., i:i+1][params_mask[..., i:i+1]] - actions_params[..., i:i+1][params_mask[..., i:i+1]])
            params_correct = (diff  < TOLERANCE).sum().item()
        return params_correct
    
    def _check_perfect_sequence(self, pred_action, action, seq_mask, i):
        if self.above[i]:
            return torch.all(pred_action[:, i:i+1][seq_mask[:, i:i+1]] - action[:, i:i+1][seq_mask[:, i:i+1]] < self.tolerances[i]) and \
                   torch.all(pred_action[:, i:i+1][seq_mask[:, i:i+1]] - action[:, i:i+1][seq_mask[:, i:i+1]] >= 0)
        else:
            return torch.all(torch.abs(pred_action[:, i:i+1][seq_mask[:, i:i+1]] - action[:, i:i+1][seq_mask[:, i:i+1]]) < TOLERANCE)

    def compute_loss(self, action_preds, actions, mse=True):
        actions = actions.long()
        pred_cmd, pred_params = action_preds

        actions_cmd = actions[..., 0]
        actions_params = actions[..., 1:]

        # Command classification loss
        loss_cmd = self.cmd_loss_fn(pred_cmd.reshape(-1, 5), actions_cmd.reshape(-1))

        loss_params = 0
        num_params = 6
        for i in range(num_params):
            pred_i = pred_params[..., i, :].reshape(-1, 1000)
            target_i = actions_params[..., i].reshape(-1)

            # Select corresponding loss function
            loss_key = self.param_loss_map[i]
            if self.use_mse:
                loss_p = self.flexible_cross_entropy(pred_i, target_i, num_classes=1000, 
                                                    tolerance=self.tolerances[i], ignore_index=-1, 
                                                    above=self.above, ignore_valid=True)
            else:
                loss_fn = self.loss_fns[loss_key]
                loss_p = loss_fn(pred_i, target_i)
            # if not nan increment
            if not torch.isnan(loss_p):
                loss_params += loss_p * self.cmd_weights[self.param_to_label[i]]

        # loss_params = self.loss_fn(pred_params.reshape(-1, 1000), actions_params.reshape(-1))

        loss = 2 * loss_cmd + loss_params

        # Get predicted actions
        cmd_predicted = torch.argmax(pred_cmd, dim=-1)
        params_predicted = torch.argmax(pred_params, dim=-1)
        pred_actions = torch.cat((cmd_predicted.unsqueeze(-1), params_predicted), dim=-1)
        
        # Compute commands correct predictions
        cmd_mask = actions_cmd != -1
        cmd_correct = (cmd_predicted[cmd_mask] == actions_cmd[cmd_mask])

        cmd_corrects = []
        cmd_counts = []
        for i in range(5):
            cmd_mask_i = actions_cmd == i
            cmd_correct_i = (cmd_predicted[cmd_mask_i] == actions_cmd[cmd_mask_i])
            cmd_corrects.append(cmd_correct_i.sum().item())
            cmd_counts.append(cmd_mask_i.sum().item())

        # Compute parameters correct predictions
        param_mask = cmd_mask.unsqueeze(-1).expand_as(actions_params) * (actions_params != -1)
        params_mask = cmd_mask.unsqueeze(-1).expand_as(actions_params) * (actions_params != -1) * (cmd_predicted == actions_cmd).unsqueeze(-1).expand_as(actions_params)

        param_corrects = []
        param_counts = []

        params_correct_all = 0
        for i in range(6):
            if self.use_mse:
                params_correct = self._count_correct_params(params_predicted, actions_params, params_mask, i)
            else:
                params_correct = (abs(params_predicted[..., i:i+1][params_mask[..., i:i+1]] - actions_params[..., i:i+1][params_mask[..., i:i+1]])<TOLERANCE).sum().item()
            params_correct_all += params_correct
            param_corrects.append(params_correct)
            param_counts.append(param_mask[..., i].sum().item())

        correct = cmd_correct.sum().item() + params_correct_all
        total = cmd_mask.sum().item() + param_mask.sum().item()

        # Compute top k correct predictions
        k = 30
        if self.use_mse:
            cmd_correct_topk = (cmd_predicted[:, :k][cmd_mask[:, :k]] == actions_cmd[:, :k][cmd_mask[:, :k]]).sum().item()
            params_correct_topk = 0
            for i in range(6):
                params_correct = self._count_correct_params(params_predicted[:, :k], actions_params[:, :k], params_mask[:, :k], i)
                params_correct_topk += params_correct
        else:
            cmd_correct_topk = (cmd_predicted[:, :k][cmd_mask[:, :k]] == actions_cmd[:, :k][cmd_mask[:, :k]]).sum().item()
            params_correct_topk = (params_predicted[:, :k][params_mask[:, :k]] == actions_params[:, :k][params_mask[:, :k]]).sum().item()

        # Compute perfect sequences
        perfect_sequences = 0
        perfect_commands = 0
        total_sequences = 0
        """
        for i in range(actions.shape[0]):
            seq_mask = actions[i] != -1
            total_sequences += 1
            if self.use_mse:
                if torch.all(torch.tensor([self._check_perfect_sequence(pred_actions[i], actions[i], seq_mask, j) for j in range(6)])):
                    perfect_sequences += 1
                if torch.all(pred_actions[i][:, 0][seq_mask[:, 0]] == actions[i][:, 0][seq_mask[:, 0]]):
                    perfect_commands += 1
            else:
                if torch.all(pred_actions[i][seq_mask] == actions[i][seq_mask]):
                    perfect_sequences += 1
                if torch.all(pred_actions[i][:, 0][seq_mask[:, 0]] == actions[i][:, 0][seq_mask[:, 0]]):
                    perfect_commands += 1
        """

        #perfect_sequence_accuracy = 100 * perfect_sequences / total_sequences if total_sequences > 0 else -1
        perfect_sequence_accuracy = 0
        metrics = {
            'correct_predictions': correct,
            'total_predictions': total,
            'cmd_corrects': cmd_corrects,
            'cmd_counts': cmd_counts,
            'param_corrects': param_corrects,
            'param_counts': param_counts,
            'cmd_correct_topk': cmd_correct_topk,
            'cmd_counts_topk': cmd_mask[:, :k].sum().item(),
            'param_correct_topk': params_correct_topk,
            'param_counts_topk': param_mask[:, :k].sum().item(),
            'perfect_sequences': perfect_sequences,
            'perfect_commands': perfect_commands,
            'total_sequences': total_sequences,
            'perfect_sequence_accuracy': perfect_sequence_accuracy,
        }

        for i in range(6):
            metrics[f'param_corrects_{i}'] = param_corrects[i]
            metrics[f'param_counts_{i}'] = param_counts[i]
        for i in range(5):
            metrics[f'cmd_corrects_{i}'] = cmd_corrects[i]
            metrics[f'cmd_counts_{i}'] = cmd_counts[i]

        return loss, metrics
    

    def sample(self, model, n=10, folder="outputs", mode="test", ablation=False):
        model.eval()
        
        if mode == "train":
            loader = self.train_loader
        elif mode == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader
        dataset = loader.dataset
        indices = random.sample(range(len(dataset)), n)
        with torch.no_grad():
            for idx in tqdm(indices):
                sample = dataset[idx]
                # print(sample)
                sample_id = os.path.basename(dataset.data_files[idx]).split("_")[0]
                save_path = os.path.join(folder, f"pred_actions_{sample_id}.csv")
                if os.path.exists(save_path):
                    continue
                batch_dict = self.prepare_batch(sample)
                if ablation:
                    batch_dict['cad_image'] = torch.zeros_like(batch_dict['cad_image'])

                model_inputs = {
                    'frames': batch_dict['frames'].unsqueeze(0)[:, :-1],
                    'actions': self.normalize_actions(batch_dict['actions'].unsqueeze(0).clone())[:, :-1],
                    'timesteps': batch_dict['timesteps'].unsqueeze(0),
                    'cad_image': batch_dict['cad_image'].unsqueeze(0),
                }

                

                action_preds = model(model_inputs)
                pred_cmd, pred_params = action_preds
                pred_cmd = torch.argmax(pred_cmd, dim=-1)
                pred_params = torch.argmax(pred_params, dim=-1)

                pred_params = self.apply_action_mask(pred_cmd, pred_params).long()
                pred_cmd = pred_cmd.long()
                pred_action = torch.cat((pred_cmd.unsqueeze(-1), pred_params), dim=-1)

                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)

                # Save actions to CSV
                actions_path = os.path.join(folder, f"pred_actions_{sample_id}.csv")
                with open(actions_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    # writer.writerow(["cmd"] + [f"param_{i}" for i in range(pred_params.shape[-1])])
                    for action in pred_action[0].cpu().numpy():
                        writer.writerow(action.tolist())

                actions_path = os.path.join(folder, f"actions_{sample_id}.csv")
                with open(actions_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    # writer.writerow(["cmd"] + [f"param_{i}" for i in range(pred_params.shape[-1])])
                    for action in batch_dict['actions'][1:].cpu().numpy():
                        writer.writerow(action.tolist())

                # Optionally save image (first frame)
                images_path = os.path.join(folder, f"images_{sample_id}.png")
                save_image(batch_dict['cad_image'][0], images_path)



    
    def _init_mistake_data(self, tol):
        """Initialize the data structure for tracking mistakes."""
        return [{
            "First Mistakes": {
                f"cmd_{i}": [] for i in range(5)
            } | {
                f"param_{i}": [] for i in range(6)
            },
            "Memory": {
                "cmd": [],
                **{f"param_{i}": [] for i in range(6)}
            },
            "Sequence Lengths": [],
            "Number of Mistakes": []
        } for _ in range(tol)]

    def _check_parameter_error(self, diff, param_idx, tolerance):
        """Check if a parameter prediction is within acceptable error bounds."""
        if param_idx in [0, 1, 5]:
            return abs(diff) > tolerance
        elif param_idx == 2:
            return diff < 0 or diff >= 50
        elif param_idx == 3:
            return diff < 0 or diff >= 200
        elif param_idx == 4:
            return diff < 0 or diff >= 500
        return False

    def _process_sequence_mistakes(self, actions_cmd, actions_params, pred_cmd, pred_params, tolerance):
        """Process mistakes for a single sequence."""
        mistakes = [0] * len(actions_cmd)
        first_mistake = False
        noted = False
        sequence_data = {
            "First Mistakes": {f"cmd_{i}": [] for i in range(5)} | {f"param_{i}": [] for i in range(6)},
            "Memory": {"cmd": [], **{f"param_{i}": [] for i in range(6)}},
            "Sequence Lengths": [],
            "Number of Mistakes": []
        }

        for j in range(len(actions_cmd)):
            any_mistake = False
            gt_cmd = actions_cmd[j].item()
            pd_cmd = pred_cmd[j].item()
            sequence_data["Memory"]["cmd"].append([gt_cmd, pd_cmd])

            # Check command mistakes
            if gt_cmd != pd_cmd:
                mistakes[j] = 1
                any_mistake = True
                if not first_mistake:
                    sequence_data["First Mistakes"][f"cmd_{gt_cmd}"].append(f"cmd_{pd_cmd}")
                    first_mistake = True

            # Check parameter mistakes
            for k in range(actions_params.size(-1)):
                gt_param = actions_params[j][k].item()
                if gt_param == -1:
                    continue

                pd_param = pred_params[j][k].item()
                sequence_data["Memory"][f"param_{k}"].append([gt_param, pd_param])

                diff = pd_param - gt_param
                if self._check_parameter_error(diff, k, tolerance) and not any_mistake:
                    mistakes[j] = 1
                    any_mistake = True

                if self._check_parameter_error(diff, k, tolerance) and not first_mistake:
                    sequence_data["First Mistakes"][f"param_{k}"].append(f"param_{pd_param}")
                    first_mistake = True

            if first_mistake and not noted:
                sequence_data["Sequence Lengths"] = [j, len(actions_cmd)]
                noted = True

        if not noted:
            sequence_data["Sequence Lengths"] = [len(actions_cmd), len(actions_cmd)]
        
        sequence_data["Number of Mistakes"] = mistakes
        return sequence_data

    def find_first_mistake(self, model, mode="test", tol=3, ablation=False):
        """Find the first mistake in each sequence for different tolerance levels."""
        model.eval()
        data = self._init_mistake_data(tol)

        loader = self.train_loader if mode == "train" else self.val_loader if mode == "val" else self.test_loader

        with torch.no_grad():
            for batch in tqdm(loader):
                batch_dict = self.prepare_batch(batch)
                if ablation:
                    batch_dict['cad_image'] = torch.zeros_like(batch_dict['cad_image'])

                model_inputs = {
                    'frames': batch_dict['frames'][:, :-1],
                    'actions': self.normalize_actions(batch_dict['actions'].clone())[:, :-1],
                    'timesteps': batch_dict['timesteps'],
                    'cad_image': batch_dict['cad_image'],
                }

                action_preds = model(model_inputs)
                pred_cmd, pred_params = action_preds
                pred_cmd = torch.argmax(pred_cmd, dim=-1)
                pred_params = torch.argmax(pred_params, dim=-1)

                actions_cmd = batch_dict['actions'][:, 1:, 0].long()
                actions_params = batch_dict['actions'][:, 1:, 1:].long()
                pred_params = self.apply_action_mask(pred_cmd, pred_params).long()
                pred_cmd = pred_cmd.long()

                for t in range(tol):
                    for i in range(len(actions_cmd)):
                        sequence_data = self._process_sequence_mistakes(
                            actions_cmd[i], actions_params[i], 
                            pred_cmd[i], pred_params[i], 
                            t
                        )
                        
                        # Merge sequence data into main data structure
                        for key in sequence_data["First Mistakes"]:
                            data[t]["First Mistakes"][key].extend(sequence_data["First Mistakes"][key])
                        for key in sequence_data["Memory"]:
                            data[t]["Memory"][key].extend(sequence_data["Memory"][key])
                        data[t]["Sequence Lengths"].append(sequence_data["Sequence Lengths"])
                        data[t]["Number of Mistakes"].append(sequence_data["Number of Mistakes"])

        return data




    
    def init_metrics(self):
        metrics =  {'correct_predictions': 0, 'total_predictions': 0,
                 'cmd_accuracy': 0, 'params_accuracy': 0,
                 'cmd_corrects': 0, 'cmd_counts': 0,
                 'param_corrects': 0, 'param_counts': 0,
                 'cmd_correct_topk': 0, 'param_correct_topk': 0,
                 'cmd_counts_topk': 0, 'param_counts_topk': 0,
                 'cmd_accuracy_topk': 0, 'param_accuracy_topk': 0,
                 'perfect_sequences': 0, 'total_sequences': 0,
                 'perfect_commands': 0, 'perfect_command_accuracy': 0,
                 'perfect_sequence_accuracy': 0}
        for i in range(6):
            metrics[f'param_accuracy_{i}'] = 0
            metrics[f'param_corrects_{i}'] = 0
            metrics[f'param_counts_{i}'] = 0
        for i in range(5):
            metrics[f'cmd_accuracy_{i}'] = 0
            metrics[f'cmd_corrects_{i}'] = 0
            metrics[f'cmd_counts_{i}'] = 0
        return metrics

    def update_metrics(self, metrics, batch_metrics):
        # Update raw counts
        metrics['cmd_correct_topk'] += batch_metrics['cmd_correct_topk']
        metrics['param_correct_topk'] += batch_metrics['param_correct_topk']
        metrics['cmd_counts_topk'] += batch_metrics['cmd_counts_topk']
        metrics['param_counts_topk'] += batch_metrics['param_counts_topk']
        
        # Update per-parameter metrics
        for i in range(6):
            metrics[f'param_corrects_{i}'] += batch_metrics[f'param_corrects_{i}']
            metrics[f'param_counts_{i}'] += batch_metrics[f'param_counts_{i}']
        
        # Update per-command metrics
        for i in range(5):
            metrics[f'cmd_corrects_{i}'] += batch_metrics[f'cmd_corrects_{i}']
            metrics[f'cmd_counts_{i}'] += batch_metrics[f'cmd_counts_{i}']
        
        # Update overall metrics
        metrics['correct_predictions'] += batch_metrics['correct_predictions']
        metrics['total_predictions'] += batch_metrics['total_predictions']
        metrics['perfect_sequences'] += batch_metrics['perfect_sequences']
        metrics['perfect_commands'] += batch_metrics['perfect_commands']
        metrics['total_sequences'] += batch_metrics['total_sequences']
        

        # Calculate top-k accuracies
        if metrics['cmd_counts_topk'] > 0:
            metrics['cmd_accuracy_topk'] = 100 * metrics['cmd_correct_topk'] / metrics['cmd_counts_topk']
        if metrics['param_counts_topk'] > 0:
            metrics['param_accuracy_topk'] = 100 * metrics['param_correct_topk'] / metrics['param_counts_topk']
        
        # Calculate per-parameter accuracies
        for i in range(6):
            if metrics[f'param_counts_{i}'] > 0:
                metrics[f'param_accuracy_{i}'] = 100 * metrics[f'param_corrects_{i}'] / metrics[f'param_counts_{i}']
        
        # Calculate per-command accuracies
        for i in range(5):
            if metrics[f'cmd_counts_{i}'] > 0:
                metrics[f'cmd_accuracy_{i}'] = 100 * metrics[f'cmd_corrects_{i}'] / metrics[f'cmd_counts_{i}']
        
        # Calculate overall accuracies
        total_cmd_counts = sum(metrics[f'cmd_counts_{i}'] for i in range(5))
        total_param_counts = sum(metrics[f'param_counts_{i}'] for i in range(6))
        
        if total_cmd_counts > 0:
            metrics['cmd_accuracy'] = 100 * sum(metrics[f'cmd_corrects_{i}'] for i in range(5)) / total_cmd_counts
        if total_param_counts > 0:
            metrics['params_accuracy'] = 100 * sum(metrics[f'param_corrects_{i}'] for i in range(6)) / total_param_counts
        if metrics['total_predictions'] > 0:
            metrics['overall_accuracy'] = 100 * metrics['correct_predictions'] / metrics['total_predictions']
        if metrics['total_sequences'] > 0:
            metrics['perfect_sequence_accuracy'] = 100 * metrics['perfect_sequences'] / metrics['total_sequences']
            metrics['perfect_command_accuracy'] = 100 * metrics['perfect_commands'] / metrics['total_sequences']

    def log_metrics(self, epoch, epochs, batch_idx, loader_len, loss, **kwargs):
        metrics = kwargs.get('metrics', {})
        cmd_accuracy = metrics['cmd_accuracy']
        params_accuracy = metrics['params_accuracy']

        self.save_metrics(metrics, ext=f"epoch_{epoch+1}")
        if not self.is_master:
            return

        print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{loader_len}], "
              f"Loss: {loss:.4f}, CMD Accuracy: {cmd_accuracy:.2f}%, Params Accuracy: {params_accuracy:.2f}%, Top-30 CMD Accuracy: {metrics['cmd_accuracy_topk']:.2f}%, Top-30 Params Accuracy: {metrics['param_accuracy_topk']:.2f}%")
        print(f"Perfect Sequence Accuracy: {metrics['perfect_sequence_accuracy']:.2f}% ({metrics['perfect_sequences']}/{metrics['total_sequences']} sequences)")
        print(f"Perfect Command Accuracy: {metrics['perfect_command_accuracy']:.2f}% ({metrics['perfect_commands']}/{metrics['total_sequences']} sequences)")
        print(f"Per-parameter accuracies:")
        for i in range(6):
            print(f"Parameter {i}: {metrics[f'param_accuracy_{i}']:.2f}%")
        print(f"Per-command accuracies:")
        for i in range(5):
            print(f"Command {i}: {metrics[f'cmd_accuracy_{i}']:.2f}%")

    def log_epoch_metrics(self, epoch, epochs, avg_loss, metrics):
        accuracy = 100 * metrics['correct_predictions'] / metrics['total_predictions']
        cmd_accuracy = metrics['cmd_accuracy']
        params_accuracy = metrics['params_accuracy']

        if not self.is_master:
            return

        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}, "
              f"Average Accuracy: {accuracy:.2f}%, CMD Accuracy: {cmd_accuracy:.2f}%, Params Accuracy: {params_accuracy:.2f}%, Top-30 CMD Accuracy: {metrics['cmd_accuracy_topk']:.2f}%, Top-30 Params Accuracy: {metrics['param_accuracy_topk']:.2f}%")
        print(f"Perfect Sequence Accuracy: {metrics['perfect_sequence_accuracy']:.2f}% ({metrics['perfect_sequences']}/{metrics['total_sequences']} sequences)")
        print(f"Perfect Command Accuracy: {metrics['perfect_command_accuracy']:.2f}% ({metrics['perfect_commands']}/{metrics['total_sequences']} sequences)")
        print(f"Per-parameter accuracies:")
        for i in range(6):
            print(f"Parameter {i}: {metrics[f'param_accuracy_{i}']:.2f}%")
        print(f"Per-command accuracies:")
        for i in range(5):
            print(f"Command {i}: {metrics[f'cmd_accuracy_{i}']:.2f}%")



# Factory function to create the appropriate trainer
def create_trainer(train_loader, val_loader, test_loader, model, training_config, device, model_type: ModelType, rank=0):
    return MultiClassesTrainer(train_loader, val_loader, test_loader, model, training_config, device, rank)
    