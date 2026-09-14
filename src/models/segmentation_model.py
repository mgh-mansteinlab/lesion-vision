"""Training/inference facade around SMPSegmentationNet."""

import os
from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.constants import CLASS_NAMES
from src.models.architecture import SMPSegmentationNet
from src.models.losses import CompoundLoss
from src.train.engine import TrainEngine


class LesionSegmentationModel(TrainEngine):
    """Class for lesion segmentation model."""
    
    def __init__(
        self,
        n_classes=4,
        backbone='tu-convnext_base',
        encoder_weights='imagenet',
        learning_rate=1e-4,
        input_shape=(3, 512, 512),
        device=None,
        use_amp=True,
        gradient_accumulation_steps=1,
        activation_checkpointing=False,
        loss_type='compound',
        class_weights=None,
        scheduler_type='cosine',
        warmup_epochs=5,
        total_epochs=100,
    ):
        """
        Initialize the model.
        
        Args:
            n_classes: Number of output classes
            backbone: Backbone model name
            encoder_weights: Pre-trained weights for encoder
            learning_rate: Initial learning rate
            input_shape: Input shape (channels, height, width)
            device: Device to use for computation (cuda or cpu)
            use_amp: Use mixed precision (autocast + GradScaler)
            gradient_accumulation_steps: Accumulate gradients over this many steps
            activation_checkpointing: Trade compute for memory (recommended for L4)
            loss_type: 'ce', 'compound', or 'focal' (default 'compound')
            class_weights: Optional per-class weight array/tensor for the CE component
            scheduler_type: 'plateau' or 'cosine' (default 'cosine')
            warmup_epochs: Warmup epochs for cosine scheduler
            total_epochs: Total training epochs (for cosine scheduler period)
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.backbone = backbone
        self.encoder_weights = encoder_weights
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.activation_checkpointing = activation_checkpointing
        
        # Define class names for visualization
        self.class_names = dict(CLASS_NAMES)
        
        self.loss_type = loss_type
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        self.model = self._build_model().to(self.device)

        cw_tensor = None
        if class_weights is not None:
            cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        if loss_type == 'compound':
            self.criterion = CompoundLoss(
                n_classes=n_classes, class_weights=cw_tensor,
                ce_weight=0.5, dice_weight=0.5,
            )
        elif loss_type == 'focal':
            self.criterion = smp.losses.FocalLoss(
                mode='multiclass', alpha=None, gamma=2.0,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=cw_tensor)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4,
        )

        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(1, total_epochs - warmup_epochs), T_mult=1, eta_min=1e-6,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
            )

        self.scaler = GradScaler('cuda') if self.use_amp else None
    
    def _build_model(self):
        """Build the segmentation model."""
        model = SMPSegmentationNet(
            n_classes=self.n_classes,
            backbone=self.backbone,
            encoder_weights=self.encoder_weights,
            activation_checkpointing=self.activation_checkpointing,
        )
        return model
    
    def predict(self, images, distributed=False):
        """
        Make segmentation predictions.
        
        Args:
            images: Input images (torch tensor [B,C,H,W])
            distributed: Whether using distributed training
        
        Returns:
            Segmentation masks (torch tensor [B,C,H,W])
        """
        model = self.model
        model.eval()
        
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images).float()
        
        if images.dim() == 3:  # Single image [C,H,W]
            images = images.unsqueeze(0)  # Add batch dimension [1,C,H,W]
        
        device = next(model.parameters()).device
        images = images.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu()
    
    def save(self, model_path, distributed=False):
        """Save the model."""
        model = self.model
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'n_classes': self.n_classes,
                        'backbone': self.backbone,
                        'encoder_weights': getattr(self, 'encoder_weights', 'imagenet'),
            'input_shape': self.input_shape,
            'learning_rate': self.learning_rate,
            'loss_type': self.loss_type,
            'scheduler_type': self.scheduler_type,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'activation_checkpointing': self.activation_checkpointing,
        }, model_path)
    
    def load(self, model_path, device=None):
        """Load the model from file."""
        map_location = device if device is not None else 'cpu'
        checkpoint = torch.load(model_path, map_location=map_location)
        
        self.n_classes = checkpoint.get('n_classes', self.n_classes)
        self.backbone = checkpoint.get('backbone', self.backbone)
        self.encoder_weights = checkpoint.get('encoder_weights', getattr(self, 'encoder_weights', 'imagenet'))
        self.input_shape = checkpoint.get('input_shape', self.input_shape)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.loss_type = checkpoint.get('loss_type', self.loss_type)
        self.scheduler_type = checkpoint.get('scheduler_type', self.scheduler_type)
        self.warmup_epochs = checkpoint.get('warmup_epochs', self.warmup_epochs)
        self.total_epochs = checkpoint.get('total_epochs', self.total_epochs)
        self.activation_checkpointing = checkpoint.get('activation_checkpointing', self.activation_checkpointing)
        
        # Rebuild model with loaded parameters
        self.model = self._build_model()
        state_dict = checkpoint['model_state_dict']
        # Checkpoints saved under DistributedDataParallel carry a 'module.'
        # prefix on every key. Strip it so the unwrapped model can load them.
        if any(k.startswith('module.') for k in state_dict):
            state_dict = {
                (k[len('module.'):] if k.startswith('module.') else k): v
                for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict)

        # Restore optimizer / scheduler / early-stopping state so that
        # --resume continues training rather than restarting the schedule.
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: could not restore optimizer state: {e}")
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"Warning: could not restore scheduler state: {e}")
        self._resume_epoch = checkpoint.get('epoch', None)
        self._resume_val_loss = checkpoint.get('val_loss', None)

        if device is not None:
            self.model = self.model.to(device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError):
            pass
        if self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=max(1, self.total_epochs - self.warmup_epochs), T_mult=1, eta_min=1e-6,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
            )
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except (ValueError, KeyError):
            pass
        
        return self.model
    
    def evaluate(self, test_dataloader, distributed=False, rank=0):
        """Evaluate the model on test data."""
        return self.validate(test_dataloader, rank, distributed)
    
    def setup_for_distributed(self, rank, world_size):
        """
        Set up the model for distributed training.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
        """
        # Move model to device
        self.model.to(self.device)
        
        # Wrap model with DDP if not already wrapped
        if not isinstance(self.model, DDP):
            # static_graph=True detects unused parameters automatically;
            # passing find_unused_parameters=True alongside it is redundant
            # (PyTorch warns) and slows the first iteration.
            print(f"Rank {rank}: Setting up DDP with static_graph=True")
            self.model = DDP(
                self.model,
                device_ids=[rank],
                static_graph=True,
            )
