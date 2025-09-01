"""Optimizer configurations and learning rate schedulers"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional
import math


def get_optimizer(model: torch.nn.Module, 
                  config: Dict) -> Optimizer:
    """
    Get optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
    
    Returns:
        Initialized optimizer
    """
    optimizer_type = config.get('type', 'adam').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=config.get('nesterov', False)
        )
    
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=config.get('alpha', 0.99),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(optimizer: Optimizer, 
                  config: Dict) -> Optional[_LRScheduler]:
    """
    Get learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration
    
    Returns:
        Initialized scheduler or None
    """
    scheduler_type = config.get('type', 'none').lower()
    
    if scheduler_type == 'none':
        return None
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [30, 60, 90]),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 0)
        )
    
    elif scheduler_type == 'cosine_warm':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 0)
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get('mode', 'min'),
            factor=config.get('factor', 0.1),
            patience=config.get('patience', 10),
            threshold=config.get('threshold', 0.0001),
            min_lr=config.get('min_lr', 0),
            verbose=config.get('verbose', True)
        )
    
    elif scheduler_type == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.get('base_lr', 0.0001),
            max_lr=config.get('max_lr', 0.001),
            step_size_up=config.get('step_size_up', 2000),
            mode=config.get('mode', 'triangular')
        )
    
    elif scheduler_type == 'one_cycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 0.001),
            total_steps=config.get('total_steps', 1000),
            pct_start=config.get('pct_start', 0.3),
            anneal_strategy=config.get('anneal_strategy', 'cos')
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with warmup"""
    
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 base_scheduler: Optional[_LRScheduler] = None,
                 warmup_start_lr: float = 1e-6,
                 last_epoch: int = -1):
        """
        Initialize warmup scheduler
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            base_scheduler: Base scheduler after warmup
            warmup_start_lr: Starting learning rate for warmup
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                    for base_lr in self.base_lrs]
        else:
            # Use base scheduler if available
            if self.base_scheduler:
                return self.base_scheduler.get_lr()
            else:
                return self.base_lrs