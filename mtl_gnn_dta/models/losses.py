"""Loss functions for multi-task learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss for multi-task learning with missing labels
    Includes task-specific weighting based on value ranges
    """
    
    def __init__(self, task_ranges: Optional[Dict[str, float]] = None):
        """
        Initialize Masked MSE Loss
        
        Args:
            task_ranges: Dictionary mapping task names to their value ranges
                        Used for automatic task weighting
        """
        super(MaskedMSELoss, self).__init__()
        self.task_ranges = task_ranges or {}
        
        # Compute normalized weights
        if self.task_ranges:
            weights = []
            for task_range in self.task_ranges.values():
                if task_range > 0:
                    weights.append(1.0 / task_range)
                else:
                    weights.append(1.0)
            
            # Normalize weights
            total_weight = sum(weights)
            self.weights = torch.tensor([w / total_weight for w in weights])
        else:
            self.weights = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MSE loss
        
        Args:
            pred: Predictions [batch_size, n_tasks]
            target: Targets [batch_size, n_tasks]
        
        Returns:
            Scalar loss value
        """
        # Create mask for non-NaN values
        mask = ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Compute per-task losses
        task_losses = []
        for i in range(target.shape[1]):
            task_mask = mask[:, i]
            if task_mask.sum() > 0:
                task_pred = pred[task_mask, i]
                task_target = target[task_mask, i]
                task_loss = F.mse_loss(task_pred, task_target)
                
                # Apply task weight if available
                if self.weights is not None:
                    task_loss = task_loss * self.weights[i].to(pred.device)
                
                task_losses.append(task_loss)
        
        if len(task_losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Average across tasks
        return torch.stack(task_losses).mean()


class MaskedMAELoss(nn.Module):
    """Masked Mean Absolute Error Loss for multi-task learning"""
    
    def __init__(self, task_weights: Optional[torch.Tensor] = None):
        super(MaskedMAELoss, self).__init__()
        self.task_weights = task_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MAE loss
        
        Args:
            pred: Predictions [batch_size, n_tasks]
            target: Targets [batch_size, n_tasks]
        
        Returns:
            Scalar loss value
        """
        mask = ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        task_losses = []
        for i in range(target.shape[1]):
            task_mask = mask[:, i]
            if task_mask.sum() > 0:
                task_pred = pred[task_mask, i]
                task_target = target[task_mask, i]
                task_loss = F.l1_loss(task_pred, task_target)
                
                if self.task_weights is not None:
                    task_loss = task_loss * self.task_weights[i].to(pred.device)
                
                task_losses.append(task_loss)
        
        if len(task_losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return torch.stack(task_losses).mean()


class MaskedHuberLoss(nn.Module):
    """Masked Huber Loss for multi-task learning (robust to outliers)"""
    
    def __init__(self, delta: float = 1.0, task_weights: Optional[torch.Tensor] = None):
        super(MaskedHuberLoss, self).__init__()
        self.delta = delta
        self.task_weights = task_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute masked Huber loss
        
        Args:
            pred: Predictions [batch_size, n_tasks]
            target: Targets [batch_size, n_tasks]
        
        Returns:
            Scalar loss value
        """
        mask = ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        task_losses = []
        for i in range(target.shape[1]):
            task_mask = mask[:, i]
            if task_mask.sum() > 0:
                task_pred = pred[task_mask, i]
                task_target = target[task_mask, i]
                task_loss = F.smooth_l1_loss(task_pred, task_target, beta=self.delta)
                
                if self.task_weights is not None:
                    task_loss = task_loss * self.task_weights[i].to(pred.device)
                
                task_losses.append(task_loss)
        
        if len(task_losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return torch.stack(task_losses).mean()


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted loss for multi-task learning
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(self, n_tasks: int):
        super(UncertaintyWeightedLoss, self).__init__()
        # Initialize log variance parameters
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-weighted loss
        
        Args:
            pred: Predictions [batch_size, n_tasks]
            target: Targets [batch_size, n_tasks]
        
        Returns:
            Scalar loss value
        """
        mask = ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        task_losses = []
        for i in range(target.shape[1]):
            task_mask = mask[:, i]
            if task_mask.sum() > 0:
                task_pred = pred[task_mask, i]
                task_target = target[task_mask, i]
                
                # Compute task loss
                task_loss = F.mse_loss(task_pred, task_target)
                
                # Weight by uncertainty
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
                
                task_losses.append(weighted_loss)
        
        if len(task_losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return torch.stack(task_losses).mean()


class FocalMSELoss(nn.Module):
    """Focal MSE Loss that focuses on hard examples"""
    
    def __init__(self, gamma: float = 2.0, task_weights: Optional[torch.Tensor] = None):
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.task_weights = task_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal MSE loss
        
        Args:
            pred: Predictions [batch_size, n_tasks]
            target: Targets [batch_size, n_tasks]
        
        Returns:
            Scalar loss value
        """
        mask = ~torch.isnan(target)
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        task_losses = []
        for i in range(target.shape[1]):
            task_mask = mask[:, i]
            if task_mask.sum() > 0:
                task_pred = pred[task_mask, i]
                task_target = target[task_mask, i]
                
                # Compute squared error
                se = (task_pred - task_target) ** 2
                
                # Apply focal weighting
                focal_weight = (1 + se) ** self.gamma
                focal_loss = (focal_weight * se).mean()
                
                if self.task_weights is not None:
                    focal_loss = focal_loss * self.task_weights[i].to(pred.device)
                
                task_losses.append(focal_loss)
        
        if len(task_losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return torch.stack(task_losses).mean()
