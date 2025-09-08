"""
Loss functions for training
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class MaskedMSELoss(nn.Module):
    """
    Masked MSE loss for multi-task learning with NaN handling.
    
    Args:
        task_ranges: Dictionary of task ranges for weighting
    """
    
    def __init__(self, task_ranges: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_ranges = task_ranges
        
        if task_ranges is not None:
            # Calculate task weights based on inverse range
            weights = []
            for range_val in task_ranges.values():
                weights.append(1.0 / range_val if range_val > 0 else 1.0)
            total_weight = sum(weights)
            self.task_weights = torch.tensor([w / total_weight for w in weights])
        else:
            self.task_weights = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate masked MSE loss.
        
        Args:
            pred: Predictions [batch_size, n_tasks]
            target: Targets [batch_size, n_tasks]
            
        Returns:
            Scalar loss value
        """
        # Create mask for non-NaN values
        mask = ~torch.isnan(target)
        
        # Handle case with no valid values
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        # Apply mask
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # Calculate squared errors
        se = (pred_masked - target_masked) ** 2
        
        # Apply task weights if available
        if self.task_weights is not None:
            task_indices = torch.where(mask)[1]
            weights = self.task_weights.to(pred.device)[task_indices]
            weighted_se = se * weights
            loss = weighted_se.mean()
        else:
            loss = se.mean()
        
        return loss