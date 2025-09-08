"""
Model-related utilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024,
        'total_size_mb': size_mb
    }


def freeze_layers(
    model: nn.Module,
    layers_to_freeze: Optional[List[str]] = None,
    freeze_all_except: Optional[List[str]] = None
) -> None:
    """
    Freeze specific layers in model.
    
    Args:
        model: PyTorch model
        layers_to_freeze: List of layer names to freeze
        freeze_all_except: Freeze all except these layers
    """
    if freeze_all_except is not None:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze specified layers
        for name, param in model.named_parameters():
            for layer_name in freeze_all_except:
                if layer_name in name:
                    param.requires_grad = True
                    break
    
    elif layers_to_freeze is not None:
        for name, param in model.named_parameters():
            for layer_name in layers_to_freeze:
                if layer_name in name:
                    param.requires_grad = False
                    break


def get_activation_stats(
    model: nn.Module,
    data_loader,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Get activation statistics for model layers.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Dictionary of activation statistics
    """
    activation_stats = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run one batch
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if 'drug' in batch and 'protein' in batch:
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                _ = model(xd, xp)
            break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats


def ensemble_predictions(
    predictions: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = 'mean'
) -> np.ndarray:
    """
    Ensemble multiple predictions.
    
    Args:
        predictions: List of prediction arrays
        weights: Optional weights for each model
        method: Ensemble method ('mean', 'median', 'weighted')
        
    Returns:
        Ensembled predictions
    """
    predictions = np.stack(predictions)
    
    if method == 'mean':
        return np.mean(predictions, axis=0)
    elif method == 'median':
        return np.median(predictions, axis=0)
    elif method == 'weighted':
        if weights is None:
            weights = [1.0 / len(predictions)] * len(predictions)
        weights = np.array(weights).reshape(-1, 1)
        return np.sum(predictions * weights, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


class ModelCheckpointer:
    """
    Model checkpointing with best model tracking.
    """
    
    def __init__(
        self,
        save_dir: str,
        metric_name: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3
    ):
        """
        Initialize checkpointer.
        
        Args:
            save_dir: Directory to save checkpoints
            metric_name: Metric to track
            mode: 'min' or 'max'
            save_top_k: Number of best models to keep
        """
        from pathlib import Path
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.save_top_k = save_top_k
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        metric_value: float,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        additional_info: Optional[Dict] = None
    ) -> bool:
        """
        Save checkpoint if it's among the best.
        
        Args:
            model: Model to save
            metric_value: Current metric value
            epoch: Current epoch
            optimizer: Optional optimizer
            additional_info: Additional information to save
            
        Returns:
            Whether checkpoint was saved
        """
        import os
        
        # Check if should save
        should_save = len(self.checkpoints) < self.save_top_k
        
        if not should_save:
            worst_idx = None
            worst_value = None
            
            for i, (_, value) in enumerate(self.checkpoints):
                if self.mode == 'min':
                    if worst_value is None or value > worst_value:
                        worst_value = value
                        worst_idx = i
                else:
                    if worst_value is None or value < worst_value:
                        worst_value = value
                        worst_idx = i
            
            if self.mode == 'min':
                should_save = metric_value < worst_value
            else:
                should_save = metric_value > worst_value
            
            if should_save and worst_idx is not None:
                # Remove worst checkpoint
                old_path, _ = self.checkpoints.pop(worst_idx)
                if os.path.exists(old_path):
                    os.remove(old_path)
        
        if should_save:
            # Save checkpoint
            checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch}_{self.metric_name}{metric_value:.4f}.pt"
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                self.metric_name: metric_value
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            if additional_info is not None:
                checkpoint.update(additional_info)
            
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append((str(checkpoint_path), metric_value))
            
            # Sort checkpoints
            if self.mode == 'min':
                self.checkpoints.sort(key=lambda x: x[1])
            else:
                self.checkpoints.sort(key=lambda x: x[1], reverse=True)
            
            return True
        
        return False
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load best checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer
            device: Device to load to
            
        Returns:
            Checkpoint dictionary
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        best_path, _ = self.checkpoints[0]
        checkpoint = torch.load(best_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint