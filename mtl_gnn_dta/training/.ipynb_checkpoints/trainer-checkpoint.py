"""Training utilities for MTL-GNN-DTA"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for MTL-GNN-DTA models"""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            scheduler: Learning rate scheduler
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            drug_batch = batch['drug'].to(self.device)
            protein_batch = batch['protein'].to(self.device)
            y = batch['y'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(drug_batch, protein_batch)
            loss = self.criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                drug_batch = batch['drug'].to(self.device)
                protein_batch = batch['protein'].to(self.device)
                y = batch['y'].to(self.device)
                
                predictions = self.model(drug_batch, protein_batch)
                loss = self.criterion(predictions, y)
                
                total_loss += loss.item()
                n_batches += 1
                
                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        metrics = {}
        n_tasks = predictions.shape[1] if len(predictions.shape) > 1 else 1
        
        for i in range(n_tasks):
            if n_tasks > 1:
                pred = predictions[:, i]
                target = targets[:, i]
            else:
                pred = predictions
                target = targets
            
            # Remove NaN values
            mask = ~np.isnan(target)
            if mask.sum() > 0:
                pred_clean = pred[mask]
                target_clean = target[mask]
                
                metrics[f'task_{i}_r2'] = r2_score(target_clean, pred_clean)
                metrics[f'task_{i}_rmse'] = np.sqrt(mean_squared_error(target_clean, pred_clean))
                metrics[f'task_{i}_mae'] = mean_absolute_error(target_clean, pred_clean)
        
        return metrics
    
    def train(self,
              train_loader,
              val_loader,
              n_epochs: int,
              patience: int = 20,
              save_best: bool = True) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            patience: Early stopping patience
            save_best: Whether to save best model
        
        Returns:
            Training history
        """
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            for key, value in val_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                if save_best and self.checkpoint_dir:
                    self.save_checkpoint(epoch, val_loss, val_metrics)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        
        # Also save as best model
        best_path = self.checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved to {path}")