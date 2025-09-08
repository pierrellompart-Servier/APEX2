"""
Training utilities for DTA models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import numpy as np
from ..models.losses import MaskedMSELoss
from ..evaluation.metrics import calculate_metrics
from .early_stopping import EarlyStopping


class MTLTrainer:
    """
    Multi-task learning trainer for DTA models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_cols: List[str],
        task_ranges: Optional[Dict[str, float]] = None,
        device: str = 'cuda',
        learning_rate: float = 0.0005,
        batch_size: int = 128
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            task_cols: List of task column names
            task_ranges: Task value ranges for weighting
            device: Device to use
            learning_rate: Learning rate
            batch_size: Batch size
        """
        self.model = model.to(device)
        self.task_cols = task_cols
        self.device = device
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = MaskedMSELoss(task_ranges=task_ranges).to(device)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: torch_geometric.loader.DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            xd = batch['drug'].to(self.device)
            xp = batch['protein'].to(self.device)
            y = batch['y'].to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(xd, xp)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Periodic cleanup
            if n_batches % 20 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / n_batches if n_batches > 0 else 0
    
    def validate(
        self,
        valid_loader: torch_geometric.loader.DataLoader
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """
        Validate model.
        
        Args:
            valid_loader: Validation data loader
            
        Returns:
            Validation loss and per-task metrics
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        task_predictions = {task: [] for task in self.task_cols}
        task_targets = {task: [] for task in self.task_cols}
        
        with torch.no_grad():
            for batch in valid_loader:
                xd = batch['drug'].to(self.device)
                xp = batch['protein'].to(self.device)
                y = batch['y'].to(self.device)
                
                pred = self.model(xd, xp)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
                n_batches += 1
                
                # Collect predictions per task
                for i, task in enumerate(self.task_cols):
                    mask = ~torch.isnan(y[:, i])
                    if mask.sum() > 0:
                        task_predictions[task].extend(pred[mask, i].cpu().numpy())
                        task_targets[task].extend(y[mask, i].cpu().numpy())
        
        # Calculate metrics
        task_metrics = {}
        for task in self.task_cols:
            if len(task_predictions[task]) > 0:
                preds = np.array(task_predictions[task])
                targets = np.array(task_targets[task])
                task_metrics[task] = calculate_metrics(targets, preds)
        
        avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
        
        return avg_loss, task_metrics
    
    def train(
        self,
        train_loader: torch_geometric.loader.DataLoader,
        valid_loader: torch_geometric.loader.DataLoader,
        n_epochs: int = 100,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            n_epochs: Number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training history and best model state
        """
        stopper = EarlyStopping(patience=patience, higher_better=False)
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, task_metrics = self.validate(valid_loader)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if stopper.update(val_loss):
                best_model_state = self.model.state_dict()
            
            if verbose:
                print(f"\nEpoch {epoch+1}/{n_epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Valid Loss: {val_loss:.4f}")
                
                for task, metrics in task_metrics.items():
                    print(f"  {task}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")
            
            if stopper.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_model_state': best_model_state
        }
    
    def predict(
        self,
        test_loader: torch_geometric.loader.DataLoader
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of predictions and targets per task
        """
        self.model.eval()
        
        task_predictions = {task: [] for task in self.task_cols}
        task_targets = {task: [] for task in self.task_cols}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                xd = batch['drug'].to(self.device)
                xp = batch['protein'].to(self.device)
                y = batch['y'].to(self.device)
                
                pred = self.model(xd, xp)
                
                # Collect predictions per task
                for i, task in enumerate(self.task_cols):
                    mask = ~torch.isnan(y[:, i])
                    if mask.sum() > 0:
                        task_predictions[task].extend(pred[mask, i].cpu().numpy())
                        task_targets[task].extend(y[mask, i].cpu().numpy())
        
        results = {}
        for task in self.task_cols:
            if len(task_predictions[task]) > 0:
                results[task] = {
                    'predictions': np.array(task_predictions[task]),
                    'targets': np.array(task_targets[task])
                }
        
        return results