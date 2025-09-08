"""
Cross-validation utilities
"""

import gc
import torch
import torch_geometric
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from ..models import MTL_DTAModel
from ..models.losses import MaskedMSELoss
from ..datasets import build_mtl_dataset_optimized
from ..evaluation import calculate_metrics
from .early_stopping import EarlyStopping


class CrossValidator:
    """
    K-fold cross-validation for DTA models.
    """
    
    def __init__(
        self,
        model_config: Dict,
        task_cols: List[str],
        task_ranges: Dict[str, float],
        n_folds: int = 5,
        batch_size: int = 128,
        n_epochs: int = 100,
        learning_rate: float = 0.0005,
        patience: int = 20,
        device: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize cross-validator.
        
        Args:
            model_config: Model configuration dictionary
            task_cols: List of task columns
            task_ranges: Task value ranges
            n_folds: Number of CV folds
            batch_size: Batch size
            n_epochs: Maximum epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            device: Device to use
            seed: Random seed
        """
        self.model_config = model_config
        self.task_cols = task_cols
        self.task_ranges = task_ranges
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.cv_results = {
            task: {
                'r2_list': [],
                'rmse_list': [],
                'all_predictions': [],
                'all_targets': []
            } for task in task_cols
        }
    
    def train_fold(
        self,
        fold_idx: int,
        train_loader: torch_geometric.loader.DataLoader,
        valid_loader: torch_geometric.loader.DataLoader,
        test_loader: torch_geometric.loader.DataLoader
    ) -> Tuple[Dict, List, List]:
        """
        Train a single fold.
        
        Args:
            fold_idx: Fold index
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Fold results, train losses, validation losses
        """
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*60}")
        
        # Create model
        model = MTL_DTAModel(
            task_names=self.task_cols,
            **self.model_config
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = MaskedMSELoss(task_ranges=self.task_ranges).to(self.device)
        
        # Training state
        best_val_loss = float('inf')
        best_model_state = None
        stopper = EarlyStopping(patience=self.patience, higher_better=False)
        train_losses = []
        val_losses = []
        
        # Training loop
        pbar = tqdm(range(self.n_epochs), desc=f"Fold {fold_idx+1}")
        
        for epoch in pbar:
            # Training phase
            model.train()
            train_loss = 0
            n_train_batches = 0
            
            for batch in train_loader:
                xd = batch['drug'].to(self.device)
                xp = batch['protein'].to(self.device)
                y = batch['y'].to(self.device)
                
                optimizer.zero_grad()
                pred = model(xd, xp)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_train_batches += 1
            
            avg_train_loss = train_loss / n_train_batches if n_train_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch in valid_loader:
                    xd = batch['drug'].to(self.device)
                    xp = batch['protein'].to(self.device)
                    y = batch['y'].to(self.device)
                    
                    pred = model(xd, xp)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('inf')
            val_losses.append(avg_val_loss)
            
            # Update progress
            pbar.set_postfix({
                'train': f"{avg_train_loss:.4f}",
                'val': f"{avg_val_loss:.4f}"
            })
            
            # Early stopping
            if stopper.update(avg_val_loss):
                best_model_state = model.state_dict()
            
            if stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluation phase
        model.eval()
        task_predictions = {task: [] for task in self.task_cols}
        task_targets = {task: [] for task in self.task_cols}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                xd = batch['drug'].to(self.device)
                xp = batch['protein'].to(self.device)
                y = batch['y'].to(self.device)
                
                pred = model(xd, xp)
                
                # Collect predictions per task
                for i, task in enumerate(self.task_cols):
                    mask = ~torch.isnan(y[:, i])
                    if mask.sum() > 0:
                        task_predictions[task].extend(pred[mask, i].cpu().numpy())
                        task_