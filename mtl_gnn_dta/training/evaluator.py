"""Model evaluation utilities"""

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    """Model evaluator for multi-task predictions"""
    
    def __init__(self, task_names: List[str]):
        """
        Initialize evaluator
        
        Args:
            task_names: List of task names
        """
        self.task_names = task_names
        self.n_tasks = len(task_names)
    
    def evaluate(self, 
                 model: torch.nn.Module,
                 data_loader,
                 device: torch.device,
                 criterion: Optional[torch.nn.Module] = None) -> Dict:
        """
        Evaluate model on data loader
        
        Args:
            model: PyTorch model
            data_loader: Data loader
            device: Device to use
            criterion: Loss function
        
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move to device
                drug_batch = batch['drug'].to(device)
                protein_batch = batch['protein'].to(device)
                y = batch['y'].to(device)
                
                # Forward pass
                predictions = model(drug_batch, protein_batch)
                
                # Calculate loss if criterion provided
                if criterion is not None:
                    loss = criterion(predictions, y)
                    total_loss += loss.item()
                    n_batches += 1
                
                # Store predictions
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        # Add loss if calculated
        if n_batches > 0:
            metrics['loss'] = total_loss / n_batches
        
        return metrics
    
    def calculate_metrics(self, 
                         predictions: np.ndarray,
                         targets: np.ndarray) -> Dict:
        """
        Calculate comprehensive metrics
        
        Args:
            predictions: Predicted values [n_samples, n_tasks]
            targets: True values [n_samples, n_tasks]
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Overall metrics
        overall_r2 = []
        overall_rmse = []
        overall_mae = []
        overall_pearson = []
        overall_spearman = []
        
        # Per-task metrics
        for i, task in enumerate(self.task_names):
            # Get task predictions and targets
            task_pred = predictions[:, i]
            task_target = targets[:, i]
            
            # Remove NaN values
            mask = ~np.isnan(task_target)
            if mask.sum() == 0:
                continue
            
            pred_clean = task_pred[mask]
            target_clean = task_target[mask]
            
            # Calculate metrics
            r2 = r2_score(target_clean, pred_clean)
            rmse = np.sqrt(mean_squared_error(target_clean, pred_clean))
            mae = mean_absolute_error(target_clean, pred_clean)
            
            # Correlation metrics
            pearson_corr, pearson_p = stats.pearsonr(target_clean, pred_clean)
            spearman_corr, spearman_p = stats.spearmanr(target_clean, pred_clean)
            
            # Store task-specific metrics
            metrics[f'{task}_r2'] = r2
            metrics[f'{task}_rmse'] = rmse
            metrics[f'{task}_mae'] = mae
            metrics[f'{task}_pearson'] = pearson_corr
            metrics[f'{task}_spearman'] = spearman_corr
            metrics[f'{task}_n_samples'] = mask.sum()
            
            # Accumulate for overall metrics
            overall_r2.append(r2)
            overall_rmse.append(rmse)
            overall_mae.append(mae)
            overall_pearson.append(pearson_corr)
            overall_spearman.append(spearman_corr)
        
        # Calculate overall metrics
        if overall_r2:
            metrics['overall_r2'] = np.mean(overall_r2)
            metrics['overall_rmse'] = np.mean(overall_rmse)
            metrics['overall_mae'] = np.mean(overall_mae)
            metrics['overall_pearson'] = np.mean(overall_pearson)
            metrics['overall_spearman'] = np.mean(overall_spearman)
        
        return metrics
    
    def evaluate_with_confidence(self,
                                 model: torch.nn.Module,
                                 data_loader,
                                 device: torch.device,
                                 n_forward: int = 10,
                                 dropout: float = 0.1) -> Dict:
        """
        Evaluate with uncertainty estimation using MC Dropout
        
        Args:
            model: PyTorch model
            data_loader: Data loader
            device: Device
            n_forward: Number of forward passes
            dropout: Dropout rate for MC Dropout
        
        Returns:
            Dictionary with predictions and uncertainties
        """
        # Enable dropout
        def enable_dropout(model):
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout
                    module.train()
        
        enable_dropout(model)
        
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating with uncertainty"):
                drug_batch = batch['drug'].to(device)
                protein_batch = batch['protein'].to(device)
                y = batch['y'].to(device)
                
                # Multiple forward passes
                batch_predictions = []
                for _ in range(n_forward):
                    pred = model(drug_batch, protein_batch)
                    batch_predictions.append(pred.cpu().numpy())
                
                batch_predictions = np.stack(batch_predictions, axis=0)
                
                # Calculate mean and std
                mean_pred = np.mean(batch_predictions, axis=0)
                std_pred = np.std(batch_predictions, axis=0)
                
                all_predictions.append(mean_pred)
                all_uncertainties.append(std_pred)
                all_targets.append(y.cpu().numpy())
        
        # Concatenate results
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_uncertainties = np.concatenate(all_uncertainties, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        # Add uncertainty metrics
        for i, task in enumerate(self.task_names):
            task_uncertainty = all_uncertainties[:, i]
            mask = ~np.isnan(all_targets[:, i])
            if mask.sum() > 0:
                metrics[f'{task}_mean_uncertainty'] = np.mean(task_uncertainty[mask])
                metrics[f'{task}_max_uncertainty'] = np.max(task_uncertainty[mask])
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'uncertainties': all_uncertainties,
            'targets': all_targets
        }