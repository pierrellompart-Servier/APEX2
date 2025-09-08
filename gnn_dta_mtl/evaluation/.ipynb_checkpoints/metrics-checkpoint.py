"""
Evaluation metrics for model performance
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Optional
import math


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}
    
    metrics = {}
    
    # Basic metrics
    metrics[f'{prefix}rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred)
    
    # Correlation metrics
    if len(y_true) > 1:
        pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = spearmanr(y_true, y_pred)
        
        metrics[f'{prefix}pearson_r'] = pearson_corr
        metrics[f'{prefix}pearson_p'] = pearson_p
        metrics[f'{prefix}spearman_r'] = spearman_corr
        metrics[f'{prefix}spearman_p'] = spearman_p
    
    # Percentage error (if no zeros in true values)
    if not np.any(y_true == 0):
        metrics[f'{prefix}mape'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # Additional statistics
    metrics[f'{prefix}mean_true'] = np.mean(y_true)
    metrics[f'{prefix}mean_pred'] = np.mean(y_pred)
    metrics[f'{prefix}std_true'] = np.std(y_true)
    metrics[f'{prefix}std_pred'] = np.std(y_pred)
    metrics[f'{prefix}n_samples'] = len(y_true)
    
    return metrics


def calculate_ci(
    values: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval.
    
    Args:
        values: Array of values
        confidence: Confidence level
        
    Returns:
        Mean, lower bound, upper bound
    """
    import scipy.stats as stats
    
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(values)
    std_err = stats.sem(values)
    interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - interval, mean + interval


def concordance_index(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calculate concordance index (C-index).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        C-index value
    """
    from lifelines.utils import concordance_index as ci
    
    try:
        return ci(y_true, y_pred)
    except:
        # Fallback implementation
        n = len(y_true)
        if n < 2:
            return 0.5
        
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if y_true[i] != y_true[j]:
                    if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                       (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                        concordant += 1
                    else:
                        discordant += 1
        
        total = concordant + discordant
        if total == 0:
            return 0.5
        
        return concordant / total


def evaluate_model(
    model,
    test_loader,
    task_cols: list,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        task_cols: List of task columns
        device: Device to use
        
    Returns:
        Dictionary of metrics per task
    """
    import torch
    from tqdm import tqdm
    
    model.eval()
    model = model.to(device)
    
    task_predictions = {task: [] for task in task_cols}
    task_targets = {task: [] for task in task_cols}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            
            pred = model(xd, xp)
            
            # Collect predictions per task
            for i, task in enumerate(task_cols):
                mask = ~torch.isnan(y[:, i])
                if mask.sum() > 0:
                    task_predictions[task].extend(pred[mask, i].cpu().numpy())
                    task_targets[task].extend(y[mask, i].cpu().numpy())
    
    # Calculate metrics for each task
    results = {}
    for task in task_cols:
        if len(task_predictions[task]) > 0:
            preds = np.array(task_predictions[task])
            targets = np.array(task_targets[task])
            
            results[task] = calculate_metrics(targets, preds)
            results[task]['c_index'] = concordance_index(targets, preds)
    
    return results


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, float, float]]:
    """
    Calculate bootstrapped confidence intervals for metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        seed: Random seed
        
    Returns:
        Dictionary of metrics with CI
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    
    # Storage for bootstrap results
    rmse_scores = []
    r2_scores = []
    mae_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metrics
        rmse_scores.append(math.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
        r2_scores.append(r2_score(y_true_boot, y_pred_boot))
        mae_scores.append(mean_absolute_error(y_true_boot, y_pred_boot))
    
    # Calculate confidence intervals
    results = {
        'rmse': calculate_ci(np.array(rmse_scores), confidence),
        'r2': calculate_ci(np.array(r2_scores), confidence),
        'mae': calculate_ci(np.array(mae_scores), confidence)
    }
    
    return results