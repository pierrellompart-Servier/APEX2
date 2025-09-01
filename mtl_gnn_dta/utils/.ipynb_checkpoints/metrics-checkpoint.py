"""Evaluation metrics for drug-target affinity prediction"""

import numpy as np
from sklearn import metrics
from scipy import stats
from typing import Dict, List, Optional, Tuple


def eval_mse(y_true: np.ndarray, y_pred: np.ndarray, squared: bool = True) -> float:
    """
    Evaluate MSE or RMSE
    
    Args:
        y_true: True values
        y_pred: Predicted values
        squared: If True return MSE, else RMSE
    
    Returns:
        MSE or RMSE value
    """
    return metrics.mean_squared_error(y_true, y_pred, squared=squared)


def eval_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate Pearson correlation"""
    return stats.pearsonr(y_true, y_pred)[0]


def eval_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate Spearman correlation"""
    return stats.spearmanr(y_true, y_pred)[0]


def eval_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate R2 score"""
    return metrics.r2_score(y_true, y_pred)


def eval_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate AUROC"""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)


def eval_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate AUPRC"""
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(rec, pre)


def evaluation_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray,
                       eval_metrics: List[str]) -> Dict[str, float]:
    """
    Evaluate multiple metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        eval_metrics: List of metric names
    
    Returns:
        Dictionary of metric values
    """
    results = {}
    
    for m in eval_metrics:
        if m == 'mse':
            s = eval_mse(y_true, y_pred, squared=True)
        elif m == 'rmse':
            s = eval_mse(y_true, y_pred, squared=False)
        elif m == 'pearson':
            s = eval_pearson(y_true, y_pred)
        elif m == 'spearman':
            s = eval_spearman(y_true, y_pred)
        elif m == 'r2':
            s = eval_r2(y_true, y_pred)
        elif m == 'auroc':
            s = eval_auroc(y_true, y_pred)
        elif m == 'auprc':
            s = eval_auprc(y_true, y_pred)
        else:
            raise ValueError(f'Unknown evaluation metric: {m}')
        
        results[m] = s
    
    return results


def calculate_task_metrics(predictions: np.ndarray,
                          targets: np.ndarray,
                          task_names: List[str]) -> Dict:
    """
    Calculate metrics for multi-task predictions
    
    Args:
        predictions: Predictions [n_samples, n_tasks]
        targets: Targets [n_samples, n_tasks]
        task_names: List of task names
    
    Returns:
        Dictionary with per-task and overall metrics
    """
    all_metrics = {}
    
    for i, task in enumerate(task_names):
        # Extract task data
        task_pred = predictions[:, i]
        task_target = targets[:, i]
        
        # Remove NaN values
        mask = ~np.isnan(task_target)
        if mask.sum() == 0:
            continue
        
        pred_clean = task_pred[mask]
        target_clean = task_target[mask]
        
        # Calculate metrics
        task_metrics = evaluation_metrics(
            target_clean, pred_clean,
            ['mse', 'rmse', 'pearson', 'spearman', 'r2']
        )
        
        # Add to results
        for metric_name, value in task_metrics.items():
            all_metrics[f'{task}_{metric_name}'] = value
        
        all_metrics[f'{task}_n_samples'] = mask.sum()
    
    # Calculate overall metrics
    for metric_type in ['mse', 'rmse', 'pearson', 'spearman', 'r2']:
        task_values = [v for k, v in all_metrics.items() 
                      if k.endswith(f'_{metric_type}')]
        if task_values:
            all_metrics[f'overall_{metric_type}'] = np.mean(task_values)
    
    return all_metrics