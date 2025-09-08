"""
Visualization utilities for model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import r2_score, mean_squared_error
import math


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str = "",
    ax: Optional[plt.Axes] = None,
    show_metrics: bool = True
) -> plt.Axes:
    """
    Plot predicted vs true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_name: Name of the task
        ax: Matplotlib axes
        show_metrics: Whether to show metrics
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Remove NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.4, s=10, color='blue')
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1, alpha=0.7)
    
    # Calculate metrics
    if show_metrics and len(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        
        # Add text with metrics
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(y_true)}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'Experimental {task_name}')
    ax.set_ylabel(f'Predicted {task_name}')
    ax.set_title(f'{task_name} Predictions' if task_name else 'Predictions')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_name: str = "",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot residuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_name: Name of the task
        ax: Matplotlib axes
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Scatter plot
    ax.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Add confidence bands
    std_residuals = np.std(residuals)
    ax.axhline(y=2*std_residuals, color='orange', linestyle=':', alpha=0.5)
    ax.axhline(y=-2*std_residuals, color='orange', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(f'Predicted {task_name}')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Residual Plot - {task_name}' if task_name else 'Residual Plot')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_results(
    cv_results: Dict,
    task_cols: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comprehensive cross-validation results.
    
    Args:
        cv_results: Cross-validation results dictionary
        task_cols: List of task columns
        save_path: Path to save figure
    """
    # Filter tasks with data
    tasks_with_data = [
        task for task in task_cols
        if len(cv_results[task]['all_targets']) > 0
    ]
    
    n_tasks = len(tasks_with_data)
    if n_tasks == 0:
        print("No data to plot")
        return
    
    # Create subplots
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_tasks == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Plot each task
    for idx, task in enumerate(tasks_with_data):
        ax = axes[idx] if n_tasks > 1 else axes[0]
        
        targets = np.array(cv_results[task]['all_targets'])
        preds = np.array(cv_results[task]['all_predictions'])
        
        plot_predictions(targets, preds, task, ax)
    
    # Hide unused subplots
    if n_tasks > 1:
        for idx in range(n_tasks, len(axes)):
            axes[idx].set_visible(False)
    
    plt.suptitle('Cross-Validation Results', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
    
    # Mark minimum validation loss
    min_val_idx = np.argmin(val_losses)
    ax.plot(min_val_idx + 1, val_losses[min_val_idx], 'r*', 
            markersize=15, label=f'Best Val Loss = {val_losses[min_val_idx]:.4f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_distribution(
    cv_results: Dict,
    task_cols: List[str],
    metric: str = 'r2',
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of metrics across folds.
    
    Args:
        cv_results: Cross-validation results
        task_cols: List of task columns
        metric: Metric to plot ('r2' or 'rmse')
        save_path: Path to save figure
    """
    # Prepare data
    data = []
    for task in task_cols:
        if f'{metric}_list' in cv_results[task]:
            for value in cv_results[task][f'{metric}_list']:
                data.append({'Task': task, metric.upper(): value})
    
    if not data:
        print(f"No {metric} data to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(x='Task', y=metric.upper(), data=df, ax=ax)
    sns.swarmplot(x='Task', y=metric.upper(), data=df, color='black', 
                  size=5, alpha=0.5, ax=ax)
    
    ax.set_title(f'{metric.upper()} Distribution Across Folds')
    ax.set_xlabel('Task')
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)
    
    # Rotate x-labels if many tasks
    if len(task_cols) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix between different task predictions.
    
    Args:
        predictions_dict: Dictionary of predictions per task
        save_path: Path to save figure
    """
    # Create DataFrame from predictions
    df = pd.DataFrame(predictions_dict)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    
    ax.set_title('Task Prediction Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_summary_report(
    cv_results: Dict,
    task_cols: List[str],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create summary report of cross-validation results.
    
    Args:
        cv_results: Cross-validation results
        task_cols: List of task columns
        output_path: Path to save report
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for task in task_cols:
        if len(cv_results[task]['r2_list']) > 0:
            summary_data.append({
                'Task': task,
                'R² (mean±std)': f"{np.mean(cv_results[task]['r2_list']):.3f}±{np.std(cv_results[task]['r2_list']):.3f}",
                'RMSE (mean±std)': f"{np.mean(cv_results[task]['rmse_list']):.3f}±{np.std(cv_results[task]['rmse_list']):.3f}",
                'N samples': len(cv_results[task]['all_targets']),
                'N folds': len(cv_results[task]['r2_list'])
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"Summary saved to {output_path}")
    
    return summary_df