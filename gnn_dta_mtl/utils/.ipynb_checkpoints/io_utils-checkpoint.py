"""
Input/Output utilities
"""

import os
import json
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


def save_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Save path
        optimizer: Optional optimizer state
        epoch: Current epoch
        metrics: Training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        path: Checkpoint path
        optimizer: Optional optimizer to load state
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {path}")
    
    return checkpoint


def save_results(
    results: Dict,
    path: str,
    format: str = 'json'
) -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        path: Save path
        format: Save format ('json', 'pickle', 'csv')
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        results_json = _convert_numpy_to_list(results)
        with open(path, 'w') as f:
            json.dump(results_json, f, indent=2)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"Results saved to {path}")


def load_results(
    path: str,
    format: str = 'auto'
) -> Union[Dict, pd.DataFrame]:
    """
    Load results from file.
    
    Args:
        path: File path
        format: File format (auto-detect if 'auto')
        
    Returns:
        Loaded results
    """
    if format == 'auto':
        ext = Path(path).suffix.lower()
        format = {
            '.json': 'json',
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.csv': 'csv'
        }.get(ext, 'json')
    
    if format == 'json':
        with open(path, 'r') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {format}")


def _convert_numpy_to_list(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_list(item) for item in obj)
    else:
        return obj


def save_predictions(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save predictions and targets.
    
    Args:
        predictions: Predictions per task
        targets: Targets per task
        path: Save path
        metadata: Additional metadata
    """
    data = {
        'predictions': predictions,
        'targets': targets
    }
    
    if metadata:
        data['metadata'] = metadata
    
    save_results(data, path, format='pickle')


def load_dataframe(
    path: str,
    format: str = 'auto'
) -> pd.DataFrame:
    """
    Load dataframe from various formats.
    
    Args:
        path: File path
        format: File format
        
    Returns:
        Loaded DataFrame
    """
    if format == 'auto':
        ext = Path(path).suffix.lower()
        
    if ext == '.parquet':
        return pd.read_parquet(path)
    elif ext == '.csv':
        return pd.read_csv(path)
    elif ext in ['.pkl', '.pickle']:
        return pd.read_pickle(path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unknown format: {ext}")


def create_output_dir(
    base_dir: str,
    experiment_name: str,
    timestamp: bool = True
) -> str:
    """
    Create output directory for experiment.
    
    Args:
        base_dir: Base directory
        experiment_name: Experiment name
        timestamp: Whether to add timestamp
        
    Returns:
        Created directory path
    """
    from datetime import datetime
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{timestamp_str}"
    else:
        dir_name = experiment_name
    
    output_dir = Path(base_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    return str(output_dir)


def save_config(
    config: Dict,
    path: str
) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Save path
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {path}")


def load_config(path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        path: Config file path
        
    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config