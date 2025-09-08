"""
Logging utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    """
    
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(
    name: str = "gnn_dta_mtl",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """
    Logger for experiment tracking.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        verbose: bool = True
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of experiment
            log_dir: Directory for log files
            verbose: Whether to print to console
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file),
            console=verbose
        )
        
        self.metrics_history = []
        self.epoch = 0
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Optional[dict] = None
    ):
        """
        Log epoch results.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics
        """
        self.epoch = epoch
        
        log_str = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        
        if metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            log_str += f", {metrics_str}"
        
        self.logger.info(log_str)
        
        # Store history
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        if metrics:
            record.update(metrics)
        self.metrics_history.append(record)
    
    def log_fold(self, fold: int, n_folds: int):
        """Log fold start."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"FOLD {fold + 1}/{n_folds}")
        self.logger.info(f"{'='*60}")
    
    def log_summary(self, summary: dict):
        """Log experiment summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("="*60)
        
        for key, value in summary.items():
            if isinstance(value, dict):
                self.logger.info(f"\n{key}:")
                for k, v in value.items():
                    self.logger.info(f"  {k}: {v}")
            else:
                self.logger.info(f"{key}: {value}")
    
    def save_history(self, path: Optional[str] = None):
        """Save metrics history to CSV."""
        if not self.metrics_history:
            return
        
        import pandas as pd
        
        if path is None:
            path = self.log_dir / f"{self.experiment_name}_history.csv"
        
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(path, index=False)
        self.logger.info(f"History saved to {path}")


def get_logger(name: str = "gnn_dta_mtl") -> logging.Logger:
    """
    Get or create logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that plays nicely with tqdm progress bars.
    """
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)