"""Logging utilities for MTL-GNN-DTA"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    name: str = "mtl_gnn_dta"
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Directory for log files
        name: Logger name
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        if not log_file.endswith('.log'):
            log_file = f"{log_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars"""
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except ImportError:
            pass


def get_logger(name: str = "mtl_gnn_dta") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


class LoggerContext:
    """Context manager for temporary logging configuration"""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        self.log_level = log_level
        self.log_file = log_file
        self.original_level = None
        self.logger = None
    
    def __enter__(self):
        self.logger = setup_logging(self.log_level, self.log_file)
        self.original_level = self.logger.level
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger and self.original_level is not None:
            self.logger.setLevel(self.original_level)
