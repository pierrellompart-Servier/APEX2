"""
Early stopping utilities for training
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping
        higher_better: Whether higher metric values are better
        delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        patience: int = 20,
        higher_better: bool = True,
        delta: float = 0
    ):
        self.patience = patience
        self.higher_better = higher_better
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = None
    
    def update(self, value: float) -> bool:
        """
        Update with new metric value.
        
        Args:
            value: Current metric value
            
        Returns:
            True if this is the best value so far
        """
        if self.best_score is None:
            self.best_score = value
            self.best_value = value
            return True
        
        if self.higher_better:
            is_better = value > self.best_score + self.delta
        else:
            is_better = value < self.best_score - self.delta
        
        if is_better:
            self.best_score = value
            self.best_value = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_value = None