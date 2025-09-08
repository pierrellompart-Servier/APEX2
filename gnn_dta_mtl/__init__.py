"""
GNN-DTA-MTL: Graph Neural Networks for Drug-Target Affinity with Multi-Task Learning
"""

__version__ = "1.0.0"

from .models import MTL_DTAModel, DTAModel
from .datasets import MTL_DTA, DTA
from .training import MTLTrainer, CrossValidator
from .evaluation import evaluate_model, plot_results

__all__ = [
    'MTL_DTAModel',
    'DTAModel', 
    'MTL_DTA',
    'DTA',
    'MTLTrainer',
    'CrossValidator',
    'evaluate_model',
    'plot_results'
]