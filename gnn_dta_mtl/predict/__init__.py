"""
Prediction for GNN-DTA-MTL
"""

from .prediction import (
    process_row_simple,
    DTAPredictor,
    predict_affinity
    )
    
__all__ = [
    'process_row_simple',
    'DTAPredictor',
    'predict_affinity'
    ]