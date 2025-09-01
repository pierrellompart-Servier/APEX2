"""
MTL-GNN-DTA: Multi-Task Learning Graph Neural Network for Drug-Target Affinity Prediction
"""

from mtl_gnn_dta.__version__ import __version__

# Core imports
from mtl_gnn_dta.core.config import Config
from mtl_gnn_dta.core.predictor import AffinityPredictor
from mtl_gnn_dta.core.trainer import ModelTrainer

# Model imports
from mtl_gnn_dta.models.dta_model import MTL_DTAModel
from mtl_gnn_dta.models.protein_encoder import ProteinGCN
from mtl_gnn_dta.models.drug_encoder import DrugGCN
from mtl_gnn_dta.models.losses import MaskedMSELoss

# Data imports
from mtl_gnn_dta.data.dataset import MTL_DTA
from mtl_gnn_dta.data.loaders import create_data_loaders

# Feature imports
from mtl_gnn_dta.features.protein_features import ProteinFeaturizer
from mtl_gnn_dta.features.drug_features import DrugFeaturizer

# Training imports
from mtl_gnn_dta.training.trainer import Trainer
from mtl_gnn_dta.training.evaluator import Evaluator

__all__ = [
    '__version__',
    'Config',
    'AffinityPredictor',
    'ModelTrainer',
    'MTL_DTAModel',
    'ProteinGCN',
    'DrugGCN',
    'MaskedMSELoss',
    'MTL_DTA',
    'create_data_loaders',
    'ProteinFeaturizer',
    'DrugFeaturizer',
    'Trainer',
    'Evaluator'
]
