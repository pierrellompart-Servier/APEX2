"""Configuration management for MTL-GNN-DTA"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration"""
    raw_dir: str = 'data/raw/'
    processed_dir: str = 'data/processed/'
    structures_dir: str = 'data/structures/'
    embeddings_dir: str = 'data/embeddings/'
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    
    # Data filtering
    min_heavy_atoms: int = 5
    max_heavy_atoms: int = 75
    max_mol_weight: float = 1000.0
    min_carbon_atoms: int = 4
    min_le: float = 0.05
    max_le_norm: float = 0.003


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Task configuration
    task_names: List[str] = field(default_factory=lambda: ['pKi', 'pEC50', 'pKd', 'pIC50'])
    
    # Protein encoder
    prot_emb_dim: int = 1280  # ESM-2 embedding dimension
    prot_gcn_dims: List[int] = field(default_factory=lambda: [128, 256, 256])
    prot_fc_dims: List[int] = field(default_factory=lambda: [1024, 128])
    
    # Drug encoder
    drug_node_in_dim: List[int] = field(default_factory=lambda: [66, 1])
    drug_node_h_dims: List[int] = field(default_factory=lambda: [128, 64])
    drug_fc_dims: List[int] = field(default_factory=lambda: [1024, 128])
    
    # MLP layers
    mlp_dims: List[int] = field(default_factory=lambda: [1024, 512])
    mlp_dropout: float = 0.25


@dataclass
class TrainingConfig:
    """Training configuration"""
    n_epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    patience: int = 20
    min_delta: float = 0.0001
    
    # Optimizer
    optimizer: str = 'adam'
    scheduler: str = 'reduce_on_plateau'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Cross-validation
    n_folds: int = 5
    valid_size: float = 0.1
    random_state: int = 42
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = 'experiments/checkpoints/'
    log_dir: str = 'experiments/logs/'


@dataclass
class FeatureConfig:
    """Feature extraction configuration"""
    # ESM model
    esm_model_name: str = 'facebook/esm2_t6_8M_UR50D'
    esm_batch_size: int = 8
    
    # Graph construction
    protein_edge_threshold: float = 8.0  # Angstroms
    
    # Molecular features
    use_polar_hydrogens: bool = True
    compute_gasteiger_charges: bool = True


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.features = FeatureConfig()
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load configuration from file"""
        path = Path(config_path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        self.update(config_dict)
    
    def save(self, config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        config_dict = self.to_dict()
        
        os.makedirs(path.parent, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        if 'training' in config_dict:
            for key, value in config_dict['training'].items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)
        
        if 'features' in config_dict:
            for key, value in config_dict['features'].items():
                if hasattr(self.features, key):
                    setattr(self.features, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'features': asdict(self.features)
        }
    
    def __repr__(self) -> str:
        return f"Config({json.dumps(self.to_dict(), indent=2)})"


def create_config_from_args(args) -> Config:
    """Create configuration from command line arguments"""
    config = Config()
    
    # Update from config file if provided
    if hasattr(args, 'config') and args.config:
        config.load(args.config)
    
    # Override with command line arguments
    for key in ['batch_size', 'n_epochs', 'learning_rate', 'n_folds']:
        if hasattr(args, key) and getattr(args, key) is not None:
            if hasattr(config.data, key):
                setattr(config.data, key, getattr(args, key))
            elif hasattr(config.training, key):
                setattr(config.training, key, getattr(args, key))
    
    return config
