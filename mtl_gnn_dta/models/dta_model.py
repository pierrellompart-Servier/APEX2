"""Main MTL-DTA model combining protein and drug encoders"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from mtl_gnn_dta.models.protein_encoder import ProteinGCN
from mtl_gnn_dta.models.drug_encoder import DrugGCN


class MTL_DTAModel(nn.Module):
    """
    Multi-Task Learning Model for Drug-Target Affinity Prediction
    Combines protein and drug graph representations for multi-task prediction
    """
    
    def __init__(self,
                 task_names: List[str],
                 prot_emb_dim: int = 1280,
                 prot_gcn_dims: List[int] = [128, 256, 256],
                 prot_fc_dims: List[int] = [1024, 128],
                 drug_node_in_dim: List[int] = [66, 1],
                 drug_node_h_dims: List[int] = [128, 64],
                 drug_fc_dims: List[int] = [1024, 128],
                 mlp_dims: List[int] = [1024, 512],
                 mlp_dropout: float = 0.25):
        """
        Initialize MTL-DTA model
        
        Args:
            task_names: List of task names (e.g., ['pKi', 'pEC50', 'pKd', 'pIC50'])
            prot_emb_dim: Protein embedding dimension (ESM-2)
            prot_gcn_dims: Protein GCN hidden dimensions
            prot_fc_dims: Protein FC layer dimensions
            drug_node_in_dim: Drug node input dimensions
            drug_node_h_dims: Drug GCN hidden dimensions
            drug_fc_dims: Drug FC layer dimensions
            mlp_dims: Shared MLP dimensions
            mlp_dropout: Dropout rate
        """
        super(MTL_DTAModel, self).__init__()
        
        self.task_names = task_names
        self.n_tasks = len(task_names)
        
        # Protein encoder
        self.protein_encoder = ProteinGCN(
            emb_dim=prot_emb_dim,
            gcn_dims=prot_gcn_dims,
            fc_dims=prot_fc_dims,
            dropout=mlp_dropout
        )
        
        # Drug encoder
        self.drug_encoder = DrugGCN(
            node_in_dim=drug_node_in_dim,
            node_h_dims=drug_node_h_dims,
            fc_dims=drug_fc_dims,
            dropout=mlp_dropout
        )
        
        # Shared MLP layers
        prot_out_dim = prot_fc_dims[-1]
        drug_out_dim = drug_fc_dims[-1]
        combined_dim = prot_out_dim + drug_out_dim
        
        self.shared_layers = nn.ModuleList()
        in_dim = combined_dim
        for out_dim in mlp_dims:
            self.shared_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(mlp_dims[-1], 1) for task in task_names
        })
        
        self.dropout = nn.Dropout(mlp_dropout)
        self.bn = nn.BatchNorm1d(combined_dim)
    
    def forward(self, drug_batch, protein_batch):
        """
        Forward pass through MTL-DTA model
        
        Args:
            drug_batch: Batched drug graphs
            protein_batch: Batched protein graphs
        
        Returns:
            Tensor of predictions [batch_size, n_tasks]
        """
        # Encode drug
        drug_repr = self.drug_encoder(
            drug_batch.x,
            drug_batch.edge_index,
            drug_batch.edge_attr if hasattr(drug_batch, 'edge_attr') else None,
            drug_batch.batch
        )
        
        # Encode protein
        protein_repr = self.protein_encoder(
            protein_batch.x,
            protein_batch.edge_index,
            protein_batch.batch
        )
        
        # Concatenate representations
        combined = torch.cat([drug_repr, protein_repr], dim=1)
        combined = self.bn(combined)
        combined = F.relu(combined)
        combined = self.dropout(combined)
        
        # Shared layers
        for i, layer in enumerate(self.shared_layers):
            combined = layer(combined)
            combined = F.relu(combined)
            combined = self.dropout(combined)
        
        # Task-specific predictions
        predictions = []
        for task in self.task_names:
            pred = self.task_heads[task](combined)
            predictions.append(pred)
        
        # Stack predictions [batch_size, n_tasks]
        return torch.cat(predictions, dim=1)
    
    def get_embeddings(self, drug_batch, protein_batch):
        """
        Get intermediate embeddings without final prediction
        
        Args:
            drug_batch: Batched drug graphs
            protein_batch: Batched protein graphs
        
        Returns:
            Drug embeddings, protein embeddings, combined embeddings
        """
        # Encode drug
        drug_repr = self.drug_encoder(
            drug_batch.x,
            drug_batch.edge_index,
            drug_batch.edge_attr if hasattr(drug_batch, 'edge_attr') else None,
            drug_batch.batch
        )
        
        # Encode protein
        protein_repr = self.protein_encoder(
            protein_batch.x,
            protein_batch.edge_index,
            protein_batch.batch
        )
        
        # Concatenate representations
        combined = torch.cat([drug_repr, protein_repr], dim=1)
        combined = self.bn(combined)
        combined = F.relu(combined)
        combined = self.dropout(combined)
        
        # Shared layers
        for i, layer in enumerate(self.shared_layers):
            combined = layer(combined)
            if i < len(self.shared_layers) - 1:  # Don't apply ReLU to last layer
                combined = F.relu(combined)
                combined = self.dropout(combined)
        
        return drug_repr, protein_repr, combined


def create_model(config: dict) -> MTL_DTAModel:
    """
    Factory function to create model with configuration
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized MTL_DTAModel
    """
    default_config = {
        'task_names': ['pKi', 'pEC50', 'pKd', 'pIC50'],
        'prot_emb_dim': 1280,
        'prot_gcn_dims': [128, 256, 256],
        'prot_fc_dims': [1024, 128],
        'drug_node_in_dim': [66, 1],
        'drug_node_h_dims': [128, 64],
        'drug_fc_dims': [1024, 128],
        'mlp_dims': [1024, 512],
        'mlp_dropout': 0.25
    }
    
    # Update with provided config
    default_config.update(config)
    
    return MTL_DTAModel(**default_config)
