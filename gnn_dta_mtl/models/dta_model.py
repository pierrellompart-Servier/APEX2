"""
Drug-Target Affinity models
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .protein_model import Prot3DGraphModel
from .drug_model import DrugGVPModel


class DTAModel(nn.Module):
    """
    Drug-Target Affinity prediction model.
    
    Args:
        prot_emb_dim: Protein ESM embedding dimension
        prot_gcn_dims: Protein GCN layer dimensions
        prot_fc_dims: Protein FC layer dimensions
        drug_node_in_dim: Drug node input dimensions
        drug_node_h_dims: Drug node hidden dimensions
        drug_fc_dims: Drug FC layer dimensions
        mlp_dims: MLP layer dimensions
        mlp_dropout: MLP dropout rate
    """
    
    def __init__(
        self,
        prot_emb_dim: int = 1280,
        prot_gcn_dims: List[int] = [128, 256, 256],
        prot_fc_dims: List[int] = [1024, 128],
        drug_node_in_dim: List[int] = [66, 1],
        drug_node_h_dims: List[int] = [128, 64],
        drug_edge_in_dim: List[int] = [16, 1],
        drug_edge_h_dims: List[int] = [32, 1],
        drug_fc_dims: List[int] = [1024, 128],
        mlp_dims: List[int] = [1024, 512],
        mlp_dropout: float = 0.25
    ):
        super().__init__()
        
        # Drug encoder
        self.drug_model = DrugGVPModel(
            node_in_dim=drug_node_in_dim,
            node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim,
            edge_h_dim=drug_edge_h_dims
        )
        drug_emb_dim = drug_node_h_dims[0]
        
        # Protein encoder
        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim,
            d_gcn=prot_gcn_dims
        )
        prot_emb_dim = prot_gcn_dims[-1]
        
        # Drug FC layers
        self.drug_fc = self._get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout,
            no_last_dropout=True,
            no_last_activation=True
        )
        
        # Protein FC layers
        self.prot_fc = self._get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout,
            no_last_dropout=True,
            no_last_activation=True
        )
        
        # Combined MLP
        self.top_fc = self._get_fc_layers(
            [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout,
            no_last_dropout=True,
            no_last_activation=True
        )
    
    def _get_fc_layers(
        self,
        hidden_sizes: List[int],
        dropout: float = 0,
        batchnorm: bool = False,
        no_last_dropout: bool = True,
        no_last_activation: bool = True
    ) -> nn.Sequential:
        """Build FC layers."""
        act_fn = nn.LeakyReLU()
        layers = []
        
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, xd, xp):
        """
        Forward pass.
        
        Args:
            xd: Drug graph data
            xp: Protein graph data
            
        Returns:
            Predicted affinity value
        """
        # Encode drug and protein
        xd = self.drug_model(xd)
        xp = self.prot_model(xp)
        
        # Process through FC layers
        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)
        
        # Concatenate and predict
        x = torch.cat([xd, xp], dim=1)
        x = self.top_fc(x)
        
        return x


class MTL_DTAModel(DTAModel):
    """
    Multi-Task Learning DTA model.
    
    Args:
        task_names: List of task names
        Other args same as DTAModel
    """
    
    def __init__(
        self,
        task_names: List[str] = ['pKi', 'pEC50', 'pKd', 'pIC50'],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.task_names = task_names
        self.n_tasks = len(task_names)
        
        # Remove the final layer from top_fc
        self.shared_fc = self.top_fc[:-1]
        
        # Task-specific heads
        mlp_dims = kwargs.get('mlp_dims', [1024, 512])
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(mlp_dims[-1], 1)
            for task in task_names
        })
    
    def forward(self, xd, xp):
        """
        Forward pass for multi-task learning.
        
        Args:
            xd: Drug graph data
            xp: Protein graph data
            
        Returns:
            Predictions for all tasks [batch_size, n_tasks]
        """
        # Encode drug and protein
        xd = self.drug_model(xd)
        xp = self.prot_model(xp)
        
        # Process through FC layers
        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)
        
        # Concatenate and process through shared layers
        x = torch.cat([xd, xp], dim=1)
        shared_repr = self.shared_fc(x)
        
        # Generate predictions for each task
        outputs = []
        for task in self.task_names:
            task_pred = self.task_heads[task](shared_repr)
            outputs.append(task_pred)
        
        # Stack outputs: [batch_size, n_tasks]
        return torch.cat(outputs, dim=1)