"""
Drug graph neural network model using GVP
"""

import torch
import torch.nn as nn
import torch_geometric
from typing import List, Tuple
from .gvp_layers import LayerNorm, GVP, GVPConvLayer


class DrugGVPModel(nn.Module):
    """
    Drug molecule GVP model.
    
    Args:
        node_in_dim: Input node dimensions (scalar, vector)
        node_h_dim: Hidden node dimensions (scalar, vector)
        edge_in_dim: Input edge dimensions (scalar, vector)
        edge_h_dim: Hidden edge dimensions (scalar, vector)
        num_layers: Number of GVP layers
        drop_rate: Dropout rate
    """
    
    def __init__(
        self,
        node_in_dim: Tuple[int, int] = [66, 1],
        node_h_dim: Tuple[int, int] = [128, 64],
        edge_in_dim: Tuple[int, int] = [16, 1],
        edge_h_dim: Tuple[int, int] = [32, 1],
        num_layers: int = 3,
        drop_rate: float = 0.1
    ):
        super().__init__()
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0))
        )
    
    def forward(self, xd):
        """
        Forward pass.
        
        Args:
            xd: PyG Data object with drug features
            
        Returns:
            Graph-level drug representation
        """
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch
        
        # Initial transformations
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        # GVP layers
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Output transformation
        out = self.W_out(h_V)
        
        # Global pooling
        out = torch_geometric.nn.global_add_pool(out, batch)
        
        return out