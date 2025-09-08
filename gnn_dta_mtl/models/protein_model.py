"""
Protein graph neural network model
"""

import torch
import torch.nn as nn
import torch_geometric
from typing import List, Optional


class Prot3DGraphModel(nn.Module):
    """
    3D protein structure graph model with ESM embeddings.
    
    Args:
        d_vocab: Vocabulary size for amino acids
        d_embed: Embedding dimension
        d_dihedrals: Dihedral features dimension
        d_pretrained_emb: Pretrained embedding dimension (ESM)
        d_edge: Edge features dimension
        d_gcn: GCN layer dimensions
    """
    
    def __init__(
        self,
        d_vocab: int = 21,
        d_embed: int = 20,
        d_dihedrals: int = 6,
        d_pretrained_emb: int = 1280,
        d_edge: int = 39,
        d_gcn: List[int] = [128, 256, 256]
    ):
        super().__init__()
        
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        
        for i in range(len(gcn_layer_sizes) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i],
                    gcn_layer_sizes[i + 1],
                    edge_dim=d_gcn_in
                ),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())
        
        self.gcn = torch_geometric.nn.Sequential('x, edge_index, edge_attr', layers)
        self.pool = torch_geometric.nn.global_mean_pool
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object with protein features
            
        Returns:
            Graph-level protein representation
        """
        x, edge_index = data.seq, data.edge_index
        batch = data.batch
        
        # Node features
        x = self.embed(x)
        s = data.node_s  # Dihedral features
        emb = data.seq_emb  # ESM embeddings
        x = torch.cat([x, s, emb], dim=-1)
        
        # Edge features
        edge_attr = data.edge_s
        
        # Projections
        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)
        
        # GCN layers
        x = self.gcn(x, edge_index, edge_attr)
        
        # Global pooling
        x = torch_geometric.nn.global_mean_pool(x, batch)
        
        return x