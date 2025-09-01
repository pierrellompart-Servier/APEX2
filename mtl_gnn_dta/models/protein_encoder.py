"""Protein encoder using Graph Convolutional Networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import List, Optional


class ProteinGCN(nn.Module):
    """Graph Convolutional Network for protein representation"""
    
    def __init__(self, 
                 emb_dim: int = 1280,  # ESM-2 embedding dimension
                 gcn_dims: List[int] = [128, 256, 256],
                 fc_dims: List[int] = [1024, 128],
                 dropout: float = 0.2):
        """
        Initialize Protein GCN
        
        Args:
            emb_dim: Input embedding dimension (ESM-2)
            gcn_dims: Hidden dimensions for GCN layers
            fc_dims: Hidden dimensions for FC layers
            dropout: Dropout rate
        """
        super(ProteinGCN, self).__init__()
        
        self.emb_dim = emb_dim
        self.dropout = dropout
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        in_dim = emb_dim
        for out_dim in gcn_dims:
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_dim = gcn_dims[-1]
        for out_dim in fc_dims:
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in gcn_dims
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass through protein GCN
        
        Args:
            x: Node features [num_nodes, emb_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for each node
        
        Returns:
            Protein representation [batch_size, fc_dims[-1]]
        """
        # GCN layers
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # FC layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x


class ProteinCNN(nn.Module):
    """1D CNN for protein sequence representation (alternative to GCN)"""
    
    def __init__(self,
                 emb_dim: int = 1280,
                 cnn_channels: List[int] = [256, 512, 1024],
                 kernel_sizes: List[int] = [7, 5, 3],
                 fc_dims: List[int] = [1024, 128],
                 dropout: float = 0.2):
        """
        Initialize Protein CNN
        
        Args:
            emb_dim: Input embedding dimension
            cnn_channels: Number of channels for each CNN layer
            kernel_sizes: Kernel sizes for each CNN layer
            fc_dims: Hidden dimensions for FC layers
            dropout: Dropout rate
        """
        super(ProteinCNN, self).__init__()
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = emb_dim
        
        for out_channels, kernel_size in zip(cnn_channels, kernel_sizes):
            self.cnn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        self.fc_layers = nn.ModuleList()
        in_dim = cnn_channels[-1]
        for out_dim in fc_dims:
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through protein CNN
        
        Args:
            x: Sequence features [batch_size, seq_len, emb_dim]
        
        Returns:
            Protein representation [batch_size, fc_dims[-1]]
        """
        # Transpose for CNN (batch, channels, length)
        x = x.transpose(1, 2)
        
        # CNN layers
        for cnn in self.cnn_layers:
            x = cnn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # FC layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling for graph representations"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-based pooling
        
        Args:
            x: Node features [num_nodes, features]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Pooled features [batch_size, features]
        """
        # Compute attention scores
        scores = self.attention(x)  # [num_nodes, 1]
        
        # Apply softmax per graph
        batch_size = batch.max().item() + 1
        pooled = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                graph_x = x[mask]
                graph_scores = scores[mask]
                graph_scores = F.softmax(graph_scores, dim=0)
                pooled_graph = (graph_x * graph_scores).sum(dim=0)
                pooled.append(pooled_graph)
        
        return torch.stack(pooled)
