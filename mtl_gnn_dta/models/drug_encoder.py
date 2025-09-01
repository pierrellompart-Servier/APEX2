"""Drug encoder using Graph Convolutional Networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from typing import List, Optional


class DrugGCN(nn.Module):
    """Graph Convolutional Network for drug/ligand representation"""
    
    def __init__(self,
                 node_in_dim: List[int] = [66, 1],  # Atom features + edge features
                 node_h_dims: List[int] = [128, 64],
                 fc_dims: List[int] = [1024, 128],
                 dropout: float = 0.2):
        """
        Initialize Drug GCN
        
        Args:
            node_in_dim: Input dimensions for node features
            node_h_dims: Hidden dimensions for GCN layers
            fc_dims: Hidden dimensions for FC layers
            dropout: Dropout rate
        """
        super(DrugGCN, self).__init__()
        
        self.dropout = dropout
        
        # Initial embedding for node features
        self.node_embedding = nn.Linear(sum(node_in_dim), node_h_dims[0])
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        in_dim = node_h_dims[0]
        for out_dim in node_h_dims[1:]:
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
            in_dim = out_dim
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_dim = node_h_dims[-1] * 2  # Concatenate mean and max pooling
        for out_dim in fc_dims:
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        # Batch normalization
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in node_h_dims[1:]
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass through drug GCN
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment for each node
        
        Returns:
            Drug representation [batch_size, fc_dims[-1]]
        """
        # Initial node embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # GCN layers
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Global pooling (concatenate mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # FC layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x


class DrugGIN(nn.Module):
    """Graph Isomorphism Network for drug representation"""
    
    def __init__(self,
                 node_in_dim: int = 66,
                 hidden_dims: List[int] = [128, 256, 128],
                 fc_dims: List[int] = [1024, 128],
                 dropout: float = 0.2):
        """
        Initialize Drug GIN
        
        Args:
            node_in_dim: Input dimension for node features
            hidden_dims: Hidden dimensions for GIN layers
            fc_dims: Hidden dimensions for FC layers
            dropout: Dropout rate
        """
        super(DrugGIN, self).__init__()
        
        # GIN layers
        self.gin_layers = nn.ModuleList()
        in_dim = node_in_dim
        
        for out_dim in hidden_dims:
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            self.gin_layers.append(GINConv(mlp))
            in_dim = out_dim
        
        # FC layers
        self.fc_layers = nn.ModuleList()
        in_dim = hidden_dims[-1]
        for out_dim in fc_dims:
            self.fc_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass through drug GIN
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for each node
        
        Returns:
            Drug representation [batch_size, fc_dims[-1]]
        """
        # GIN layers
        for gin in self.gin_layers:
            x = gin(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_add_pool(x, batch)
        
        # FC layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class MessagePassingNet(nn.Module):
    """Custom message passing network for drug representation"""
    
    def __init__(self,
                 node_features: int = 66,
                 edge_features: int = 6,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 output_dim: int = 128,
                 dropout: float = 0.2):
        """
        Initialize Message Passing Network
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension
            n_layers: Number of message passing layers
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(MessagePassingNet, self).__init__()
        
        self.n_layers = n_layers
        
        # Node embedding
        self.node_embed = nn.Linear(node_features, hidden_dim)
        
        # Edge embedding
        self.edge_embed = nn.Linear(edge_features, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(n_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass through message passing network
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch assignment
        
        Returns:
            Drug representation
        """
        # Initial embeddings
        h = self.node_embed(x)
        edge_feat = self.edge_embed(edge_attr)
        
        # Message passing
        for i in range(self.n_layers):
            # Aggregate messages
            row, col = edge_index
            messages = []
            
            for j in range(edge_index.shape[1]):
                src, dst = row[j], col[j]
                msg = torch.cat([h[src], h[dst], edge_feat[j]], dim=0)
                msg = self.mp_layers[i](msg)
                messages.append(msg)
            
            # Update node representations
            h = h + torch.stack(messages).mean(dim=0)
            h = self.layer_norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        # Global pooling
        h = global_mean_pool(h, batch)
        
        # Output
        return self.output(h)
