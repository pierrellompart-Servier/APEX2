"""
Protein graph featurization functions
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_cluster
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ..utils.constants import LETTER_TO_NUM


def _normalize(tensor, dim=-1):
    """Normalize tensor along dimension without NaNs."""
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """
    Returns an RBF embedding of distances.
    
    Args:
        D: Distance tensor
        D_min: Minimum distance
        D_max: Maximum distance
        D_count: Number of RBF kernels
        device: Device to use
        
    Returns:
        RBF embedding tensor
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _dihedrals(X, eps=1e-7):
    """
    Calculate dihedral angles from backbone coordinates.
    
    Args:
        X: Coordinates tensor [n_residues, 4, 3]
        eps: Small value for numerical stability
        
    Returns:
        Dihedral angle features
    """
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]
    
    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)
    
    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    
    # Pad to maintain shape
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    
    # Lift to circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index, num_embeddings=None, period_range=[2, 1000]):
    """
    Generate sinusoidal positional embeddings.
    
    Args:
        edge_index: Edge indices [2, n_edges]
        num_embeddings: Number of embedding dimensions
        period_range: Range of periods for sinusoidal functions
        
    Returns:
        Positional embeddings
    """
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X):
    """
    Calculate orientation features from CA coordinates.
    
    Args:
        X: CA coordinates [n_residues, 3]
        
    Returns:
        Orientation features
    """
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    """
    Calculate virtual sidechain vectors.
    
    Args:
        X: Backbone coordinates [n_residues, 4, 3]
        
    Returns:
        Sidechain vectors
    """
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _quaternions(R):
    """
    Convert rotation matrix to quaternions.
    
    Args:
        R: Rotation matrices
        
    Returns:
        Quaternion representations
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    
    _R = lambda i,j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q


def _local_frame(X, edge_index, eps=1e-6):
    """
    Calculate local coordinate frame features.
    
    Args:
        X: CA coordinates
        edge_index: Edge indices
        eps: Small value for numerical stability
        
    Returns:
        Local frame features
    """
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]
    
    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)
    
    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)
    
    dX = X[edge_index[1]] - X[edge_index[0]]
    dX = _normalize(dX, dim=-1)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)
    R = torch.bmm(O[edge_index[0]].transpose(-1,-2), O[edge_index[1]])
    Q = _quaternions(R)
    O_features = torch.cat((dU, Q), dim=-1)
    
    return O_features


def featurize_protein_graph(
    protein: Dict,
    name: Optional[str] = None,
    num_pos_emb: int = 16,
    num_rbf: int = 16,
    contact_cutoff: float = 8.0
) -> torch_geometric.data.Data:
    """
    Featurize protein structure as a graph.
    
    Args:
        protein: Dictionary with protein data
        name: Protein name
        num_pos_emb: Number of positional embeddings
        num_rbf: Number of RBF kernels
        contact_cutoff: Distance cutoff for contacts
        
    Returns:
        PyG Data object with protein graph
    """
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)
        
        # Load ESM embeddings if available
        if 'embed' in protein:
            seq_emb = torch.load(protein['embed'])
        else:
            seq_emb = torch.zeros(len(seq), 1280)
        
        # Handle missing coordinates
        mask = torch.isfinite(coords.sum(dim=(1,2)))
        coords[~mask] = np.inf
        
        # Extract CA coordinates
        X_ca = coords[:, 1]
        ca_mask = torch.isfinite(X_ca.sum(dim=1))
        ca_mask = ca_mask.float()
        ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)
        
        # Calculate distances
        dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)
        D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca**2, 2) + 1e-6)
        
        # Build contact graph
        edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1))
        edge_index = edge_index.t().contiguous()
        
        # Calculate features
        O_feature = _local_frame(X_ca, edge_index)
        pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)
        
        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)
        
        # Node features
        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        
        # Edge features
        edge_s = torch.cat([rbf, O_feature, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        
        # Handle NaNs
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                             (node_s, node_v, edge_s, edge_v))
        
    data = torch_geometric.data.Data(
        x=X_ca,
        seq=seq,
        name=name,
        node_s=node_s,
        node_v=node_v,
        edge_s=edge_s,
        edge_v=edge_v,
        edge_index=edge_index,
        mask=mask,
        seq_emb=seq_emb
    )
    
    return data


def pdb_to_graphs(
    prot_data: Dict[str, Dict],
    params: Dict
) -> Dict[str, torch_geometric.data.Data]:
    """
    Convert multiple protein structures to graphs.
    
    Args:
        prot_data: Dictionary of protein data
        params: Featurization parameters
        
    Returns:
        Dictionary of protein graphs
    """
    graphs = {}
    for key, struct in tqdm(prot_data.items(), desc='Featurizing proteins'):
        graphs[key] = featurize_protein_graph(struct, name=key, **params)
    return graphs