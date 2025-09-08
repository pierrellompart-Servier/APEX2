"""
Drug molecule graph featurization
"""

import torch
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
import torch_geometric
import torch_cluster
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from ..utils.constants import ATOM_VOCAB


def onehot_encoder(
    a: List,
    alphabet: List,
    default: Optional[str] = None,
    drop_first: bool = False
) -> np.ndarray:
    """
    One-hot encode categorical features.
    
    Args:
        a: Array of feature values
        alphabet: Valid feature values
        default: Default value for out-of-vocabulary
        drop_first: Whether to drop first column
        
    Returns:
        One-hot encoded array
    """
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]
    
    a = pd.Categorical(a, categories=alphabet)
    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def _build_atom_feature(mol: Chem.Mol) -> np.ndarray:
    """
    Build atom features for molecule.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Atom feature matrix
    """
    feature_alphabet = {
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1)
    }
    
    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(
            feature,
            alphabet=feature_alphabet[attr][0],
            default=feature_alphabet[attr][1],
            drop_first=(attr in ['GetIsAromatic'])
        )
        atom_feature = feature if atom_feature is None else \
                      np.concatenate((atom_feature, feature), axis=1)
    
    return atom_feature.astype(np.float32)


def _build_edge_feature(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    D_max: float = 4.5,
    num_rbf: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge features from coordinates.
    
    Args:
        coords: Atom coordinates
        edge_index: Edge indices
        D_max: Maximum distance
        num_rbf: Number of RBF kernels
        
    Returns:
        Edge scalar and vector features
    """
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    
    # RBF encoding of distances
    distances = E_vectors.norm(dim=-1)
    D_mu = torch.linspace(0, D_max, num_rbf)
    D_mu = D_mu.view([1, -1])
    D_sigma = D_max / num_rbf
    D_expand = distances.unsqueeze(-1)
    rbf = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    
    edge_s = rbf
    edge_v = E_vectors.unsqueeze(-2) / (E_vectors.norm(dim=-1, keepdim=True).unsqueeze(-2) + 1e-8)
    
    # Handle NaNs
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    
    return edge_s, edge_v


def featurize_drug(
    sdf_path: str,
    name: Optional[str] = None,
    edge_cutoff: float = 4.5,
    num_rbf: int = 16
) -> torch_geometric.data.Data:
    """
    Featurize drug molecule from SDF file.
    
    Args:
        sdf_path: Path to SDF file
        name: Molecule name
        edge_cutoff: Distance cutoff for edges
        num_rbf: Number of RBF kernels
        
    Returns:
        PyG Data object with drug graph
    """
    mol = rdkit.Chem.MolFromMolFile(sdf_path)
    if mol is None:
        raise ValueError(f"Could not read molecule from {sdf_path}")
    
    conf = mol.GetConformer()
    
    with torch.no_grad():
        # Get coordinates
        coords = conf.GetPositions()
        coords = torch.as_tensor(coords, dtype=torch.float32)
        
        # Build atom features
        atom_feature = _build_atom_feature(mol)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        
        # Build edge index using radius graph
        edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)
        
        # Build edge features
        edge_s, edge_v = _build_edge_feature(
            coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf
        )
        
        # Node features
        node_s = atom_feature
        node_v = coords.unsqueeze(1)
    
    data = torch_geometric.data.Data(
        x=coords,
        edge_index=edge_index,
        name=name,
        node_v=node_v,
        node_s=node_s,
        edge_v=edge_v,
        edge_s=edge_s
    )
    
    return data


def sdf_to_graphs(
    data_list: Dict[str, str]
) -> Dict[str, torch_geometric.data.Data]:
    """
    Convert multiple SDF files to graphs.
    
    Args:
        data_list: Dictionary mapping drug keys to SDF paths
        
    Returns:
        Dictionary of drug graphs
    """
    graphs = {}
    for key, sdf_path in tqdm(data_list.items(), desc='Featurizing drugs'):
        try:
            graphs[key] = featurize_drug(sdf_path, name=key)
        except Exception as e:
            print(f"Failed to featurize {key}: {e}")
    return graphs