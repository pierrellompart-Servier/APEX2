"""
Feature extraction utilities for GNN-DTA-MTL
"""

from .drug_graph import featurize_drug, sdf_to_graphs
from .protein_graph import featurize_protein_graph, pdb_to_graphs
from .esm_embeddings import ESMEmbedder, get_esm_embedding
from .structure_processing import (
    StructureProcessor,
    StructureChunkLoader,
    extract_backbone_coords
)

__all__ = [
    # Drug features
    'featurize_drug',
    'sdf_to_graphs',
    
    # Protein features
    'featurize_protein_graph',
    'pdb_to_graphs',
    
    # ESM embeddings
    'ESMEmbedder',
    'get_esm_embedding',
    
    # Structure processing
    'StructureProcessor',
    'StructureChunkLoader',
    'extract_backbone_coords'
]