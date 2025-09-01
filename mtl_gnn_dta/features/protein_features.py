"""Protein featurization module"""

import torch
import numpy as np
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class ProteinFeaturizer:
    """Featurize protein structures as graphs"""
    
    def __init__(self, 
                 edge_threshold: float = 8.0,
                 use_esm_embeddings: bool = True):
        """
        Initialize protein featurizer
        
        Args:
            edge_threshold: Distance threshold for creating edges (Angstroms)
            use_esm_embeddings: Whether to use ESM embeddings
        """
        self.parser = PDBParser(QUIET=True)
        self.edge_threshold = edge_threshold
        self.use_esm_embeddings = use_esm_embeddings
    
    @staticmethod
    def extract_backbone_coords(structure, pdb_id: str, pdb_path: str) -> Tuple:
        """Extract backbone coordinates from protein structure"""
        try:
            model = structure[0]
            
            for chain in model:
                residues = [r for r in chain if is_aa(r)]
                if len(residues) == 0:
                    continue
                
                # Extract sequence
                seq = ''.join([seq1(r.resname) for r in residues])
                
                # Extract coordinates for backbone atoms
                coords = {'N': [], 'CA': [], 'C': [], 'O': []}
                
                for residue in residues:
                    for atom_name in ['N', 'CA', 'C', 'O']:
                        if atom_name in residue:
                            coords[atom_name].append(residue[atom_name].coord.tolist())
                        else:
                            coords[atom_name].append([0.0, 0.0, 0.0])
                
                return seq, coords, chain.id
            
            return None, None, None
            
        except Exception as e:
            logger.warning(f"Failed to extract backbone from {pdb_id}: {e}")
            return None, None, None
    
    @staticmethod
    def compute_distance_matrix(coords: List) -> np.ndarray:
        """Compute distance matrix from coordinates"""
        coords_array = np.array(coords)
        n_residues = len(coords_array)
        dist_matrix = np.zeros((n_residues, n_residues))
        
        for i in range(n_residues):
            for j in range(i+1, n_residues):
                # Use CA atom for distance calculation
                ca_i = coords_array[i][1]  # CA is second in [N, CA, C, O]
                ca_j = coords_array[j][1]
                dist = np.linalg.norm(np.array(ca_i) - np.array(ca_j))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    @staticmethod
    def distance_to_adjacency(dist_matrix: np.ndarray, threshold: float = 8.0) -> torch.Tensor:
        """Convert distance matrix to adjacency matrix"""
        adjacency = (dist_matrix < threshold) & (dist_matrix > 0)
        edge_list = np.array(np.where(adjacency))
        return torch.tensor(edge_list, dtype=torch.long)
    
    def featurize_protein_graph(self, protein_json: Dict) -> Optional[Data]:
        """
        Create graph representation of protein
        
        Args:
            protein_json: Dictionary with protein information
        
        Returns:
            PyTorch Geometric Data object
        """
        try:
            # Load ESM embedding if available
            if self.use_esm_embeddings and 'embed' in protein_json:
                embed_path = protein_json['embed']
                if os.path.exists(embed_path):
                    embedding = torch.load(embed_path, map_location='cpu')
                else:
                    # Use one-hot encoding as fallback
                    seq_len = len(protein_json['seq'])
                    embedding = self.sequence_to_onehot(protein_json['seq'])
            else:
                # Use one-hot encoding
                embedding = self.sequence_to_onehot(protein_json['seq'])
            
            # Handle different embedding shapes
            if len(embedding.shape) == 3:
                embedding = embedding[0]  # Remove batch dimension
            elif len(embedding.shape) == 1:
                embedding = embedding.unsqueeze(0)
            
            # Compute adjacency from coordinates
            coords = protein_json['coords']
            dist_matrix = self.compute_distance_matrix(coords)
            edge_index = self.distance_to_adjacency(dist_matrix, self.edge_threshold)
            
            # Add 3D coordinates as node positions
            pos = torch.tensor(coords, dtype=torch.float)
            if len(pos.shape) == 3:
                # Flatten to [n_residues, 12] (4 atoms * 3 coords)
                pos = pos.reshape(pos.shape[0], -1)
            
            # Create PyG Data object
            data = Data(
                x=embedding, 
                edge_index=edge_index,
                pos=pos
            )
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to featurize protein {protein_json.get('name', 'unknown')}: {e}")
            return None
    
    def featurize_from_pdb(self, 
                          pdb_path: str,
                          sequence: Optional[str] = None,
                          embedding: Optional[torch.Tensor] = None) -> Optional[Data]:
        """
        Featurize protein from PDB file
        
        Args:
            pdb_path: Path to PDB file
            sequence: Optional protein sequence
            embedding: Optional pre-computed embedding
        
        Returns:
            PyTorch Geometric Data object
        """
        try:
            pdb_id = os.path.basename(pdb_path).split('.')[0]
            structure = self.parser.get_structure(pdb_id, pdb_path)
            
            # Extract backbone coordinates
            seq, coords, chain_id = self.extract_backbone_coords(structure, pdb_id, pdb_path)
            
            if seq is None:
                return None
            
            # Use provided sequence if available
            if sequence is not None:
                seq = sequence
            
            # Stack coordinates
            coords_stacked = []
            for i in range(len(coords["N"])):
                coord_group = []
                for atom in ["N", "CA", "C", "O"]:
                    coord_group.append(coords[atom][i])
                coords_stacked.append(coord_group)
            
            # Create protein info dictionary
            protein_json = {
                'name': pdb_id,
                'seq': seq,
                'coords': coords_stacked,
                'chain': chain_id
            }
            
            # Add embedding if provided
            if embedding is not None:
                protein_json['embed'] = embedding
            
            return self.featurize_protein_graph(protein_json)
            
        except Exception as e:
            logger.warning(f"Failed to featurize from PDB {pdb_path}: {e}")
            return None
    
    @staticmethod
    def sequence_to_onehot(sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to one-hot encoding"""
        aa_dict = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20  # Unknown
        }
        
        seq_len = len(sequence)
        encoding = torch.zeros(seq_len, 21)
        
        for i, aa in enumerate(sequence):
            if aa in aa_dict:
                encoding[i, aa_dict[aa]] = 1
            else:
                encoding[i, 20] = 1  # Unknown amino acid
        
        return encoding
    
    @staticmethod
    def compute_secondary_structure(pdb_path: str) -> Optional[np.ndarray]:
        """
        Compute secondary structure using DSSP (if available)
        
        Args:
            pdb_path: Path to PDB file
        
        Returns:
            Secondary structure encoding
        """
        try:
            from Bio.PDB import DSSP
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            model = structure[0]
            dssp = DSSP(model, pdb_path)
            
            # Extract secondary structure
            ss_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
            ss_encoding = []
            
            for residue in dssp:
                ss = residue[2]
                if ss in ss_dict:
                    ss_encoding.append(ss_dict[ss])
                else:
                    ss_encoding.append(7)  # Coil/unknown
            
            return np.array(ss_encoding)
            
        except Exception as e:
            logger.debug(f"Could not compute secondary structure: {e}")
            return None
    
    @staticmethod
    def compute_residue_features(residue) -> List[float]:
        """
        Compute additional residue-level features
        
        Args:
            residue: Bio.PDB residue object
        
        Returns:
            List of features
        """
        features = []
        
        # Hydrophobicity (Kyte-Doolittle scale)
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        resname = residue.resname
        if resname in hydrophobicity:
            features.append(hydrophobicity[resname])
        else:
            features.append(0.0)
        
        # Charge
        charged = {'R': 1, 'K': 1, 'D': -1, 'E': -1}
        if resname in charged:
            features.append(charged[resname])
        else:
            features.append(0.0)
        
        # Size (number of heavy atoms)
        size = {'G': 1, 'A': 2, 'S': 3, 'C': 3, 'P': 4, 'T': 4, 'V': 5,
                'N': 5, 'D': 5, 'I': 6, 'L': 6, 'E': 6, 'Q': 6, 'K': 6,
                'M': 6, 'H': 7, 'F': 9, 'R': 8, 'Y': 10, 'W': 12}
        
        if resname in size:
            features.append(size[resname] / 12.0)  # Normalize
        else:
            features.append(0.5)
        
        return features
