"""Drug featurization module"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from torch_geometric.data import Data
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class DrugFeaturizer:
    """Featurize drug molecules as graphs"""
    
    # Atom features
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    HYBRIDIZATIONS = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    
    @staticmethod
    def one_hot_encoding(value, categories):
        """Create one-hot encoding"""
        encoding = [0] * len(categories)
        if value in categories:
            encoding[categories.index(value)] = 1
        return encoding
    
    @staticmethod
    def atom_features(atom):
        """Extract features for a single atom"""
        features = []
        
        # Atom type (one-hot)
        features.extend(DrugFeaturizer.one_hot_encoding(
            atom.GetSymbol(), DrugFeaturizer.ATOM_TYPES
        ))
        
        # Degree
        features.extend(DrugFeaturizer.one_hot_encoding(
            atom.GetDegree(), [0, 1, 2, 3, 4, 5]
        ))
        
        # Hybridization
        features.extend(DrugFeaturizer.one_hot_encoding(
            atom.GetHybridization(), DrugFeaturizer.HYBRIDIZATIONS
        ))
        
        # Implicit valence
        features.extend(DrugFeaturizer.one_hot_encoding(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]
        ))
        
        # Formal charge
        features.append(atom.GetFormalCharge())
        
        # Number of radical electrons
        features.append(atom.GetNumRadicalElectrons())
        
        # Aromatic
        features.append(int(atom.GetIsAromatic()))
        
        # Ring membership
        features.append(int(atom.IsInRing()))
        
        # Chirality
        features.append(int(atom.HasProp('_ChiralityPossible')))
        
        # Additional features
        features.append(atom.GetMass())
        features.append(atom.GetTotalNumHs())
        features.append(atom.GetNumExplicitHs())
        
        return features
    
    @staticmethod
    def bond_features(bond):
        """Extract features for a single bond"""
        bond_type = bond.GetBondType()
        features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        return features
    
    @staticmethod
    def featurize_drug(sdf_path: str) -> Optional[Data]:
        """
        Featurize drug molecule from SDF file
        
        Args:
            sdf_path: Path to SDF file
        
        Returns:
            PyTorch Geometric Data object
        """
        try:
            mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
            if mol is None:
                return None
            
            # Node features
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(DrugFeaturizer.atom_features(atom))
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Edge features and connectivity
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions
                edge_indices.extend([[i, j], [j, i]])
                
                bond_feat = DrugFeaturizer.bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # Handle molecules with no bonds
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 6), dtype=torch.float)
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return data
            
        except Exception as e:
            logger.warning(f"Failed to featurize drug from {sdf_path}: {e}")
            return None
    
    @staticmethod
    def featurize_from_smiles(smiles: str, add_3d: bool = True) -> Optional[Data]:
        """
        Featurize drug molecule from SMILES string
        
        Args:
            smiles: SMILES string
            add_3d: Whether to add 3D coordinates
        
        Returns:
            PyTorch Geometric Data object
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if add_3d:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(mol)
            
            # Node features
            node_features = []
            for atom in mol.GetAtoms():
                node_features.append(DrugFeaturizer.atom_features(atom))
            
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Edge features and connectivity
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                edge_indices.extend([[i, j], [j, i]])
                bond_feat = DrugFeaturizer.bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 6), dtype=torch.float)
            
            # Add 3D coordinates if available
            if add_3d and mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                pos = []
                for i in range(mol.GetNumAtoms()):
                    pos.append(list(conf.GetAtomPosition(i)))
                pos = torch.tensor(pos, dtype=torch.float)
            else:
                pos = None
            
            # Create PyG Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            if pos is not None:
                data.pos = pos
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to featurize from SMILES {smiles}: {e}")
            return None
    
    @staticmethod
    def compute_molecular_descriptors(mol) -> Dict[str, float]:
        """
        Compute molecular descriptors for a molecule
        
        Args:
            mol: RDKit molecule object
        
        Returns:
            Dictionary of molecular descriptors
        """
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'RingCount': Descriptors.RingCount(mol),
            'FractionCsp3': Descriptors.FractionCsp3(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'MolMR': Descriptors.MolMR(mol),
            'BertzCT': Descriptors.BertzCT(mol)
        }
        return descriptors
