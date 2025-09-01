"""SDF processing utilities for ligand structure standardization"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Define polar atoms
POLAR_ATOMS = {7, 8, 9, 15, 16}  # N, O, F, P, S


def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Standardize RDKit molecule
    
    Args:
        mol: RDKit molecule
    
    Returns:
        Standardized molecule or None
    """
    try:
        # Clean up molecule
        mol = rdMolStandardize.Cleanup(mol)
        
        # Normalize
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        
        # Remove fragments
        mol = rdMolStandardize.FragmentParent(mol)
        
        # Tautomer canonicalization
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
        
        # Clear isotopes
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        
        return mol
        
    except Exception as e:
        logger.warning(f"Failed to standardize molecule: {e}")
        return None


def keep_only_polar_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove non-polar hydrogens from molecule
    
    Args:
        mol: RDKit molecule with explicit hydrogens
    
    Returns:
        Molecule with only polar hydrogens
    """
    try:
        # Find hydrogens to remove
        h_to_remove = []
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:  # Hydrogen
                # Check neighbors
                neighbors = atom.GetNeighbors()
                if neighbors:
                    neighbor = neighbors[0]
                    if neighbor.GetAtomicNum() not in POLAR_ATOMS:
                        h_to_remove.append(atom.GetIdx())
        
        # Remove non-polar hydrogens
        if h_to_remove:
            em = Chem.EditableMol(mol)
            for idx in sorted(h_to_remove, reverse=True):
                em.RemoveAtom(idx)
            mol = em.GetMol()
        
        # Add missing polar hydrogens
        mol.UpdatePropertyCache(strict=False)
        targets = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in POLAR_ATOMS and atom.GetNumImplicitHs() > 0:
                targets.append(atom.GetIdx())
        
        if targets:
            mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=targets)
        
        return mol
        
    except Exception as e:
        logger.warning(f"Failed to process hydrogens: {e}")
        return mol


def standardize_ligand(
    sdf_path: str,
    output_path: str,
    keep_polar_h: bool = True,
    add_charges: bool = True,
    optimize_3d: bool = False
) -> bool:
    """
    Standardize ligand structure from SDF file
    
    Args:
        sdf_path: Input SDF file path
        output_path: Output SDF file path
        keep_polar_h: Whether to keep only polar hydrogens
        add_charges: Whether to compute Gasteiger charges
        optimize_3d: Whether to optimize 3D conformation
    
    Returns:
        Success status
    """
    try:
        # Read molecule
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=False)
        if mol is None:
            logger.error(f"Failed to read molecule from {sdf_path}")
            return False
        
        # Sanitize
        Chem.SanitizeMol(mol)
        
        # Standardize
        mol = standardize_mol(mol)
        if mol is None:
            return False
        
        # Handle hydrogens
        if keep_polar_h:
            mol = keep_only_polar_hydrogens(mol)
        
        # Assign stereochemistry
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
        
        # Add 3D coordinates if missing
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            if optimize_3d:
                AllChem.UFFOptimizeMolecule(mol)
        
        # Compute charges
        if add_charges:
            ComputeGasteigerCharges(mol)
        
        # Write to file
        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to standardize ligand {sdf_path}: {e}")
        return False


def validate_ligand(sdf_path: str) -> Dict[str, any]:
    """
    Validate ligand structure and return statistics
    
    Args:
        sdf_path: Path to SDF file
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': False,
        'smiles': None,
        'n_atoms': 0,
        'n_heavy_atoms': 0,
        'mol_weight': 0,
        'logp': 0,
        'has_3d': False
    }
    
    try:
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol is None:
            return results
        
        results['smiles'] = Chem.MolToSmiles(mol)
        results['n_atoms'] = mol.GetNumAtoms()
        results['n_heavy_atoms'] = mol.GetNumHeavyAtoms()
        results['mol_weight'] = Descriptors.MolWt(mol)
        results['logp'] = Descriptors.MolLogP(mol)
        results['has_3d'] = mol.GetNumConformers() > 0
        results['valid'] = True
        
    except Exception as e:
        logger.error(f"Failed to validate {sdf_path}: {e}")
    
    return results


def generate_3d_coordinates(
    smiles: str,
    output_path: str,
    optimize: bool = True,
    add_hydrogens: bool = True
) -> bool:
    """
    Generate 3D coordinates from SMILES
    
    Args:
        smiles: SMILES string
        output_path: Output SDF file path
        optimize: Whether to optimize geometry
        add_hydrogens: Whether to add hydrogens
    
    Returns:
        Success status
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        if optimize:
            AllChem.UFFOptimizeMolecule(mol)
        
        # Write to file
        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate 3D coordinates for {smiles}: {e}")
        return False