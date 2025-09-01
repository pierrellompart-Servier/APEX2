"""PDB processing utilities for protein structure standardization"""

import os
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

from Bio.PDB import PDBParser, PDBIO, Select
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="openmm.app")

# Define polar heavy atoms
POLAR_HEAVY = {7, 8, 15, 16}  # N, O, P, S


class StandardResidueSelect(Select):
    """Select only standard amino acid residues"""
    
    def accept_residue(self, residue):
        from Bio.PDB import is_aa
        return is_aa(residue)


def clean_protein_structure(
    pdb_path: str,
    output_path: str,
    ph: float = 7.4,
    remove_water: bool = True,
    keep_only_polar_h: bool = True
) -> bool:
    """
    Clean and standardize protein structure
    
    Args:
        pdb_path: Input PDB file path
        output_path: Output PDB file path
        ph: pH for protonation
        remove_water: Whether to remove water molecules
        keep_only_polar_h: Whether to keep only polar hydrogens
    
    Returns:
        Success status
    """
    try:
        # Use PDBFixer for cleaning
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=not remove_water)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=ph)
        
        if keep_only_polar_h:
            # Remove non-polar hydrogens
            from openmm.app import Modeller, element as elem
            
            mod = Modeller(fixer.topology, fixer.positions)
            to_delete = []
            
            for atom in mod.topology.atoms():
                if atom.element == elem.hydrogen:
                    # Check if bonded to non-polar atom
                    for bond in mod.topology.bonds():
                        a1, a2 = bond
                        if a1 == atom:
                            if a2.element == elem.carbon:
                                to_delete.append(atom)
                        elif a2 == atom:
                            if a1.element == elem.carbon:
                                to_delete.append(atom)
            
            if to_delete:
                mod.delete(to_delete)
            
            # Save cleaned structure
            with open(output_path, 'w') as f:
                PDBFile.writeFile(mod.topology, mod.positions, f)
        else:
            with open(output_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to clean protein structure {pdb_path}: {e}")
        return False


def extract_protein_sequence(pdb_path: str) -> Optional[str]:
    """
    Extract amino acid sequence from PDB file
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        Protein sequence string
    """
    try:
        from Bio.PDB import PDBParser, is_aa
        from Bio.SeqUtils import seq1
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if is_aa(residue):
                        sequence.append(seq1(residue.resname))
        
        return ''.join(sequence)
        
    except Exception as e:
        logger.error(f"Failed to extract sequence from {pdb_path}: {e}")
        return None


def validate_protein_structure(pdb_path: str) -> Dict[str, any]:
    """
    Validate protein structure and return statistics
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': False,
        'n_residues': 0,
        'n_atoms': 0,
        'n_chains': 0,
        'has_missing_residues': False,
        'has_missing_atoms': False,
        'sequence': None
    }
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        n_residues = 0
        n_atoms = 0
        chains = []
        
        for model in structure:
            for chain in model:
                chains.append(chain.id)
                for residue in chain:
                    n_residues += 1
                    for atom in residue:
                        n_atoms += 1
        
        results['n_residues'] = n_residues
        results['n_atoms'] = n_atoms
        results['n_chains'] = len(set(chains))
        results['sequence'] = extract_protein_sequence(pdb_path)
        results['valid'] = n_residues > 0 and n_atoms > 0
        
    except Exception as e:
        logger.error(f"Failed to validate {pdb_path}: {e}")
    
    return results