"""
Molecular structure standardization functions
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from rdkit.Chem.MolStandardize import rdMolStandardize
from Bio.PDB import PDBParser, PDBIO, is_aa
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torch

from ..utils.constants import POLAR_HEAVY


def keep_only_polar_H_rdkit(mol: Chem.Mol) -> Chem.Mol:
    """
    Keep only hydrogens bonded to polar atoms (N, O, P, S).
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Modified molecule with only polar hydrogens
    """
    # Remove explicit H attached to non-polar heavy atoms
    h_to_del = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 1:
            continue
        nbrs = a.GetNeighbors()
        if not nbrs:
            continue
        if nbrs[0].GetAtomicNum() not in POLAR_HEAVY:
            h_to_del.append(a.GetIdx())
    
    if h_to_del:
        em = Chem.EditableMol(mol)
        for idx in sorted(h_to_del, reverse=True):
            em.RemoveAtom(idx)
        mol = em.GetMol()
    
    # Add missing H on polar atoms
    mol.UpdatePropertyCache(strict=False)
    targets = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetAtomicNum() in POLAR_HEAVY and a.GetImplicitHCount() > 0]
    if targets:
        mol = Chem.AddHs(mol,
                         addCoords=(mol.GetNumConformers() > 0),
                         onlyOnAtoms=targets)
    return mol


def standardize_ligand(path: str, output_path: str) -> bool:
    """
    Standardize a ligand structure from SDF or PDB file.
    
    Args:
        path: Input file path
        output_path: Output SDF file path
        
    Returns:
        Success status
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.sdf':
        mol = Chem.MolFromMolFile(path, removeHs=False)
    elif ext == '.pdb':
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    else:
        return False
        
    if mol is None:
        return False
    
    mol = keep_only_polar_H_rdkit(mol)
    
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
    rdPartialCharges.ComputeGasteigerCharges(mol)
    Chem.MolToMolFile(mol, output_path)
    
    return True


def standardize_smiles_from_sdf(sdf_path: str) -> Optional[str]:
    """
    Generate standardized SMILES from SDF file.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        Standardized SMILES string or None if failed
    """
    try:
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol is None:
            return None
        
        # Keep existing polar H; drop non-polar H
        to_del = []
        for a in mol.GetAtoms():
            if a.GetAtomicNum() != 1:
                continue
            nbs = a.GetNeighbors()
            if nbs and nbs[0].GetAtomicNum() not in POLAR_HEAVY:
                to_del.append(a.GetIdx())
        
        if to_del:
            em = Chem.EditableMol(mol)
            for idx in sorted(to_del, reverse=True):
                em.RemoveAtom(idx)
            mol = em.GetMol()
        
        mol.UpdatePropertyCache(strict=False)
        AllChem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=False)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
        
        # Add missing H on polar atoms
        mol = Chem.AddHs(mol, addCoords=False, onlyOnPolarAtoms=True)
        
        # Standardize
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.Normalizer().normalize(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        
    except Exception as e:
        print(f"Error processing {sdf_path}: {e}")
        return None


def clean_protein_structure(
    pdb_path: str,
    output_path: str,
    ph: float = 7.4,
    remove_water: bool = True
) -> None:
    """
    Clean and standardize protein structure using pdbfixer.
    
    Args:
        pdb_path: Input PDB file path
        output_path: Output PDB file path
        ph: pH for adding hydrogens
        remove_water: Whether to remove water molecules
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile, Modeller, element as elem
    
    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=not remove_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=ph)
    
    mod = Modeller(fixer.topology, fixer.positions)
    
    # Remove non-polar hydrogens
    to_delete = []
    for bond in mod.topology.bonds():
        a1, a2 = bond
        if a1.element == elem.hydrogen and a2.element == elem.carbon:
            to_delete.append(a1)
        elif a2.element == elem.hydrogen and a1.element == elem.carbon:
            to_delete.append(a2)
    
    if to_delete:
        mod.delete(to_delete)
    
    with open(output_path, 'w') as f:
        PDBFile.writeFile(mod.topology, mod.positions, f)


class StructureStandardizer:
    """
    Batch standardization of molecular structures
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize standardizer.
        
        Args:
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.n_workers = n_workers or cpu_count() - 1
    
    def standardize_ligands(
        self,
        df: pd.DataFrame,
        input_col: str,
        output_dir: str
    ) -> pd.DataFrame:
        """
        Standardize all ligands in dataframe.
        
        Args:
            df: DataFrame with ligand paths
            input_col: Column name with input paths
            output_dir: Output directory for standardized files
            
        Returns:
            DataFrame with added standardized paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        args = []
        for idx, row in df.iterrows():
            in_path = row[input_col]
            out_path = os.path.join(output_dir, f"{idx}.sdf")
            args.append((in_path, out_path))
        
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.starmap(standardize_ligand, args),
                total=len(args),
                desc="Standardizing ligands"
            ))
        
        df['standardized_ligand_sdf'] = [
            args[i][1] if results[i] else None
            for i in range(len(results))
        ]
        
        return df
    
    def standardize_proteins(
        self,
        df: pd.DataFrame,
        input_col: str,
        output_dir: str
    ) -> pd.DataFrame:
        """
        Standardize all proteins in dataframe.
        
        Args:
            df: DataFrame with protein paths
            input_col: Column name with input paths
            output_dir: Output directory for standardized files
            
        Returns:
            DataFrame with added standardized paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        args = []
        for idx, row in df.iterrows():
            in_path = row[input_col]
            out_path = os.path.join(output_dir, f"{idx}.pdb")
            args.append((in_path, out_path))
        
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.starmap(clean_protein_structure, args),
                total=len(args),
                desc="Standardizing proteins"
            ))
        
        df['standardized_protein_pdb'] = [
            args[i][1] for i in range(len(args))
        ]
        
        return df
    
    def standardize_smiles(
        self,
        df: pd.DataFrame,
        sdf_col: str = 'standardized_ligand_sdf'
    ) -> pd.DataFrame:
        """
        Generate standardized SMILES for all ligands.
        
        Args:
            df: DataFrame with ligand SDF paths
            sdf_col: Column name with SDF paths
            
        Returns:
            DataFrame with added standardized SMILES
        """
        with Pool(self.n_workers) as pool:
            smiles = list(tqdm(
                pool.map(standardize_smiles_from_sdf, df[sdf_col]),
                total=len(df),
                desc="Generating SMILES"
            ))
        
        df['std_smiles'] = smiles
        return df