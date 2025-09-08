"""
Molecular property calculations and ligand efficiency metrics
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, QED
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Optional


def compute_molecular_properties(smiles: str) -> Dict:
    """
    Compute molecular properties for a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular properties
    """
    if not isinstance(smiles, str) or smiles.strip() == '':
        return {k: None for k in [
            'InChIKey', 'MolWt', 'HeavyAtomCount', 'QED', 'NumHDonors',
            'NumHAcceptors', 'NumRotatableBonds', 'TPSA', 'LogP'
        ]}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: None for k in [
            'InChIKey', 'MolWt', 'HeavyAtomCount', 'QED', 'NumHDonors',
            'NumHAcceptors', 'NumRotatableBonds', 'TPSA', 'LogP'
        ]}
    
    return {
        'InChIKey': Chem.MolToInchiKey(mol),
        'MolWt': Descriptors.MolWt(mol),
        'HeavyAtomCount': mol.GetNumHeavyAtoms(),
        'QED': QED.qed(mol),
        'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'TPSA': rdMolDescriptors.CalcTPSA(mol),
        'LogP': Crippen.MolLogP(mol)
    }


def add_molecular_properties_parallel(
    df: pd.DataFrame,
    smiles_col: str = 'std_smiles',
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Add molecular properties to dataframe in parallel.
    
    Args:
        df: Input dataframe
        smiles_col: Column containing SMILES
        n_jobs: Number of parallel jobs
        
    Returns:
        DataFrame with added molecular properties
    """
    smiles_list = df[smiles_col].tolist()
    
    props = Parallel(n_jobs=n_jobs)(
        delayed(compute_molecular_properties)(smi) 
        for smi in tqdm(smiles_list, desc="Computing properties")
    )
    
    props_df = pd.DataFrame(props)
    return pd.concat([df.reset_index(drop=True), props_df], axis=1)


def compute_ligand_efficiency(
    df: pd.DataFrame,
    activity_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute ligand efficiency metrics.
    
    Args:
        df: DataFrame with activity data
        activity_cols: List of activity columns (e.g., ['pKi', 'pEC50'])
        
    Returns:
        DataFrame with added LE metrics
    """
    if activity_cols is None:
        activity_cols = [col for col in df.columns 
                        if col.strip().lower() in ['pki', 'pkd', 'pec50', 'pic50']]
    
    for col in activity_cols:
        if col not in df.columns:
            continue
            
        le_col = f'LE_{col}'
        le_norm_col = f'LEnorm_{col}'
        
        # Ligand efficiency
        df[le_col] = df.apply(
            lambda row, col=col: row[col] / row['HeavyAtomCount']
            if pd.notnull(row.get(col)) and pd.notnull(row.get('HeavyAtomCount')) 
               and row['HeavyAtomCount'] > 0
            else None,
            axis=1
        )
        
        # Normalized ligand efficiency
        df[le_norm_col] = df.apply(
            lambda row, col=col: row.get(f'LE_{col}') / row['MolWt']
            if pd.notnull(row.get(f'LE_{col}')) and pd.notnull(row.get('MolWt')) 
               and row['MolWt'] > 0
            else None,
            axis=1
        )
    
    return df


def compute_mean_ligand_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ligand efficiency across all activity types.
    
    Args:
        df: DataFrame with LE metrics
        
    Returns:
        DataFrame with mean LE values
    """
    le_cols = [c for c in df.columns if c.startswith("LE_") and not c.startswith("LEnorm_")]
    le_norm_cols = [c for c in df.columns if c.startswith("LEnorm_")]
    
    if le_cols:
        df['LE'] = df[le_cols].mean(axis=1, skipna=True)
    
    if le_norm_cols:
        df['LE_norm'] = df[le_norm_cols].mean(axis=1, skipna=True)
    
    return df


def filter_by_properties(
    df: pd.DataFrame,
    min_heavy_atoms: int = 5,
    max_heavy_atoms: int = 75,
    max_mw: float = 1000,
    min_carbons: int = 4,
    min_le: Optional[float] = 0.05,
    max_le_norm: Optional[float] = 0.003
) -> pd.DataFrame:
    """
    Filter molecules based on property criteria.
    
    Args:
        df: DataFrame with molecular properties
        min_heavy_atoms: Minimum heavy atom count
        max_heavy_atoms: Maximum heavy atom count
        max_mw: Maximum molecular weight
        min_carbons: Minimum carbon atoms
        min_le: Minimum ligand efficiency
        max_le_norm: Maximum normalized ligand efficiency
        
    Returns:
        Filtered DataFrame
    """
    # Count carbon atoms
    df['carbon_count'] = df['std_smiles'].apply(
        lambda x: x.count('C') + x.count('c') if pd.notnull(x) else 0
    )
    
    # Build filter conditions
    conditions = [
        df["HeavyAtomCount"] >= min_heavy_atoms,
        df["HeavyAtomCount"] <= max_heavy_atoms,
        df["MolWt"] <= max_mw,
        df['carbon_count'] >= min_carbons
    ]
    
    if min_le is not None and 'LE' in df.columns:
        conditions.append(df["LE"] >= min_le)
    
    if max_le_norm is not None and 'LE_norm' in df.columns:
        conditions.append(df["LE_norm"] <= max_le_norm)
    
    # Combine conditions
    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        mask &= condition
    
    return df[mask].copy()