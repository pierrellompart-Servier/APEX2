"""
Data preprocessing utilities
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def prepare_dta_data(
    df: pd.DataFrame,
    protein_col: str = 'standardized_protein_pdb',
    ligand_col: str = 'standardized_ligand_sdf',
    activity_cols: Optional[List[str]] = None,
    min_activity_value: Optional[float] = None,
    max_activity_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Prepare DTA data for training.
    
    Args:
        df: Input DataFrame
        protein_col: Column with protein PDB paths
        ligand_col: Column with ligand SDF paths
        activity_cols: Activity columns to use
        min_activity_value: Minimum activity value threshold
        max_activity_value: Maximum activity value threshold
        
    Returns:
        Prepared DataFrame
    """
    # Remove rows with missing structure files
    df = df.dropna(subset=[protein_col, ligand_col])
    
    # Check file existence
    print("Checking file existence...")
    valid_mask = df[protein_col].apply(os.path.exists) & df[ligand_col].apply(os.path.exists)
    df = df[valid_mask].reset_index(drop=True)
    print(f"Valid structures: {len(df)}")
    
    # Filter by activity values if specified
    if activity_cols and (min_activity_value is not None or max_activity_value is not None):
        for col in activity_cols:
            if col in df.columns:
                if min_activity_value is not None:
                    df = df[(df[col].isna()) | (df[col] >= min_activity_value)]
                if max_activity_value is not None:
                    df = df[(df[col].isna()) | (df[col] <= max_activity_value)]
    
    # Add protein ID if not present
    if 'protein_id' not in df.columns:
        df['protein_id'] = df[protein_col].apply(
            lambda p: os.path.splitext(os.path.basename(p))[0]
        )
    
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate entries.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
        keep: Which duplicates to keep ('first', 'last', False)
        
    Returns:
        DataFrame without duplicates
    """
    if subset is None:
        subset = ['protein_id', 'std_smiles']
    
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    print(f"Removed {initial_len - len(df)} duplicates")
    
    return df


def filter_by_quality(
    df: pd.DataFrame,
    min_resolution: Optional[float] = None,
    max_resolution: Optional[float] = 3.0,
    min_chain_length: int = 50,
    allowed_organisms: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter structures by quality criteria.
    
    Args:
        df: Input DataFrame
        min_resolution: Minimum resolution (Å)
        max_resolution: Maximum resolution (Å)
        min_chain_length: Minimum protein chain length
        allowed_organisms: List of allowed organisms
        
    Returns:
        Filtered DataFrame
    """
    initial_len = len(df)
    
    # Filter by resolution if column exists
    if 'resolution' in df.columns:
        if min_resolution is not None:
            df = df[df['resolution'] >= min_resolution]
        if max_resolution is not None:
            df = df[df['resolution'] <= max_resolution]
    
    # Filter by chain length if column exists
    if 'chain_length' in df.columns:
        df = df[df['chain_length'] >= min_chain_length]
    
    # Filter by organism if specified
    if allowed_organisms and 'organism' in df.columns:
        df = df[df['organism'].isin(allowed_organisms)]
    
    df = df.reset_index(drop=True)
    print(f"Quality filter: {initial_len} -> {len(df)} entries")
    
    return df


def balance_dataset(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int = 10,
    strategy: str = 'undersample',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance dataset by target distribution.
    
    Args:
        df: Input DataFrame
        target_col: Target column to balance
        n_bins: Number of bins for stratification
        strategy: 'undersample' or 'oversample'
        random_state: Random seed
        
    Returns:
        Balanced DataFrame
    """
    # Remove rows with missing target
    df_valid = df.dropna(subset=[target_col])
    
    # Create bins
    df_valid['bin'] = pd.qcut(df_valid[target_col], n_bins, labels=False, duplicates='drop')
    
    # Get counts per bin
    bin_counts = df_valid['bin'].value_counts()
    
    if strategy == 'undersample':
        min_count = bin_counts.min()
        balanced_dfs = []
        for bin_idx in range(n_bins):
            bin_df = df_valid[df_valid['bin'] == bin_idx]
            if len(bin_df) > min_count:
                bin_df = bin_df.sample(n=min_count, random_state=random_state)
            balanced_dfs.append(bin_df)
    else:  # oversample
        max_count = bin_counts.max()
        balanced_dfs = []
        for bin_idx in range(n_bins):
            bin_df = df_valid[df_valid['bin'] == bin_idx]
            if len(bin_df) < max_count:
                bin_df = bin_df.sample(n=max_count, replace=True, random_state=random_state)
            balanced_dfs.append(bin_df)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.drop('bin', axis=1)
    
    print(f"Balanced dataset: {len(df_valid)} -> {len(df_balanced)} entries")
    
    return df_balanced


def create_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = 'std_smiles',
    frac: List[float] = [0.7, 0.1, 0.2],
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create scaffold-based train/valid/test split.
    
    Args:
        df: Input DataFrame
        smiles_col: Column with SMILES
        frac: Train/valid/test fractions
        random_state: Random seed
        
    Returns:
        Train, valid, test DataFrames
    """
    from rdkit import Chem
    from rdkit.Chem import Scaffolds
    from collections import defaultdict
    
    # Generate scaffolds
    scaffolds = defaultdict(list)
    
    for idx, smiles in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            scaffold = smiles  # Use SMILES as scaffold if parsing fails
        else:
            scaffold = Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        scaffolds[scaffold].append(idx)
    
    # Sort scaffolds by size
    scaffold_sets = [set(idxs) for idxs in scaffolds.values()]
    scaffold_sets = sorted(scaffold_sets, key=len, reverse=True)
    
    # Assign molecules to sets
    train_size = int(len(df) * frac[0])
    valid_size = int(len(df) * frac[1])
    
    train_idxs, valid_idxs, test_idxs = [], [], []
    
    for scaffold_set in scaffold_sets:
        if len(train_idxs) < train_size:
            train_idxs.extend(scaffold_set)
        elif len(valid_idxs) < valid_size:
            valid_idxs.extend(scaffold_set)
        else:
            test_idxs.extend(scaffold_set)
    
    # Create splits
    train_df = df.iloc[train_idxs].reset_index(drop=True)
    valid_df = df.iloc[valid_idxs].reset_index(drop=True)
    test_df = df.iloc[test_idxs].reset_index(drop=True)
    
    print(f"Scaffold split: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    
    return train_df, valid_df, test_df


def normalize_activity_values(
    df: pd.DataFrame,
    activity_cols: List[str],
    method: str = 'standard',
    clip_outliers: bool = True,
    outlier_std: float = 3.0
) -> pd.DataFrame:
    """
    Normalize activity values.
    
    Args:
        df: Input DataFrame
        activity_cols: Activity columns to normalize
        method: 'standard' or 'minmax'
        clip_outliers: Whether to clip outliers
        outlier_std: Number of stds for outlier detection
        
    Returns:
        DataFrame with normalized values
    """
    df = df.copy()
    
    for col in activity_cols:
        if col not in df.columns:
            continue
        
        values = df[col].dropna()
        if len(values) == 0:
            continue
        
        # Clip outliers if requested
        if clip_outliers:
            mean = values.mean()
            std = values.std()
            lower = mean - outlier_std * std
            upper = mean + outlier_std * std
            df[col] = df[col].clip(lower, upper)
        
        # Normalize
        if method == 'standard':
            df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
    
    return df