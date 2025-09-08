"""
Dataset building and splitting utilities
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from ..features.protein_graph import featurize_protein_graph
from ..features.drug_graph import featurize_drug
from ..features.structure_processing import StructureChunkLoader
from .dta_dataset import DTA, MTL_DTA


def build_dataset(
    df_fold: pd.DataFrame,
    pdb_structures: Dict,
    exp_col: str = "pKi",
    is_pred: bool = False
) -> DTA:
    """
    Build single-task DTA dataset.
    
    Args:
        df_fold: DataFrame with fold data
        pdb_structures: Dictionary of protein structures
        exp_col: Experimental data column
        is_pred: Whether this is for prediction (no labels)
        
    Returns:
        DTA dataset
    """
    data_list = []
    
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building dataset"):
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        
        if protein_json is None:
            continue
        
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        if is_pred:
            data_list.append({
                "protein": protein,
                "drug": drug,
                "y": 0
            })
        else:
            data_list.append({
                "protein": protein,
                "drug": drug,
                "y": float(row[exp_col])
            })
    
    return DTA(df=df_fold, data_list=data_list)


def build_mtl_dataset(
    df_fold: pd.DataFrame,
    pdb_structures: Dict,
    task_cols: List[str] = ['pKi', 'pEC50', 'pKd', 'pIC50']
) -> MTL_DTA:
    """
    Build multi-task DTA dataset.
    
    Args:
        df_fold: DataFrame with fold data
        pdb_structures: Dictionary of protein structures
        task_cols: List of task columns
        
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building MTL dataset"):
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        
        if protein_json is None:
            continue
        
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)


def build_mtl_dataset_optimized(
    df_fold: pd.DataFrame,
    chunk_loader: StructureChunkLoader,
    task_cols: List[str] = ['pKi', 'pEC50', 'pKd', 'pIC50']
) -> MTL_DTA:
    """
    Build MTL dataset efficiently using chunked structure loader.
    
    Args:
        df_fold: DataFrame with fold data
        chunk_loader: StructureChunkLoader instance
        task_cols: List of task columns
        
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    # Get protein IDs
    protein_ids = []
    for _, row in df_fold.iterrows():
        protein_id = os.path.basename(row["standardized_protein_pdb"]).split(".")[0]
        protein_ids.append(protein_id)
    
    # Batch load structures
    print(f"Loading structures for {len(set(protein_ids))} unique proteins...")
    unique_ids = list(set(protein_ids))
    structures_batch = chunk_loader.get_batch(unique_ids)
    
    # Process each row
    skipped = 0
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building dataset"):
        protein_id = os.path.basename(row["standardized_protein_pdb"]).split(".")[0]
        
        if protein_id not in structures_batch:
            skipped += 1
            continue
        
        protein_json = structures_batch[protein_id]
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries due to missing structures")
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)


def create_data_splits(
    df: pd.DataFrame,
    split_method: str = 'random',
    split_frac: List[float] = [0.7, 0.1, 0.2],
    seed: int = 42,
    entity_col: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create train/valid/test splits.
    
    Args:
        df: Full dataset
        split_method: 'random', 'protein', 'drug', or 'both'
        split_frac: [train, valid, test] fractions
        seed: Random seed
        entity_col: Column for entity-based splitting
        
    Returns:
        Dictionary with train/valid/test DataFrames
    """
    train_frac, val_frac, test_frac = split_frac
    
    if split_method == 'random':
        # Random split
        test = df.sample(frac=test_frac, random_state=seed)
        train_val = df[~df.index.isin(test.index)]
        val = train_val.sample(frac=val_frac/(1-test_frac), random_state=seed)
        train = train_val[~train_val.index.isin(val.index)]
        
    elif split_method in ['protein', 'drug']:
        # Entity-based split
        entity_col = entity_col or split_method
        entities = df[entity_col].unique()
        
        # Split entities
        test_entities = np.random.RandomState(seed).choice(
            entities, size=int(len(entities) * test_frac), replace=False
        )
        remaining_entities = [e for e in entities if e not in test_entities]
        val_entities = np.random.RandomState(seed).choice(
            remaining_entities, size=int(len(entities) * val_frac), replace=False
        )
        
        # Create splits
        test = df[df[entity_col].isin(test_entities)]
        val = df[df[entity_col].isin(val_entities)]
        train = df[~df[entity_col].isin(test_entities) & ~df[entity_col].isin(val_entities)]
        
    elif split_method == 'both':
        # Both drug and protein are unseen in test
        test_drugs = df['drug'].drop_duplicates().sample(frac=test_frac, random_state=seed).values
        test_prots = df['protein'].drop_duplicates().sample(frac=test_frac, random_state=seed).values
        
        test = df[(df['drug'].isin(test_drugs)) & (df['protein'].isin(test_prots))]
        train_val = df[(~df['drug'].isin(test_drugs)) & (~df['protein'].isin(test_prots))]
        
        val = train_val.sample(frac=val_frac/(1-test_frac), random_state=seed)
        train = train_val[~train_val.index.isin(val.index)]
    
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    
    return {
        'train': train.reset_index(drop=True),
        'valid': val.reset_index(drop=True),
        'test': test.reset_index(drop=True)
    }


def prepare_mtl_experiment(
    df: pd.DataFrame,
    task_cols: List[str] = ['pKi', 'pEC50', 'pKd', 'pIC50']
) -> Dict[str, float]:
    """
    Prepare multi-task learning experiment.
    
    Args:
        df: DataFrame with task data
        task_cols: List of task columns
        
    Returns:
        Dictionary of task ranges for weighting
    """
    task_ranges = {}
    
    for task in task_cols:
        if task in df.columns:
            valid_values = df[task].dropna()
            if len(valid_values) > 0:
                task_ranges[task] = valid_values.max() - valid_values.min()
            else:
                task_ranges[task] = 1.0
        else:
            task_ranges[task] = 1.0
    
    print("Task ranges for weighting:")
    for task, range_val in task_ranges.items():
        weight = 1.0 / range_val if range_val > 0 else 1.0
        normalized_weight = weight / sum(1.0/r if r > 0 else 1.0 for r in task_ranges.values())
        print(f"  {task}: range={range_val:.2f}, weight={normalized_weight:.4f}")
    
    return task_ranges