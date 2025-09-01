


"""Dataset classes for MTL-GNN-DTA"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MTL_DTA(Dataset):
    """Multi-Task Learning Dataset for Drug-Target Affinity"""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 data_list: List[Dict], 
                 task_cols: List[str],
                 transform=None):
        """
        Initialize MTL-DTA dataset
        
        Args:
            df: DataFrame with metadata
            data_list: List of dictionaries containing protein/drug graphs and targets
            task_cols: List of task column names
            transform: Optional data transformation
        """
        super(MTL_DTA, self).__init__()
        self.df = df
        self.data_list = data_list
        self.task_cols = task_cols
        self.n_tasks = len(task_cols)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """Get a single data point"""
        data_dict = self.data_list[idx]
        
        # Prepare target tensor with NaN for missing values
        y = torch.zeros(self.n_tasks)
        for i, task in enumerate(self.task_cols):
            if task in data_dict and not pd.isna(data_dict[task]):
                y[i] = float(data_dict[task])
            else:
                y[i] = float('nan')
        
        sample = {
            'protein': data_dict['protein'],
            'drug': data_dict['drug'],
            'y': y,
            'idx': idx
        }
        
        # Apply transformation if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample"""
        if idx < len(self.df):
            return self.df.iloc[idx].to_dict()
        return {}
    
    def get_task_statistics(self) -> Dict:
        """Get statistics for each task"""
        stats = {}
        for task in self.task_cols:
            values = []
            for data in self.data_list:
                if task in data and not pd.isna(data[task]):
                    values.append(data[task])
            
            if values:
                stats[task] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'missing': len(self.data_list) - len(values)
                }
            else:
                stats[task] = {
                    'count': 0,
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'missing': len(self.data_list)
                }
        
        return stats


class ProteinLigandDataset(Dataset):
    """Dataset for protein-ligand pairs with file paths"""
    
    def __init__(self,
                 protein_paths: List[str],
                 ligand_paths: List[str],
                 targets: Optional[np.ndarray] = None,
                 task_cols: Optional[List[str]] = None,
                 protein_featurizer=None,
                 drug_featurizer=None,
                 cache_features: bool = False):
        """
        Initialize dataset from file paths
        
        Args:
            protein_paths: List of protein PDB file paths
            ligand_paths: List of ligand SDF file paths
            targets: Optional target values [n_samples, n_tasks]
            task_cols: Optional task column names
            protein_featurizer: Protein featurizer object
            drug_featurizer: Drug featurizer object
            cache_features: Whether to cache featurized data
        """
        super().__init__()
        
        assert len(protein_paths) == len(ligand_paths), \
            "Number of proteins and ligands must match"
        
        self.protein_paths = protein_paths
        self.ligand_paths = ligand_paths
        self.targets = targets
        self.task_cols = task_cols or ['task']
        self.n_tasks = len(self.task_cols)
        
        self.protein_featurizer = protein_featurizer
        self.drug_featurizer = drug_featurizer
        
        self.cache_features = cache_features
        self.cache = {} if cache_features else None
    
    def __len__(self):
        return len(self.protein_paths)
    
    def __getitem__(self, idx):
        """Get a single data point"""
        # Check cache
        if self.cache_features and idx in self.cache:
            return self.cache[idx]
        
        # Featurize protein
        protein_path = self.protein_paths[idx]
        if self.protein_featurizer:
            protein_data = self.protein_featurizer.featurize_from_pdb(protein_path)
        else:
            # Return path if no featurizer
            protein_data = protein_path
        
        # Featurize drug
        ligand_path = self.ligand_paths[idx]
        if self.drug_featurizer:
            drug_data = self.drug_featurizer.featurize_drug(ligand_path)
        else:
            # Return path if no featurizer
            drug_data = ligand_path
        
        # Prepare target
        if self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float)
        else:
            y = torch.zeros(self.n_tasks)
            y[:] = float('nan')
        
        sample = {
            'protein': protein_data,
            'drug': drug_data,
            'y': y,
            'protein_path': protein_path,
            'ligand_path': ligand_path,
            'idx': idx
        }
        
        # Cache if requested
        if self.cache_features:
            self.cache[idx] = sample
        
        return sample


class CrossValidationDataset:
    """Helper class for cross-validation data splitting"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 task_cols: List[str],
                 n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize cross-validation dataset
        
        Args:
            df: DataFrame with all data
            task_cols: List of task columns
            n_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.df = df
        self.task_cols = task_cols
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Create fold assignments
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        self.fold_indices = list(kf.split(df))
    
    def get_fold(self, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train and test data for a specific fold
        
        Args:
            fold_idx: Fold index (0 to n_folds-1)
        
        Returns:
            train_df, test_df
        """
        if fold_idx >= self.n_folds:
            raise ValueError(f"Fold index {fold_idx} >= n_folds {self.n_folds}")
        
        train_idx, test_idx = self.fold_indices[fold_idx]
        train_df = self.df.iloc[train_idx].reset_index(drop=True)
        test_df = self.df.iloc[test_idx].reset_index(drop=True)
        
        return train_df, test_df
    
    def get_fold_with_validation(self, 
                                 fold_idx: int, 
                                 val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get train, validation, and test data for a specific fold
        
        Args:
            fold_idx: Fold index
            val_size: Fraction of training data to use for validation
        
        Returns:
            train_df, val_df, test_df
        """
        train_df, test_df = self.get_fold(fold_idx)
        
        # Split train into train and validation
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, 
            test_size=val_size, 
            random_state=self.random_state + fold_idx
        )
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_dataset(df: pd.DataFrame,
                   protein_structures: Dict,
                   task_cols: List[str],
                   protein_featurizer,
                   drug_featurizer) -> MTL_DTA:
    """
    Create MTL_DTA dataset from DataFrame and protein structures
    
    Args:
        df: DataFrame with data
        protein_structures: Dictionary of protein structures
        task_cols: List of task columns
        protein_featurizer: Protein featurizer
        drug_featurizer: Drug featurizer
    
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    for idx, row in df.iterrows():
        # Get protein structure
        if 'protein_id' in row:
            protein_id = row['protein_id']
        else:
            # Extract from path
            protein_id = os.path.basename(row['standardized_protein_pdb']).split('.')[0]
        
        protein_json = protein_structures.get(protein_id)
        if protein_json is None:
            logger.warning(f"Protein structure not found for {protein_id}")
            continue
        
        # Featurize protein
        protein_data = protein_featurizer.featurize_protein_graph(protein_json)
        if protein_data is None:
            continue
        
        # Featurize drug
        drug_data = drug_featurizer.featurize_drug(row['standardized_ligand_sdf'])
        if drug_data is None:
            continue
        
        # Collect task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            'protein': protein_data,
            'drug': drug_data,
            **task_values
        }
        
        data_list.append(data_entry)
    
    return MTL_DTA(df=df, data_list=data_list, task_cols=task_cols)
