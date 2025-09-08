"""
Dataset classes for drug-target affinity prediction
"""

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from functools import partial

from ..features.protein_graph import featurize_protein_graph
from ..features.drug_graph import featurize_drug


class DTA(data.Dataset):
    """
    Base dataset for drug-target affinity.
    """
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        data_list: Optional[List[Dict]] = None,
        onthefly: bool = False,
        prot_featurize_fn: Optional[callable] = None,
        drug_featurize_fn: Optional[callable] = None
    ):
        """
        Initialize DTA dataset.
        
        Args:
            df: DataFrame with data
            data_list: List of data dictionaries
            onthefly: Whether to featurize on the fly
            prot_featurize_fn: Protein featurization function
            drug_featurize_fn: Drug featurization function
        """
        super().__init__()
        self.data_df = df
        self.data_list = data_list
        self.onthefly = onthefly
        
        if onthefly:
            assert prot_featurize_fn is not None, 'prot_featurize_fn required for onthefly'
            assert drug_featurize_fn is not None, 'drug_featurize_fn required for onthefly'
        
        self.prot_featurize_fn = prot_featurize_fn
        self.drug_featurize_fn = drug_featurize_fn
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.onthefly:
            drug = self.drug_featurize_fn(
                self.data_list[idx]['drug'],
                name=self.data_list[idx].get('drug_name')
            )
            prot = self.prot_featurize_fn(
                self.data_list[idx]['protein'],
                name=self.data_list[idx].get('protein_name')
            )
        else:
            drug = self.data_list[idx]['drug']
            prot = self.data_list[idx]['protein']
        
        y = self.data_list[idx]['y']
        
        return {'drug': drug, 'protein': prot, 'y': y}


class MTL_DTA(data.Dataset):
    """
    Multi-task learning dataset for drug-target affinity.
    """
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        data_list: Optional[List[Dict]] = None,
        task_cols: Optional[List[str]] = None,
        onthefly: bool = False,
        prot_featurize_fn: Optional[callable] = None,
        drug_featurize_fn: Optional[callable] = None
    ):
        """
        Initialize MTL-DTA dataset.
        
        Args:
            df: DataFrame with data
            data_list: List of data dictionaries
            task_cols: List of task column names
            onthefly: Whether to featurize on the fly
            prot_featurize_fn: Protein featurization function
            drug_featurize_fn: Drug featurization function
        """
        super().__init__()
        self.data_df = df
        self.data_list = data_list
        self.task_cols = task_cols or ['pKi', 'pEC50', 'pKd', 'pIC50']
        self.onthefly = onthefly
        self.prot_featurize_fn = prot_featurize_fn
        self.drug_featurize_fn = drug_featurize_fn
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.onthefly:
            drug = self.drug_featurize_fn(
                self.data_list[idx]['drug'],
                name=self.data_list[idx].get('drug_name')
            )
            prot = self.prot_featurize_fn(
                self.data_list[idx]['protein'],
                name=self.data_list[idx].get('protein_name')
            )
        else:
            drug = self.data_list[idx]['drug']
            prot = self.data_list[idx]['protein']
        
        # Get multi-task targets
        y_multi = []
        for task in self.task_cols:
            val = self.data_list[idx].get(task, np.nan)
            y_multi.append(val if not pd.isna(val) else np.nan)
        
        y = torch.tensor(y_multi, dtype=torch.float32)
        
        return {'drug': drug, 'protein': prot, 'y': y}