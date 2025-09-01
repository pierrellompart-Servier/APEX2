"""Data loader utilities for MTL-GNN-DTA"""

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def collate_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching protein-drug pairs
    
    Args:
        batch: List of dictionaries with 'protein', 'drug', and 'y' keys
    
    Returns:
        Dictionary with batched data
    """
    proteins = [item['protein'] for item in batch]
    drugs = [item['drug'] for item in batch]
    ys = torch.stack([item['y'] for item in batch])
    
    # Batch graphs
    protein_batch = Batch.from_data_list(proteins)
    drug_batch = Batch.from_data_list(drugs)
    
    return {
        'protein': protein_batch,
        'drug': drug_batch,
        'y': ys
    }


def create_data_loaders(train_dataset, val_dataset=None, test_dataset=None,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    # Training loader
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_batch
    )
    
    # Validation loader
    if val_dataset is not None:
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_batch
        )
    
    # Test loader
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_batch
        )
    
    return loaders


class ChunkedDataLoader:
    """Data loader that handles chunked data for large datasets"""
    
    def __init__(self, chunk_loader, df, batch_size: int = 32,
                 task_cols: List[str] = None, shuffle: bool = False):
        self.chunk_loader = chunk_loader
        self.df = df
        self.batch_size = batch_size
        self.task_cols = task_cols or ['pKi', 'pEC50', 'pKd', 'pIC50']
        self.shuffle = shuffle
        
        # Generate indices
        self.indices = list(range(len(df)))
        if shuffle:
            import random
            random.shuffle(self.indices)
    
    def __iter__(self):
        """Iterate through batches"""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch_df = self.df.iloc[batch_indices]
            
            # Load structures for batch
            protein_ids = batch_df['protein_id'].tolist()
            structures = self.chunk_loader.get_batch(protein_ids)
            
            # Create batch data
            batch_data = []
            for idx, row in batch_df.iterrows():
                if row['protein_id'] in structures:
                    # Here you would featurize the protein and drug
                    # This is a placeholder
                    batch_data.append({
                        'protein': structures[row['protein_id']],
                        'drug': row['standardized_ligand_sdf'],
                        'y': torch.tensor([row.get(task, float('nan')) 
                                         for task in self.task_cols])
                    })
            
            if batch_data:
                yield collate_batch(batch_data)
    
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
