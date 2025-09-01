"""Data chunking utilities for handling large datasets"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataChunker:
    """Chunk large datasets for efficient processing"""
    
    def __init__(self, chunk_size: int = 10000, cache_dir: str = "cache/chunks"):
        """
        Initialize data chunker
        
        Args:
            chunk_size: Number of samples per chunk
            cache_dir: Directory to cache chunks
        """
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_index = {}
    
    def chunk_dataframe(self, df: pd.DataFrame, prefix: str = "chunk") -> List[str]:
        """
        Split DataFrame into chunks and save
        
        Args:
            df: DataFrame to chunk
            prefix: Prefix for chunk files
        
        Returns:
            List of chunk file paths
        """
        n_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        chunk_paths = []
        
        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(df))
            
            chunk_df = df.iloc[start_idx:end_idx]
            chunk_path = self.cache_dir / f"{prefix}_{i:04d}.parquet"
            
            chunk_df.to_parquet(chunk_path, index=False)
            chunk_paths.append(str(chunk_path))
            
            # Update index
            self.chunk_index[f"{prefix}_{i:04d}"] = {
                'path': str(chunk_path),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_samples': len(chunk_df)
            }
        
        # Save index
        index_path = self.cache_dir / f"{prefix}_index.json"
        with open(index_path, 'w') as f:
            json.dump(self.chunk_index, f, indent=2)
        
        logger.info(f"Created {n_chunks} chunks for {prefix}")
        return chunk_paths
    
    def load_chunk(self, chunk_path: str) -> pd.DataFrame:
        """Load a single chunk"""
        return pd.read_parquet(chunk_path)
    
    def iterate_chunks(self, chunk_paths: List[str]) -> Iterator[pd.DataFrame]:
        """Iterate through chunks"""
        for path in chunk_paths:
            yield self.load_chunk(path)
    
    def merge_chunks(self, chunk_paths: List[str]) -> pd.DataFrame:
        """Merge multiple chunks into single DataFrame"""
        dfs = []
        for path in chunk_paths:
            dfs.append(self.load_chunk(path))
        return pd.concat(dfs, ignore_index=True)


class ProteinStructureChunker:
    """Specialized chunker for protein structure data"""
    
    def __init__(self, chunk_size: int = 1000, cache_dir: str = "cache/proteins"):
        """
        Initialize protein structure chunker
        
        Args:
            chunk_size: Number of proteins per chunk
            cache_dir: Directory to cache protein chunks
        """
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def chunk_proteins(self, protein_dict: Dict, prefix: str = "proteins") -> List[str]:
        """
        Chunk protein structure dictionary
        
        Args:
            protein_dict: Dictionary of protein structures
            prefix: Prefix for chunk files
        
        Returns:
            List of chunk file paths
        """
        protein_ids = list(protein_dict.keys())
        n_chunks = (len(protein_ids) + self.chunk_size - 1) // self.chunk_size
        chunk_paths = []
        
        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(protein_ids))
            
            chunk_ids = protein_ids[start_idx:end_idx]
            chunk_data = {pid: protein_dict[pid] for pid in chunk_ids}
            
            chunk_path = self.cache_dir / f"{prefix}_{i:04d}.pkl"
            with open(chunk_path, 'wb') as f:
                pickle.dump(chunk_data, f)
            
            chunk_paths.append(str(chunk_path))
        
        logger.info(f"Created {n_chunks} protein chunks")
        return chunk_paths
    
    def load_protein_chunk(self, chunk_path: str) -> Dict:
        """Load a protein chunk"""
        with open(chunk_path, 'rb') as f:
            return pickle.load(f)
    
    def get_protein(self, protein_id: str, chunk_paths: List[str]) -> Optional[Dict]:
        """
        Get specific protein from chunks
        
        Args:
            protein_id: Protein identifier
            chunk_paths: List of chunk file paths
        
        Returns:
            Protein data or None
        """
        for path in chunk_paths:
            chunk = self.load_protein_chunk(path)
            if protein_id in chunk:
                return chunk[protein_id]
        return None


class FeatureChunker:
    """Chunker for pre-computed features"""
    
    def __init__(self, feature_type: str = "embeddings", cache_dir: str = "cache/features"):
        """
        Initialize feature chunker
        
        Args:
            feature_type: Type of features (embeddings, graphs, etc.)
            cache_dir: Directory to cache features
        """
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir) / feature_type
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_features(self, features: Dict, chunk_id: str):
        """
        Save features to chunk
        
        Args:
            features: Dictionary of features
            chunk_id: Chunk identifier
        """
        chunk_path = self.cache_dir / f"{chunk_id}.pkl"
        with open(chunk_path, 'wb') as f:
            pickle.dump(features, f)
    
    def load_features(self, chunk_id: str) -> Optional[Dict]:
        """
        Load features from chunk
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            Features dictionary or None
        """
        chunk_path = self.cache_dir / f"{chunk_id}.pkl"
        if chunk_path.exists():
            with open(chunk_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def exists(self, chunk_id: str) -> bool:
        """Check if chunk exists"""
        chunk_path = self.cache_dir / f"{chunk_id}.pkl"
        return chunk_path.exists()