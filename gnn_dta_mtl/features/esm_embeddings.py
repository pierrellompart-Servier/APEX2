"""
ESM protein embedding generation
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import EsmModel, EsmTokenizer
from tqdm import tqdm
import numpy as np


class ESMEmbedder:
    """
    Generate ESM embeddings for protein sequences.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize ESM embedder.
        
        Args:
            model_name: ESM model name
            device: Device to use (auto-detect if None)
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def embed_sequence(
        self,
        sequence: str,
        return_contacts: bool = False
    ) -> torch.Tensor:
        """
        Generate embedding for a single sequence.
        
        Args:
            sequence: Protein sequence
            return_contacts: Whether to return contact predictions
            
        Returns:
            Sequence embedding tensor
        """
        # Tokenize
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get sequence representation (remove CLS and EOS tokens)
            sequence_embedding = outputs.last_hidden_state[0, 1:-1]
            
            if return_contacts and hasattr(outputs, 'contacts'):
                contacts = outputs.contacts[0, 1:-1, 1:-1]
                return sequence_embedding, contacts
        
        return sequence_embedding
    
    def embed_batch(
        self,
        sequences: List[str],
        batch_size: int = 8
    ) -> List[torch.Tensor]:
        """
        Generate embeddings for multiple sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            
        Returns:
            List of embedding tensors
        """
        embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
            batch = sequences[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Process each sequence in batch
                for j, seq_len in enumerate([len(s) for s in batch]):
                    # Remove padding and special tokens
                    embedding = outputs.last_hidden_state[j, 1:seq_len+1]
                    embeddings.append(embedding.cpu())
        
        return embeddings
    
    def embed_and_save(
        self,
        sequence: str,
        save_path: str,
        protein_id: Optional[str] = None
    ) -> str:
        """
        Generate embedding and save to file.
        
        Args:
            sequence: Protein sequence
            save_path: Path to save embedding
            protein_id: Optional protein ID for caching
            
        Returns:
            Path where embedding was saved
        """
        # Check cache if ID provided
        if protein_id and self.cache_dir:
            cache_path = self.cache_dir / f"{protein_id}.pt"
            if cache_path.exists():
                return str(cache_path)
        
        # Generate embedding
        embedding = self.embed_sequence(sequence)
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.cpu(), save_path)
        
        # Cache if ID provided
        if protein_id and self.cache_dir:
            cache_path = self.cache_dir / f"{protein_id}.pt"
            torch.save(embedding.cpu(), cache_path)
        
        return str(save_path)
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()


def get_esm_embedding(
    seq: str,
    esm_model: EsmModel,
    tokenizer: EsmTokenizer,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Standalone function to get ESM embedding.
    
    Args:
        seq: Protein sequence
        esm_model: ESM model
        tokenizer: ESM tokenizer
        device: Device to use
        
    Returns:
        Embedding tensor
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = esm_model(**inputs)
        # Remove CLS and EOS tokens
        embedding = outputs.last_hidden_state[0, 1:-1]
    
    return embedding