"""
Parallel structure processing for large datasets
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1

from .esm_embeddings import get_esm_embedding


def extract_backbone_coords(
    structure,
    pdb_id: str,
    pdb_path: str
) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    """
    Extract backbone coordinates from protein structure.
    
    Args:
        structure: BioPython structure object
        pdb_id: PDB identifier
        pdb_path: Path to PDB file
        
    Returns:
        Tuple of (sequence, coordinates dict, chain_id)
    """
    coords = {"N": [], "CA": [], "C": [], "O": []}
    seq = ""
    
    model = structure[0]
    
    # Find valid chain
    valid_chain = None
    for chain in model:
        if any(is_aa(res, standard=True) for res in chain):
            valid_chain = chain
            break
    
    if valid_chain is None:
        return None, None, None
    
    chain_id = valid_chain.id
    
    # Extract coordinates and sequence
    for res in valid_chain:
        if not is_aa(res, standard=True):
            continue
        
        # Get one-letter code
        try:
            seq += seq1(res.resname)
        except:
            seq += 'X'  # Unknown residue
        
        # Get backbone atom coordinates
        for atom_name in ["N", "CA", "C", "O"]:
            if atom_name in res:
                coords[atom_name].append(res[atom_name].coord.tolist())
            else:
                coords[atom_name].append([float("nan")] * 3)
    
    return seq, coords, chain_id


def _process_pdb_path(pdb_path: str) -> Tuple:
    """
    Process a single PDB file (for parallel processing).
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        Tuple with processing status and data
    """
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        seq, coords, chain_id = extract_backbone_coords(structure, pdb_id, pdb_path)
        
        if seq is None:
            return ("skip", pdb_id, "No valid chain found")
        
        if not coords or len(coords["N"]) == 0:
            return ("skip", pdb_id, "No valid coordinates")
        
        # Stack coordinates in order: N, CA, C, O
        coords_stacked = []
        for i in range(len(coords["N"])):
            coord_group = []
            for atom in ["N", "CA", "C", "O"]:
                coord_group.append(coords[atom][i])
            coords_stacked.append(coord_group)
        
        return ("ok", pdb_id, seq, coords_stacked, chain_id)
        
    except Exception as e:
        return ("error", pdb_id, str(e))


class StructureProcessor:
    """
    Process protein structures in parallel with chunking support.
    """
    
    def __init__(
        self,
        esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
        chunk_size: int = 100000,
        max_workers: Optional[int] = None,
        embed_dir: str = "esm_embeddings",
        out_dir: str = "structure_chunks"
    ):
        """
        Initialize structure processor.
        
        Args:
            esm_model_name: ESM model to use
            chunk_size: Maximum structures per chunk
            max_workers: Number of parallel workers
            embed_dir: Directory for embeddings
            out_dir: Directory for structure chunks
        """
        self.esm_model_name = esm_model_name
        self.chunk_size = chunk_size
        self.max_workers = max_workers or (cpu_count() - 1)
        self.embed_dir = Path(embed_dir)
        self.out_dir = Path(out_dir)
        
        # Create directories
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def process_chunk(
        self,
        chunk_idx: int,
        pdb_paths: List[str],
        esm_model = None,
        tokenizer = None
    ) -> Dict:
        """
        Process a single chunk of PDB files.
        
        Args:
            chunk_idx: Chunk index
            pdb_paths: List of PDB paths
            esm_model: ESM model (will load if None)
            tokenizer: ESM tokenizer (will load if None)
            
        Returns:
            Chunk processing results
        """
        print(f"\n[Chunk {chunk_idx}] Processing {len(pdb_paths)} structures")
        
        # Load ESM model if needed
        if esm_model is None or tokenizer is None:
            from transformers import EsmModel, EsmTokenizer
            tokenizer = EsmTokenizer.from_pretrained(self.esm_model_name)
            esm_model = EsmModel.from_pretrained(self.esm_model_name)
            esm_model.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            esm_model = esm_model.to(device)
        else:
            device = next(esm_model.parameters()).device
        
        structure_dict = {}
        results = []
        
        # Phase 1: Parse PDB files in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_process_pdb_path, p): p for p in pdb_paths}
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Chunk {chunk_idx} - PDB parsing"
            ):
                try:
                    status_tuple = fut.result(timeout=30)
                    results.append(status_tuple)
                except Exception as e:
                    print(f"[Chunk {chunk_idx}] Error: {e}")
        
        # Process results
        ok_items = []
        for r in results:
            tag = r[0]
            if tag == "ok":
                _, pdb_id, seq, coords_stacked, chain_id = r
                ok_items.append((pdb_id, seq, coords_stacked, chain_id))
            elif tag == "skip":
                _, pdb_id, msg = r
                print(f"[SKIP] {pdb_id}: {msg}")
            elif tag == "error":
                _, pdb_id, err = r
                print(f"[ERROR] {pdb_id}: {err}")
        
        # Phase 2: Generate ESM embeddings
        print(f"[Chunk {chunk_idx}] Generating embeddings for {len(ok_items)} proteins")
        
        for pdb_id, seq, coords_stacked, chain_id in tqdm(ok_items):
            try:
                # Generate embedding
                embedding = get_esm_embedding(seq, esm_model, tokenizer, device)
                
                # Save embedding
                embed_path = self.embed_dir / f"{pdb_id}.pt"
                torch.save(embedding.cpu(), embed_path)
                
                # Store structure data
                structure_dict[pdb_id] = {
                    "name": pdb_id,
                    "UniProt_id": "UNKNOWN",
                    "PDB_id": pdb_id,
                    "chain": chain_id,
                    "seq": seq,
                    "coords": coords_stacked,
                    "embed": str(embed_path)
                }
                
            except Exception as e:
                print(f"[Chunk {chunk_idx}] Embedding failed for {pdb_id}: {e}")
        
        # Save chunk
        chunk_filename = f"structures_chunk_{chunk_idx:04d}.json"
        chunk_path = self.out_dir / chunk_filename
        with open(chunk_path, "w") as f:
            json.dump(structure_dict, f, indent=2)
        
        print(f"[Chunk {chunk_idx}] ✅ Saved {len(structure_dict)} structures")
        
        # Clean up
        if esm_model is not None and device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "chunk_idx": chunk_idx,
            "filename": chunk_filename,
            "path": str(chunk_path),
            "num_structures": len(structure_dict),
            "num_errors": len(results) - len(ok_items)
        }
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        pdb_col: str = "standardized_protein_pdb"
    ) -> Dict:
        """
        Process all structures in dataframe.
        
        Args:
            df: DataFrame with PDB paths
            pdb_col: Column containing PDB paths
            
        Returns:
            Processing metadata
        """
        # Get unique PDB paths
        pdb_paths = df[pdb_col].unique().tolist()
        total_pdbs = len(pdb_paths)
        num_chunks = (total_pdbs + self.chunk_size - 1) // self.chunk_size
        
        print(f"Processing {total_pdbs} unique PDBs in {num_chunks} chunks")
        
        chunk_results = []
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, total_pdbs)
            chunk_paths = pdb_paths[start_idx:end_idx]
            
            result = self.process_chunk(chunk_idx, chunk_paths)
            chunk_results.append(result)
        
        # Create metadata
        metadata = {
            "num_chunks": num_chunks,
            "chunk_size": self.chunk_size,
            "total_structures": sum(r["num_structures"] for r in chunk_results),
            "chunks": chunk_results
        }
        
        # Save metadata
        metadata_path = self.out_dir / "chunk_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"  Total structures: {metadata['total_structures']}")
        print(f"  Metadata saved: {metadata_path}")
        
        return metadata


class StructureChunkLoader:
    """
    Efficient loader for chunked structure dictionaries.
    """
    
    def __init__(
        self,
        chunk_dir: str = "structure_chunks",
        cache_size: int = 2
    ):
        """
        Initialize chunk loader.
        
        Args:
            chunk_dir: Directory containing chunks
            cache_size: Number of chunks to cache in memory
        """
        self.chunk_dir = Path(chunk_dir)
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
        # Load metadata
        metadata_path = self.chunk_dir / "chunk_metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Build PDB to chunk index
        self.pdb_to_chunk = {}
        for chunk_info in self.metadata["chunks"]:
            chunk_path = self.chunk_dir / chunk_info["filename"]
            if chunk_path.exists():
                with open(chunk_path, "r") as f:
                    chunk_data = json.load(f)
                    for pdb_id in chunk_data.keys():
                        self.pdb_to_chunk[pdb_id] = chunk_info["chunk_idx"]
        
        print(f"Loaded {len(self.pdb_to_chunk)} structures from {self.metadata['num_chunks']} chunks")
    
    def _load_chunk(self, chunk_idx: int) -> Dict:
        """Load a chunk into cache."""
        if chunk_idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.cache[chunk_idx]
        
        # Load chunk
        chunk_info = self.metadata["chunks"][chunk_idx]
        chunk_path = self.chunk_dir / chunk_info["filename"]
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
        
        # Add to cache
        self.cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            gc.collect()
        
        return chunk_data
    
    def get(self, pdb_id: str) -> Optional[Dict]:
        """Get structure for a specific PDB ID."""
        if pdb_id not in self.pdb_to_chunk:
            return None
        
        chunk_idx = self.pdb_to_chunk[pdb_id]
        chunk_data = self._load_chunk(chunk_idx)
        return chunk_data.get(pdb_id)
    
    def get_batch(self, pdb_ids: List[str]) -> Dict[str, Dict]:
        """Get multiple structures efficiently."""
        # Group by chunk
        chunk_groups = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.pdb_to_chunk:
                chunk_idx = self.pdb_to_chunk[pdb_id]
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(pdb_id)
        
        # Load from chunks
        results = {}
        for chunk_idx, chunk_pdb_ids in chunk_groups.items():
            chunk_data = self._load_chunk(chunk_idx)
            for pdb_id in chunk_pdb_ids:
                if pdb_id in chunk_data:
                    results[pdb_id] = chunk_data[pdb_id]
        
        return results
    
    def get_available_pdb_ids(self) -> set:
        """Return set of all available PDB IDs."""
        return set(self.pdb_to_chunk.keys())