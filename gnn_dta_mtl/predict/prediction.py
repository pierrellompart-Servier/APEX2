import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Batch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import tempfile
import os
import sys
import json
import gc
import warnings
warnings.filterwarnings('ignore')
from Bio.PDB import PDBParser
from Bio.PDB import MMCIFParser, PDBIO, Select

import pandas as pd
import numpy as np
import torch
import torch_geometric
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch_geometric
from torch_geometric.data import Batch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os
import tempfile
import gc
import warnings
warnings.filterwarnings('ignore')

# Import from parent package
from ..models.dta_model import MTL_DTAModel, DTAModel
from ..features.drug_graph import featurize_drug
from ..features.protein_graph import featurize_protein_graph
from ..features.esm_embeddings import ESMEmbedder
from ..training.cross_validation import CrossValidator
from ..training.trainer import MTLTrainer

# Import only if these exist in your data module
# Comment out if they don't exist
try:
    from ..data.standardization import StructureStandardizer
except ImportError:
    pass

try:
    from ..data.preprocessing import StructureProcessor, StructureChunkLoader
except ImportError:
    pass

try:
    from ..data.molecular_properties import (
        add_molecular_properties_parallel,
        compute_ligand_efficiency,
        compute_mean_ligand_efficiency,
        filter_by_properties
    )
except ImportError:
    pass

try:
    from ..datasets.data_utils import prepare_mtl_experiment, build_mtl_dataset, build_mtl_dataset_optimized
except ImportError:
    pass

try:
    from ..evaluation.metrics import evaluate_model
    from ..evaluation.visualization import plot_results, plot_predictions, create_summary_report
except ImportError:
    pass

try:
    from ..utils.logger import ExperimentLogger
    from ..utils.io_utils import save_model, save_results, create_output_dir
except ImportError:
    pass

# Use joblib for parallel processing
from joblib import Parallel, delayed



def process_row_simple(idx, row_data):
    """Simple row processor that avoids complex imports"""
    import os
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    
    parser = PDBParser(QUIET=True)
    
    def extract_backbone_coords_simple(structure, pdb_id, pdb_path):
        """Extract backbone coordinates from protein structure."""
        coords = {"N": [], "CA": [], "C": [], "O": []}
        seq = ""
        
        model = structure[0]
        
        # Find valid chain
        valid_chain = None
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # Standard amino acid
                    valid_chain = chain
                    break
            if valid_chain:
                break
        
        if valid_chain is None:
            return None, None, None
        
        chain_id = valid_chain.id
        
        # Extract coordinates and sequence
        for res in valid_chain:
            if res.id[0] != ' ':  # Skip non-standard residues
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
    
    try:
        pdb_path = row_data['protein_path']
        pdb_id = os.path.basename(pdb_path).split('.')[0]
        structure = parser.get_structure(pdb_id, pdb_path)
        seq, coords, chain_id = extract_backbone_coords_simple(structure, pdb_id, pdb_path)
        if seq is None:
            raise ValueError(f"No valid chain found in {pdb_path}")
        return {
            'idx': idx,
            'pdb_path': pdb_path,
            'seq': seq,
            'coords': coords,
            'chain_id': chain_id,
            'pdb_id': pdb_id
        }
    except Exception as e:
        print(f"Error parsing PDB {idx} ({row_data['protein_path']}): {e}")
        return None

    
class DTAPredictor:
    """Simple predictor for drug-target affinity."""
    
    def __init__(self, model, model_path, config_path=None, device='cuda', esm_model=None):
        """
        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to config file (optional)
            device: Device to run on
            esm_model: Pre-loaded ESM model (optional)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                'task_cols': ['pKi', 'pEC50', 'pKd', 'pIC50', 'pKd (Wang, FEP)', 'potency'],
                'model_config': {
                    'prot_emb_dim': 1280,
                    'prot_gcn_dims': [128, 256, 256],
                    'prot_fc_dims': [1024, 128],
                    'drug_node_in_dim': [66, 1],
                    'drug_node_h_dims': [128, 64],
                    'drug_edge_in_dim': [16, 1],
                    'drug_edge_h_dims': [32, 1],
                    'drug_fc_dims': [1024, 128],
                    'mlp_dims': [1024, 512],
                    'mlp_dropout': 0.25
                }
            }
        
        # Load model
        self.model = model
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Initialize ESM model for protein embeddings
        if esm_model is not None:
            self.esm_model = esm_model
            self.tokenizer = None
        else:
            from transformers import EsmModel, EsmTokenizer
            model_name = "facebook/esm2_t33_650M_UR50D"
            self.tokenizer = EsmTokenizer.from_pretrained(model_name)
            self.esm_model = EsmModel.from_pretrained(model_name)
            self.esm_model.eval()
            self.esm_model = self.esm_model.to(self.device)
        
        # Initialize parser
        self.parser = PDBParser(QUIET=True)
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Using device: {self.device}")
        
    def extract_backbone_coords(self, structure, pdb_id, pdb_path):
        """Extract backbone coordinates from protein structure (from your code)."""
        coords = {"N": [], "CA": [], "C": [], "O": []}
        seq = ""
        
        model = structure[0]
        
        # Find valid chain
        valid_chain = None
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # Standard amino acid
                    valid_chain = chain
                    break
            if valid_chain:
                break
        
        if valid_chain is None:
            return None, None, None
        
        chain_id = valid_chain.id
        
        # Extract coordinates and sequence
        for res in valid_chain:
            if res.id[0] != ' ':  # Skip non-standard residues
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
    
    def get_esm_embedding(self, seq):
        """Get ESM embedding for sequence (from your code)."""
        if self.tokenizer is None:
            from transformers import EsmTokenizer
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        inputs = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            # Remove CLS and EOS tokens
            embedding = outputs.last_hidden_state[0, 1:-1]
        
        return embedding
    
    def create_protein_structure_dict(self, pdb_path):
        """Create protein structure dictionary with real ESM embeddings."""
        pdb_id = os.path.basename(pdb_path).split('.')[0]
        
        structure = self.parser.get_structure(pdb_id, pdb_path)
        seq, coords, chain_id = self.extract_backbone_coords(structure, pdb_id, pdb_path)
        
        if seq is None:
            raise ValueError(f"No valid chain found in {pdb_path}")
        
        # Stack coordinates in order: N, CA, C, O
        coords_stacked = []
        for i in range(len(coords["N"])):
            coord_group = []
            for atom in ["N", "CA", "C", "O"]:
                coord_group.append(coords[atom][i])
            coords_stacked.append(coord_group)
        
        # Get ESM embedding
        embedding = self.get_esm_embedding(seq)
        
        # Save embedding temporarily
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(embedding.cpu(), f.name)
            embed_path = f.name
        
        structure_dict = {
            "name": pdb_id,
            "UniProt_id": "UNKNOWN",
            "PDB_id": pdb_id,
            "chain": chain_id,
            "seq": seq,
            "coords": coords_stacked,
            "embed": embed_path
        }
        
        return structure_dict
    
    def predict(self, protein_ligand_pairs):
        """
        Predict affinities for protein-ligand pairs.

        Args:
            protein_ligand_pairs: List of tuples (protein_path, ligand_path) or
                                 DataFrame with 'protein_path' and 'ligand_path' columns

        Returns:
            DataFrame with predictions for each task
        """
        # Convert to DataFrame if needed
        if isinstance(protein_ligand_pairs, list):
            df = pd.DataFrame(protein_ligand_pairs, 
                            columns=['protein_path', 'ligand_path'])
        else:
            df = protein_ligand_pairs.copy()

        # Step 1: Featurize all pairs
        drug_graphs = []
        prot_graphs = []
        valid_indices = []
        temp_embed_files = []

        print("Featurizing all protein-ligand pairs...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
            try:
                # Featurize drug
                drug_graph = featurize_drug(row['ligand_path'])

                # Create protein structure with real ESM embeddings
                protein_struct = self.create_protein_structure_dict(row['protein_path'])
                temp_embed_files.append(protein_struct['embed'])

                # Featurize protein
                prot_graph = featurize_protein_graph(protein_struct)

                # Store graphs
                drug_graphs.append(drug_graph)
                prot_graphs.append(prot_graph)
                valid_indices.append(idx)

            except Exception as e:
                print(f"Error featurizing {idx} ({row['protein_path']}): {e}")

        # Step 2: Batch all data
        if drug_graphs:
            drug_batch = Batch.from_data_list(drug_graphs).to(self.device)
            prot_batch = Batch.from_data_list(prot_graphs).to(self.device)

            # Step 3: Predict on full batch
            print("Running batch prediction...")
            with torch.no_grad():
                batch_preds = self.model(drug_batch, prot_batch)
                batch_preds = batch_preds.cpu().numpy()

        # Step 4: Format results
        predictions = []
        pred_idx = 0

        for idx in range(len(df)):
            if idx in valid_indices:
                # Get predictions for this sample
                pred = batch_preds[pred_idx]
                pred_dict = {task: float(pred[i]) for i, task in enumerate(self.config['task_cols'])}
                pred_idx += 1
            else:
                # Failed featurization - use NaN
                pred_dict = {task: np.nan for task in self.config['task_cols']}

            predictions.append(pred_dict)

        # Clean up temp embedding files
        for embed_file in temp_embed_files:
            if os.path.exists(embed_file):
                os.remove(embed_file)

        # Create results DataFrame
        results_df = pd.DataFrame(predictions)
        results_df = pd.concat([df[['protein_path', 'ligand_path']], results_df], axis=1)

        return results_df

    
    def predict_ll(self, protein_ligand_pairs, prediction_batch_size=100):
        """
        Predict affinities for protein-ligand pairs with batched ESM and prediction processing.

        Args:
            protein_ligand_pairs: List of tuples (protein_path, ligand_path) or
                                 DataFrame with 'protein_path' and 'ligand_path' columns
            prediction_batch_size: Number of samples to predict at once (default 100)

        Returns:
            DataFrame with predictions for each task
        """
        # Import featurization functions
        from gnn_dta_mtl import featurize_drug, featurize_protein_graph
        
        # Convert to DataFrame if needed
        if isinstance(protein_ligand_pairs, list):
            df = pd.DataFrame(protein_ligand_pairs, 
                            columns=['protein_path', 'ligand_path'])
        else:
            df = protein_ligand_pairs.copy()

        # Step 1: Extract all protein sequences in parallel
        print("Extracting protein sequences in parallel...")
        
        # Prepare data for parallel processing (convert DataFrame rows to dicts)
        row_data = [{'protein_path': row['protein_path']} for _, row in df.iterrows()]
        
        # Use the simple function that's defined outside the class
        results = Parallel(n_jobs=-1)(
            delayed(process_row_simple)(idx, row_data[idx])
            for idx in tqdm(range(len(row_data)), desc="Parsing PDBs")
        )
        
        protein_structures = []
        protein_sequences = []
        sequence_to_indices = {}
        
        for res in results:
            if res is None:
                protein_structures.append(None)
                continue
        
            seq = res['seq']
            protein_structures.append(res)
        
            if seq not in sequence_to_indices:
                sequence_to_indices[seq] = []
                protein_sequences.append(seq)
            sequence_to_indices[seq].append(res['idx'])
        
        # Step 2: Get ESM embeddings for all unique sequences
        print(f"Getting ESM embeddings for {len(protein_sequences)} unique sequences...")
        unique_embeddings = self.get_esm_embeddings_batch(protein_sequences, batch_size=250)
        
        # Create mapping from sequence to embedding
        seq_to_embedding = {seq: emb for seq, emb in zip(protein_sequences, unique_embeddings)}
        
        # Step 3: Process predictions in batches
        all_predictions = []
        total_samples = len(df)
        
        print(f"Processing predictions in batches of {prediction_batch_size}...")
        
        for batch_start in tqdm(range(0, total_samples, prediction_batch_size), 
                                desc="Prediction batches"):
            batch_end = min(batch_start + prediction_batch_size, total_samples)
            batch_indices = range(batch_start, batch_end)
            
            # Featurize current batch
            drug_graphs = []
            prot_graphs = []
            valid_indices = []
            temp_embed_files = []
            
            for idx in batch_indices:
                struct_info = protein_structures[idx]
                if struct_info is None:
                    continue
                
                row = df.iloc[idx]
                
                try:
                    # Featurize drug
                    drug_graph = featurize_drug(row['ligand_path'])
                    
                    # Get embedding for this sequence
                    embedding = seq_to_embedding[struct_info['seq']]
                    
                    # Create protein structure dict with pre-computed embedding
                    protein_struct = self.create_protein_structure_dict_with_embedding(
                        struct_info, 
                        embedding
                    )
                    temp_embed_files.append(protein_struct['embed'])
                    
                    # Featurize protein
                    prot_graph = featurize_protein_graph(protein_struct)
                    
                    # Store graphs
                    drug_graphs.append(drug_graph)
                    prot_graphs.append(prot_graph)
                    valid_indices.append(idx)
                    
                except Exception as e:
                    print(f"Error featurizing {idx} ({row['protein_path']}): {e}")
            
            # Predict on current batch if we have valid samples
            if drug_graphs:
                drug_batch = Batch.from_data_list(drug_graphs).to(self.device)
                prot_batch = Batch.from_data_list(prot_graphs).to(self.device)
                
                with torch.no_grad():
                    batch_preds = self.model(drug_batch, prot_batch)
                    batch_preds = batch_preds.cpu().numpy()
                
                # Clear GPU memory after each batch
                del drug_batch, prot_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Format batch results
            pred_idx = 0
            for idx in batch_indices:
                if idx in valid_indices:
                    # Get predictions for this sample
                    pred = batch_preds[pred_idx]
                    pred_dict = {task: float(pred[i]) for i, task in enumerate(self.config['task_cols'])}
                    pred_idx += 1
                else:
                    # Failed featurization - use NaN
                    pred_dict = {task: np.nan for task in self.config['task_cols']}
                
                all_predictions.append(pred_dict)
            
            # Clean up temp embedding files for this batch
            for embed_file in temp_embed_files:
                if os.path.exists(embed_file):
                    os.remove(embed_file)
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_predictions)
        results_df = pd.concat([df[['protein_path', 'ligand_path']], results_df], axis=1)
        
        return results_df
    
    def get_esm_embeddings_batch(self, sequences, batch_size=250):
        """Get ESM embeddings for a batch of sequences."""
        if self.tokenizer is None:
            from transformers import EsmTokenizer
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing ESM batches"):
            batch_seqs = sequences[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True,
                add_special_tokens=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.esm_model(**inputs)
                # Remove CLS and EOS tokens for each sequence
                batch_embeddings = []
                for j, seq in enumerate(batch_seqs):
                    # Get the actual sequence length (excluding padding)
                    seq_len = len(seq)
                    # Extract embedding without CLS and EOS tokens, up to actual sequence length
                    embedding = outputs.last_hidden_state[j, 1:seq_len+1]  # Skip CLS, take only actual sequence
                    batch_embeddings.append(embedding.cpu())
                
                all_embeddings.extend(batch_embeddings)
            
            # Clear GPU memory
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_embeddings
    
    def create_protein_structure_dict_with_embedding(self, struct_info, embedding):
        """Create protein structure dictionary with pre-computed ESM embedding."""
        # Stack coordinates in order: N, CA, C, O
        coords = struct_info['coords']
        coords_stacked = []
        for i in range(len(coords["N"])):
            coord_group = []
            for atom in ["N", "CA", "C", "O"]:
                coord_group.append(coords[atom][i])
            coords_stacked.append(coord_group)
        
        # Save embedding temporarily
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(embedding, f.name)
            embed_path = f.name
        
        structure_dict = {
            "name": struct_info['pdb_id'],
            "UniProt_id": "UNKNOWN",
            "PDB_id": struct_info['pdb_id'],
            "chain": struct_info['chain_id'],
            "seq": struct_info['seq'],
            "coords": coords_stacked,
            "embed": embed_path
        }
        
        return structure_dict


def predict_affinity(
    protein_ligand_pairs,
    output_path=None,
    device='cuda',
    predictor=None,
    esm_model=None,
    fast=True
):
    """
    Simple function to predict affinities with real featurization.
    
    Args:
        model_path: Path to trained model
        protein_ligand_pairs: List of (protein_pdb, ligand_sdf) or DataFrame
        output_path: Optional path to save predictions
        device: Device to use
        esm_model: Pre-loaded ESM model (optional, will load if not provided)
    
    Returns:
        DataFrame with predictions
    """
    
    # Predict
    if fast:
        results = predictor.predict_ll(protein_ligand_pairs)
    else:
        results = predictor.predict(protein_ligand_pairs)
    
    # Save if requested
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return results