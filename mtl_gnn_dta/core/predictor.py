"""Main predictor class for drug-target affinity prediction"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

from mtl_gnn_dta.core.config import Config
from mtl_gnn_dta.models.dta_model import MTL_DTAModel
from mtl_gnn_dta.features.protein_features import ProteinFeaturizer
from mtl_gnn_dta.features.drug_features import DrugFeaturizer
from mtl_gnn_dta.data.loaders import collate_batch

logger = logging.getLogger(__name__)


class AffinityPredictor:
    """High-level interface for drug-target affinity prediction"""
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 model_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize predictor
        
        Args:
            config: Configuration object
            model_path: Path to trained model checkpoint
            device: PyTorch device
        """
        self.config = config or Config()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize featurizers
        self.protein_featurizer = ProteinFeaturizer()
        self.drug_featurizer = DrugFeaturizer()
        
        # Initialize model
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract configuration if available
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = self.config.model
        
        # Create model
        self.model = MTL_DTAModel(
            task_names=model_config.task_names if hasattr(model_config, 'task_names') 
                      else checkpoint.get('task_cols', ['pKi', 'pEC50', 'pKd', 'pIC50']),
            prot_emb_dim=model_config.prot_emb_dim if hasattr(model_config, 'prot_emb_dim') else 1280,
            prot_gcn_dims=model_config.prot_gcn_dims if hasattr(model_config, 'prot_gcn_dims') 
                         else [128, 256, 256],
            prot_fc_dims=model_config.prot_fc_dims if hasattr(model_config, 'prot_fc_dims') 
                        else [1024, 128],
            drug_node_in_dim=model_config.drug_node_in_dim if hasattr(model_config, 'drug_node_in_dim') 
                            else [66, 1],
            drug_node_h_dims=model_config.drug_node_h_dims if hasattr(model_config, 'drug_node_h_dims') 
                            else [128, 64],
            drug_fc_dims=model_config.drug_fc_dims if hasattr(model_config, 'drug_fc_dims') 
                        else [1024, 128],
            mlp_dims=model_config.mlp_dims if hasattr(model_config, 'mlp_dims') else [1024, 512],
            mlp_dropout=model_config.mlp_dropout if hasattr(model_config, 'mlp_dropout') else 0.25
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store task information
        self.task_cols = checkpoint.get('task_cols', ['pKi', 'pEC50', 'pKd', 'pIC50'])
        self.task_ranges = checkpoint.get('task_ranges', {})
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict_from_files(self, 
                           protein_path: str,
                           ligand_path: str,
                           protein_sequence: Optional[str] = None,
                           protein_embedding: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Predict affinity from protein and ligand files
        
        Args:
            protein_path: Path to protein PDB file
            ligand_path: Path to ligand SDF file
            protein_sequence: Optional protein sequence
            protein_embedding: Optional pre-computed protein embedding
        
        Returns:
            Dictionary with predictions for each task
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Featurize protein
        protein_data = self.protein_featurizer.featurize_from_pdb(
            protein_path, 
            sequence=protein_sequence,
            embedding=protein_embedding
        )
        
        # Featurize drug
        drug_data = self.drug_featurizer.featurize_drug(ligand_path)
        
        if protein_data is None or drug_data is None:
            raise ValueError("Failed to featurize protein or drug")
        
        # Create batch
        from torch_geometric.data import Batch
        protein_batch = Batch.from_data_list([protein_data])
        drug_batch = Batch.from_data_list([drug_data])
        
        # Move to device
        protein_batch = protein_batch.to(self.device)
        drug_batch = drug_batch.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(drug_batch, protein_batch)
        
        # Convert to dictionary
        predictions = predictions.cpu().numpy()[0]
        result = {}
        for i, task in enumerate(self.task_cols):
            result[task] = float(predictions[i])
        
        return result
    
    def predict_batch(self,
                     protein_paths: List[str],
                     ligand_paths: List[str],
                     batch_size: int = 32) -> pd.DataFrame:
        """
        Predict affinities for multiple protein-ligand pairs
        
        Args:
            protein_paths: List of protein PDB file paths
            ligand_paths: List of ligand SDF file paths
            batch_size: Batch size for prediction
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if len(protein_paths) != len(ligand_paths):
            raise ValueError("Number of proteins and ligands must match")
        
        results = []
        
        # Process in batches
        for i in range(0, len(protein_paths), batch_size):
            batch_proteins = protein_paths[i:i+batch_size]
            batch_ligands = ligand_paths[i:i+batch_size]
            
            protein_data_list = []
            drug_data_list = []
            
            # Featurize batch
            for prot_path, lig_path in zip(batch_proteins, batch_ligands):
                try:
                    protein_data = self.protein_featurizer.featurize_from_pdb(prot_path)
                    drug_data = self.drug_featurizer.featurize_drug(lig_path)
                    
                    if protein_data is not None and drug_data is not None:
                        protein_data_list.append(protein_data)
                        drug_data_list.append(drug_data)
                except Exception as e:
                    logger.warning(f"Failed to process {prot_path} and {lig_path}: {e}")
                    continue
            
            if len(protein_data_list) == 0:
                continue
            
            # Create batches
            from torch_geometric.data import Batch
            protein_batch = Batch.from_data_list(protein_data_list).to(self.device)
            drug_batch = Batch.from_data_list(drug_data_list).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(drug_batch, protein_batch)
            
            # Store results
            predictions = predictions.cpu().numpy()
            for j in range(predictions.shape[0]):
                result = {
                    'protein_path': batch_proteins[j],
                    'ligand_path': batch_ligands[j]
                }
                for k, task in enumerate(self.task_cols):
                    result[task] = float(predictions[j, k])
                results.append(result)
        
        return pd.DataFrame(results)
    
    def predict_from_smiles(self,
                           protein_path: str,
                           smiles: str,
                           protein_sequence: Optional[str] = None) -> Dict[str, float]:
        """
        Predict affinity from protein file and SMILES string
        
        Args:
            protein_path: Path to protein PDB file
            smiles: SMILES string of the ligand
            protein_sequence: Optional protein sequence
        
        Returns:
            Dictionary with predictions for each task
        """
        # Convert SMILES to temporary SDF
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
            writer = Chem.SDWriter(tmp.name)
            writer.write(mol)
            writer.close()
            
            # Predict
            result = self.predict_from_files(
                protein_path, 
                tmp.name,
                protein_sequence=protein_sequence
            )
        
        # Clean up
        Path(tmp.name).unlink()
        
        return result
    
    def explain_prediction(self,
                          protein_path: str,
                          ligand_path: str) -> Dict:
        """
        Get feature importance and attention weights for a prediction
        
        Args:
            protein_path: Path to protein PDB file
            ligand_path: Path to ligand SDF file
        
        Returns:
            Dictionary with explanations
        """
        # This would implement attention visualization and feature importance
        # For now, return a placeholder
        raise NotImplementedError("Explanation functionality will be implemented in future versions")


def load_predictor(model_path: str, 
                  config_path: Optional[str] = None) -> AffinityPredictor:
    """
    Convenience function to load a predictor
    
    Args:
        model_path: Path to trained model
        config_path: Optional path to configuration file
    
    Returns:
        Initialized AffinityPredictor
    """
    config = Config(config_path) if config_path else None
    predictor = AffinityPredictor(config=config, model_path=model_path)
    return predictor
