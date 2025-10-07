#!/usr/bin/env python3
"""
APEX2 Affinity Prediction Script

Usage:
    python run.py -i input.csv -o output.csv [--fast] [--gpu 0]

Arguments:
    -i, --input     Input CSV file with 'target' and 'ligand' columns
    -o, --output    Output CSV file for predictions
    --fast          Enable fast mode for >1000 complexes (optional)
    --gpu           GPU device ID (default: 0)
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import EsmModel, EsmTokenizer
from rdkit import RDLogger

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import APEX modules
from gnn_dta_mtl import (
    MTL_DTAModel,
    DTAPredictor,
    predict_affinity,
)

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='APEX2: Affinity Prediction with Embedded X-structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py -i examples.csv -o predictions.csv
    python run.py -i examples.csv -o predictions.csv --fast --gpu 1
    python run.py -i examples.csv -o predictions.csv --gpu 0
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input CSV file with "target" and "ligand" columns containing file paths'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output CSV file for affinity predictions'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        default=False,
        help='Enable fast mode for processing >1000 complexes (default: False)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )
    
    return parser.parse_args()


def load_models(gpu_id, model_checkpoint='./models/best_model.pt'):
    """
    Load ESM and APEX models.
    
    Args:
        gpu_id: GPU device ID
        model_checkpoint: Path to APEX model checkpoint
        
    Returns:
        predictor: DTAPredictor instance
        device: torch device
    """
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        torch.cuda.manual_seed_all(SEED)
        print(f"Using device: cuda:{gpu_id}")
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("CUDA not available. Using CPU.")
    
    # Load ESM model
    print("\nLoading ESM model...")
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    esm_model.eval()
    
    if torch.cuda.is_available():
        esm_model = esm_model.to(device)
    
    print("✓ ESM model loaded")
    
    # Load APEX model
    print("\nLoading APEX model...")
    config = {
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
    
    # Initialize model
    model = MTL_DTAModel(
        task_names=config['task_cols'],
        **config['model_config']
    ).to(device)
    
    # Create predictor
    predictor = DTAPredictor(
        model, 
        model_checkpoint, 
        device=device, 
        esm_model=esm_model
    )
    
    print("✓ APEX model loaded\n")
    
    return predictor, device


def validate_input_file(input_path):
    """
    Validate input CSV file.
    
    Args:
        input_path: Path to input CSV file
        
    Returns:
        df: Pandas DataFrame with validated data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read CSV
    df = pd.read_csv(input_path)
    
    # Check required columns
    required_cols = ['target', 'ligand']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Input CSV missing required columns: {missing_cols}\n"
            f"Required columns: {required_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Check if files exist
    print(f"Validating input files...")
    missing_files = []
    
    for idx, row in df.iterrows():
        if not os.path.exists(row['target']):
            missing_files.append(f"Row {idx}: target file not found: {row['target']}")
        if not os.path.exists(row['ligand']):
            missing_files.append(f"Row {idx}: ligand file not found: {row['ligand']}")
    
    if missing_files:
        print("\nWarning: Some files not found:")
        for msg in missing_files[:10]:  # Show first 10
            print(f"  {msg}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print(f"✓ Validated {len(df)} protein-ligand pairs\n")
    
    return df


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 60)
    print("APEX2: Affinity Prediction with Embedded X-structures")
    print("=" * 60)
    print(f"\nInput file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Fast mode:   {args.fast}")
    print(f"GPU device:  {args.gpu}")
    print()
    
    # Validate input
    df = validate_input_file(args.input)
    
    # Load models
    predictor, device = load_models(args.gpu)
    
    # Prepare protein-ligand pairs
    test_complexes = list(df[['target', 'ligand']].itertuples(index=False, name=None))
    
    print(f"Running predictions on {len(test_complexes)} complexes...")
    print(f"Fast mode: {'Enabled' if args.fast else 'Disabled'}")
    print()
    
    # Run predictions
    predictions = predict_affinity(
        protein_ligand_pairs=test_complexes,
        output_path=args.output,
        device=device,
        predictor=predictor,
        esm_model=predictor.esm_model,
        fast=args.fast
    )
    
    print(f"\n{'=' * 60}")
    print(f"✓ Predictions complete!")
    print(f"✓ Results saved to: {args.output}")
    print(f"{'=' * 60}\n")
    
    # Display summary
    print("Prediction Summary:")
    print(f"  Total complexes: {len(predictions)}")
    print(f"\n  Task predictions (mean ± std):")
    for col in ['pKi', 'pEC50', 'pKd', 'pIC50', 'pKd (Wang, FEP)']:
        if col in predictions.columns:
            mean_val = predictions[col].mean()
            std_val = predictions[col].std()
            print(f"    {col:20s}: {mean_val:6.3f} ± {std_val:5.3f}")
    
    print("\nNote: The 'potency' task is a future head and will be covered once SAIR is integrated.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)