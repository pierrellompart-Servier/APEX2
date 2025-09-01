#!/usr/bin/env python3
"""
Cross-validation script for MTL-GNN-DTA
"""

import argparse
import os
import sys
sys.path.append('../')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

from mtl_gnn_dta import (
    Config,
    MTL_DTAModel,
    create_data_loaders
)
from mtl_gnn_dta.data.dataset import CrossValidationDataset
from mtl_gnn_dta.models.losses import MaskedMSELoss
from mtl_gnn_dta.training import Trainer
from mtl_gnn_dta.training.evaluator import Evaluator
from mtl_gnn_dta.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-validation for MTL-GNN-DTA')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to full dataset')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--output_dir', type=str, default='../experiments/cv/',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def train_fold(fold_idx: int,
               train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               test_df: pd.DataFrame,
               config: Config,
               task_ranges: dict,
               device: torch.device,
               output_dir: Path):
    """Train a single CV fold"""
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets (implement based on your data)
    # train_dataset = create_dataset(train_df, ...)
    # val_dataset = create_dataset(val_df, ...)
    # test_dataset = create_dataset(test_df, ...)
    
    # For demonstration
    from mtl_gnn_dta.data.dataset import MTL_DTA
    train_dataset = MTL_DTA(train_df, [], config.model.task_names)
    val_dataset = MTL_DTA(val_df, [], config.model.task_names)
    test_dataset = MTL_DTA(test_df, [], config.model.task_names)
    
    # Create data loaders
    loaders = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.data.batch_size
    )
    
    # Create model
    model = MTL_DTAModel(
        task_names=config.model.task_names,
        prot_emb_dim=config.model.prot_emb_dim,
        prot_gcn_dims=config.model.prot_gcn_dims,
        prot_fc_dims=config.model.prot_fc_dims,
        drug_node_in_dim=config.model.drug_node_in_dim,
        drug_node_h_dims=config.model.drug_node_h_dims,
        drug_fc_dims=config.model.drug_fc_dims,
        mlp_dims=config.model.mlp_dims,
        mlp_dropout=config.model.mlp_dropout
    ).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate
    )
    
    criterion = MaskedMSELoss(task_ranges=task_ranges).to(device)
    
    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(output_dir / f'fold_{fold_idx}')
    )
    
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        n_epochs=config.training.n_epochs,
        patience=config.training.patience
    )
    
    # Evaluate
    evaluator = Evaluator(config.model.task_names)
    test_metrics = evaluator.evaluate(
        model,
        loaders['test'],
        device,
        criterion
    )
    
    return test_metrics, history


def main():
    args = parse_args()
    
    # Setup
    setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    config.training.n_folds = args.n_folds
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    # Calculate task ranges
    task_ranges = {}
    for task in config.model.task_names:
        if task in df.columns:
            valid_values = df[task].dropna()
            if len(valid_values) > 0:
                task_ranges[task] = valid_values.max() - valid_values.min()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize cross-validation
    cv_dataset = CrossValidationDataset(
        df,
        config.model.task_names,
        n_folds=args.n_folds,
        random_state=args.seed
    )
    
    # Train each fold
    all_results = []
    
    for fold_idx in range(args.n_folds):
        # Get fold data
        train_df, val_df, test_df = cv_dataset.get_fold_with_validation(
            fold_idx,
            val_size=0.1
        )
        
        # Train fold
        fold_metrics, fold_history = train_fold(
            fold_idx,
            train_df,
            val_df,
            test_df,
            config,
            task_ranges,
            device,
            output_dir
        )
        
        fold_metrics['fold'] = fold_idx
        all_results.append(fold_metrics)
        
        print(f"\nFold {fold_idx + 1} Results:")
        for key, value in fold_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    # Aggregate results
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    
    # Calculate mean and std for each metric
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'fold']
    
    for col in numeric_cols:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)
    
    # Save summary
    summary = {
        'n_folds': args.n_folds,
        'config': config.to_dict(),
        'results': results_df.to_dict('records'),
        'summary': {
            col: {
                'mean': float(results_df[col].mean()),
                'std': float(results_df[col].std())
            }
            for col in numeric_cols
        }
    }
    
    import json
    with open(output_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()