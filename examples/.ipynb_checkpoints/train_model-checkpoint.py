#!/usr/bin/env python3
"""
Training script for MTL-GNN-DTA model
"""

import argparse
import os
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

from mtl_gnn_dta import (
    Config,
    MTL_DTAModel,
    Trainer,
    create_data_loaders,
    create_dataset
)
from mtl_gnn_dta.models.losses import MaskedMSELoss
from mtl_gnn_dta.training.callbacks import EarlyStopping, ModelCheckpoint
from mtl_gnn_dta.training.evaluator import Evaluator
from mtl_gnn_dta.features import ProteinFeaturizer, DrugFeaturizer
from mtl_gnn_dta.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train MTL-GNN-DTA model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='../data/processed/',
                       help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='../experiments/',
                       help='Output directory')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    setup_logging(log_file=f"training_{args.seed}.log")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print(f"Using device: {device}")
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    config.training.n_epochs = args.n_epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.patience = args.patience
    
    # Load data
    print("Loading data...")
    data_dir = Path(args.data_dir)
    
    train_df = pd.read_parquet(data_dir / 'train_data.parquet')
    val_df = pd.read_parquet(data_dir / 'val_data.parquet')
    test_df = pd.read_parquet(data_dir / 'test_data.parquet')
    
    # Load task ranges
    import json
    with open(data_dir / 'task_ranges.json', 'r') as f:
        task_ranges = json.load(f)
    
    print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    # Initialize featurizers
    print("Initializing featurizers...")
    protein_featurizer = ProteinFeaturizer()
    drug_featurizer = DrugFeaturizer()
    
    # Load protein structures (placeholder - implement based on your data)
    # protein_structures = load_protein_structures(...)
    
    # Create datasets (placeholder - implement based on your data)
    print("Creating datasets...")
    # train_dataset = create_dataset(train_df, protein_structures, 
    #                               config.model.task_names, 
    #                               protein_featurizer, drug_featurizer)
    # val_dataset = create_dataset(val_df, protein_structures,
    #                             config.model.task_names,
    #                             protein_featurizer, drug_featurizer)
    
    # For demonstration, create mock datasets
    from mtl_gnn_dta.data.dataset import MTL_DTA
    train_dataset = MTL_DTA(train_df, [], config.model.task_names)
    val_dataset = MTL_DTA(val_df, [], config.model.task_names)
    test_dataset = MTL_DTA(test_df, [], config.model.task_names)
    
    # Create data loaders
    print("Creating data loaders...")
    loaders = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Create model
    print("Creating model...")
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        verbose=True
    )
    
    criterion = MaskedMSELoss(task_ranges=task_ranges).to(device)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"experiment_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=str(output_dir / 'checkpoints')
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        loaders['train'],
        loaders['val'],
        n_epochs=config.training.n_epochs,
        patience=config.training.patience,
        save_best=True
    )
    
    print(f"Training completed. Best epoch: {history['best_epoch']}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    evaluator = Evaluator(config.model.task_names)
    test_metrics = evaluator.evaluate(
        model,
        loaders['test'],
        device,
        criterion
    )
    
    print("Test set results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Save results
    results = {
        'config': config.to_dict(),
        'history': history,
        'test_metrics': test_metrics
    }
    
    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()