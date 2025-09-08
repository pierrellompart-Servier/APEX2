#!/usr/bin/env python3
"""
ddp_training.py - Complete DDP Training Module
Save this as a separate Python file to avoid multiprocessing issues
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import your custom modules here
# These need to be importable from the Python path
try:
    from your_modules import (
        MTL_DTAModel, 
        MaskedMSELoss,
        build_mtl_dataset_optimized
    )
except ImportError:
    print("Warning: Custom modules not found. Define them in this file or ensure they're importable.")

# ============= DDP SETUP FUNCTIONS =============

def setup_ddp(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


# ============= DDP WORKER FUNCTION =============

def train_fold_ddp_worker(rank, world_size, args):
    """
    Worker function for DDP training
    Args passed as dict to avoid pickling issues
    """
    # Unpack arguments
    fold_idx = args['fold_idx']
    n_folds = args['n_folds']
    df_train = args['df_train']
    df_valid = args['df_valid']
    df_test = args['df_test']
    chunk_loader = args['chunk_loader']
    task_cols = args['task_cols']
    task_ranges = args['task_ranges']
    n_epochs = args['n_epochs']
    batch_size = args['batch_size']
    lr = args['lr']
    patience = args['patience']
    fold_results_path = args['fold_results_path']
    
    # Setup DDP
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Only rank 0 prints
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds} - DDP with {world_size} GPUs")
        print(f"{'='*60}")
    
    try:
        # Create datasets
        train_dataset = build_mtl_dataset_optimized(df_train, chunk_loader, task_cols)
        valid_dataset = build_mtl_dataset_optimized(df_valid, chunk_loader, task_cols)
        test_dataset = build_mtl_dataset_optimized(df_test, chunk_loader, task_cols)
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers=4,
            pin_memory=True, persistent_workers=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size,
            sampler=valid_sampler, num_workers=4,
            pin_memory=True, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            sampler=test_sampler, num_workers=4,
            pin_memory=True, persistent_workers=True
        )
        
        # Create model
        model = MTL_DTAModel(
            task_names=task_cols,
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1],
            drug_node_h_dims=[128, 64],
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512],
            mlp_dropout=0.25
        ).to(device)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = MaskedMSELoss(task_ranges=task_ranges).to(device)
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        
        # Training loop
        if rank == 0:
            pbar = tqdm(range(n_epochs), desc=f"Training Fold {fold_idx+1}", ncols=100)
        else:
            pbar = range(n_epochs)
        
        for epoch in pbar:
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)
            
            # Training
            model.train()
            train_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                y = batch['y'].to(device)
                
                optimizer.zero_grad()
                pred = model(xd, xp)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            # Average loss across processes
            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0
            train_loss_tensor = torch.tensor([avg_train_loss]).to(device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / world_size
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch in valid_loader:
                    xd = batch['drug'].to(device)
                    xp = batch['protein'].to(device)
                    y = batch['y'].to(device)
                    
                    pred = model(xd, xp)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
                    n_val_batches += 1
            
            # Average validation loss
            avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('inf')
            val_loss_tensor = torch.tensor([avg_val_loss]).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / world_size
            val_losses.append(avg_val_loss)
            
            # Update progress
            if rank == 0:
                pbar.set_postfix({
                    'train': f"{avg_train_loss:.4f}",
                    'val': f"{avg_val_loss:.4f}"
                })
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.module.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if rank == 0:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.module.load_state_dict(best_model_state)
        
        # Testing (gather results from all processes)
        # ... [rest of testing code]
        
        # Save results (only rank 0)
        if rank == 0:
            fold_results = {
                'fold_idx': fold_idx,
                'train_losses': train_losses,
                'val_losses': val_losses,
                # Add test results here
            }
            with open(fold_results_path, 'wb') as f:
                pickle.dump(fold_results, f)
        
    finally:
        cleanup_ddp()


# ============= MAIN CV FUNCTION =============

def run_ddp_cross_validation_safe(
    df_clean, chunk_loader, task_cols, task_ranges,
    n_folds=5, n_epochs=100, batch_size=256, 
    lr=0.0005, patience=20
):
    """
    Safe DDP cross-validation that works with Jupyter
    """
    world_size = torch.cuda.device_count()
    
    print(f"Starting DDP CV with {world_size} GPUs")
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_clean)):
        # Prepare data
        df_train = df_clean.iloc[train_idx].reset_index(drop=True)
        df_test = df_clean.iloc[test_idx].reset_index(drop=True)
        
        valid_size = int(0.1 * len(df_train))
        df_valid = df_train.sample(n=valid_size, random_state=42)
        df_train = df_train.drop(df_valid.index).reset_index(drop=True)
        
        # Pack arguments
        fold_results_path = f'/tmp/fold_{fold_idx}_results.pkl'
        args = {
            'fold_idx': fold_idx,
            'n_folds': n_folds,
            'df_train': df_train,
            'df_valid': df_valid,
            'df_test': df_test,
            'chunk_loader': chunk_loader,
            'task_cols': task_cols,
            'task_ranges': task_ranges,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'patience': patience,
            'fold_results_path': fold_results_path
        }
        
        # Launch DDP training
        mp.spawn(
            train_fold_ddp_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
        
        # Load results
        with open(fold_results_path, 'rb') as f:
            fold_results = pickle.load(f)
        all_results.append(fold_results)
        
        # Clean up
        os.remove(fold_results_path)
        torch.cuda.empty_cache()
        gc.collect()
    
    return all_results


if __name__ == "__main__":
    print("DDP Training module loaded successfully!")
    print("Import this module in your notebook:")
    print("  from ddp_training import run_ddp_cross_validation_safe")
