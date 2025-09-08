#!/usr/bin/env python3
"""
run_ddp_cv.py - Standalone DDP training script
Run this from terminal: python run_ddp_cv.py
"""

import os
import sys
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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import gc

# Add notebook directory to path
sys.path.append('/home/HX46_FR5/application/IScoreIt/notebook')

# Now import your modules
from parallel_structure_processing_optimized import StructureChunkLoader

# Import all your model components here
exec(open('model_definitions.py').read())  # Save your model classes to this file

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_fold_ddp(rank, world_size, fold_data):
    """DDP training for one fold"""
    setup_ddp(rank, world_size)
    
    # Unpack fold data
    train_data = fold_data['train']
    valid_data = fold_data['valid']
    test_data = fold_data['test']
    task_cols = fold_data['task_cols']
    task_ranges = fold_data['task_ranges']
    fold_idx = fold_data['fold_idx']
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank)
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=256, sampler=train_sampler)
    valid_loader = DataLoader(valid_data, batch_size=256, sampler=valid_sampler)
    
    # Create model
    model = MTL_DTAModel(task_names=task_cols).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = MaskedMSELoss(task_ranges).cuda(rank)
    
    # Training loop
    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        
        # Train
        model.train()
        for batch in train_loader:
            xd = batch['drug'].cuda(rank)
            xp = batch['protein'].cuda(rank)
            y = batch['y'].cuda(rank)
            
            optimizer.zero_grad()
            pred = model(xd, xp)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                xd = batch['drug'].cuda(rank)
                xp = batch['protein'].cuda(rank)
                y = batch['y'].cuda(rank)
                pred = model(xd, xp)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        if rank == 0:
            print(f"Fold {fold_idx}, Epoch {epoch}: val_loss={val_loss:.4f}")
    
    cleanup_ddp()

def main():
    # Load data
    df_clean = pd.read_pickle('df_clean.pkl')
    chunk_loader = StructureChunkLoader(chunk_dir="../data/structure_chunks/")
    
    task_cols = ['pKi', 'pEC50', 'pKd', 'pKd (Wang, FEP)', 'pIC50', 'potency']
    task_ranges = calculate_task_ranges(df_clean, task_cols)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df_clean)):
        # Prepare fold data
        fold_data = prepare_fold_data(df_clean, train_idx, test_idx, chunk_loader, task_cols, task_ranges, fold_idx)
        
        # Launch DDP training
        world_size = torch.cuda.device_count()
        mp.spawn(train_fold_ddp, args=(world_size, fold_data), nprocs=world_size, join=True)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
