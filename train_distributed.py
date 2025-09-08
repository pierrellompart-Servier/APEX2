#!/usr/bin/env python3
import os
import sys
import gc
import json
import pickle
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch_geometric
from torch_geometric.loader import DataLoader  # Use torch_geometric DataLoader!
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def train_on_gpu(gpu_id, config, return_dict):
    """Train a single model on a specific GPU"""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        
        print(f"[GPU {gpu_id}] Starting training process...")
        
        # Load datasets
        with open(config['train_data_path'], 'rb') as f:
            train_dataset = pickle.load(f)
        with open(config['valid_data_path'], 'rb') as f:
            valid_dataset = pickle.load(f)
        
        print(f"[GPU {gpu_id}] Creating data loaders...")
        
        # IMPORTANT: Use torch_geometric.loader.DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"[GPU {gpu_id}] Loading model and criterion...")
        
        # Load the actual model code
        import importlib.util
        spec = importlib.util.spec_from_file_location("models", config['model_code_path'])
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        
        # Create model
        model = models_module.MTL_DTAModel(
            task_names=config['task_cols'],
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1],
            drug_node_h_dims=[128, 64],
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512],
            mlp_dropout=0.25
        ).to(device)
        
        # Different initialization for each GPU
        torch.manual_seed(config['seed'] + gpu_id)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        # Load criterion
        with open(config['criterion_path'], 'rb') as f:
            criterion = pickle.load(f).to(device)
        
        # Training state
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        train_losses = []
        val_losses = []
        
        print(f"[GPU {gpu_id}] Starting training loop...")
        
        for epoch in range(config['n_epochs']):
            # Training
            model.train()
            train_loss = 0
            n_train_batches = 0
            
            for batch in train_loader:
                try:
                    xd = batch['drug'].to(device)
                    xp = batch['protein'].to(device)
                    y = batch['y'].to(device)
                    
                    optimizer.zero_grad()
                    pred = model(xd, xp)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    n_train_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[GPU {gpu_id}] OOM, skipping batch")
                        optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
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
            
            avg_train_loss = train_loss / max(n_train_batches, 1)
            avg_val_loss = val_loss / max(n_val_batches, 1)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"[GPU {gpu_id}] Epoch {epoch+1}/{config['n_epochs']} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= config['patience']:
                print(f"[GPU {gpu_id}] Early stopping at epoch {epoch+1}")
                break
            
            # Periodic cleanup
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save model
        model_path = f"{config['output_dir']}/model_gpu_{gpu_id}.pt"
        torch.save({
            'model_state': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }, model_path)
        
        print(f"[GPU {gpu_id}] Training complete. Model saved to {model_path}")
        return_dict[gpu_id] = {'success': True, 'best_loss': best_val_loss}
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = {'success': False, 'error': str(e)}

def main():
    with open('train_config.json', 'r') as f:
        config = json.load(f)
    
    print("="*70)
    print("DISTRIBUTED GNN TRAINING")
    print("="*70)
    print(f"Configuration:")
    for key, value in config.items():
        if not key.endswith('_path'):
            print(f"  {key}: {value}")
    print("="*70)
    
    n_gpus = min(config['n_gpus'], torch.cuda.device_count())
    print(f"\nUsing {n_gpus} GPUs for training")
    
    manager = mp.Manager()
    return_dict = manager.dict()
    
    if n_gpus > 1:
        mp.set_start_method('spawn', force=True)
        processes = []
        for gpu_id in range(n_gpus):
            p = mp.Process(target=train_on_gpu, args=(gpu_id, config, return_dict))
            p.start()
            processes.append(p)
            print(f"Started training process on GPU {gpu_id}")
        
        for p in processes:
            p.join()
    else:
        # Single GPU fallback
        train_on_gpu(0, config, return_dict)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    for gpu_id, result in return_dict.items():
        if result['success']:
            print(f"GPU {gpu_id}: Success! Best Val Loss = {result['best_loss']:.4f}")
        else:
            print(f"GPU {gpu_id}: Failed - {result['error']}")

if __name__ == '__main__':
    main()
