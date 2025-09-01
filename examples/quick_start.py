#!/usr/bin/env python3
"""
Quick Start Example for MTL-GNN-DTA
This script demonstrates basic usage of the package
"""

import os
import sys
sys.path.append('../')

from mtl_gnn_dta import (
    Config,
    AffinityPredictor,
    MTL_DTAModel,
    DrugFeaturizer,
    ProteinFeaturizer
)
import torch
import pandas as pd
import numpy as np


def main():
    """Main function demonstrating MTL-GNN-DTA usage"""
    
    print("MTL-GNN-DTA Quick Start Example")
    print("="*60)
    
    # 1. Initialize configuration
    print("\n1. Initializing configuration...")
    config = Config()
    print(f"   Tasks: {config.model.task_names}")
    print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 2. Initialize featurizers
    print("\n2. Initializing featurizers...")
    drug_featurizer = DrugFeaturizer()
    protein_featurizer = ProteinFeaturizer()
    print("   ✓ Drug and protein featurizers initialized")
    
    # 3. Example: Featurize a drug from SMILES
    print("\n3. Example drug featurization...")
    smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O"  # Ibuprofen
    drug_graph = drug_featurizer.featurize_from_smiles(smiles)
    if drug_graph:
        print(f"   SMILES: {smiles}")
        print(f"   Nodes: {drug_graph.x.shape[0]}")
        print(f"   Edges: {drug_graph.edge_index.shape[1]}")
    
    # 4. Create and initialize model
    print("\n4. Creating MTL-DTA model...")
    model = MTL_DTAModel(
        task_names=['pKi', 'pEC50', 'pKd', 'pIC50'],
        prot_emb_dim=1280,
        prot_gcn_dims=[128, 256, 256],
        prot_fc_dims=[1024, 128],
        drug_node_in_dim=[66, 1],
        drug_node_h_dims=[128, 64],
        drug_fc_dims=[1024, 128],
        mlp_dims=[1024, 512],
        mlp_dropout=0.25
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # 5. Example prediction (with mock data)
    print("\n5. Example prediction with mock data...")
    
    # Create mock batch data
    from torch_geometric.data import Data, Batch
    
    # Mock protein batch (batch size = 2)
    protein_batch = Batch.from_data_list([
        Data(x=torch.randn(100, 1280), edge_index=torch.randint(0, 100, (2, 200))),
        Data(x=torch.randn(120, 1280), edge_index=torch.randint(0, 120, (2, 240)))
    ])
    
    # Mock drug batch (batch size = 2)
    drug_batch = Batch.from_data_list([
        Data(x=torch.randn(20, 66), edge_index=torch.randint(0, 20, (2, 40)), 
             edge_attr=torch.randn(40, 6)),
        Data(x=torch.randn(25, 66), edge_index=torch.randint(0, 25, (2, 50)),
             edge_attr=torch.randn(50, 6))
    ])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(drug_batch, protein_batch)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions for sample 1:")
    for i, task in enumerate(['pKi', 'pEC50', 'pKd', 'pIC50']):
        print(f"     {task}: {predictions[0, i].item():.3f}")
    
    # 6. Training example (simplified)
    print("\n6. Training example...")
    
    # Create mock training data
    train_data = []
    for _ in range(10):
        protein = Data(x=torch.randn(100, 1280), 
                      edge_index=torch.randint(0, 100, (2, 200)))
        drug = Data(x=torch.randn(20, 66), 
                   edge_index=torch.randint(0, 20, (2, 40)),
                   edge_attr=torch.randn(40, 6))
        y = torch.randn(4)  # 4 tasks
        train_data.append({'protein': protein, 'drug': drug, 'y': y})
    
    # Create data loader
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        proteins = Batch.from_data_list([item['protein'] for item in batch])
        drugs = Batch.from_data_list([item['drug'] for item in batch])
        ys = torch.stack([item['y'] for item in batch])
        return {'protein': proteins, 'drug': drugs, 'y': ys}
    
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, 
                             collate_fn=collate_fn)
    
    # Training loop (1 epoch for demonstration)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch['drug'], batch['protein'])
        loss = criterion(predictions, batch['y'])
        loss.backward()
        optimizer.step()
        print(f"   Batch loss: {loss.item():.4f}")
        break  # Just one batch for demo
    
    # 7. Save and load model
    print("\n7. Saving and loading model...")
    
    # Save
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'task_cols': ['pKi', 'pEC50', 'pKd', 'pIC50']
    }
    
    checkpoint_path = 'demo_model.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"   Model saved to {checkpoint_path}")
    
    # Load
    loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    print(f"   Model loaded successfully")
    
    # Clean up
    os.remove(checkpoint_path)
    
    print("\n" + "="*60)
    print("Quick start example completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your protein PDB and ligand SDF files")
    print("2. Load and process your affinity data")
    print("3. Train the model on your dataset")
    print("4. Use the trained model for predictions")
    print("\nSee the example notebooks for detailed workflows.")


if __name__ == "__main__":
    main()
