# MTL-GNN-DTA: Multi-Task Learning Graph Neural Network for Drug-Target Affinity Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)

## 🔬 Overview

MTL-GNN-DTA is a state-of-the-art deep learning framework for predicting drug-target binding affinity using multi-task learning and graph neural networks. The model combines:
- **Geometric Vector Perceptrons (GVP)** for processing 3D molecular structures
- **Graph Convolutional Networks (GCN)** for protein and drug representations
- **Multi-task learning** for simultaneous prediction of multiple affinity metrics (pKi, pKd, pIC50, pEC50)
- **ESM-2 protein embeddings** for enhanced protein representation

## ✨ Features

- 🧬 **3D Structure Processing**: Handles both PDB (protein) and SDF (ligand) files
- 🎯 **Multi-Task Learning**: Simultaneous prediction of multiple affinity metrics
- 📊 **Comprehensive Evaluation**: Multiple metrics including R², RMSE, Pearson correlation
- 🔄 **Cross-Validation Support**: Built-in k-fold cross-validation
- 📈 **Uncertainty Quantification**: MC Dropout for prediction confidence
- 🚀 **Optimized Performance**: GPU acceleration, batch processing, data chunking
- 📝 **Extensive Documentation**: Jupyter notebooks and examples included

## 💻 System Requirements

### Hardware Requirements
- **Minimum**: 16GB RAM, NVIDIA GPU with 8GB VRAM
- **Recommended**: 32GB+ RAM, NVIDIA GPU with 16GB+ VRAM (A100, V100, RTX 3090/4090)
- **Storage**: 50GB+ free space for data and models

### Software Requirements
- Ubuntu 20.04+ / macOS 11+ / Windows 10+ (WSL2)
- CUDA 11.7+ (for GPU support)
- Conda/Mamba package manager

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/MTL-GNN-DTA.git
cd MTL-GNN-DTA
```

### Step 2: Create Conda Environment

#### Option A: Using the provided environment.yml (Recommended)
```bash
# Using conda (slower but stable)
conda env create -f environment.yml

# OR using mamba (faster)
mamba env create -f environment.yml

# Activate the environment
conda activate mtl-gnn-dta
```

#### Option B: Manual installation
```bash
# Create new environment
conda create -n mtl-gnn-dta python=3.10 -y
conda activate mtl-gnn-dta

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install RDKit and BioPython
conda install -c conda-forge rdkit biopython

# Install other dependencies
pip install -r requirements.txt
```

### Step 3: Install Additional Tools

```bash
# Install PDBFixer for protein structure cleaning
conda install -c conda-forge pdbfixer openmm

# Install ESM for protein embeddings
pip install fair-esm

# Install the package in development mode
pip install -e .
```

### Step 4: Verify Installation

```bash
# Run verification script
python -c "
import torch
import torch_geometric
import rdkit
import Bio
print('✓ PyTorch:', torch.__version__)
print('✓ PyTorch Geometric:', torch_geometric.__version__)
print('✓ RDKit:', rdkit.__version__)
print('✓ BioPython:', Bio.__version__)
print('✓ CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
"
```

## 🎯 Quick Start

### 1. Download Example Data
```bash
# Download sample data
bash scripts/download_sample_data.sh
```

### 2. Run Quick Example
```python
from mtl_gnn_dta import AffinityPredictor, Config

# Initialize predictor
config = Config()
predictor = AffinityPredictor(config=config)

# Load pre-trained model (download from releases)
predictor.load_model('models/mtl_dta_pretrained.pt')

# Predict affinity
result = predictor.predict_from_files(
    protein_path='data/examples/1a2b.pdb',
    ligand_path='data/examples/ligand.sdf'
)

print(f"Predicted pKi: {result['pKi']:.2f}")
print(f"Predicted pKd: {result['pKd']:.2f}")
```

### 3. Train on Your Data
```bash
# Prepare your data
python scripts/prepare_data.py \
    --protein_dir data/proteins \
    --ligand_dir data/ligands \
    --affinity_file data/affinities.csv \
    --output_dir data/processed

# Train model
python examples/train_model.py \
    --data_dir data/processed \
    --output_dir experiments/my_model \
    --n_epochs 100 \
    --batch_size 32
```

## 📂 Project Structure

```
MTL-GNN-DTA/
├── mtl_gnn_dta/              # Main package
│   ├── core/                 # Core functionality
│   ├── models/               # Neural network models
│   ├── data/                 # Data handling
│   ├── features/             # Feature extraction
│   ├── preprocessing/        # Data preprocessing
│   ├── training/             # Training utilities
│   └── utils/                # General utilities
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data
│   └── structures/           # PDB/SDF files
├── experiments/              # Experiment outputs
│   ├── configs/              # Configuration files
│   ├── logs/                 # Training logs
│   └── checkpoints/          # Model checkpoints
├── examples/                 # Example scripts
│   └── notebooks/            # Jupyter notebooks
├── scripts/                  # Utility scripts
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## 📊 Data Preparation

### Input Data Format

1. **Protein Structures**: PDB format files
2. **Ligand Structures**: SDF format files with 3D coordinates
3. **Affinity Data**: CSV file with columns:
   - `protein_id`: Protein identifier
   - `ligand_id`: Ligand identifier
   - `pKi`, `pKd`, `pIC50`, `pEC50`: Affinity values (optional, can have missing values)

### Data Processing Pipeline

```python
# Run complete data preparation pipeline
python examples/notebooks/01_data_preparation-v2.ipynb
```

This will:
- Clean and standardize protein structures
- Standardize ligand structures
- Calculate molecular properties
- Filter by quality criteria
- Split into train/validation/test sets

## 🏋️ Training

### Basic Training
```bash
python examples/train_model.py --config configs/default.yaml
```

### Advanced Training with Custom Configuration
```python
from mtl_gnn_dta import Config, Trainer, MTL_DTAModel
from mtl_gnn_dta.data import create_dataset

# Load configuration
config = Config('configs/custom.yaml')

# Create model
model = MTL_DTAModel(
    task_names=['pKi', 'pKd', 'pIC50', 'pEC50'],
    prot_emb_dim=1280,
    drug_node_in_dim=66
)

# Train
trainer = Trainer(model, config)
trainer.train(train_loader, val_loader, n_epochs=100)
```

### Cross-Validation
```bash
python examples/cross_validation.py \
    --data_path data/processed/all_data.parquet \
    --n_folds 5 \
    --output_dir experiments/cv
```

## 📈 Evaluation

### Evaluate Trained Model
```python
from mtl_gnn_dta.training import Evaluator

evaluator = Evaluator(task_names=['pKi', 'pKd', 'pIC50', 'pEC50'])
metrics = evaluator.evaluate(model, test_loader, device)

print(f"Overall R²: {metrics['overall_r2']:.3f}")
print(f"Overall RMSE: {metrics['overall_rmse']:.3f}")
```

### Generate Predictions with Uncertainty
```python
results = evaluator.evaluate_with_confidence(
    model, test_loader, device,
    n_forward=10,  # MC Dropout iterations
    dropout=0.1
)
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=mtl_gnn_dta --cov-report=html
```

## 📝 Configuration

Create custom configuration files in YAML format:

```yaml
# configs/custom.yaml
model:
  task_names: ['pKi', 'pKd', 'pIC50', 'pEC50']
  prot_emb_dim: 1280
  drug_node_in_dim: 66
  mlp_dropout: 0.25

training:
  n_epochs: 100
  learning_rate: 0.0001
  batch_size: 32
  patience: 20

data:
  train_path: 'data/processed/train.parquet'
  val_path: 'data/processed/val.parquet'
  test_path: 'data/processed/test.parquet'
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch_size 16
   
   # Use gradient accumulation
   python train.py --accumulation_steps 4
   ```

2. **Missing Dependencies**
   ```bash
   # Update conda environment
   conda env update -f environment.yml
   
   # Reinstall PyTorch Geometric
   pip install torch-geometric --upgrade
   ```

3. **Slow Data Loading**
   ```bash
   # Increase number of workers
   python train.py --num_workers 8
   
   # Use data chunking for large datasets
   python scripts/chunk_data.py --chunk_size 10000
   ```

## 📖 Documentation

- [API Documentation](docs/api/)
- [Tutorial Notebooks](examples/notebooks/)
- [Model Architecture](docs/architecture.md)
- [Data Format Guide](docs/data_format.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you use MTL-GNN-DTA in your research, please cite:

```bibtex
@article{mtl-gnn-dta2024,
  title={Multi-Task Learning Graph Neural Networks for Drug-Target Affinity Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ESM team for protein language models
- PyTorch Geometric team for graph neural network framework
- RDKit community for molecular processing tools

## 📧 Contact

- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/MTL-GNN-DTA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/MTL-GNN-DTA/discussions)