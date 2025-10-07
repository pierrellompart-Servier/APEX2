# APEX2: Affinity Prediction with Embedded X-structures

**Version 2.0** - Trained on 400,000 protein-ligand complexes

APEX2 is a deep learning model for predicting protein-ligand binding affinities from 3D structures. It provides multi-task predictions across six different affinity measurement types using graph neural networks and ESM2 protein embeddings.

---

## What APEX2 Does

APEX2 predicts binding affinities for protein-ligand complexes without requiring molecular dynamics simulations or docking calculations. Given a protein structure (PDB) and ligand structure (SDF), the model predicts:

- **pKi** - Inhibition constant
- **pEC50** - Half-maximal effective concentration  
- **pKd** - Dissociation constant
- **pIC50** - Half-maximal inhibitory concentration
- **pKd (Wang, FEP)** - Free energy perturbation-derived Kd
- **Potency** - Experimental potency (future feature)

Higher values indicate stronger binding affinity.

---

## Installation

### Setup

```bash
# Clone repository
git clone https://github.com/pierrellompart-Servier/APEX2.git
cd APEX2

# Activate your conda environment
conda activate bioml

# CRITICAL: Fix library path (required for scipy/sklearn compatibility)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify it's set
echo $LD_LIBRARY_PATH

# Make it permanent (optional but recommended)
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Quick Start

### Command Line Usage

```bash
# Basic prediction (default: GPU 0, standard mode)
python run.py -i ./input/examples.csv -o predictions.csv

# Using different GPU
python run.py -i ./input/examples.csv -o predictions.csv --gpu 1

# Fast mode for large datasets (>1000 complexes)
python run.py -i ./input/examples.csv -o predictions.csv --fast

# Combined options
python run.py -i ./input/examples.csv -o predictions.csv --fast --gpu 2
```

### Command Options

```
-i, --input    Input CSV with 'target' and 'ligand' columns (required)
-o, --output   Output CSV for predictions (required)
--fast         Enable fast mode for >1000 complexes (optional)
--gpu          GPU device ID: 0, 1, 2, etc. (default: 0)
```

---

## Input Format

Your input CSV must have exactly two columns: `target` and `ligand`

**Example: input/examples.csv**
```csv
target,ligand
./test/245158.pdb,./test/245158.sdf
./test/279794.pdb,./test/279794.sdf
./test/203418.pdb,./test/203418.sdf
./test/48008.pdb,./test/48008.sdf
./test/434516.pdb,./test/434516.sdf
./test/3705.pdb,./test/3705.sdf
./test/323023.pdb,./test/323023.sdf
./test/251630.pdb,./test/251630.sdf
./test/472746.pdb,./test/472746.sdf
./test/24704.pdb,./test/24704.sdf
```

### File Requirements (I will soon integrate my standardization protocol to the predict function)

**Protein files (PDB):**
- Standard PDB format with 3D coordinates
- Clean structures (remove waters, ions, other ligands)
- Complete residues recommended

**Ligand files (SDF):**
- Standard SDF format with 3D coordinates
- Single conformer per file
- Explicit hydrogens added
- Valid chemical structure

---

## Output Format

The output CSV contains your input paths plus predictions for all tasks:

**Example: predictions.csv**
```csv
protein_path,ligand_path,pKi,pEC50,pKd,pIC50,pKd (Wang, FEP),potency
./test/245158.pdb,./test/245158.sdf,7.637,6.892,6.342,6.879,7.292,0.054
./test/279794.pdb,./test/279794.sdf,8.614,7.790,8.436,7.967,8.197,0.069
./test/203418.pdb,./test/203418.sdf,6.523,5.891,5.445,5.923,6.134,0.041
```

**Note:** The potency column is experimental and will be fully integrated with SAIR in future releases.

---

## Repository Structure

```
APEX2/
├── gnn_dta_mtl/          # Main package
│   ├── data/             # Data processing utilities
│   ├── datasets/         # Dataset classes
│   ├── evaluation/       # Metrics and visualization
│   ├── features/         # Featurization (graphs, embeddings)
│   ├── models/           # Model architectures
│   ├── predict/          # Prediction functions
│   ├── training/         # Training utilities
│   └── utils/            # Helper functions
├── models/               # Model checkpoints
│   └── best_model.pt     # Trained APEX2 model (542 MB)
├── test/                 # Example protein-ligand complexes
├── input/                # Example input files
│   └── examples.csv
├── run.py                # Command-line interface
├── example.ipynb         # Jupyter notebook tutorial
└── README.md             # This file
```
