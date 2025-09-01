#!/bin/bash
# File: scripts/setup_environment.sh
# Complete environment setup script for MTL-GNN-DTA

set -e  # Exit on error

echo "=========================================="
echo "MTL-GNN-DTA Environment Setup"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if conda/mamba is installed
check_conda() {
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        print_status "Found mamba"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        print_status "Found conda"
    else
        print_error "Neither conda nor mamba found. Please install Anaconda/Miniconda/Mambaforge first."
        echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "CUDA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        CUDA_AVAILABLE=true
    else
        print_warning "No CUDA GPU detected. Will install CPU-only version."
        CUDA_AVAILABLE=false
    fi
}

# Create conda environment
create_environment() {
    echo ""
    echo "Creating conda environment..."
    
    if $CONDA_CMD env list | grep -q "mtl-gnn-dta"; then
        print_warning "Environment 'mtl-gnn-dta' already exists."
        read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $CONDA_CMD env remove -n mtl-gnn-dta -y
        else
            echo "Exiting..."
            exit 0
        fi
    fi
    
    # Create environment from yml file
    if [ -f "environment.yml" ]; then
        print_status "Creating environment from environment.yml..."
        $CONDA_CMD env create -f environment.yml
    else
        print_warning "environment.yml not found. Creating environment manually..."
        
        # Create base environment
        $CONDA_CMD create -n mtl-gnn-dta python=3.10 -y
        
        # Activate environment
        eval "$($CONDA_CMD shell.bash hook)"
        conda activate mtl-gnn-dta
        
        # Install packages
        install_packages_manual
    fi
}

# Manual package installation
install_packages_manual() {
    echo ""
    echo "Installing packages manually..."
    
    # Install PyTorch
    if [ "$CUDA_AVAILABLE" = true ]; then
        print_status "Installing PyTorch with CUDA support..."
        $CONDA_CMD install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        print_status "Installing PyTorch CPU version..."
        $CONDA_CMD install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    # Install PyTorch Geometric
    print_status "Installing PyTorch Geometric..."
    $CONDA_CMD install pyg -c pyg -y
    
    # Install scientific packages
    print_status "Installing scientific computing packages..."
    $CONDA_CMD install -c conda-forge \
        numpy scipy pandas scikit-learn \
        matplotlib seaborn plotly \
        jupyter ipykernel jupyterlab -y
    
    # Install molecular processing tools
    print_status "Installing molecular processing tools..."
    $CONDA_CMD install -c conda-forge rdkit biopython pdbfixer openmm -y
    
    # Install remaining packages via pip
    print_status "Installing additional packages via pip..."
    pip install fair-esm transformers wandb tensorboard optuna \
        tqdm pyyaml click rich python-dotenv \
        pytest pytest-cov black flake8 mypy
}

# Download ESM models
download_esm_models() {
    echo ""
    echo "Downloading ESM protein models..."
    
    python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Downloading ESM-2 model (650M parameters)...')
model_name = 'facebook/esm2_t33_650M_UR50D'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print('✓ ESM-2 model downloaded successfully')
"
}

# Download sample data
download_sample_data() {
    echo ""
    echo "Setting up sample data..."
    
    # Create data directories
    mkdir -p data/{raw,processed,structures,embeddings}
    mkdir -p data/structures/{proteins,ligands}
    mkdir -p experiments/{configs,logs,checkpoints}
    mkdir -p models
    
    print_status "Directory structure created"
    
    # Download sample PDB and SDF files (if you have URLs)
    # wget -O data/structures/proteins/sample.pdb https://example.com/sample.pdb
    # wget -O data/structures/ligands/sample.sdf https://example.com/sample.sdf
    
    # Create sample configuration
    cat > experiments/configs/default.yaml << EOF
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
  batch_size: 32
  num_workers: 4
  pin_memory: true
EOF
    
    print_status "Sample configuration created"
}

# Verify installation
verify_installation() {
    echo ""
    echo "Verifying installation..."
    
    python << EOF
import sys
import warnings
warnings.filterwarnings('ignore')

def check_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name}: {e}")
        return False

print("\n=== Checking Core Dependencies ===")
success = True
success &= check_import('torch', 'PyTorch')
success &= check_import('torch_geometric', 'PyTorch Geometric')
success &= check_import('rdkit', 'RDKit')
success &= check_import('Bio', 'BioPython')

print("\n=== Checking Additional Dependencies ===")
check_import('esm', 'ESM')
check_import('transformers', 'Transformers')
check_import('pandas', 'Pandas')
check_import('numpy', 'NumPy')
check_import('sklearn', 'Scikit-learn')

print("\n=== Checking GPU Support ===")
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
else:
    print("! CUDA not available - CPU only mode")

print("\n=== System Information ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

if success:
    print("\n✓ All core dependencies installed successfully!")
else:
    print("\n✗ Some dependencies failed to install. Please check the errors above.")
    sys.exit(1)
EOF
}

# Install package in development mode
install_package() {
    echo ""
    echo "Installing MTL-GNN-DTA package..."
    
    if [ -f "setup.py" ]; then
        pip install -e .
        print_status "Package installed in development mode"
    else
        print_warning "setup.py not found. Skipping package installation."
    fi
}

# Main installation flow
main() {
    echo "Starting installation process..."
    echo ""
    
    # Check prerequisites
    check_conda
    check_cuda
    
    # Create and setup environment
    create_environment
    
    # Activate environment
    eval "$($CONDA_CMD shell.bash hook)"
    conda activate mtl-gnn-dta
    
    # Download models and data
    download_esm_models
    download_sample_data
    
    # Install package
    install_package
    
    # Verify installation
    verify_installation
    
    echo ""
    echo "=========================================="
    print_status "Installation completed successfully!"
    echo "=========================================="
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate mtl-gnn-dta"
    echo ""
    echo "To start Jupyter Lab, run:"
    echo "  jupyter lab"
    echo ""
    echo "To run the quick start example:"
    echo "  python examples/quick_start.py"
    echo ""
}

# Run main function
main
