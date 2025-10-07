conda activate bioml

# Set the conda library path to take precedence
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify it's set
echo $LD_LIBRARY_PATH

# Now run the script
python run.py -i ./input/examples.csv -o predictions.csv

# Basic usage (default: fast=False, gpu=0)
python run.py -i examples.csv -o predictions.csv

# With fast mode enabled
python run.py -i examples.csv -o predictions.csv --fast

# Using GPU 1
python run.py -i examples.csv -o predictions.csv --gpu 1

# Fast mode on GPU 2
python run.py -i examples.csv -o predictions.csv --fast --gpu 2


target,ligand
./test/245158.pdb,./test/245158.sdf
./test/279794.pdb,./test/279794.sdf