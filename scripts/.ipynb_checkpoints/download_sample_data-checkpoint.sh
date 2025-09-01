
# ===================================================
# File: scripts/download_sample_data.sh
# Download sample data for testing
# ===================================================

#!/bin/bash

echo "Downloading sample data for MTL-GNN-DTA..."

# Create directories
mkdir -p data/examples
mkdir -p models

# Download sample PDB files (using PDB database)
echo "Downloading sample protein structures..."
wget -q -O data/examples/1a2b.pdb https://files.rcsb.org/download/1A2B.pdb
wget -q -O data/examples/3htb.pdb https://files.rcsb.org/download/3HTB.pdb

echo "✓ Sample data downloaded"

# Create sample ligand SDF file
cat > data/examples/ligand.sdf << 'EOF'
ligand_001
  ChemDraw08142415112D

 24 25  0  0  0  0  0  0  0  0999 V2000
    1.4289    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7145   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7145    1.2375    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4289    0.8250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
M  END
$$$$
EOF

echo "✓ Sample ligand created"

# Create sample affinity data
cat > data/examples/affinities.csv << EOF
protein_id,ligand_id,pKi,pKd,pIC50,pEC50
1a2b,ligand_001,7.2,7.5,6.8,
3htb,ligand_001,8.1,,7.9,7.2
EOF

echo "✓ Sample affinity data created"

echo "Sample data setup complete!"
echo "Files created in data/examples/"

