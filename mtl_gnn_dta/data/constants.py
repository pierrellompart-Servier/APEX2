"""Constants for molecular and protein data processing"""

# Amino acid mapping
LETTER_TO_NUM = {
    'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
    'N': 2, 'Y': 18, 'M': 12, 'X': 20
}

NUM_TO_LETTER = {v: k for k, v in LETTER_TO_NUM.items()}

# Atom vocabulary for drug molecules
ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'unk'
]

# Standard amino acids
STANDARD_AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]

# Hydrophobicity scale (Kyte-Doolittle)
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Amino acid charges
AMINO_ACID_CHARGES = {
    'R': 1, 'K': 1, 'D': -1, 'E': -1,
    'H': 0.5  # Partially charged at physiological pH
}

# Amino acid sizes (number of heavy atoms)
AMINO_ACID_SIZES = {
    'G': 1, 'A': 2, 'S': 3, 'C': 3, 'P': 4, 'T': 4, 'V': 5,
    'N': 5, 'D': 5, 'I': 6, 'L': 6, 'E': 6, 'Q': 6, 'K': 6,
    'M': 6, 'H': 7, 'F': 9, 'R': 8, 'Y': 10, 'W': 12
}
