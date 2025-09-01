APEX

MTL-GNN-DTA/
│
├── README.md                      # Main documentation
├── LICENSE                        # MIT or your preferred license
├── setup.py                       # Package setup
├── requirements.txt               # Core dependencies
├── environment.yml                # Conda environment file
├── .gitignore                     # Git ignore patterns
├── pyproject.toml                 # Modern Python packaging config
├── MANIFEST.in                    # Include data files in package
│
├── mtl_gnn_dta/                   # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── __version__.py            # Version info
│   │
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py             # Configuration management
│   │   ├── predictor.py         # Main predictor class
│   │   └── trainer.py           # Training orchestration
│   │
│   ├── models/                    # Neural network models
│   │   ├── __init__.py
│   │   ├── dta_model.py         # Main MTL-DTAModel
│   │   ├── protein_encoder.py   # Prot3DGraphModel
│   │   ├── drug_encoder.py      # DrugGVPModel
│   │   ├── gvp_layers.py        # GVP components
│   │   └── losses.py            # MaskedMSELoss, etc.
│   │
│   ├── data/                      # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py           # MTL_DTA dataset class
│   │   ├── loaders.py           # DataLoader utilities
│   │   ├── splits.py            # Cross-validation splits
│   │   └── constants.py         # ATOM_VOCAB, LETTER_TO_NUM, etc.
│   │
│   ├── features/                  # Feature extraction
│   │   ├── __init__.py
│   │   ├── protein_features.py  # PDB processing, ESM embeddings
│   │   ├── drug_features.py     # SDF processing, molecular graphs
│   │   ├── graph_builder.py     # Graph construction utilities
│   │   └── embeddings.py        # ESM2 embedding handler
│   │
│   ├── preprocessing/             # Data preprocessing
│   │   ├── __init__.py
│   │   ├── pdb_processor.py     # PDB cleaning, standardization
│   │   ├── sdf_processor.py     # SDF validation, 3D generation
│   │   ├── chunker.py           # Large dataset chunking
│   │   └── validator.py         # Data validation utilities
│   │
│   ├── training/                  # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loops
│   │   ├── evaluator.py         # Evaluation metrics
│   │   ├── callbacks.py         # Early stopping, checkpointing
│   │   └── optimizers.py        # Optimizer configurations
│   │
│   ├── utils/                     # General utilities
│   │   ├── __init__.py
│   │   ├── io.py                # File I/O utilities
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── logging.py           # Logging configuration
│   │   └── visualization.py     # Plotting utilities
│   │
│   └── cli/                       # Command-line interface
│       ├── __init__.py
│       ├── main.py              # Main CLI entry point
│       ├── train.py             # Training commands
│       ├── predict.py           # Prediction commands
│       └── preprocess.py        # Data preprocessing commands
│
├── data/                          # Data directory (not in package)
│   ├── raw/                      # Raw data files
│   ├── processed/                # Processed data
│   ├── structures/               # PDB/SDF files
│   ├── embeddings/               # Pre-computed embeddings
│   └── configs/                  # Configuration files
│
├── experiments/                   # Experiment configurations
│   ├── configs/                  # YAML/JSON configs
│   ├── logs/                     # Training logs
│   └── checkpoints/              # Model checkpoints
│
├── examples/                      # Example scripts
│   ├── quick_start.py           # Simple usage example
│   ├── train_model.py           # Training example
│   ├── predict_affinity.py     # Prediction example
│   ├── cross_validation.py     # CV example
│   └── notebooks/               # Jupyter notebooks
│       ├── 01_data_preparation.ipynb
│       ├── 02_model_training.ipynb
│       └── 03_analysis.ipynb
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_features.py
│   ├── test_data.py
│   └── test_training.py
│
├── docs/                          # Documentation
│   ├── source/
│   │   ├── conf.py              # Sphinx configuration
│   │   ├── index.rst            # Documentation index
│   │   ├── installation.rst     # Installation guide
│   │   ├── quickstart.rst       # Quick start guide
│   │   ├── api/                 # API documentation
│   │   └── tutorials/           # Detailed tutorials
│   └── Makefile                 # Documentation build
│
└── scripts/                       # Utility scripts
    ├── download_esm.sh          # Download ESM models
    ├── setup_environment.sh     # Environment setup
    ├── prepare_data.py          # Data preparation script
    └── benchmark.py             # Performance benchmarking