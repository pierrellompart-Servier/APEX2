# gnn_dta_mtl/__init__.py
"""
GNN-DTA-MTL: Graph Neural Networks for Drug-Target Affinity with Multi-Task Learning
"""

__version__ = "1.0.0"

from .models.dta_model import DTAModel, MTL_DTAModel
from .models.drug_model import DrugGVPModel
from .models.protein_model import Prot3DGraphModel
from .models.losses import MaskedMSELoss

from .datasets.dta_dataset import DTA, MTL_DTA
from .datasets.data_utils import (
    build_dataset,
    build_mtl_dataset,
    build_mtl_dataset_optimized,
    create_data_splits,
    prepare_mtl_experiment
)

from .training.trainer import MTLTrainer
from .training.cross_validation import CrossValidator
from .training.early_stopping import EarlyStopping

from .evaluation.metrics import (
    calculate_metrics,
    evaluate_model,
    concordance_index,
    bootstrap_metrics
)
from .evaluation.visualization import (
    plot_results,
    plot_predictions,
    plot_residuals,
    plot_training_history,
    create_summary_report
)

from .features.drug_graph import featurize_drug
from .features.protein_graph import featurize_protein_graph
from .features.esm_embeddings import ESMEmbedder, get_esm_embedding
from .features.structure_processing import (
    StructureProcessor,
    StructureChunkLoader
)

from .data.standardization import StructureStandardizer
from .data.molecular_properties import (
    compute_molecular_properties,
    add_molecular_properties_parallel,
    compute_ligand_efficiency,
    compute_mean_ligand_efficiency,  # This was missing!
    filter_by_properties
)

from .utils.constants import LETTER_TO_NUM, NUM_TO_LETTER, ATOM_VOCAB
from .utils.io_utils import (
    save_model, 
    load_model, 
    save_results, 
    load_results,
    create_output_dir  # This was also missing!
)
from .utils.logger import setup_logger, ExperimentLogger
from .utils.model_utils import ModelCheckpointer, count_parameters

__all__ = [
    # Models
    'DTAModel',
    'MTL_DTAModel', 
    'DrugGVPModel',
    'Prot3DGraphModel',
    'MaskedMSELoss',
    
    # Datasets
    'DTA',
    'MTL_DTA',
    'build_dataset',
    'build_mtl_dataset',
    'build_mtl_dataset_optimized',
    'create_data_splits',
    'prepare_mtl_experiment',
    
    # Training
    'MTLTrainer',
    'CrossValidator',
    'EarlyStopping',
    
    # Evaluation
    'calculate_metrics',
    'evaluate_model',
    'plot_results',
    'plot_predictions',
    'create_summary_report',
    'concordance_index',
    'bootstrap_metrics',
    'plot_residuals',
    'plot_training_history',
    
    # Features
    'featurize_drug',
    'featurize_protein_graph',
    'ESMEmbedder',
    'get_esm_embedding',
    'StructureProcessor',
    'StructureChunkLoader',
    
    # Data processing
    'StructureStandardizer',
    'compute_molecular_properties',
    'add_molecular_properties_parallel',
    'compute_ligand_efficiency',
    'compute_mean_ligand_efficiency',  # Added
    'filter_by_properties',
    
    # Utils
    'save_model',
    'load_model',
    'save_results',
    'load_results',
    'create_output_dir',  # Added
    'setup_logger',
    'ExperimentLogger',
    'ModelCheckpointer',
    'count_parameters',
    'LETTER_TO_NUM',
    'NUM_TO_LETTER',
    'ATOM_VOCAB'
]