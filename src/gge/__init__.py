"""
GGE: Generated Genetic Expression Evaluator

Comprehensive evaluation of generated gene expression data.

A modular, object-oriented framework for computing metrics between real
and generated gene expression datasets stored in AnnData (h5ad) format.

Features:
- Multiple distance and correlation metrics (per-gene and aggregate)
- Condition-based matching (perturbation, cell type, etc.)
- Train/test split support
- Publication-quality visualizations
- Command-line interface

Installation:
    pip install gge-eval

Quick Start:
    >>> from gge import evaluate
    >>> results = evaluate(
    ...     real_path="real.h5ad",
    ...     generated_path="generated.h5ad", 
    ...     condition_columns=["perturbation"],
    ...     output_dir="output/"
    ... )

CLI Usage:
    $ gge --real real.h5ad --generated generated.h5ad \\
          --conditions perturbation cell_type --output results/
"""

__version__ = "0.1.7"
__author__ = "GGE Team"

# Main evaluation interface
from .evaluator import (
    evaluate,
    evaluate_lazy,
    GeneEvalEvaluator,
    MetricRegistry,
)

# Space transformation
from .spaces import (
    SpaceType,
    SpaceConfig,
    SpaceTransformer,
    get_space_config,
)

# Data loading
from .data.loader import (
    GeneExpressionDataLoader,
    load_data,
)

# Results
from .results import (
    EvaluationResult,
    SplitResult,
    ConditionResult,
)

# Metrics
from .metrics.base_metric import (
    BaseMetric,
    MetricResult,
    DistributionMetric,
    CorrelationMetric,
)
from .metrics.correlation import (
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
    RSquared,
    PerturbationEffectCorrelation,
    compute_perturbation_effect_correlation,
)
from .metrics.distances import (
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MSEDistance,
    MultivariateWasserstein,
    MultivariateMMD,
)

# Visualization
from .visualization.visualizer import (
    EvaluationVisualizer,
    visualize,
)

# DEG utilities
from .utils.deg import (
    identify_degs,
    evaluate_deg_space,
    DEGSpaceEvaluator,
    filter_to_degs,
    get_deg_mask,
    compute_perturbation_effects,
)

# PC-space utilities
from .utils.pca import (
    compute_pca,
    get_pc_coordinates,
    project_to_pc_space,
    PCSpaceEvaluator,
    evaluate_pc_space,
    compute_pc_variance_explained,
)

# Legacy support
from .data.gene_expression_datamodule import GeneExpressionDataModule

# Testing utilities (for users to generate test data)
from .testing import (
    MockDataGenerator,
    MockMetricData,
    create_test_data,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "evaluate",
    "evaluate_lazy",
    "GeneEvalEvaluator",
    "MetricRegistry",
    # Space transformation
    "SpaceType",
    "SpaceConfig",
    "SpaceTransformer",
    "get_space_config",
    # Data loading
    "GeneExpressionDataLoader",
    "load_data",
    # Results
    "EvaluationResult",
    "SplitResult", 
    "ConditionResult",
    # Base metrics
    "BaseMetric",
    "MetricResult",
    "DistributionMetric",
    "CorrelationMetric",
    # Correlation metrics
    "PearsonCorrelation",
    "SpearmanCorrelation",
    "MeanPearsonCorrelation",
    "MeanSpearmanCorrelation",
    "RSquared",
    "PerturbationEffectCorrelation",
    "compute_perturbation_effect_correlation",
    # Distance metrics
    "Wasserstein1Distance",
    "Wasserstein2Distance",
    "MMDDistance",
    "EnergyDistance",
    "MSEDistance",
    "MultivariateWasserstein",
    "MultivariateMMD",
    # Visualization
    "EvaluationVisualizer",
    "visualize",
    # DEG utilities
    "identify_degs",
    "evaluate_deg_space",
    "DEGSpaceEvaluator",
    "filter_to_degs",
    "get_deg_mask",
    "compute_perturbation_effects",
    # PC-space utilities
    "compute_pca",
    "get_pc_coordinates",
    "project_to_pc_space",
    "PCSpaceEvaluator",
    "evaluate_pc_space",
    "compute_pc_variance_explained",
    # Testing utilities
    "MockDataGenerator",
    "MockMetricData",
    "create_test_data",
    # Legacy
    "GeneExpressionDataModule",
]