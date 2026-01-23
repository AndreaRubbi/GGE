"""
GenEval: Comprehensive evaluation of generated gene expression data.

A modular, object-oriented framework for computing metrics between real
and generated gene expression datasets stored in AnnData (h5ad) format.

Features:
- Multiple distance and correlation metrics (per-gene and aggregate)
- Condition-based matching (perturbation, cell type, etc.)
- Train/test split support
- Publication-quality visualizations
- Command-line interface

Quick Start:
    >>> from geneval import evaluate
    >>> results = evaluate(
    ...     real_path="real.h5ad",
    ...     generated_path="generated.h5ad", 
    ...     condition_columns=["perturbation"],
    ...     output_dir="output/"
    ... )

CLI Usage:
    $ geneval --real real.h5ad --generated generated.h5ad \\
              --conditions perturbation cell_type --output results/
"""

__version__ = "0.1.1"
__author__ = "GenEval Team"

# Main evaluation interface
from .evaluator import (
    evaluate,
    GeneEvalEvaluator,
    MetricRegistry,
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
)
from .metrics.distances import (
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MultivariateWasserstein,
    MultivariateMMD,
)

# Visualization
from .visualization.visualizer import (
    EvaluationVisualizer,
    visualize,
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
    "GeneEvalEvaluator",
    "MetricRegistry",
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
    # Distance metrics
    "Wasserstein1Distance",
    "Wasserstein2Distance",
    "MMDDistance",
    "EnergyDistance",
    "MultivariateWasserstein",
    "MultivariateMMD",
    # Visualization
    "EvaluationVisualizer",
    "visualize",
    # Testing utilities
    "MockDataGenerator",
    "MockMetricData",
    "create_test_data",
    # Legacy
    "GeneExpressionDataModule",
]