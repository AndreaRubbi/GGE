"""
Metrics module for gene expression evaluation.

Provides per-gene and aggregate metrics for comparing distributions:
- Correlation metrics (Pearson, Spearman)
- Distribution distances (Wasserstein, MMD, Energy)
- Multivariate distances
"""

from .base_metric import (
    BaseMetric,
    MetricResult,
    DistributionMetric,
    CorrelationMetric,
)
from .correlation import (
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
)
from .distances import (
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MultivariateWasserstein,
    MultivariateMMD,
)

# All available metrics
ALL_METRICS = [
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MultivariateWasserstein,
    MultivariateMMD,
]

__all__ = [
    # Base classes
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
    # Collections
    "ALL_METRICS",
]