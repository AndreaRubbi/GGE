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
    RSquared,
    PerturbationEffectCorrelation,
    compute_perturbation_effect_correlation,
)
from .distances import (
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MultivariateWasserstein,
    MultivariateMMD,
    MSEDistance,
)

# All available metrics
ALL_METRICS = [
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
    RSquared,
    PerturbationEffectCorrelation,
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MSEDistance,
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
    "RSquared",
    "PerturbationEffectCorrelation",
    "compute_perturbation_effect_correlation",
    # Distance metrics
    "Wasserstein1Distance",
    "Wasserstein2Distance",
    "MMDDistance",
    "EnergyDistance",
    "MSEDistance",
    # Multivariate metrics
    "MultivariateWasserstein",
    "MultivariateMMD",
    # Registry
    "ALL_METRICS",
]