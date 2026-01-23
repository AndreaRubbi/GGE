"""
Base metric classes for gene expression evaluation.

Provides abstract interface for all metrics with per-gene and aggregate computation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np


@dataclass
class MetricResult:
    """
    Container for metric computation results.
    
    Stores both per-gene and aggregate values.
    """
    name: str
    per_gene_values: np.ndarray  # Shape: (n_genes,)
    gene_names: List[str]
    aggregate_value: float
    aggregate_method: str = "mean"  # mean, median, etc.
    condition: Optional[str] = None
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "aggregate_value": float(self.aggregate_value),
            "aggregate_method": self.aggregate_method,
            "per_gene_mean": float(np.nanmean(self.per_gene_values)),
            "per_gene_std": float(np.nanstd(self.per_gene_values)),
            "per_gene_median": float(np.nanmedian(self.per_gene_values)),
            "n_genes": len(self.gene_names),
            "condition": self.condition,
            "split": self.split,
            **self.metadata
        }
    
    def top_genes(self, n: int = 10, ascending: bool = True) -> Dict[str, float]:
        """Get top n genes by metric value."""
        order = np.argsort(self.per_gene_values)
        if not ascending:
            order = order[::-1]
        indices = order[:n]
        return {self.gene_names[i]: float(self.per_gene_values[i]) for i in indices}


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    
    Metrics can be computed per-gene (returning a vector) or as aggregates.
    All metrics should inherit from this class.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        higher_is_better: bool = True,
        requires_distribution: bool = False,
    ):
        """
        Initialize metric.
        
        Parameters
        ----------
        name : str
            Unique identifier for the metric
        description : str
            Human-readable description
        higher_is_better : bool
            Whether higher values indicate better performance
        requires_distribution : bool
            Whether metric needs full distribution (not just means)
        """
        self.name = name
        self.description = description
        self.higher_is_better = higher_is_better
        self.requires_distribution = requires_distribution
    
    @abstractmethod
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute metric for each gene.
        
        Parameters
        ----------
        real : np.ndarray
            Real data matrix, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data matrix, shape (n_samples_gen, n_genes)
            
        Returns
        -------
        np.ndarray
            Metric value per gene, shape (n_genes,)
        """
        pass
    
    def compute_aggregate(
        self,
        per_gene_values: np.ndarray,
        method: str = "mean",
    ) -> float:
        """
        Aggregate per-gene values to single metric.
        
        Parameters
        ----------
        per_gene_values : np.ndarray
            Per-gene metric values
        method : str
            Aggregation method: "mean", "median", "std", "min", "max"
            
        Returns
        -------
        float
            Aggregated metric value
        """
        methods = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "std": np.nanstd,
            "min": np.nanmin,
            "max": np.nanmax,
        }
        if method not in methods:
            raise ValueError(f"Unknown aggregation method: {method}")
        return float(methods[method](per_gene_values))
    
    def compute(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        gene_names: Optional[List[str]] = None,
        aggregate_method: str = "mean",
        condition: Optional[str] = None,
        split: Optional[str] = None,
    ) -> MetricResult:
        """
        Compute full metric result with per-gene and aggregate values.
        
        Parameters
        ----------
        real : np.ndarray
            Real data matrix, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data matrix, shape (n_samples_gen, n_genes)
        gene_names : List[str], optional
            Names of genes (columns)
        aggregate_method : str
            How to aggregate per-gene values
        condition : str, optional
            Condition identifier
        split : str, optional
            Split identifier (train/test)
            
        Returns
        -------
        MetricResult
            Complete metric result
        """
        n_genes = real.shape[1] if real.ndim > 1 else 1
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        per_gene = self.compute_per_gene(real, generated)
        aggregate = self.compute_aggregate(per_gene, method=aggregate_method)
        
        return MetricResult(
            name=self.name,
            per_gene_values=per_gene,
            gene_names=gene_names,
            aggregate_value=aggregate,
            aggregate_method=aggregate_method,
            condition=condition,
            split=split,
            metadata={
                "higher_is_better": self.higher_is_better,
                "description": self.description,
            }
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class DistributionMetric(BaseMetric):
    """
    Base class for distribution-based metrics (Wasserstein, MMD, Energy).
    
    These metrics require the full sample distributions, not just means.
    """
    
    def __init__(self, name: str, description: str = "", higher_is_better: bool = False):
        super().__init__(
            name=name,
            description=description,
            higher_is_better=higher_is_better,
            requires_distribution=True,
        )


class CorrelationMetric(BaseMetric):
    """
    Base class for correlation-based metrics (Pearson, Spearman).
    
    These compare mean profiles between real and generated data.
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(
            name=name,
            description=description,
            higher_is_better=True,
            requires_distribution=False,
        )
