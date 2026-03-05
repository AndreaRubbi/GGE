"""
Base metric classes for gene expression evaluation.

Provides abstract interface for all metrics with per-gene and aggregate computation.
All metrics support computation in different spaces: raw, pca, deg.

Reference: GGE Paper Section 3.3 "The Space Question"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable, Literal, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..spaces import SpaceConfig, SpaceTransformer

SpaceType = Literal["raw", "pca", "deg"]


@dataclass
class MetricResult:
    """
    Container for metric computation results.
    
    Stores both per-gene and aggregate values, along with space information.
    """
    name: str
    per_gene_values: np.ndarray  # Shape: (n_features,) - genes or PCs
    gene_names: List[str]  # Feature names (gene names or PC names)
    aggregate_value: float
    aggregate_method: str = "mean"  # mean, median, etc.
    space: SpaceType = "raw"  # Computation space
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
            "space": self.space,
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
    All metrics support computation in different spaces: raw, pca, deg.
    
    Parameters
    ----------
    space : str
        Computation space: "raw", "pca", or "deg" (default: "raw")
    n_components : int
        Number of PCA components (for space="pca", default: 50)
    deg_lfc : float
        log2 fold change threshold (for space="deg", default: 1.0)
    deg_pval : float
        p-value threshold (for space="deg", default: 0.05)
    n_top_degs : int, optional
        Select top N DEGs by absolute fold change (for space="deg")
    
    Examples
    --------
    >>> # Raw space (default)
    >>> metric = PearsonCorrelation()
    
    >>> # PCA space
    >>> metric = MMD(space="pca", n_components=50)
    
    >>> # DEG space with custom thresholds
    >>> metric = PearsonCorrelation(space="deg", deg_lfc=0.25, deg_pval=0.1)
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        higher_is_better: bool = True,
        requires_distribution: bool = False,
        # Space parameters
        space: SpaceType = "raw",
        n_components: int = 50,
        deg_lfc: float = 1.0,
        deg_pval: float = 0.05,
        n_top_degs: Optional[int] = None,
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
        space : str
            Computation space: "raw", "pca", or "deg"
        n_components : int
            PCA components (for space="pca")
        deg_lfc : float
            log2 fold change threshold (for space="deg")
        deg_pval : float
            p-value threshold (for space="deg")
        n_top_degs : int, optional
            Select top N DEGs (for space="deg")
        """
        self._base_name = name
        self.description = description
        self.higher_is_better = higher_is_better
        self.requires_distribution = requires_distribution
        
        # Space configuration
        self.space = space
        self.n_components = n_components
        self.deg_lfc = deg_lfc
        self.deg_pval = deg_pval
        self.n_top_degs = n_top_degs
        
        # Transformer (created on first use)
        self._transformer: Optional["SpaceTransformer"] = None
    
    @property
    def name(self) -> str:
        """Get metric name with space suffix."""
        if self.space == "raw":
            return self._base_name
        elif self.space == "pca":
            return f"{self._base_name}_pca{self.n_components}"
        else:  # deg
            return f"{self._base_name}_deg"
    
    @property
    def space_config(self) -> "SpaceConfig":
        """Get space configuration."""
        from ..spaces import SpaceConfig
        return SpaceConfig(
            space=self.space,
            n_components=self.n_components,
            deg_lfc=self.deg_lfc,
            deg_pval=self.deg_pval,
            n_top_degs=self.n_top_degs,
        )
    
    def _get_transformer(
        self, 
        control_data: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
    ) -> "SpaceTransformer":
        """Get or create space transformer."""
        from ..spaces import SpaceTransformer
        
        if self._transformer is None:
            self._transformer = SpaceTransformer(
                config=self.space_config,
                control_data=control_data,
                gene_names=gene_names,
            )
        return self._transformer
    
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
        control_data: Optional[np.ndarray] = None,
    ) -> MetricResult:
        """
        Compute full metric result with per-gene and aggregate values.
        
        Automatically applies space transformation based on metric configuration.
        
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
        control_data : np.ndarray, optional
            Control data for DEG space (required if space="deg")
            
        Returns
        -------
        MetricResult
            Complete metric result
        """
        n_genes = real.shape[1] if real.ndim > 1 else 1
        if gene_names is None:
            gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Apply space transformation if not raw
        if self.space != "raw":
            transformer = self._get_transformer(
                control_data=control_data,
                gene_names=gene_names,
            )
            real_t, gen_t, feature_names = transformer.transform(
                real, generated, gene_names
            )
        else:
            real_t, gen_t, feature_names = real, generated, gene_names
        
        per_gene = self.compute_per_gene(real_t, gen_t)
        aggregate = self.compute_aggregate(per_gene, method=aggregate_method)
        
        return MetricResult(
            name=self.name,
            per_gene_values=per_gene,
            gene_names=feature_names,
            aggregate_value=aggregate,
            aggregate_method=aggregate_method,
            space=self.space,
            condition=condition,
            split=split,
            metadata={
                "higher_is_better": self.higher_is_better,
                "description": self.description,
                "base_name": self._base_name,
            }
        )
    
    def __repr__(self) -> str:
        if self.space == "raw":
            return f"{self.__class__.__name__}(name='{self._base_name}')"
        elif self.space == "pca":
            return f"{self.__class__.__name__}(name='{self._base_name}', space='pca', n_components={self.n_components})"
        else:
            return f"{self.__class__.__name__}(name='{self._base_name}', space='deg', deg_lfc={self.deg_lfc}, deg_pval={self.deg_pval})"


class DistributionMetric(BaseMetric):
    """
    Base class for distribution-based metrics (Wasserstein, MMD, Energy).
    
    These metrics require the full sample distributions, not just means.
    Supports computation in raw, pca, or deg space.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = "", 
        higher_is_better: bool = False,
        space: SpaceType = "raw",
        n_components: int = 50,
        deg_lfc: float = 1.0,
        deg_pval: float = 0.05,
        n_top_degs: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            higher_is_better=higher_is_better,
            requires_distribution=True,
            space=space,
            n_components=n_components,
            deg_lfc=deg_lfc,
            deg_pval=deg_pval,
            n_top_degs=n_top_degs,
        )


class CorrelationMetric(BaseMetric):
    """
    Base class for correlation-based metrics (Pearson, Spearman).
    
    These compare mean profiles between real and generated data.
    Supports computation in raw, pca, or deg space.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = "",
        space: SpaceType = "raw",
        n_components: int = 50,
        deg_lfc: float = 1.0,
        deg_pval: float = 0.05,
        n_top_degs: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            higher_is_better=True,
            requires_distribution=False,
            space=space,
            n_components=n_components,
            deg_lfc=deg_lfc,
            deg_pval=deg_pval,
            n_top_degs=n_top_degs,
        )
