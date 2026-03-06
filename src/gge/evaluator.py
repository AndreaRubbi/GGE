"""
Comprehensive evaluator for gene expression data.

Computes all metrics between real and generated data, organized by conditions and splits.
All metrics support computation in different spaces: raw, pca, deg.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Type, Any, Literal, TYPE_CHECKING
from pathlib import Path
import numpy as np
import warnings
from dataclasses import dataclass

if TYPE_CHECKING:
    import anndata as ad

from .data.loader import GeneExpressionDataLoader, load_data
from .metrics.base_metric import BaseMetric, MetricResult
from .metrics.correlation import (
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
    RSquared,
)
from .metrics.distances import (
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MultivariateWasserstein,
    MultivariateMMD,
    MSEDistance,
)
from .results import EvaluationResult, SplitResult, ConditionResult


# Default metrics to compute (paper metrics only - no mean_ variants or multivariate)
DEFAULT_METRICS = [
    PearsonCorrelation,
    SpearmanCorrelation,
    RSquared,
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
]


class GeneEvalEvaluator:
    """
    Main evaluator class for gene expression data.
    
    Computes comprehensive metrics between real and generated datasets,
    supporting multiple conditions, splits, and metric types.
    
    Each metric can be configured to compute in a different space:
    - raw: Full gene space (default)
    - pca: PCA-reduced space
    - deg: Differentially expressed genes only
    
    Parameters
    ----------
    data_loader : GeneExpressionDataLoader
        Loaded and aligned data loader
    metrics : List[BaseMetric or Type[BaseMetric]], optional
        Metrics to compute. If None, uses default set.
        Each metric can have its own space configuration.
    aggregate_method : str
        How to aggregate per-gene values (mean, median, etc.)
    include_multivariate : bool
        Whether to include multivariate (whole-space) metrics
    control_key : str, optional
        Value identifying control samples for DEG space metrics.
        Required if any metric uses space="deg".
    control_column : str, optional
        Column name containing control information (default: first condition column)
    verbose : bool
        Whether to print progress
        
    Examples
    --------
    >>> # Basic usage with default raw space
    >>> loader = load_data("real.h5ad", "generated.h5ad", ["perturbation"])
    >>> evaluator = GeneEvalEvaluator(loader)
    >>> results = evaluator.evaluate()
    
    >>> # With mixed space metrics (Paper example)
    >>> from gge.metrics import PearsonCorrelation, MMD, W2
    >>> metrics = [
    ...     PearsonCorrelation(space="deg", deg_lfc=0.25, deg_pval=0.1),
    ...     MMD(space="pca", n_components=50),
    ...     W2(space="pca", n_components=50),
    ... ]
    >>> evaluator = GeneEvalEvaluator(loader, metrics=metrics, control_key="ctrl")
    """
    
    def __init__(
        self,
        data_loader: GeneExpressionDataLoader,
        metrics: Optional[List[Union[BaseMetric, Type[BaseMetric]]]] = None,
        aggregate_method: str = "mean",
        include_multivariate: bool = False,
        control_key: Optional[str] = None,
        control_column: Optional[str] = None,
        verbose: bool = True,
    ):
        self.data_loader = data_loader
        self.aggregate_method = aggregate_method
        self.include_multivariate = include_multivariate
        self.control_key = control_key
        self.control_column = control_column or (
            data_loader.condition_columns[0] if data_loader.condition_columns else None
        )
        self.verbose = verbose
        
        # Cache for control data
        self._control_data: Optional[np.ndarray] = None
        
        # Initialize metrics
        self.metrics: List[BaseMetric] = []
        metric_classes = metrics or DEFAULT_METRICS
        
        for m in metric_classes:
            if isinstance(m, str):
                # It's a string name, look up in registry
                metric_class = MetricRegistry.get(m)
                if metric_class is None:
                    raise ValueError(f"Unknown metric: {m}. Available: {MetricRegistry.list_all()}")
                self.metrics.append(metric_class())
            elif isinstance(m, type):
                # It's a class, instantiate it
                self.metrics.append(m())
            else:
                # It's already an instance
                self.metrics.append(m)
        
        # Add multivariate metrics if requested
        if include_multivariate:
            self.metrics.extend([
                MultivariateWasserstein(),
                MultivariateMMD(),
            ])
        
        # Check if any metric needs DEG space
        self._needs_control = any(m.space == "deg" for m in self.metrics)
        if self._needs_control and self.control_key is None:
            warnings.warn(
                "Some metrics use space='deg' but no control_key provided. "
                "DEG space metrics will return NaN. Set control_key to enable."
            )
    
    def _get_control_data(self, split: Optional[str] = None) -> Optional[np.ndarray]:
        """Get control data for DEG space computation."""
        if self._control_data is not None:
            return self._control_data
        
        if self.control_key is None or self.control_column is None:
            return None
        
        try:
            # Get control samples from real data
            real_adata = self.data_loader._real_aligned
            mask = real_adata.obs[self.control_column].astype(str) == self.control_key
            if mask.sum() == 0:
                warnings.warn(f"No control samples found with {self.control_column}={self.control_key}")
                return None
            
            self._control_data = real_adata[mask].X
            if hasattr(self._control_data, 'toarray'):
                self._control_data = self._control_data.toarray()
            return self._control_data
        except Exception as e:
            warnings.warn(f"Failed to get control data: {e}")
            return None
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)
    
    def evaluate(
        self,
        splits: Optional[List[str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation on all conditions and splits.
        
        Parameters
        ----------
        splits : List[str], optional
            Splits to evaluate. If None, evaluates all available splits.
        save_dir : str or Path, optional
            If provided, save results to this directory
            
        Returns
        -------
        EvaluationResult
            Complete evaluation results
        """
        # Get available splits
        available_splits = self.data_loader.get_splits()
        
        if splits is None:
            splits = available_splits
        else:
            # Validate requested splits
            invalid = set(splits) - set(available_splits)
            if invalid:
                warnings.warn(f"Requested splits not found: {invalid}")
                splits = [s for s in splits if s in available_splits]
        
        self._log(f"Evaluating {len(splits)} splits: {splits}")
        self._log(f"Using {len(self.metrics)} metrics: {[m.name for m in self.metrics]}")
        
        # Create result container
        result = EvaluationResult(
            gene_names=self.data_loader.gene_names,
            condition_columns=self.data_loader.condition_columns,
            metadata={
                "real_path": str(self.data_loader.real_path),
                "generated_path": str(self.data_loader.generated_path),
                "aggregate_method": self.aggregate_method,
                "metric_names": [m.name for m in self.metrics],
            }
        )
        
        # Evaluate each split
        for split in splits:
            split_key = split if split != "all" else None
            split_result = self._evaluate_split(split, split_key)
            result.add_split(split_result)
        
        # Compute aggregate metrics
        for split_result in result.splits.values():
            split_result.compute_aggregates()
        
        # Print summary
        if self.verbose:
            self._print_summary(result)
        
        # Save if requested
        if save_dir is not None:
            result.save(save_dir)
            self._log(f"Results saved to: {save_dir}")
        
        return result
    
    def _evaluate_split(
        self,
        split_name: str,
        split_filter: Optional[str]
    ) -> SplitResult:
        """Evaluate a single split."""
        split_result = SplitResult(split_name=split_name)
        
        conditions = list(self.data_loader.iterate_conditions(split_filter))
        self._log(f"\n  Split '{split_name}': {len(conditions)} conditions")
        
        for i, (cond_key, real_data, gen_data, cond_info) in enumerate(conditions):
            if self.verbose and (i + 1) % 10 == 0:
                self._log(f"    Processing condition {i + 1}/{len(conditions)}")
            
            # Create condition result
            cond_result = ConditionResult(
                condition_key=cond_key,
                split=split_name,
                n_real_samples=real_data.shape[0],
                n_generated_samples=gen_data.shape[0],
                n_genes=real_data.shape[1],
                gene_names=self.data_loader.gene_names,
                perturbation=cond_info.get(self.data_loader.condition_columns[0]),
                covariates=cond_info,
            )
            
            # Store mean profiles
            cond_result.real_mean = real_data.mean(axis=0)
            cond_result.generated_mean = gen_data.mean(axis=0)
            
            # Get control data for DEG space metrics
            control_data = self._get_control_data(split_filter) if self._needs_control else None
            
            # Compute all metrics
            for metric in self.metrics:
                try:
                    metric_result = metric.compute(
                        real=real_data,
                        generated=gen_data,
                        gene_names=self.data_loader.gene_names,
                        aggregate_method=self.aggregate_method,
                        condition=cond_key,
                        split=split_name,
                        control_data=control_data,
                    )
                    cond_result.add_metric(metric.name, metric_result)
                except Exception as e:
                    warnings.warn(
                        f"Failed to compute {metric.name} for {cond_key}: {e}"
                    )
            
            split_result.add_condition(cond_result)
        
        return split_result
    
    def _print_summary(self, result: EvaluationResult):
        """Print summary of results."""
        self._log("\n" + "=" * 60)
        self._log("EVALUATION SUMMARY")
        self._log("=" * 60)
        
        for split_name, split in result.splits.items():
            self._log(f"\nSplit: {split_name} ({split.n_conditions} conditions)")
            self._log("-" * 40)
            
            # Print aggregate metrics
            for key, value in sorted(split.aggregate_metrics.items()):
                if key.endswith("_mean"):
                    metric_name = key[:-5]
                    std_key = f"{metric_name}_std"
                    std = split.aggregate_metrics.get(std_key, 0)
                    self._log(f"  {metric_name}: {value:.4f} ± {std:.4f}")
        
        self._log("=" * 60)


def evaluate(
    real_data: Union[str, Path, "ad.AnnData"],
    generated_data: Union[str, Path, "ad.AnnData"],
    condition_columns: List[str],
    split_column: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    metrics: Optional[List[Union[BaseMetric, Type[BaseMetric], str]]] = None,
    include_multivariate: bool = False,
    control_key: Optional[str] = None,
    control_column: Optional[str] = None,
    verbose: bool = True,
    **loader_kwargs
) -> EvaluationResult:
    """
    Convenience function to run full evaluation.
    
    All metrics support computation in different spaces (raw, pca, deg)
    through their space parameter.
    
    Parameters
    ----------
    real_data : str, Path, or AnnData
        Path to real data h5ad file or AnnData object
    generated_data : str, Path, or AnnData
        Path to generated data h5ad file or AnnData object
    condition_columns : List[str]
        Columns to match between datasets
    split_column : str, optional
        Column indicating train/test split
    output_dir : str or Path, optional
        Directory to save results
    metrics : List, optional
        Metrics to compute
    include_multivariate : bool
        Whether to include multivariate metrics
    verbose : bool
        Print progress
    **loader_kwargs
        Additional arguments for data loader
        
    Returns
    -------
    EvaluationResult
        Complete evaluation results
        
    Examples
    --------
    >>> # From paths
    >>> results = evaluate(
    ...     "real.h5ad",
    ...     "generated.h5ad",
    ...     condition_columns=["perturbation", "cell_type"],
    ...     output_dir="evaluation_output/"
    ... )
    >>> 
    >>> # From AnnData objects
    >>> results = evaluate(
    ...     real_adata,
    ...     generated_adata,
    ...     condition_columns=["perturbation"],
    ... )
    >>> 
    >>> # Mixed (path + AnnData)
    >>> results = evaluate(
    ...     "real.h5ad",
    ...     generated_adata,
    ...     condition_columns=["perturbation"],
    ... )
    """
    # Load data
    loader = load_data(
        real_data=real_data,
        generated_data=generated_data,
        condition_columns=condition_columns,
        split_column=split_column,
        **loader_kwargs
    )
    
    # Create evaluator
    evaluator = GeneEvalEvaluator(
        data_loader=loader,
        metrics=metrics,
        include_multivariate=include_multivariate,
        control_key=control_key,
        control_column=control_column,
        verbose=verbose,
    )
    
    # Run evaluation
    return evaluator.evaluate(save_dir=output_dir)


def evaluate_lazy(
    real_path: str,
    generated_path: str,
    condition_columns: Union[str, List[str]],
    metrics: List[BaseMetric],
    control_key: Optional[str] = None,
    control_column: Optional[str] = None,
    split_column: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    **loader_kwargs
) -> EvaluationResult:
    """
    Lazy-loading evaluation function for gene expression data.
    
    This function provides a streamlined API for evaluating generative models
    with explicit metric configuration. Each metric can specify its own space
    (raw, pca, or deg) for computation.
    
    Args:
        real_path: Path to real gene expression data
        generated_path: Path to generated gene expression data
        condition_columns: Column(s) for condition-wise stratification (string or list)
        metrics: List of metric instances with their space configurations
        control_key: Value identifying control samples for DEG space (required if any metric uses "deg" space)
        control_column: Column containing control identifier (defaults to first condition column)
        split_column: Optional column for train/test split evaluation
        output_dir: Optional directory to save evaluation results
        verbose: Whether to show progress information
        **loader_kwargs: Additional arguments passed to data loader
    
    Returns:
        EvaluationResult object containing computed metrics
    
    Example:
        >>> from gge import evaluate_lazy
        >>> from gge.metrics import PearsonCorrelation, Wasserstein2Distance, MMDDistance
        >>> 
        >>> metrics = [
        ...     PearsonCorrelation(space="deg", deg_lfc=0.25, deg_pval=0.1),
        ...     Wasserstein2Distance(space="pca", n_components=50),
        ...     MMDDistance(space="pca", n_components=50),
        ... ]
        >>> 
        >>> results = evaluate_lazy(
        ...     "real_data.h5ad",
        ...     "generated_data.h5ad", 
        ...     condition_columns="perturbation",  # Can be string or list
        ...     control_key="ctrl",
        ...     metrics=metrics
        ... )
    """
    # Convert string to list if needed
    if isinstance(condition_columns, str):
        condition_columns = [condition_columns]
    
    return evaluate(
        real_data=real_path,
        generated_data=generated_path,
        condition_columns=condition_columns,
        split_column=split_column,
        output_dir=output_dir,
        metrics=metrics,
        include_multivariate=False,  # User controls this via metrics list
        control_key=control_key,
        control_column=control_column,
        verbose=verbose,
        **loader_kwargs
    )


class MetricRegistry:
    """
    Registry of available metrics.
    
    Allows registration of custom metrics and retrieval by name.
    """
    
    _metrics: Dict[str, Type[BaseMetric]] = {}
    
    @classmethod
    def register(cls, metric_class: Type[BaseMetric]):
        """Register a metric class."""
        instance = metric_class()
        cls._metrics[instance.name] = metric_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseMetric]]:
        """Get metric class by name."""
        return cls._metrics.get(name)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())
    
    @classmethod
    def get_all(cls) -> List[Type[BaseMetric]]:
        """Get all registered metric classes."""
        return list(cls._metrics.values())


# Register default metrics
for metric_class in DEFAULT_METRICS:
    MetricRegistry.register(metric_class)

MetricRegistry.register(MultivariateWasserstein)
MetricRegistry.register(MultivariateMMD)
MetricRegistry.register(RSquared)
MetricRegistry.register(MSEDistance)
