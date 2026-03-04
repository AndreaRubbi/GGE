"""
Comprehensive evaluator for gene expression data.

Computes all metrics between real and generated data, organized by conditions and splits.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union, Type, Any
from pathlib import Path
import numpy as np
import warnings
from dataclasses import dataclass

from .data.loader import GeneExpressionDataLoader, load_data
from .metrics.base_metric import BaseMetric, MetricResult
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
from .results import EvaluationResult, SplitResult, ConditionResult


# Default metrics to compute
DEFAULT_METRICS = [
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
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
    
    Parameters
    ----------
    data_loader : GeneExpressionDataLoader
        Loaded and aligned data loader
    metrics : List[BaseMetric or Type[BaseMetric]], optional
        Metrics to compute. If None, uses default set.
    aggregate_method : str
        How to aggregate per-gene values (mean, median, etc.)
    include_multivariate : bool
        Whether to include multivariate (whole-space) metrics
    verbose : bool
        Whether to print progress
        
    Examples
    --------
    >>> loader = load_data("real.h5ad", "generated.h5ad", ["perturbation"])
    >>> evaluator = GeneEvalEvaluator(loader)
    >>> results = evaluator.evaluate()
    >>> results.save("output/")
    """
    
    def __init__(
        self,
        data_loader: GeneExpressionDataLoader,
        metrics: Optional[List[Union[BaseMetric, Type[BaseMetric]]]] = None,
        aggregate_method: str = "mean",
        include_multivariate: bool = True,
        verbose: bool = True,
    ):
        self.data_loader = data_loader
        self.aggregate_method = aggregate_method
        self.include_multivariate = include_multivariate
        self.verbose = verbose
        
        # Initialize metrics
        self.metrics: List[BaseMetric] = []
        metric_classes = metrics or DEFAULT_METRICS
        
        for m in metric_classes:
            if isinstance(m, type):
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
    real_path: Union[str, Path],
    generated_path: Union[str, Path],
    condition_columns: List[str],
    split_column: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    metrics: Optional[List[Union[BaseMetric, Type[BaseMetric]]]] = None,
    include_multivariate: bool = True,
    verbose: bool = True,
    **loader_kwargs
) -> EvaluationResult:
    """
    Convenience function to run full evaluation.
    
    Parameters
    ----------
    real_path : str or Path
        Path to real data h5ad file
    generated_path : str or Path
        Path to generated data h5ad file
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
    >>> results = evaluate(
    ...     "real.h5ad",
    ...     "generated.h5ad",
    ...     condition_columns=["perturbation", "cell_type"],
    ...     split_column="split",
    ...     output_dir="evaluation_output/"
    ... )
    """
    # Load data
    loader = load_data(
        real_path=real_path,
        generated_path=generated_path,
        condition_columns=condition_columns,
        split_column=split_column,
        **loader_kwargs
    )
    
    # Create evaluator
    evaluator = GeneEvalEvaluator(
        data_loader=loader,
        metrics=metrics,
        include_multivariate=include_multivariate,
        verbose=verbose,
    )
    
    # Run evaluation
    return evaluator.evaluate(save_dir=output_dir)


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
