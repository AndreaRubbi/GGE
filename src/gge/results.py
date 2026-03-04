"""
Results container classes for evaluation outputs.

Provides structured storage for metrics, conditions, and visualization data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import json
from pathlib import Path


@dataclass
class ConditionResult:
    """
    Results for a single condition (perturbation + covariates).
    """
    condition_key: str
    split: str
    n_real_samples: int
    n_generated_samples: int
    n_genes: int
    gene_names: List[str]
    metrics: Dict[str, "MetricResult"] = field(default_factory=dict)
    
    # Mean expression profiles
    real_mean: Optional[np.ndarray] = None
    generated_mean: Optional[np.ndarray] = None
    
    # Parsed condition components
    perturbation: Optional[str] = None
    covariates: Dict[str, str] = field(default_factory=dict)
    
    def add_metric(self, name: str, result: "MetricResult"):
        """Add a metric result."""
        self.metrics[name] = result
    
    def get_metric_value(self, name: str) -> Optional[float]:
        """Get aggregate value for a metric."""
        if name in self.metrics:
            return self.metrics[name].aggregate_value
        return None
    
    def get_per_gene_values(self, name: str) -> Optional[np.ndarray]:
        """Get per-gene values for a metric."""
        if name in self.metrics:
            return self.metrics[name].per_gene_values
        return None
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        result = {
            "condition_key": self.condition_key,
            "split": self.split,
            "perturbation": self.perturbation,
            "n_real_samples": self.n_real_samples,
            "n_generated_samples": self.n_generated_samples,
            "n_genes": self.n_genes,
        }
        result.update(self.covariates)
        
        for name, metric in self.metrics.items():
            result[name] = metric.aggregate_value
        
        return result


@dataclass
class SplitResult:
    """
    Results for a single split (train/test/all).
    """
    split_name: str
    conditions: Dict[str, ConditionResult] = field(default_factory=dict)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_condition(self, condition: ConditionResult):
        """Add a condition result."""
        self.conditions[condition.condition_key] = condition
    
    def compute_aggregates(self):
        """Compute aggregate metrics across all conditions."""
        if not self.conditions:
            return
        
        # Collect all metric names
        metric_names = set()
        for cond in self.conditions.values():
            metric_names.update(cond.metrics.keys())
        
        # Compute mean across conditions for each metric
        for name in metric_names:
            values = []
            for cond in self.conditions.values():
                if name in cond.metrics:
                    values.append(cond.metrics[name].aggregate_value)
            if values:
                self.aggregate_metrics[f"{name}_mean"] = float(np.nanmean(values))
                self.aggregate_metrics[f"{name}_std"] = float(np.nanstd(values))
                self.aggregate_metrics[f"{name}_median"] = float(np.nanmedian(values))
    
    @property
    def n_conditions(self) -> int:
        return len(self.conditions)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert condition results to DataFrame."""
        rows = [cond.summary for cond in self.conditions.values()]
        return pd.DataFrame(rows)


@dataclass
class EvaluationResult:
    """
    Complete evaluation results container.
    
    Stores results per split and provides serialization methods.
    """
    splits: Dict[str, SplitResult] = field(default_factory=dict)
    gene_names: List[str] = field(default_factory=list)
    condition_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Paths to saved outputs
    output_dir: Optional[Path] = None
    
    def add_split(self, split: SplitResult):
        """Add a split result."""
        self.splits[split.split_name] = split
    
    def get_split(self, name: str) -> Optional[SplitResult]:
        """Get results for a specific split."""
        return self.splits.get(name)
    
    def get_all_conditions(self) -> List[ConditionResult]:
        """Get all condition results across splits."""
        conditions = []
        for split in self.splits.values():
            conditions.extend(split.conditions.values())
        return conditions
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Dict[str, float]]:
        """
        Get summary of a metric across all splits.
        
        Returns dict: split_name -> {mean, std, median}
        """
        summary = {}
        for split_name, split in self.splits.items():
            values = []
            for cond in split.conditions.values():
                if metric_name in cond.metrics:
                    values.append(cond.metrics[metric_name].aggregate_value)
            if values:
                summary[split_name] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                    "median": float(np.nanmedian(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "n_conditions": len(values),
                }
        return summary
    
    def to_dataframe(self, include_split: bool = True) -> pd.DataFrame:
        """
        Convert all results to a single DataFrame.
        
        Parameters
        ----------
        include_split : bool
            Whether to include split column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with one row per condition
        """
        dfs = []
        for split_name, split in self.splits.items():
            df = split.to_dataframe()
            if include_split:
                df["split"] = split_name
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def to_per_gene_dataframe(self, metric_name: str) -> pd.DataFrame:
        """
        Get per-gene metric values as DataFrame.
        
        Parameters
        ----------
        metric_name : str
            Name of metric to extract
            
        Returns
        -------
        pd.DataFrame
            DataFrame with genes as rows, conditions as columns
        """
        data = {}
        for split in self.splits.values():
            for cond_key, cond in split.conditions.items():
                if metric_name in cond.metrics:
                    col_name = f"{split.split_name}_{cond_key}"
                    data[col_name] = cond.metrics[metric_name].per_gene_values
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, index=self.gene_names)
        return df
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive summary."""
        result = {
            "n_splits": len(self.splits),
            "n_genes": len(self.gene_names),
            "condition_columns": self.condition_columns,
            "splits": {},
        }
        
        for split_name, split in self.splits.items():
            split.compute_aggregates()
            result["splits"][split_name] = {
                "n_conditions": split.n_conditions,
                "aggregates": split.aggregate_metrics,
            }
        
        return result
    
    def save(self, output_dir: Union[str, Path]):
        """
        Save results to directory.
        
        Saves:
        - summary.json: Aggregate metrics and metadata
        - results.csv: Per-condition metrics
        - per_gene_*.csv: Per-gene metrics for each metric type
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        
        # Save summary
        summary = self.summary()
        summary["metadata"] = self.metadata
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save condition-level results
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(output_dir / "results.csv", index=False)
        
        # Save per-gene metrics
        metric_names = set()
        for split in self.splits.values():
            for cond in split.conditions.values():
                metric_names.update(cond.metrics.keys())
        
        for metric_name in metric_names:
            df_gene = self.to_per_gene_dataframe(metric_name)
            if not df_gene.empty:
                df_gene.to_csv(output_dir / f"per_gene_{metric_name}.csv")
        
        return output_dir
    
    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> "EvaluationResult":
        """
        Load results from directory.
        
        Note: Currently loads summary only, not full per-gene data.
        """
        output_dir = Path(output_dir)
        
        with open(output_dir / "summary.json") as f:
            summary = json.load(f)
        
        result = cls(
            gene_names=[],
            condition_columns=summary.get("condition_columns", []),
            metadata=summary.get("metadata", {}),
        )
        result.output_dir = output_dir
        
        # Load condition-level results if available
        results_path = output_dir / "results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            # Reconstruct splits and conditions from DataFrame
            for split_name in df["split"].unique() if "split" in df.columns else ["all"]:
                split_df = df[df["split"] == split_name] if "split" in df.columns else df
                split_result = SplitResult(split_name=split_name)
                
                for _, row in split_df.iterrows():
                    cond = ConditionResult(
                        condition_key=row.get("condition_key", ""),
                        split=split_name,
                        n_real_samples=row.get("n_real_samples", 0),
                        n_generated_samples=row.get("n_generated_samples", 0),
                        n_genes=row.get("n_genes", 0),
                        gene_names=[],
                        perturbation=row.get("perturbation"),
                    )
                    split_result.add_condition(cond)
                
                result.add_split(split_result)
        
        return result
    
    def __repr__(self) -> str:
        n_conds = sum(s.n_conditions for s in self.splits.values())
        return (
            f"EvaluationResult(n_splits={len(self.splits)}, "
            f"n_conditions={n_conds}, n_genes={len(self.gene_names)})"
        )


# Import MetricResult here to avoid circular import
from .metrics.base_metric import MetricResult

# Update forward references
ConditionResult.__annotations__["metrics"] = Dict[str, MetricResult]
