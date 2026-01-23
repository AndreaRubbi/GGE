"""
Configuration settings for GenEval.

Provides centralized configuration for metrics, paths, and defaults.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MetricConfig:
    """Configuration for metric computation."""
    
    # Default metrics to compute
    default_metrics: List[str] = field(default_factory=lambda: [
        "pearson",
        "spearman", 
        "mean_pearson",
        "mean_spearman",
        "wasserstein_1",
        "wasserstein_2",
        "mmd",
        "energy",
    ])
    
    # Whether to include multivariate metrics
    include_multivariate: bool = True
    
    # Aggregation method for per-gene metrics
    aggregate_method: str = "mean"
    
    # Wasserstein parameters
    wasserstein_blur: float = 0.01
    
    # MMD parameters
    mmd_kernel: str = "rbf"
    mmd_sigma: Optional[float] = None  # None = median heuristic


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    # Minimum samples per condition
    min_samples_per_condition: int = 2
    
    # Default split column name
    default_split_column: str = "split"
    
    # Standard split values
    train_split_values: List[str] = field(default_factory=lambda: ["train", "training"])
    test_split_values: List[str] = field(default_factory=lambda: ["test", "testing", "val", "validation"])


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    
    # Figure DPI
    dpi: int = 150
    
    # Default figure sizes
    figure_small: tuple = (8, 6)
    figure_medium: tuple = (12, 8)
    figure_large: tuple = (16, 12)
    figure_wide: tuple = (16, 6)
    
    # Style settings
    style: str = "whitegrid"
    context: str = "paper"
    font_scale: float = 1.2
    
    # Colors
    real_color: str = "#1f77b4"  # Blue
    generated_color: str = "#ff7f0e"  # Orange
    
    # Output formats
    default_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])


@dataclass
class Config:
    """
    Main configuration class for GenEval.
    
    Combines all configuration settings.
    """
    metrics: MetricConfig = field(default_factory=MetricConfig)
    data: DataConfig = field(default_factory=DataConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    
    # Output settings
    output_dir: Path = Path("output/")
    log_dir: Path = Path("logs/")
    
    # Verbosity
    verbose: bool = True
    
    @classmethod
    def default(cls) -> "Config":
        """Get default configuration."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "metrics": {
                "default_metrics": self.metrics.default_metrics,
                "include_multivariate": self.metrics.include_multivariate,
                "aggregate_method": self.metrics.aggregate_method,
            },
            "data": {
                "min_samples_per_condition": self.data.min_samples_per_condition,
                "default_split_column": self.data.default_split_column,
            },
            "plot": {
                "dpi": self.plot.dpi,
                "style": self.plot.style,
                "default_formats": self.plot.default_formats,
            },
            "output_dir": str(self.output_dir),
            "verbose": self.verbose,
        }


# Global default config instance
DEFAULT_CONFIG = Config.default()


def get_config() -> Config:
    """Get the current configuration."""
    return DEFAULT_CONFIG


def set_config(config: Config):
    """Set the global configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config