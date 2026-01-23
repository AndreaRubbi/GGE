"""
Visualization module for gene expression evaluation.

Provides publication-quality plots:
- Boxplots and violin plots for metric distributions
- Radar plots for multi-metric comparison
- Scatter plots for real vs generated expression
- Embedding plots (PCA, UMAP)
- Heatmaps for per-gene metrics
"""

from .plots import (
    EvaluationPlotter,
    create_boxplot,
    create_violin_plot,
    create_heatmap,
    create_scatter,
    create_radar_chart,
)
from .visualizer import (
    EvaluationVisualizer,
    PlotStyle,
    visualize,
)

__all__ = [
    # Classes
    "EvaluationPlotter",
    "EvaluationVisualizer",
    "PlotStyle",
    # Functions
    "visualize",
    "create_boxplot",
    "create_violin_plot",
    "create_heatmap",
    "create_scatter",
    "create_radar_chart",
]