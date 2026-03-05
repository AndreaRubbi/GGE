"""
Visualization module for gene expression evaluation.

Provides publication-quality plots:
- Boxplots and violin plots for metric distributions
- Radar plots for multi-metric comparison
- Scatter plots for real vs generated expression
- Embedding plots (PCA, UMAP)
- Heatmaps for per-gene metrics
- Interactive Plotly visualizations
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

# Interactive visualizations (optional - requires plotly)
try:
    from .interactive import (
        InteractiveVisualizer,
        density_overlay,
        embedding_interactive,
        PLOTLY_AVAILABLE,
    )
except ImportError:
    PLOTLY_AVAILABLE = False
    InteractiveVisualizer = None
    density_overlay = None
    embedding_interactive = None

__all__ = [
    # Classes
    "EvaluationPlotter",
    "EvaluationVisualizer",
    "PlotStyle",
    "InteractiveVisualizer",
    # Functions
    "visualize",
    "create_boxplot",
    "create_violin_plot",
    "create_heatmap",
    "create_scatter",
    "create_radar_chart",
    # Interactive functions
    "density_overlay",
    "embedding_interactive",
    "PLOTLY_AVAILABLE",
]