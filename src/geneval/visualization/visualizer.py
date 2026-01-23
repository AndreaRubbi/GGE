"""
Comprehensive visualization module for gene expression evaluation.

Provides publication-quality plots for evaluation results:
- Boxplots and violin plots for metric distributions
- Radar plots for multi-metric comparison
- Scatter plots for real vs generated expression
- Embedding plots (PCA, UMAP) for data visualization
- Heatmaps for per-gene metrics
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

if TYPE_CHECKING:
    from geneval.results import EvaluationResult
    from geneval.data.loader import GeneExpressionDataLoader

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    import seaborn as sns
except ImportError:
    raise ImportError(
        "matplotlib and seaborn are required for visualization. "
        "Install with: pip install matplotlib seaborn"
    )


class PlotStyle:
    """Plot styling configuration."""
    
    # Color palettes
    REAL_COLOR = "#1f77b4"  # Blue
    GENERATED_COLOR = "#ff7f0e"  # Orange
    
    METRIC_PALETTE = {
        "pearson": "#2ecc71",      # Green
        "spearman": "#27ae60",     # Dark green
        "mean_pearson": "#3498db",  # Blue
        "mean_spearman": "#2980b9", # Dark blue
        "wasserstein_1": "#e74c3c", # Red
        "wasserstein_2": "#c0392b", # Dark red
        "mmd": "#9b59b6",          # Purple
        "energy": "#8e44ad",       # Dark purple
        "multivariate_wasserstein": "#f39c12",  # Yellow
        "multivariate_mmd": "#d35400",          # Orange
    }
    
    # Default figure sizes
    FIGURE_SMALL = (8, 6)
    FIGURE_MEDIUM = (12, 8)
    FIGURE_LARGE = (16, 12)
    FIGURE_WIDE = (16, 6)
    
    # Style settings
    STYLE = "whitegrid"
    CONTEXT = "paper"
    FONT_SCALE = 1.2


class EvaluationVisualizer:
    """
    Comprehensive visualizer for evaluation results.
    
    Generates all plots from EvaluationResult objects.
    
    Parameters
    ----------
    results : EvaluationResult
        Evaluation results to visualize
    style : str
        Seaborn style
    context : str
        Seaborn context
    font_scale : float
        Font scale multiplier
    dpi : int
        Resolution for saved figures
    """
    
    def __init__(
        self,
        results: "EvaluationResult",
        style: str = PlotStyle.STYLE,
        context: str = PlotStyle.CONTEXT,
        font_scale: float = PlotStyle.FONT_SCALE,
        dpi: int = 150,
    ):
        self.results = results
        self.style = style
        self.context = context
        self.font_scale = font_scale
        self.dpi = dpi
        
        # Apply style
        sns.set_style(style)
        sns.set_context(context, font_scale=font_scale)
    
    def _get_metric_data(
        self,
        metric_name: str,
        split: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract metric data as DataFrame."""
        rows = []
        
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                if metric_name in cond.metrics:
                    value = cond.metrics[metric_name].aggregate_value
                    rows.append({
                        "split": split_name,
                        "condition": cond_key,
                        "perturbation": cond.perturbation or cond_key,
                        "value": value,
                        "metric": metric_name,
                    })
        
        return pd.DataFrame(rows)
    
    def _get_all_metrics_data(
        self,
        split: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract all metrics as DataFrame."""
        rows = []
        
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                row = {
                    "split": split_name,
                    "condition": cond_key,
                    "perturbation": cond.perturbation or cond_key,
                }
                for metric_name, metric_result in cond.metrics.items():
                    row[metric_name] = metric_result.aggregate_value
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    # ==================== BOXPLOTS ====================
    
    def boxplot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        split: Optional[str] = None,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_WIDE,
        palette: Optional[Dict[str, str]] = None,
    ) -> Figure:
        """
        Create boxplot of metric values across conditions.
        
        Parameters
        ----------
        metrics : List[str], optional
            Metrics to include. If None, uses all available.
        split : str, optional
            Filter to specific split
        figsize : Tuple[int, int]
            Figure size
        palette : Dict[str, str], optional
            Color mapping for metrics
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        df = self._get_all_metrics_data(split)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Get metric columns
        meta_cols = ["split", "condition", "perturbation"]
        available_metrics = [c for c in df.columns if c not in meta_cols]
        
        if metrics is not None:
            available_metrics = [m for m in metrics if m in available_metrics]
        
        # Melt to long format
        df_long = df.melt(
            id_vars=meta_cols,
            value_vars=available_metrics,
            var_name="metric",
            value_name="value"
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = palette or PlotStyle.METRIC_PALETTE
        color_list = [colors.get(m, "#95a5a6") for m in available_metrics]
        
        sns.boxplot(
            data=df_long,
            x="metric",
            y="value",
            palette=color_list,
            ax=ax,
        )
        
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title(f"Metric Distributions{' (' + split + ')' if split else ''}")
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    def boxplot_by_condition(
        self,
        metric_name: str,
        split: Optional[str] = None,
        max_conditions: int = 20,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_WIDE,
    ) -> Figure:
        """
        Create boxplot of a single metric across conditions.
        
        Shows per-gene distribution for each condition.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        rows = []
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                if metric_name in cond.metrics:
                    per_gene = cond.metrics[metric_name].per_gene_values
                    for val in per_gene[:1000]:  # Limit for performance
                        rows.append({
                            "condition": cond.perturbation or cond_key[:20],
                            "value": val,
                        })
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            ax.text(0.5, 0.5, f"No data for {metric_name}", ha='center', va='center')
            return fig
        
        # Limit conditions
        top_conditions = df.groupby("condition")["value"].median().nlargest(max_conditions).index
        df = df[df["condition"].isin(top_conditions)]
        
        sns.boxplot(
            data=df,
            x="condition",
            y="value",
            ax=ax,
            color=PlotStyle.METRIC_PALETTE.get(metric_name, "#3498db"),
        )
        
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Per-Gene {metric_name} by Condition")
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    # ==================== VIOLIN PLOTS ====================
    
    def violin_metrics(
        self,
        metrics: Optional[List[str]] = None,
        split: Optional[str] = None,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_WIDE,
    ) -> Figure:
        """
        Create violin plot of metric values across conditions.
        """
        df = self._get_all_metrics_data(split)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        meta_cols = ["split", "condition", "perturbation"]
        available_metrics = [c for c in df.columns if c not in meta_cols]
        
        if metrics is not None:
            available_metrics = [m for m in metrics if m in available_metrics]
        
        df_long = df.melt(
            id_vars=meta_cols,
            value_vars=available_metrics,
            var_name="metric",
            value_name="value"
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = [PlotStyle.METRIC_PALETTE.get(m, "#95a5a6") for m in available_metrics]
        
        sns.violinplot(
            data=df_long,
            x="metric",
            y="value",
            palette=colors,
            ax=ax,
            inner="box",
        )
        
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title(f"Metric Distributions (Violin){' (' + split + ')' if split else ''}")
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    def violin_per_gene(
        self,
        metric_name: str,
        split: Optional[str] = None,
        max_conditions: int = 10,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_MEDIUM,
    ) -> Figure:
        """
        Create violin plot showing per-gene metric distributions.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        rows = []
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                if metric_name in cond.metrics:
                    per_gene = cond.metrics[metric_name].per_gene_values
                    for val in per_gene:
                        rows.append({
                            "condition": cond.perturbation or cond_key[:15],
                            "value": val,
                        })
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            ax.text(0.5, 0.5, f"No data for {metric_name}", ha='center', va='center')
            return fig
        
        # Limit to top conditions by median
        top_conditions = df.groupby("condition")["value"].median().nlargest(max_conditions).index
        df = df[df["condition"].isin(top_conditions)]
        
        sns.violinplot(
            data=df,
            x="condition",
            y="value",
            ax=ax,
            color=PlotStyle.METRIC_PALETTE.get(metric_name, "#3498db"),
            inner="quartile",
        )
        
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Per-Gene {metric_name} Distribution")
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    # ==================== RADAR PLOTS ====================
    
    def radar_plot(
        self,
        metrics: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        split: Optional[str] = None,
        max_conditions: int = 6,
        normalize: bool = True,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_MEDIUM,
    ) -> Figure:
        """
        Create radar plot comparing multiple metrics across conditions.
        
        Parameters
        ----------
        metrics : List[str], optional
            Metrics to include (should be 3+)
        conditions : List[str], optional
            Conditions to compare
        split : str, optional
            Filter to specific split
        max_conditions : int
            Maximum conditions to show
        normalize : bool
            Whether to normalize metrics to [0, 1]
        figsize : Tuple[int, int]
            Figure size
            
        Returns
        -------
        Figure
            Matplotlib figure with radar plot
        """
        df = self._get_all_metrics_data(split)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        meta_cols = ["split", "condition", "perturbation"]
        available_metrics = [c for c in df.columns if c not in meta_cols]
        
        if metrics is not None:
            available_metrics = [m for m in metrics if m in available_metrics]
        
        if len(available_metrics) < 3:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Need at least 3 metrics for radar plot", ha='center', va='center')
            return fig
        
        # Select conditions
        if conditions is not None:
            df = df[df["perturbation"].isin(conditions)]
        else:
            # Take top conditions by mean metric
            df["_mean"] = df[available_metrics].mean(axis=1)
            top = df.nlargest(max_conditions, "_mean")["perturbation"].unique()
            df = df[df["perturbation"].isin(top)]
        
        # Normalize if requested
        if normalize:
            for m in available_metrics:
                col = df[m]
                min_val, max_val = col.min(), col.max()
                if max_val > min_val:
                    df[m] = (col - min_val) / (max_val - min_val)
        
        # Set up radar chart
        n_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Plot each condition
        colors = sns.color_palette("husl", len(df["perturbation"].unique()))
        
        for i, (_, row) in enumerate(df.iterrows()):
            values = [row[m] for m in available_metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row["perturbation"][:20], color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, size=10)
        
        ax.set_title("Multi-Metric Comparison", size=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        fig.tight_layout()
        return fig
    
    def radar_split_comparison(
        self,
        metrics: Optional[List[str]] = None,
        normalize: bool = True,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_MEDIUM,
    ) -> Figure:
        """
        Radar plot comparing aggregate metrics across splits.
        """
        # Collect aggregate metrics per split
        data = {}
        for split_name, split_result in self.results.splits.items():
            split_result.compute_aggregates()
            data[split_name] = {}
            for key, value in split_result.aggregate_metrics.items():
                if key.endswith("_mean"):
                    metric_name = key[:-5]
                    data[split_name][metric_name] = value
        
        if not data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        df = pd.DataFrame(data).T
        
        if metrics is not None:
            df = df[[m for m in metrics if m in df.columns]]
        
        available_metrics = df.columns.tolist()
        
        if len(available_metrics) < 3:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Need at least 3 metrics for radar plot", ha='center', va='center')
            return fig
        
        # Normalize
        if normalize:
            for col in df.columns:
                min_val, max_val = df[col].min(), df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        # Create radar
        n_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        colors = [PlotStyle.REAL_COLOR, PlotStyle.GENERATED_COLOR, "#2ecc71", "#9b59b6"]
        
        for i, (split_name, row) in enumerate(df.iterrows()):
            values = [row[m] for m in available_metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=split_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics, size=10)
        ax.set_title("Split Comparison", size=14, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        fig.tight_layout()
        return fig
    
    # ==================== SCATTER PLOTS ====================
    
    def scatter_real_vs_generated(
        self,
        condition: str,
        split: Optional[str] = None,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_MEDIUM,
        alpha: float = 0.5,
    ) -> Figure:
        """
        Scatter plot of real vs generated mean expression.
        
        Parameters
        ----------
        condition : str
            Condition key or perturbation name
        split : str, optional
            Filter to specific split
        figsize : Tuple[int, int]
            Figure size
        alpha : float
            Point transparency
            
        Returns
        -------
        Figure
            Scatter plot figure
        """
        # Find the condition
        cond_result = None
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                if cond_key == condition or cond.perturbation == condition:
                    cond_result = cond
                    break
        
        if cond_result is None or cond_result.real_mean is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data for condition: {condition}", ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=figsize)
        
        real_mean = cond_result.real_mean
        gen_mean = cond_result.generated_mean
        
        ax.scatter(real_mean, gen_mean, alpha=alpha, s=10, c=PlotStyle.REAL_COLOR)
        
        # Add diagonal line
        lims = [
            min(real_mean.min(), gen_mean.min()),
            max(real_mean.max(), gen_mean.max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        
        # Add correlation
        if "pearson" in cond_result.metrics:
            r = cond_result.metrics["pearson"].aggregate_value
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top')
        
        ax.set_xlabel("Real Mean Expression")
        ax.set_ylabel("Generated Mean Expression")
        ax.set_title(f"Real vs Generated: {cond_result.perturbation or condition}")
        
        fig.tight_layout()
        return fig
    
    def scatter_grid(
        self,
        split: Optional[str] = None,
        max_conditions: int = 12,
        ncols: int = 4,
        figsize_per_panel: Tuple[float, float] = (4, 4),
    ) -> Figure:
        """
        Grid of scatter plots for multiple conditions.
        """
        # Collect conditions
        conditions = []
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond in split_result.conditions.values():
                if cond.real_mean is not None:
                    conditions.append(cond)
        
        conditions = conditions[:max_conditions]
        n = len(conditions)
        
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
            squeeze=False
        )
        
        for i, cond in enumerate(conditions):
            ax = axes[i // ncols, i % ncols]
            
            real_mean = cond.real_mean
            gen_mean = cond.generated_mean
            
            ax.scatter(real_mean, gen_mean, alpha=0.4, s=5, c=PlotStyle.REAL_COLOR)
            
            lims = [
                min(real_mean.min(), gen_mean.min()),
                max(real_mean.max(), gen_mean.max()),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
            
            if "pearson" in cond.metrics:
                r = cond.metrics["pearson"].aggregate_value
                ax.text(0.05, 0.95, f'r={r:.2f}', transform=ax.transAxes, 
                       fontsize=8, verticalalignment='top')
            
            ax.set_title(cond.perturbation or cond.condition_key[:20], fontsize=9)
            ax.tick_params(labelsize=7)
        
        # Hide empty panels
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].axis('off')
        
        fig.suptitle("Real vs Generated Expression", fontsize=12, y=1.02)
        fig.tight_layout()
        return fig
    
    # ==================== HEATMAPS ====================
    
    def heatmap_per_gene(
        self,
        metric_name: str,
        split: Optional[str] = None,
        max_genes: int = 50,
        max_conditions: int = 20,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_LARGE,
        cmap: str = "RdYlBu_r",
    ) -> Figure:
        """
        Heatmap of per-gene metric values.
        
        Parameters
        ----------
        metric_name : str
            Metric to visualize
        split : str, optional
            Filter to specific split
        max_genes : int
            Maximum genes to show (selects most variable)
        max_conditions : int
            Maximum conditions to show
        figsize : Tuple[int, int]
            Figure size
        cmap : str
            Colormap name
            
        Returns
        -------
        Figure
            Heatmap figure
        """
        # Collect data
        data = {}
        gene_names = None
        
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                if metric_name in cond.metrics:
                    data[cond.perturbation or cond_key[:15]] = cond.metrics[metric_name].per_gene_values
                    if gene_names is None:
                        gene_names = cond.metrics[metric_name].gene_names
        
        if not data:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data for {metric_name}", ha='center', va='center')
            return fig
        
        df = pd.DataFrame(data, index=gene_names)
        
        # Select conditions
        if df.shape[1] > max_conditions:
            # Keep conditions with highest variance
            var = df.var()
            top_conds = var.nlargest(max_conditions).index
            df = df[top_conds]
        
        # Select genes
        if df.shape[0] > max_genes:
            # Keep genes with highest variance
            gene_var = df.var(axis=1)
            top_genes = gene_var.nlargest(max_genes).index
            df = df.loc[top_genes]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            df,
            ax=ax,
            cmap=cmap,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": metric_name},
        )
        
        ax.set_xlabel("Condition")
        ax.set_ylabel("Gene")
        ax.set_title(f"Per-Gene {metric_name}")
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(fontsize=8)
        
        fig.tight_layout()
        return fig
    
    def heatmap_metrics_summary(
        self,
        split: Optional[str] = None,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_MEDIUM,
        cmap: str = "RdYlBu",
    ) -> Figure:
        """
        Heatmap summarizing all metrics across conditions.
        """
        df = self._get_all_metrics_data(split)
        
        if df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        meta_cols = ["split", "condition", "perturbation"]
        metric_cols = [c for c in df.columns if c not in meta_cols]
        
        # Pivot for heatmap
        df_pivot = df.set_index("perturbation")[metric_cols]
        
        # Normalize columns for visualization
        df_norm = (df_pivot - df_pivot.mean()) / df_pivot.std()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            df_norm,
            ax=ax,
            cmap=cmap,
            xticklabels=True,
            yticklabels=True,
            center=0,
            cbar_kws={"label": "Z-score"},
        )
        
        ax.set_xlabel("Metric")
        ax.set_ylabel("Condition")
        ax.set_title("Metrics Summary (Z-scored)")
        
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        return fig
    
    # ==================== EMBEDDING PLOTS ====================
    
    def embedding_plot(
        self,
        data_loader: "GeneExpressionDataLoader",
        method: str = "pca",
        split: Optional[str] = None,
        max_samples: int = 5000,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_MEDIUM,
        alpha: float = 0.6,
    ) -> Figure:
        """
        Plot embedded data (PCA or UMAP) comparing real and generated.
        
        Parameters
        ----------
        data_loader : GeneExpressionDataLoader
            Data loader with real and generated data
        method : str
            Embedding method: "pca" or "umap"
        split : str, optional
            Filter to specific split
        max_samples : int
            Maximum samples to plot
        figsize : Tuple[int, int]
            Figure size
        alpha : float
            Point transparency
            
        Returns
        -------
        Figure
            Embedding plot figure
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy is required for embedding plots")
        
        # Combine real and generated data
        real = data_loader.real.copy()
        gen = data_loader.generated.copy()
        
        # Apply split filter if needed
        if split is not None and data_loader.split_column is not None:
            if data_loader.split_column in real.obs.columns:
                mask = real.obs[data_loader.split_column].astype(str) == split
                real = real[mask].copy()
        
        # Subsample if needed
        if real.n_obs > max_samples // 2:
            idx = np.random.choice(real.n_obs, max_samples // 2, replace=False)
            real = real[idx].copy()
        if gen.n_obs > max_samples // 2:
            idx = np.random.choice(gen.n_obs, max_samples // 2, replace=False)
            gen = gen[idx].copy()
        
        # Add source label
        real.obs["_source"] = "Real"
        gen.obs["_source"] = "Generated"
        
        # Concatenate
        combined = real.concatenate(gen, batch_key="_batch")
        
        # Compute embedding
        sc.pp.pca(combined, n_comps=50)
        
        if method.lower() == "umap":
            sc.pp.neighbors(combined)
            sc.tl.umap(combined)
            x_key = "X_umap"
            x_label, y_label = "UMAP1", "UMAP2"
        else:
            x_key = "X_pca"
            x_label, y_label = "PC1", "PC2"
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        coords = combined.obsm[x_key][:, :2]
        source = combined.obs["_source"]
        
        for label, color in [("Real", PlotStyle.REAL_COLOR), ("Generated", PlotStyle.GENERATED_COLOR)]:
            mask = source == label
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, label=label, alpha=alpha, s=10
            )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{method.upper()} Embedding: Real vs Generated")
        ax.legend()
        
        fig.tight_layout()
        return fig
    
    def embedding_by_condition(
        self,
        data_loader: "GeneExpressionDataLoader",
        method: str = "pca",
        condition_column: Optional[str] = None,
        max_samples: int = 5000,
        figsize: Tuple[int, int] = PlotStyle.FIGURE_LARGE,
    ) -> Figure:
        """
        Embedding plot colored by condition.
        """
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy is required for embedding plots")
        
        # Use first condition column if not specified
        if condition_column is None:
            condition_column = data_loader.condition_columns[0]
        
        # Combine data
        real = data_loader.real.copy()
        gen = data_loader.generated.copy()
        
        real.obs["_source"] = "Real"
        gen.obs["_source"] = "Generated"
        
        # Subsample
        if real.n_obs > max_samples // 2:
            idx = np.random.choice(real.n_obs, max_samples // 2, replace=False)
            real = real[idx].copy()
        if gen.n_obs > max_samples // 2:
            idx = np.random.choice(gen.n_obs, max_samples // 2, replace=False)
            gen = gen[idx].copy()
        
        combined = real.concatenate(gen, batch_key="_batch")
        
        # Compute embedding
        sc.pp.pca(combined, n_comps=50)
        if method.lower() == "umap":
            sc.pp.neighbors(combined)
            sc.tl.umap(combined)
            x_key = "X_umap"
        else:
            x_key = "X_pca"
        
        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        coords = combined.obsm[x_key][:, :2]
        
        for ax, source in zip(axes, ["Real", "Generated"]):
            mask = combined.obs["_source"] == source
            
            conditions = combined.obs.loc[mask, condition_column].astype(str)
            unique_conds = conditions.unique()
            colors = sns.color_palette("husl", len(unique_conds))
            color_map = dict(zip(unique_conds, colors))
            
            for cond in unique_conds:
                cond_mask = (combined.obs["_source"] == source) & (combined.obs[condition_column].astype(str) == cond)
                ax.scatter(
                    coords[cond_mask, 0], coords[cond_mask, 1],
                    c=[color_map[cond]], label=cond[:15], alpha=0.6, s=10
                )
            
            ax.set_title(f"{source}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        
        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        fig.suptitle(f"{method.upper()} Embedding by {condition_column}", fontsize=12)
        fig.tight_layout()
        return fig
    
    # ==================== SAVE ALL ====================
    
    def save_all(
        self,
        output_dir: Union[str, Path],
        formats: List[str] = ["png", "pdf"],
        data_loader: Optional["GeneExpressionDataLoader"] = None,
    ):
        """
        Generate and save all plots.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save plots
        formats : List[str]
            Image formats to save
        data_loader : GeneExpressionDataLoader, optional
            If provided, also generates embedding plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = {}
        
        # Generate all plots
        try:
            plots["boxplot_metrics"] = self.boxplot_metrics()
        except Exception as e:
            warnings.warn(f"Failed to generate boxplot_metrics: {e}")
        
        try:
            plots["violin_metrics"] = self.violin_metrics()
        except Exception as e:
            warnings.warn(f"Failed to generate violin_metrics: {e}")
        
        try:
            plots["radar_split"] = self.radar_split_comparison()
        except Exception as e:
            warnings.warn(f"Failed to generate radar_split: {e}")
        
        try:
            plots["scatter_grid"] = self.scatter_grid()
        except Exception as e:
            warnings.warn(f"Failed to generate scatter_grid: {e}")
        
        try:
            plots["heatmap_summary"] = self.heatmap_metrics_summary()
        except Exception as e:
            warnings.warn(f"Failed to generate heatmap_summary: {e}")
        
        # Per-metric plots
        for metric_name in ["pearson", "wasserstein_1", "mmd"]:
            try:
                plots[f"violin_{metric_name}"] = self.violin_per_gene(metric_name)
            except Exception:
                pass
        
        # Embedding plots if data loader provided
        if data_loader is not None:
            try:
                plots["embedding_pca"] = self.embedding_plot(data_loader, method="pca")
            except Exception as e:
                warnings.warn(f"Failed to generate PCA embedding: {e}")
            
            try:
                plots["embedding_umap"] = self.embedding_plot(data_loader, method="umap")
            except Exception as e:
                warnings.warn(f"Failed to generate UMAP embedding: {e}")
        
        # Save all plots
        for name, fig in plots.items():
            for fmt in formats:
                path = output_dir / f"{name}.{fmt}"
                fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Saved {len(plots)} plots to {output_dir}")


# Convenience function
def visualize(
    results: "EvaluationResult",
    output_dir: Union[str, Path],
    data_loader: Optional["GeneExpressionDataLoader"] = None,
    **kwargs
):
    """
    Generate and save all visualizations.
    
    Parameters
    ----------
    results : EvaluationResult
        Evaluation results to visualize
    output_dir : str or Path
        Directory to save plots
    data_loader : GeneExpressionDataLoader, optional
        If provided, generates embedding plots
    **kwargs
        Additional arguments for EvaluationVisualizer
    """
    viz = EvaluationVisualizer(results, **kwargs)
    viz.save_all(output_dir, data_loader=data_loader)
