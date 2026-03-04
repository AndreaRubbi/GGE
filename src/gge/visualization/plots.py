from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


# ==================== STANDALONE PLOTTING FUNCTIONS ====================

def create_boxplot(
    data: Dict[str, np.ndarray],
    title: str = "Boxplot",
    xlabel: str = "Group",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    show_points: bool = True,
    rotation: int = 45,
) -> Figure:
    """
    Create a boxplot from dictionary of arrays.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary mapping group names to arrays of values.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size.
    show_points : bool
        Whether to overlay individual points.
    rotation : int
        X-tick label rotation angle.
    
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Prepare data for seaborn
    rows = []
    for group, values in data.items():
        for val in np.asarray(values).flatten():
            rows.append({"group": group, "value": val})
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    with sns.axes_style("whitegrid"):
        sns.boxplot(data=df, x="group", y="value", ax=ax, palette="Set2")
        
        if show_points:
            sns.stripplot(
                data=df, x="group", y="value", ax=ax,
                color="black", alpha=0.3, size=3, jitter=True,
            )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    
    fig.tight_layout()
    return fig


def create_violin_plot(
    data: Dict[str, np.ndarray],
    title: str = "Violin Plot",
    xlabel: str = "Group",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    show_box: bool = True,
    rotation: int = 45,
) -> Figure:
    """
    Create a violin plot from dictionary of arrays.
    
    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary mapping group names to arrays of values.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size.
    show_box : bool
        Whether to overlay a boxplot inside the violin.
    rotation : int
        X-tick label rotation angle.
    
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    # Prepare data for seaborn
    rows = []
    for group, values in data.items():
        for val in np.asarray(values).flatten():
            rows.append({"group": group, "value": val})
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    with sns.axes_style("whitegrid"):
        sns.violinplot(
            data=df, x="group", y="value", ax=ax,
            palette="Set2", inner=None, cut=0,
        )
        
        if show_box:
            sns.boxplot(
                data=df, x="group", y="value", ax=ax,
                width=0.15, showcaps=True,
                boxprops={"facecolor": "white", "edgecolor": "black"},
                whiskerprops={"color": "black"},
                medianprops={"color": "red"},
                showfliers=False,
            )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if rotation:
        plt.xticks(rotation=rotation, ha="right")
    
    fig.tight_layout()
    return fig


def create_heatmap(
    data: np.ndarray,
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    title: str = "Heatmap",
    cmap: str = "RdYlBu_r",
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = False,
    center: Optional[float] = None,
) -> Figure:
    """
    Create a heatmap from a 2D array.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of values.
    row_labels : Sequence[str], optional
        Row labels.
    col_labels : Sequence[str], optional
        Column labels.
    title : str
        Plot title.
    cmap : str
        Colormap name.
    figsize : Tuple[int, int]
        Figure size.
    annot : bool
        Whether to annotate cells with values.
    center : float, optional
        Center value for diverging colormaps.
    
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    with sns.axes_style("white"):
        sns.heatmap(
            data, ax=ax,
            xticklabels=col_labels if col_labels else False,
            yticklabels=row_labels if row_labels else False,
            cmap=cmap,
            annot=annot,
            center=center,
            cbar_kws={"shrink": 0.8},
        )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def create_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.6,
    size: int = 20,
    add_diagonal: bool = False,
) -> Figure:
    """
    Create a scatter plot.
    
    Parameters
    ----------
    x : np.ndarray
        X values.
    y : np.ndarray
        Y values.
    labels : np.ndarray, optional
        Labels for coloring points.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : Tuple[int, int]
        Figure size.
    alpha : float
        Point transparency.
    size : int
        Point size.
    add_diagonal : bool
        Whether to add y=x diagonal line.
    
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(x[mask], y[mask], c=[colors[i]], s=size, alpha=alpha, label=str(label))
        
        ax.legend(frameon=False, loc="best")
    else:
        ax.scatter(x, y, s=size, alpha=alpha)
    
    if add_diagonal:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    fig.tight_layout()
    return fig


def create_radar_chart(
    values: Dict[str, float],
    title: str = "Radar Chart",
    figsize: Tuple[int, int] = (8, 8),
    fill: bool = True,
    alpha: float = 0.25,
) -> Figure:
    """
    Create a radar/spider chart.
    
    Parameters
    ----------
    values : Dict[str, float]
        Dictionary mapping metric names to values (should be normalized 0-1).
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    fill : bool
        Whether to fill the radar area.
    alpha : float
        Fill transparency.
    
    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    labels = list(values.keys())
    stats = list(values.values())
    
    # Close the plot
    stats = stats + stats[:1]
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    ax.plot(angles, stats, 'o-', linewidth=2, label="Metrics")
    
    if fill:
        ax.fill(angles, stats, alpha=alpha)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=12, fontweight="bold", y=1.08)
    
    fig.tight_layout()
    return fig


# ==================== PLOTTER CLASS ====================

class EvaluationPlotter:
    """
    Plotting helper for evaluation outputs.
    Produces meaningful, compact figures that summarize fit quality.
    """

    def __init__(self, style: str = "whitegrid"):
        self.style = style

    @staticmethod
    def _deg_set(deg: Optional[object]) -> Optional[set]:
        if deg is None:
            return None
        names = None
        if isinstance(deg, dict):
            names = deg.get("names", None)
        elif hasattr(deg, "columns") and "names" in getattr(deg, "columns", []):
            names = deg["names"]
        else:
            names = deg
        if names is None:
            return None
        if hasattr(names, "tolist"):
            names = names.tolist()
        return set([str(x) for x in names])

    def scatter_means_grid(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray, Sequence[str]]],
        stats: Optional[Mapping[str, Dict[str, float]]] = None,
        deg_map: Optional[Mapping[str, object]] = None,
        max_panels: int = 12,
        ncols: int = 3,
        figsize: Tuple[int, int] = (15, 12),
        alpha_other: float = 0.4,
        alpha_deg: float = 0.9,
    ):
        """
        Grid of scatter plots: mean(real) vs mean(generated) per condition key.
        data: key -> (real_means, gen_means, gene_names)
        stats: key -> {'pearson': float, 'mse': float}
        deg_map: key -> DEG-like object (iterable or dict with 'names')
        """
        keys = list(data.keys())[:max_panels]
        n = len(keys)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols)) if n > 0 else 1

        with sns.axes_style(self.style):
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        lims = None
        # compute shared limits for comparability
        all_vals = []
        for k in keys:
            rm, gm, _ = data[k]
            all_vals.append(rm)
            all_vals.append(gm)
        if all_vals:
            v = np.concatenate(all_vals)
            lo, hi = np.nanpercentile(v, [0.5, 99.5])
            pad = (hi - lo) * 0.05
            lims = (lo - pad, hi + pad)

        for i, k in enumerate(keys):
            r, g, genes = data[k]
            ax = axes[i // ncols, i % ncols]
            # highlight DEGs if provided
            degs = self._deg_set(deg_map.get(k)) if deg_map else None
            if degs:
                mask = np.isin(np.asarray(genes).astype(str), list(degs))
                ax.scatter(r[~mask], g[~mask], s=8, alpha=alpha_other, label="Other")
                ax.scatter(r[mask], g[mask], s=10, alpha=alpha_deg, label="DEGs", color="#d62728")
                ax.legend(frameon=False, fontsize=8, loc="upper left")
            else:
                ax.scatter(r, g, s=8, alpha=alpha_other)

            ax.plot(lims, lims, ls="--", c="gray", lw=1) if lims else None
            if lims:
                ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_title(k, fontsize=10)
            ax.set_xlabel("Mean expression (real)", fontsize=9)
            ax.set_ylabel("Mean expression (generated)", fontsize=9)

            if stats and k in stats:
                s = stats[k]
                txt = f"r={s.get('pearson', np.nan):.2f}  MSE={s.get('mse', np.nan):.2e}"
                ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8)

        # hide empty axes
        for j in range(n, nrows * ncols):
            ax = axes[j // ncols, j % ncols]
            ax.axis("off")

        fig.tight_layout()
        return fig

    def residuals_violin(
        self,
        residuals: Dict[str, np.ndarray],
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        figsize: Tuple[int, int] = (12, 4),
        rotate_xticks: bool = True,
    ):
        """
        Violin/box overlay of residuals (generated - real), per condition key.
        """
        rows = []
        for k, v in residuals.items():
            v = np.asarray(v, dtype=float)
            lo, hi = np.nanpercentile(v, clip_percentiles)
            v = np.clip(v, lo, hi)
            rows.extend([(k, x) for x in v])
        df = pd.DataFrame(rows, columns=["condition", "residual"])

        with sns.axes_style(self.style):
            fig, ax = plt.subplots(figsize=figsize)
            sns.violinplot(data=df, x="condition", y="residual", inner=None, cut=0, ax=ax, color="#9ecae1")
            sns.boxplot(data=df, x="condition", y="residual", ax=ax, width=0.15, showcaps=True,
                        boxprops={"facecolor": "white"}, showfliers=False)
            ax.axhline(0, ls="--", c="gray", lw=1)
            ax.set_title("Residual distributions per condition (generated − real)")
            ax.set_xlabel("Condition")
            ax.set_ylabel("Residual")
            if rotate_xticks:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
        return fig

    def metrics_bar(
        self,
        metrics_per_key: Mapping[str, Mapping[str, float]],
        order: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 4),
    ):
        """
        Grouped bar chart of metrics per condition key.
        metrics_per_key: key -> {metric_name: value}
        """
        rows = []
        metric_names = set()
        for k, m in metrics_per_key.items():
            for name, val in m.items():
                metric_names.add(name)
                rows.append((k, name, val))
        df = pd.DataFrame(rows, columns=["condition", "metric", "value"])

        # default order: descending by pearson if present else by first metric
        if order is None and "pearson" in (metric_names or []):
            agg = df[df.metric == "pearson"].sort_values("value", ascending=True)
            order = agg["condition"].tolist()
        elif order is None:
            first = df.metric.iloc[0] if not df.empty else None
            if first:
                agg = df[df.metric == first].sort_values("value", ascending=False)
                order = agg["condition"].tolist()

        with sns.axes_style(self.style):
            fig, ax = plt.subplots(figsize=figsize)
            sns.barplot(data=df, x="condition", y="value", hue="metric", ax=ax)
            ax.set_title("Evaluation metrics per condition")
            ax.set_xlabel("Condition")
            ax.set_ylabel("Metric value")
            if order:
                ax.set_xticklabels(order)
            ax.legend(frameon=False, ncols=min(4, len(metric_names)))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
        return fig