"""
Interactive Plotly/Dash visualizations for gene expression evaluation.

Provides interactive plots including:
- Density overlays of generated data on real data 
- Interactive scatter plots with metadata coloring
- Embedding plots with hover information
- Metric comparison dashboards
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any, TYPE_CHECKING
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

if TYPE_CHECKING:
    from gge.results import EvaluationResult
    from gge.data.loader import GeneExpressionDataLoader
    import anndata as ad

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None
    make_subplots = None


def _check_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )


class InteractiveVisualizer:
    """
    Interactive visualizer using Plotly for gene expression evaluation.
    
    Generates interactive plots that support:
    - Hover tooltips with detailed information
    - Zooming and panning
    - Metadata-based coloring
    - Export to HTML
    
    Parameters
    ----------
    results : EvaluationResult, optional
        Evaluation results to visualize
    loader : GeneExpressionDataLoader, optional
        Data loader with real and generated data
    """
    
    def __init__(
        self,
        results: Optional["EvaluationResult"] = None,
        loader: Optional["GeneExpressionDataLoader"] = None,
    ):
        _check_plotly()
        self.results = results
        self.loader = loader
        
        # Default color scheme
        self.real_color = "#1f77b4"  # Blue
        self.generated_color = "#ff7f0e"  # Orange
    
    # ==================== DENSITY PLOTS ====================
    
    def density_overlay(
        self,
        gene: Optional[str] = None,
        gene_index: Optional[int] = None,
        condition: Optional[str] = None,
        split: Optional[str] = None,
        nbins: int = 50,
        title: Optional[str] = None,
        show_rug: bool = False,
    ) -> go.Figure:
        """
        Create density overlay plot of generated data on real data.
        
        Visualizes how well the generated distribution matches the real
        distribution for a specific gene.
        
        Parameters
        ----------
        gene : str, optional
            Gene name to plot. If None, uses gene_index or first gene.
        gene_index : int, optional
            Gene index to plot (alternative to gene name).
        condition : str, optional
            Filter to specific condition
        split : str, optional
            Filter to specific split
        nbins : int
            Number of histogram bins
        title : str, optional
            Custom plot title
        show_rug : bool
            Show rug plot on x-axis
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.loader is None:
            raise ValueError("Loader required for density plots")
        
        if not self.loader._is_loaded:
            self.loader.load()
            self.loader.align_genes()
        
        # Get gene data
        real_data = self.loader.real
        gen_data = self.loader.generated
        
        # Filter by condition if specified
        if condition is not None:
            condition_mask_real = self._get_condition_mask(real_data, condition)
            condition_mask_gen = self._get_condition_mask(gen_data, condition)
            real_data = real_data[condition_mask_real]
            gen_data = gen_data[condition_mask_gen]
        
        # Filter by split if specified
        if split is not None and self.loader.split_column:
            split_mask_real = real_data.obs[self.loader.split_column] == split
            split_mask_gen = gen_data.obs[self.loader.split_column] == split
            real_data = real_data[split_mask_real]
            gen_data = gen_data[split_mask_gen]
        
        # Get gene expression values
        if gene is not None:
            if gene not in real_data.var_names:
                raise ValueError(f"Gene '{gene}' not found")
            gene_idx = list(real_data.var_names).index(gene)
        elif gene_index is not None:
            gene_idx = gene_index
            gene = real_data.var_names[gene_idx]
        else:
            gene_idx = 0
            gene = real_data.var_names[0]
        
        real_values = np.asarray(real_data.X[:, gene_idx]).flatten()
        gen_values = np.asarray(gen_data.X[:, gene_idx]).flatten()
        
        # Create figure
        fig = go.Figure()
        
        # Add real data histogram (density normalized)
        fig.add_trace(go.Histogram(
            x=real_values,
            nbinsx=nbins,
            name="Real",
            opacity=0.7,
            histnorm="probability density",
            marker_color=self.real_color,
        ))
        
        # Add generated data histogram
        fig.add_trace(go.Histogram(
            x=gen_values,
            nbinsx=nbins,
            name="Generated",
            opacity=0.7,
            histnorm="probability density",
            marker_color=self.generated_color,
        ))
        
        # Add rug plot if requested
        if show_rug:
            fig.add_trace(go.Scatter(
                x=real_values,
                y=np.zeros_like(real_values) - 0.02,
                mode='markers',
                marker=dict(symbol='line-ns', size=8, color=self.real_color),
                name='Real (rug)',
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=gen_values,
                y=np.zeros_like(gen_values) - 0.04,
                mode='markers',
                marker=dict(symbol='line-ns', size=8, color=self.generated_color),
                name='Generated (rug)',
                showlegend=False,
            ))
        
        # Layout
        plot_title = title or f"Expression Density: {gene}"
        if condition:
            plot_title += f" ({condition})"
        
        fig.update_layout(
            title=plot_title,
            xaxis_title="Expression Level",
            yaxis_title="Density",
            barmode="overlay",
            template="plotly_white",
            legend=dict(x=0.8, y=0.95),
        )
        
        return fig
    
    def density_grid(
        self,
        genes: Optional[List[str]] = None,
        n_genes: int = 9,
        condition: Optional[str] = None,
        split: Optional[str] = None,
        ncols: int = 3,
        nbins: int = 30,
    ) -> go.Figure:
        """
        Create grid of density overlay plots for multiple genes.
        
        Parameters
        ----------
        genes : List[str], optional
            Specific genes to plot. If None, uses top variable genes.
        n_genes : int
            Number of genes if genes not specified
        condition : str, optional
            Filter to specific condition
        split : str, optional
            Filter to specific split
        ncols : int
            Number of columns in grid
        nbins : int
            Number of histogram bins
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.loader is None:
            raise ValueError("Loader required for density plots")
        
        if not self.loader._is_loaded:
            self.loader.load()
            self.loader.align_genes()
        
        # Select genes
        if genes is None:
            # Use first n_genes (could also select most variable)
            genes = list(self.loader.real.var_names[:n_genes])
        
        n_genes = len(genes)
        nrows = (n_genes + ncols - 1) // ncols
        
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=genes,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )
        
        for i, gene in enumerate(genes):
            row = i // ncols + 1
            col = i % ncols + 1
            
            # Get data for this gene
            single_fig = self.density_overlay(
                gene=gene,
                condition=condition,
                split=split,
                nbins=nbins,
            )
            
            # Add traces to subplot
            for trace in single_fig.data:
                trace_copy = trace
                trace_copy.showlegend = (i == 0)  # Only show legend once
                fig.add_trace(trace_copy, row=row, col=col)
        
        fig.update_layout(
            height=300 * nrows,
            width=350 * ncols,
            title="Expression Density Comparison",
            barmode="overlay",
            template="plotly_white",
        )
        
        return fig
    
    # ==================== EMBEDDING PLOTS ====================
    
    def embedding_plot(
        self,
        method: str = "pca",
        color_by: Optional[str] = None,
        condition: Optional[str] = None,
        split: Optional[str] = None,
        n_components: int = 2,
        title: Optional[str] = None,
        point_size: int = 5,
        opacity: float = 0.7,
    ) -> go.Figure:
        """
        Create interactive embedding plot (PCA/UMAP) with metadata coloring.
        
        Parameters
        ----------
        method : str
            Dimensionality reduction method: 'pca' or 'umap'
        color_by : str, optional
            Metadata column to color by. If None, colors by real/generated.
        condition : str, optional
            Filter to specific condition
        split : str, optional  
            Filter to specific split
        n_components : int
            Number of dimensions (2 or 3)
        title : str, optional
            Custom plot title
        point_size : int
            Size of scatter points
        opacity : float
            Point opacity
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.loader is None:
            raise ValueError("Loader required for embedding plots")
        
        if not self.loader._is_loaded:
            self.loader.load()
            self.loader.align_genes()
        
        real_data = self.loader.real
        gen_data = self.loader.generated
        
        # Filter by condition if specified
        if condition is not None:
            condition_mask_real = self._get_condition_mask(real_data, condition)
            condition_mask_gen = self._get_condition_mask(gen_data, condition)
            real_data = real_data[condition_mask_real]
            gen_data = gen_data[condition_mask_gen]
        
        # Filter by split if specified
        if split is not None and self.loader.split_column:
            split_mask_real = real_data.obs[self.loader.split_column] == split
            split_mask_gen = gen_data.obs[self.loader.split_column] == split  
            real_data = real_data[split_mask_real]
            gen_data = gen_data[split_mask_gen]
        
        # Combine data
        X_real = np.asarray(real_data.X)
        X_gen = np.asarray(gen_data.X)
        X_combined = np.vstack([X_real, X_gen])
        
        # Compute embedding
        if method.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            embedding = reducer.fit_transform(X_combined)
            axis_prefix = "PC"
            variance_explained = reducer.explained_variance_ratio_
        elif method.lower() == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                embedding = reducer.fit_transform(X_combined)
                axis_prefix = "UMAP"
                variance_explained = None
            except ImportError:
                raise ImportError("umap-learn required for UMAP. Install with: pip install umap-learn")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'umap'.")
        
        # Create dataframe for plotting
        n_real = len(X_real)
        df = pd.DataFrame({
            f"{axis_prefix}1": embedding[:, 0],
            f"{axis_prefix}2": embedding[:, 1],
            "source": ["Real"] * n_real + ["Generated"] * (len(embedding) - n_real),
        })
        
        if n_components == 3:
            df[f"{axis_prefix}3"] = embedding[:, 2]
        
        # Add metadata columns
        if color_by is not None:
            real_meta = real_data.obs[color_by].values if color_by in real_data.obs.columns else ["N/A"] * n_real
            gen_meta = gen_data.obs[color_by].values if color_by in gen_data.obs.columns else ["N/A"] * len(X_gen)
            df[color_by] = list(real_meta) + list(gen_meta)
        
        # Add hover info
        hover_cols = ["source"]
        if color_by:
            hover_cols.append(color_by)
        
        # Create plot
        plot_title = title or f"{method.upper()} Embedding: Real vs Generated"
        
        if n_components == 2:
            if color_by:
                fig = px.scatter(
                    df, x=f"{axis_prefix}1", y=f"{axis_prefix}2",
                    color=color_by, symbol="source",
                    hover_data=hover_cols,
                    title=plot_title,
                    opacity=opacity,
                )
            else:
                fig = px.scatter(
                    df, x=f"{axis_prefix}1", y=f"{axis_prefix}2",
                    color="source",
                    color_discrete_map={"Real": self.real_color, "Generated": self.generated_color},
                    hover_data=hover_cols,
                    title=plot_title,
                    opacity=opacity,
                )
            
            # Update axis labels with variance if PCA
            if variance_explained is not None:
                fig.update_xaxes(title=f"{axis_prefix}1 ({variance_explained[0]:.1%})")
                fig.update_yaxes(title=f"{axis_prefix}2 ({variance_explained[1]:.1%})")
        else:
            # 3D plot
            if color_by:
                fig = px.scatter_3d(
                    df, x=f"{axis_prefix}1", y=f"{axis_prefix}2", z=f"{axis_prefix}3",
                    color=color_by, symbol="source",
                    hover_data=hover_cols,
                    title=plot_title,
                    opacity=opacity,
                )
            else:
                fig = px.scatter_3d(
                    df, x=f"{axis_prefix}1", y=f"{axis_prefix}2", z=f"{axis_prefix}3",
                    color="source",
                    color_discrete_map={"Real": self.real_color, "Generated": self.generated_color},
                    hover_data=hover_cols,
                    title=plot_title,
                    opacity=opacity,
                )
        
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(template="plotly_white")
        
        return fig
    
    # ==================== METRIC PLOTS ====================
    
    def metric_comparison(
        self,
        metrics: Optional[List[str]] = None,
        split: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive bar chart comparing metrics across conditions.
        
        Parameters
        ----------
        metrics : List[str], optional
            Metrics to include. If None, uses all available.
        split : str, optional
            Filter to specific split
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.results is None:
            raise ValueError("Results required for metric comparison")
        
        rows = []
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                for metric_name, metric_result in cond.metrics.items():
                    if metrics is None or metric_name in metrics:
                        rows.append({
                            "split": split_name,
                            "condition": cond.perturbation or cond_key,
                            "metric": metric_name,
                            "value": metric_result.aggregate_value,
                        })
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = px.bar(
            df,
            x="condition",
            y="value",
            color="metric",
            barmode="group",
            title="Metric Comparison by Condition",
            hover_data=["split", "condition", "metric", "value"],
        )
        
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Condition",
            yaxis_title="Metric Value",
            legend_title="Metric",
        )
        
        return fig
    
    def metric_heatmap(
        self,
        metric: str = "pearson",
        split: Optional[str] = None,
        n_genes: int = 50,
    ) -> go.Figure:
        """
        Create interactive heatmap of per-gene metric values.
        
        Parameters
        ----------
        metric : str
            Metric to visualize
        split : str, optional
            Filter to specific split
        n_genes : int
            Number of top genes to show
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.results is None:
            raise ValueError("Results required for heatmap")
        
        # Collect per-gene values
        data = {}
        for split_name, split_result in self.results.splits.items():
            if split is not None and split_name != split:
                continue
            
            for cond_key, cond in split_result.conditions.items():
                if metric in cond.metrics:
                    metric_result = cond.metrics[metric]
                    label = cond.perturbation or cond_key
                    data[label] = metric_result.per_gene_values[:n_genes]
        
        if not data:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Get gene names
        first_result = next(iter(self.results.splits.values()))
        first_cond = next(iter(first_result.conditions.values()))
        gene_names = first_cond.gene_names[:n_genes] if first_cond.gene_names else [f"Gene_{i}" for i in range(n_genes)]
        
        # Create matrix
        conditions = list(data.keys())
        matrix = np.array([data[c] for c in conditions])
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=gene_names,
            y=conditions,
            colorscale="RdBu_r" if metric in ["pearson", "spearman"] else "Viridis",
            text=np.round(matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="Gene: %{x}<br>Condition: %{y}<br>Value: %{z:.3f}<extra></extra>",
        ))
        
        fig.update_layout(
            title=f"Per-Gene {metric.title()} Values",
            xaxis_title="Gene",
            yaxis_title="Condition",
            template="plotly_white",
            height=100 + 30 * len(conditions),
        )
        
        return fig
    
    def scatter_real_vs_generated(
        self,
        gene: Optional[str] = None,
        gene_index: Optional[int] = None,
        condition: Optional[str] = None,
        split: Optional[str] = None,
        color_by: Optional[str] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create scatter plot of real vs generated mean expression.
        
        Parameters
        ----------
        gene : str, optional
            Specific gene (if None, plots all genes)
        gene_index : int, optional
            Gene index
        condition : str, optional
            Filter to specific condition
        split : str, optional
            Filter to specific split
        color_by : str, optional
            Metadata column to color by (for sample-level plots)
        title : str, optional
            Custom title
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        if self.loader is None:
            raise ValueError("Loader required for scatter plots")
        
        if not self.loader._is_loaded:
            self.loader.load()
            self.loader.align_genes()
        
        real_data = self.loader.real
        gen_data = self.loader.generated
        
        # Filter by condition
        if condition is not None:
            condition_mask_real = self._get_condition_mask(real_data, condition)
            condition_mask_gen = self._get_condition_mask(gen_data, condition)
            real_data = real_data[condition_mask_real]
            gen_data = gen_data[condition_mask_gen]
        
        # Filter by split
        if split is not None and self.loader.split_column:
            split_mask_real = real_data.obs[self.loader.split_column] == split
            split_mask_gen = gen_data.obs[self.loader.split_column] == split
            real_data = real_data[split_mask_real]
            gen_data = gen_data[split_mask_gen]
        
        # If single gene, plot sample-level
        if gene is not None or gene_index is not None:
            if gene is not None:
                gene_idx = list(real_data.var_names).index(gene)
            else:
                gene_idx = gene_index
                gene = real_data.var_names[gene_idx]
            
            real_vals = np.asarray(real_data.X[:, gene_idx]).flatten()
            gen_vals = np.asarray(gen_data.X[:, gene_idx]).flatten()
            
            # Match samples (use means for comparison)
            df = pd.DataFrame({
                "Real Mean": [real_vals.mean()],
                "Generated Mean": [gen_vals.mean()],
            })
            
            plot_title = title or f"Expression: {gene}"
        else:
            # Plot gene-level means
            real_means = np.asarray(real_data.X.mean(axis=0)).flatten()
            gen_means = np.asarray(gen_data.X.mean(axis=0)).flatten()
            
            df = pd.DataFrame({
                "Gene": list(real_data.var_names),
                "Real Mean": real_means,
                "Generated Mean": gen_means,
            })
            
            plot_title = title or "Mean Expression: Real vs Generated"
            
            fig = px.scatter(
                df,
                x="Real Mean",
                y="Generated Mean",
                hover_data=["Gene"],
                title=plot_title,
            )
            
            # Add diagonal line
            max_val = max(df["Real Mean"].max(), df["Generated Mean"].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='y=x',
                showlegend=True,
            ))
            
            fig.update_layout(template="plotly_white")
            return fig
        
        # For single gene, show distributions
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=real_vals, name="Real", opacity=0.7))
        fig.add_trace(go.Histogram(x=gen_vals, name="Generated", opacity=0.7))
        fig.update_layout(
            title=plot_title,
            xaxis_title="Expression",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_white",
        )
        return fig
    
    # ==================== HELPER METHODS ====================
    
    def _get_condition_mask(self, adata: "ad.AnnData", condition_value: str) -> np.ndarray:
        """Get boolean mask for samples matching a condition."""
        # Try exact match first
        for col in self.loader.condition_columns:
            if col in adata.obs.columns:
                mask = adata.obs[col] == condition_value
                if mask.sum() > 0:
                    return mask
        
        # Try partial match on combined condition
        combined = adata.obs[self.loader.condition_columns].astype(str).agg("_".join, axis=1)
        mask = combined == condition_value
        if mask.sum() > 0:
            return mask
        
        # Return all if no match
        warnings.warn(f"Condition '{condition_value}' not found, using all samples")
        return np.ones(len(adata), dtype=bool)
    
    def save_html(
        self,
        fig: go.Figure,
        path: Union[str, Path],
        include_plotlyjs: bool = True,
    ) -> None:
        """
        Save interactive plot to HTML file.
        
        Parameters
        ----------
        fig : go.Figure
            Plotly figure to save
        path : str or Path
            Output file path
        include_plotlyjs : bool
            Whether to include plotly.js in the HTML
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=include_plotlyjs)
    
    def generate_all_interactive(
        self,
        output_dir: Union[str, Path],
        include_density: bool = True,
        include_embedding: bool = True,
        include_metrics: bool = True,
        n_genes_density: int = 9,
    ) -> Dict[str, Path]:
        """
        Generate all interactive plots and save to HTML files.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory
        include_density : bool
            Generate density overlay plots
        include_embedding : bool
            Generate embedding plots
        include_metrics : bool
            Generate metric comparison plots
        n_genes_density : int
            Number of genes for density grid
            
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved = {}
        
        if include_density and self.loader is not None:
            try:
                fig = self.density_grid(n_genes=n_genes_density)
                path = output_dir / "density_grid.html"
                self.save_html(fig, path)
                saved["density_grid"] = path
            except Exception as e:
                warnings.warn(f"Could not generate density grid: {e}")
        
        if include_embedding and self.loader is not None:
            try:
                fig = self.embedding_plot(method="pca")
                path = output_dir / "embedding_pca.html"
                self.save_html(fig, path)
                saved["embedding_pca"] = path
            except Exception as e:
                warnings.warn(f"Could not generate PCA embedding: {e}")
            
            try:
                fig = self.embedding_plot(method="pca", n_components=3)
                path = output_dir / "embedding_pca_3d.html"
                self.save_html(fig, path)
                saved["embedding_pca_3d"] = path
            except Exception as e:
                warnings.warn(f"Could not generate 3D PCA embedding: {e}")
        
        if include_metrics and self.results is not None:
            try:
                fig = self.metric_comparison()
                path = output_dir / "metric_comparison.html"
                self.save_html(fig, path)
                saved["metric_comparison"] = path
            except Exception as e:
                warnings.warn(f"Could not generate metric comparison: {e}")
            
            try:
                fig = self.metric_heatmap()
                path = output_dir / "metric_heatmap.html"
                self.save_html(fig, path)
                saved["metric_heatmap"] = path
            except Exception as e:
                warnings.warn(f"Could not generate metric heatmap: {e}")
        
        if self.loader is not None:
            try:
                fig = self.scatter_real_vs_generated()
                path = output_dir / "scatter_expression.html"
                self.save_html(fig, path)
                saved["scatter_expression"] = path
            except Exception as e:
                warnings.warn(f"Could not generate scatter plot: {e}")
        
        return saved


# ==================== CONVENIENCE FUNCTIONS ====================

def density_overlay(
    real_data: "ad.AnnData",
    generated_data: "ad.AnnData",
    gene: str,
    **kwargs,
) -> go.Figure:
    """
    Quick density overlay plot for a single gene.
    
    Parameters
    ----------
    real_data : AnnData
        Real expression data
    generated_data : AnnData
        Generated expression data
    gene : str
        Gene name to plot
    **kwargs
        Additional arguments for density_overlay
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    _check_plotly()
    
    from gge.data.loader import GeneExpressionDataLoader
    
    # Create temporary loader
    loader = GeneExpressionDataLoader(
        real_data=real_data,
        generated_data=generated_data,
        condition_columns=list(real_data.obs.columns[:1]),  # Use first column
    )
    loader.load()
    loader.align_genes()
    
    viz = InteractiveVisualizer(loader=loader)
    return viz.density_overlay(gene=gene, **kwargs)


def embedding_interactive(
    real_data: "ad.AnnData",
    generated_data: "ad.AnnData",
    method: str = "pca",
    color_by: Optional[str] = None,
    **kwargs,
) -> go.Figure:
    """
    Quick interactive embedding plot.
    
    Parameters
    ----------
    real_data : AnnData
        Real expression data
    generated_data : AnnData
        Generated expression data
    method : str
        'pca' or 'umap'
    color_by : str, optional
        Metadata column to color by
    **kwargs
        Additional arguments for embedding_plot
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    _check_plotly()
    
    from gge.data.loader import GeneExpressionDataLoader
    
    # Create temporary loader
    loader = GeneExpressionDataLoader(
        real_data=real_data,
        generated_data=generated_data,
        condition_columns=list(real_data.obs.columns[:1]),
    )
    loader.load()
    loader.align_genes()
    
    viz = InteractiveVisualizer(loader=loader)
    return viz.embedding_plot(method=method, color_by=color_by, **kwargs)
