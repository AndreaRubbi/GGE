"""
PC-Space (Principal Component) utilities for GGE.

Provides functionality for:
- Computing PCA on gene expression data
- Evaluating models in PC space for lower-dimensional comparison
- Computing PC-space metrics (Wasserstein, MMD, etc.)

Reference: GGE Paper Section 3.3 "The Space Question: A Theoretical Analysis"

The key insight is that comparing data in PC space captures global structure
while being computationally more tractable for optimal transport metrics.
"""
from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import warnings
import numpy as np

try:
    import anndata as ad
except ImportError:
    ad = None  # type: ignore


def compute_pca(
    adata: "ad.AnnData",
    n_components: int = 50,
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    copy: bool = True,
) -> "ad.AnnData":
    """
    Compute PCA on AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data
    n_components : int
        Number of principal components to compute
    use_highly_variable : bool
        If True and n_vars > n_top_genes, filter to highly variable genes first
    n_top_genes : int
        Number of highly variable genes to select if use_highly_variable=True
    copy : bool
        If True, return a copy; otherwise modify in place
        
    Returns
    -------
    AnnData
        AnnData with PCA in adata.obsm['X_pca']
        
    Examples
    --------
    >>> adata_pca = compute_pca(adata, n_components=50)
    >>> pc_coords = adata_pca.obsm['X_pca']
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for PCA computation: pip install scanpy")
    
    if copy:
        adata = adata.copy()
    
    # Filter to highly variable genes if needed
    if use_highly_variable and adata.n_vars > n_top_genes:
        sc.pp.highly_variable_genes(
            adata, 
            n_top_genes=n_top_genes, 
            flavor='seurat_v3',
            subset=True
        )
    
    # Compute PCA
    n_comps = min(n_components, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comps)
    
    return adata


def get_pc_coordinates(
    adata: "ad.AnnData",
    n_components: Optional[int] = None,
) -> np.ndarray:
    """
    Get PC coordinates from AnnData.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with PCA computed (in obsm['X_pca'])
    n_components : int, optional
        Number of components to return. If None, returns all.
        
    Returns
    -------
    np.ndarray
        PC coordinates of shape (n_samples, n_components)
        
    Raises
    ------
    ValueError
        If PCA has not been computed
    """
    if 'X_pca' not in adata.obsm:
        raise ValueError("PCA not found. Run compute_pca or sc.tl.pca first.")
    
    pca = adata.obsm['X_pca']
    
    if n_components is not None:
        return pca[:, :n_components]
    return pca


def project_to_pc_space(
    adata: "ad.AnnData",
    reference_adata: "ad.AnnData",
    n_components: Optional[int] = None,
) -> np.ndarray:
    """
    Project data onto reference PC space.
    
    Uses the PCA loadings from reference_adata to project adata.
    Useful for comparing generated data to real data in a consistent PC space.
    
    Parameters
    ----------
    adata : AnnData
        Data to project
    reference_adata : AnnData
        Reference data with computed PCA (provides loadings)
    n_components : int, optional
        Number of components to use
        
    Returns
    -------
    np.ndarray
        Projected coordinates
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for PCA projection: pip install scanpy")
    
    if 'X_pca' not in reference_adata.obsm:
        raise ValueError("Reference AnnData must have PCA computed")
    
    if 'pca' not in reference_adata.varm:
        raise ValueError("Reference AnnData must have PCA loadings in varm['pca']")
    
    # Get loadings from reference
    loadings = reference_adata.varm['pca']
    
    if n_components is not None:
        loadings = loadings[:, :n_components]
    
    # Find common genes
    common_genes = adata.var_names.intersection(reference_adata.var_names)
    if len(common_genes) == 0:
        raise ValueError("No common genes between adata and reference")
    
    # Get data matrix for common genes (matching reference order)
    ref_gene_order = reference_adata.var_names[reference_adata.var_names.isin(common_genes)]
    X = adata[:, ref_gene_order].X
    
    if hasattr(X, 'toarray'):  # sparse matrix
        X = X.toarray()
    
    # Center using reference mean if available
    if 'mean' in reference_adata.var:
        mean = reference_adata.var.loc[ref_gene_order, 'mean'].values
        X = X - mean
    
    # Project
    projected = X @ loadings[reference_adata.var_names.isin(common_genes)]
    
    return projected


class PCSpaceEvaluator:
    """
    Evaluator for PC-space metrics.
    
    Computes metrics in reduced PC space for more efficient comparison
    of global structure between real and generated data.
    
    Parameters
    ----------
    n_components : int
        Number of principal components to use
    use_highly_variable : bool
        Filter to highly variable genes before PCA
    n_top_genes : int
        Number of HVGs if use_highly_variable=True
        
    Examples
    --------
    >>> evaluator = PCSpaceEvaluator(n_components=50)
    >>> results = evaluator.evaluate(
    ...     real_adata, 
    ...     generated_adata,
    ...     condition_columns=["perturbation"]
    ... )
    """
    
    def __init__(
        self,
        n_components: int = 50,
        use_highly_variable: bool = True,
        n_top_genes: int = 2000,
    ):
        self.n_components = n_components
        self.use_highly_variable = use_highly_variable
        self.n_top_genes = n_top_genes
    
    def transform_to_pc_space(
        self,
        real_data: "ad.AnnData",
        generated_data: "ad.AnnData",
    ) -> Tuple["ad.AnnData", "ad.AnnData"]:
        """
        Transform both datasets to PC space.
        
        Computes PCA on real data and projects both datasets.
        
        Parameters
        ----------
        real_data : AnnData
            Real/reference data
        generated_data : AnnData
            Generated data
            
        Returns
        -------
        tuple of AnnData
            (real_pc, generated_pc) with PC coordinates in obsm['X_pca']
        """
        # Compute PCA on real data
        real_pca = compute_pca(
            real_data,
            n_components=self.n_components,
            use_highly_variable=self.use_highly_variable,
            n_top_genes=self.n_top_genes,
            copy=True,
        )
        
        # Project generated data onto same PC space
        try:
            gen_projected = project_to_pc_space(
                generated_data,
                real_pca,
                n_components=self.n_components,
            )
            
            generated_pca = generated_data.copy()
            generated_pca.obsm['X_pca'] = gen_projected
        except (ValueError, KeyError):
            # Fallback: compute PCA separately on generated data
            # (less ideal but still useful)
            warnings.warn(
                "Could not project to reference PC space. "
                "Computing PCA separately on generated data."
            )
            generated_pca = compute_pca(
                generated_data,
                n_components=self.n_components,
                use_highly_variable=self.use_highly_variable,
                n_top_genes=self.n_top_genes,
                copy=True,
            )
        
        return real_pca, generated_pca
    
    def evaluate(
        self,
        real_data: "ad.AnnData",
        generated_data: "ad.AnnData",
        condition_columns: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> "EvaluationResult":
        """
        Evaluate in PC space.
        
        Parameters
        ----------
        real_data : AnnData
            Real/reference data
        generated_data : AnnData
            Generated data
        condition_columns : list of str, optional
            Columns for condition matching
        metrics : list of str, optional
            Metrics to compute. Defaults to Wasserstein and MMD.
        verbose : bool
            Print progress messages
            
        Returns
        -------
        EvaluationResult
            Evaluation results in PC space
        """
        from ..evaluator import evaluate, GeneEvalEvaluator, MetricRegistry
        
        if verbose:
            print(f"Transforming data to {self.n_components}-component PC space...")
        
        real_pc, gen_pc = self.transform_to_pc_space(real_data, generated_data)
        
        # Create AnnData objects from PC coordinates
        import anndata as ad
        
        # Use PC coordinates as the expression matrix
        real_pc_adata = ad.AnnData(X=real_pc.obsm['X_pca'])
        real_pc_adata.obs = real_pc.obs.copy()
        real_pc_adata.var_names = [f"PC{i+1}" for i in range(real_pc.obsm['X_pca'].shape[1])]
        
        gen_pc_adata = ad.AnnData(X=gen_pc.obsm['X_pca'])
        gen_pc_adata.obs = gen_pc.obs.copy()
        gen_pc_adata.var_names = [f"PC{i+1}" for i in range(gen_pc.obsm['X_pca'].shape[1])]
        
        # Default to multivariate metrics for PC space
        if metrics is None:
            metrics = ["wasserstein_1", "wasserstein_2", "mmd", "energy"]
        
        # Convert string metric names to metric classes
        metric_classes = []
        for m in metrics:
            if isinstance(m, str):
                metric_class = MetricRegistry.get(m)
                if metric_class is None:
                    warnings.warn(f"Unknown metric: {m}")
                else:
                    metric_classes.append(metric_class)
            else:
                metric_classes.append(m)
        
        if verbose:
            print(f"Evaluating with metrics: {metrics}")
        
        # Evaluate
        results = evaluate(
            real_data=real_pc_adata,
            generated_data=gen_pc_adata,
            condition_columns=condition_columns,
            metrics=metric_classes if metric_classes else None,
            verbose=verbose,
        )
        
        return results


def evaluate_pc_space(
    real_data: Union["ad.AnnData", str, Path],
    generated_data: Union["ad.AnnData", str, Path],
    condition_columns: Optional[List[str]] = None,
    n_components: int = 50,
    use_highly_variable: bool = True,
    n_top_genes: int = 2000,
    metrics: Optional[List[str]] = None,
    verbose: bool = True,
) -> "EvaluationResult":
    """
    Evaluate generative model in PC space.
    
    Convenience function that transforms data to PC space and evaluates.
    
    Parameters
    ----------
    real_data : AnnData or str or Path
        Real/reference data or path to h5ad file
    generated_data : AnnData or str or Path
        Generated data or path to h5ad file
    condition_columns : list of str, optional
        Columns for condition matching
    n_components : int
        Number of principal components
    use_highly_variable : bool
        Filter to HVGs before PCA
    n_top_genes : int
        Number of HVGs to use
    metrics : list of str, optional
        Metrics to compute
    verbose : bool
        Print progress
        
    Returns
    -------
    EvaluationResult
        Evaluation results
        
    Examples
    --------
    >>> results = evaluate_pc_space(
    ...     real_data="real.h5ad",
    ...     generated_data="gen.h5ad",
    ...     condition_columns=["perturbation"],
    ...     n_components=50,
    ... )
    >>> print(results.summary())
    """
    from ..data.loader import load_data
    
    # Load data if paths provided
    if isinstance(real_data, (str, Path)):
        real_data = load_data(real_data)
    if isinstance(generated_data, (str, Path)):
        generated_data = load_data(generated_data)
    
    evaluator = PCSpaceEvaluator(
        n_components=n_components,
        use_highly_variable=use_highly_variable,
        n_top_genes=n_top_genes,
    )
    
    return evaluator.evaluate(
        real_data=real_data,
        generated_data=generated_data,
        condition_columns=condition_columns,
        metrics=metrics,
        verbose=verbose,
    )


def compute_pc_variance_explained(
    adata: "ad.AnnData",
    n_components: int = 50,
) -> Dict[str, float]:
    """
    Compute variance explained by principal components.
    
    Parameters
    ----------
    adata : AnnData
        Data with PCA computed
    n_components : int
        Number of components to report
        
    Returns
    -------
    dict
        {"per_component": array, "cumulative": array, "total": float}
    """
    if 'pca' not in adata.uns:
        raise ValueError("PCA not computed. Run compute_pca first.")
    
    var_ratio = adata.uns['pca']['variance_ratio']
    n = min(n_components, len(var_ratio))
    
    return {
        "per_component": var_ratio[:n],
        "cumulative": np.cumsum(var_ratio[:n]),
        "total": float(np.sum(var_ratio[:n])),
    }
