"""
Differentially Expressed Gene (DEG) utilities for GGE.

Provides functionality for:
- Identifying DEGs between conditions (perturbation vs control)
- Filtering data to DEG space for targeted evaluation
- Computing perturbation effects

Reference: GGE Paper Section 4.3 "DEG-Space Evaluation with Perturbation Effects"

The key insight is that evaluating models only on DEGs (genes that respond to
perturbations) provides a more targeted assessment of how well generative models
capture actual biological effects, rather than overall gene expression patterns.
"""
from __future__ import annotations

from typing import Optional, List, Union, Dict, Tuple, Set
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

try:
    import anndata as ad
except ImportError:
    ad = None  # type: ignore


def identify_degs(
    adata: "ad.AnnData",
    condition_column: str,
    control_value: str,
    treatment_value: Optional[str] = None,
    log2fc_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    method: str = "ttest",
    use_fdr: bool = True,
) -> pd.DataFrame:
    """
    Identify differentially expressed genes between conditions.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data with condition annotations
    condition_column : str
        Column in adata.obs containing condition labels
    control_value : str
        Value identifying control/baseline samples
    treatment_value : str, optional
        Value identifying treatment samples. If None, compares all non-control
        samples to control.
    log2fc_threshold : float
        Minimum absolute log2 fold change to consider a gene differentially expressed
    pvalue_threshold : float
        Maximum p-value (or FDR) threshold
    method : str
        Statistical test: 'ttest' (Welch's t-test) or 'wilcoxon' (rank-sum test)
    use_fdr : bool
        Apply Benjamini-Hochberg FDR correction
        
    Returns
    -------
    pd.DataFrame
        DEG results with columns: gene, log2fc, pvalue, fdr, is_deg
        
    Examples
    --------
    >>> degs = identify_degs(
    ...     adata, 
    ...     condition_column="perturbation",
    ...     control_value="control",
    ...     treatment_value="drug_A",
    ...     log2fc_threshold=1.0
    ... )
    >>> deg_genes = degs[degs['is_deg']]['gene'].tolist()
    """
    if condition_column not in adata.obs.columns:
        raise ValueError(f"Condition column '{condition_column}' not found in adata.obs")
    
    # Get control and treatment samples
    control_mask = adata.obs[condition_column] == control_value
    
    if treatment_value is not None:
        treatment_mask = adata.obs[condition_column] == treatment_value
    else:
        treatment_mask = ~control_mask
    
    if control_mask.sum() == 0:
        raise ValueError(f"No samples found with {condition_column}='{control_value}'")
    if treatment_mask.sum() == 0:
        raise ValueError(f"No treatment samples found")
    
    # Extract expression matrices
    X_control = np.asarray(adata[control_mask].X)
    X_treatment = np.asarray(adata[treatment_mask].X)
    
    # Handle sparse matrices
    if hasattr(X_control, 'toarray'):
        X_control = X_control.toarray()
    if hasattr(X_treatment, 'toarray'):
        X_treatment = X_treatment.toarray()
    
    n_genes = X_control.shape[1]
    gene_names = list(adata.var_names)
    
    # Compute log2 fold change (mean treatment / mean control)
    # Add small pseudocount to avoid log(0)
    epsilon = 1e-10
    mean_control = X_control.mean(axis=0) + epsilon
    mean_treatment = X_treatment.mean(axis=0) + epsilon
    log2fc = np.log2(mean_treatment / mean_control)
    
    # Compute p-values
    pvalues = np.zeros(n_genes)
    
    for i in range(n_genes):
        ctrl_vals = X_control[:, i]
        treat_vals = X_treatment[:, i]
        
        # Skip if no variance
        if np.std(ctrl_vals) == 0 and np.std(treat_vals) == 0:
            pvalues[i] = 1.0
            continue
        
        if method == "ttest":
            # Welch's t-test (unequal variances)
            _, pval = stats.ttest_ind(treat_vals, ctrl_vals, equal_var=False)
        elif method == "wilcoxon":
            # Mann-Whitney U / Wilcoxon rank-sum test
            _, pval = stats.mannwhitneyu(treat_vals, ctrl_vals, alternative='two-sided')
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ttest' or 'wilcoxon'.")
        
        pvalues[i] = pval if not np.isnan(pval) else 1.0
    
    # FDR correction
    if use_fdr:
        fdr = _benjamini_hochberg(pvalues)
        significance = fdr
    else:
        fdr = pvalues  # No correction
        significance = pvalues
    
    # Determine DEGs
    is_deg = (np.abs(log2fc) >= log2fc_threshold) & (significance <= pvalue_threshold)
    
    results = pd.DataFrame({
        'gene': gene_names,
        'log2fc': log2fc,
        'pvalue': pvalues,
        'fdr': fdr,
        'is_deg': is_deg,
        'mean_control': mean_control - epsilon,
        'mean_treatment': mean_treatment - epsilon,
    })
    
    return results.sort_values('pvalue')


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    
    # BH adjustment
    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        adjusted[i] = min(cummin, sorted_pvals[i] * n / (i + 1))
        cummin = adjusted[i]
    
    # Restore original order
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result


def get_deg_mask(
    adata: "ad.AnnData",
    deg_genes: List[str],
) -> np.ndarray:
    """
    Get boolean mask for DEG genes in an AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data
    deg_genes : List[str]
        List of DEG gene names
        
    Returns
    -------
    np.ndarray
        Boolean mask for genes in adata.var_names that are DEGs
    """
    gene_set = set(deg_genes)
    return np.array([g in gene_set for g in adata.var_names])


def filter_to_degs(
    adata: "ad.AnnData",
    deg_genes: List[str],
) -> "ad.AnnData":
    """
    Filter AnnData to only include DEG genes.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data
    deg_genes : List[str]
        List of DEG gene names
        
    Returns
    -------
    AnnData
        Filtered AnnData with only DEG genes
    """
    mask = get_deg_mask(adata, deg_genes)
    return adata[:, mask].copy()


def compute_perturbation_effects(
    adata: "ad.AnnData",
    condition_column: str,
    control_value: str,
    genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute perturbation effects (fold changes) for each condition vs control.
    
    Parameters
    ----------
    adata : AnnData
        Gene expression data
    condition_column : str
        Column containing condition labels
    control_value : str
        Value identifying control samples
    genes : List[str], optional
        Subset of genes to compute effects for. If None, uses all genes.
        
    Returns
    -------
    pd.DataFrame
        Matrix of log2 fold changes: rows=genes, columns=conditions
    """
    if genes is not None:
        mask = get_deg_mask(adata, genes)
        adata = adata[:, mask]
    
    gene_names = list(adata.var_names)
    conditions = [c for c in adata.obs[condition_column].unique() if c != control_value]
    
    # Control mean
    control_mask = adata.obs[condition_column] == control_value
    X_control = np.asarray(adata[control_mask].X)
    if hasattr(X_control, 'toarray'):
        X_control = X_control.toarray()
    mean_control = X_control.mean(axis=0) + 1e-10
    
    # Compute fold changes for each condition
    fc_data = {}
    for cond in conditions:
        cond_mask = adata.obs[condition_column] == cond
        X_cond = np.asarray(adata[cond_mask].X)
        if hasattr(X_cond, 'toarray'):
            X_cond = X_cond.toarray()
        mean_cond = X_cond.mean(axis=0) + 1e-10
        fc_data[cond] = np.log2(mean_cond / mean_control)
    
    return pd.DataFrame(fc_data, index=gene_names)


class DEGSpaceEvaluator:
    """
    Evaluator that operates in DEG space.
    
    This class wraps the standard evaluation but filters data to only
    differentially expressed genes, providing targeted assessment of
    how well models capture perturbation effects.
    
    Parameters
    ----------
    deg_genes : List[str]
        List of DEG gene names to filter to
    metrics : List, optional
        Metrics to compute (default: all correlation and distance metrics)
        
    Examples
    --------
    >>> # First identify DEGs from real data
    >>> degs = identify_degs(real_adata, "perturbation", "control")
    >>> deg_genes = degs[degs['is_deg']]['gene'].tolist()
    >>> 
    >>> # Evaluate in DEG space
    >>> deg_evaluator = DEGSpaceEvaluator(deg_genes)
    >>> results = deg_evaluator.evaluate(real_adata, gen_adata, ["perturbation"])
    """
    
    def __init__(
        self,
        deg_genes: List[str],
        metrics: Optional[list] = None,
    ):
        self.deg_genes = deg_genes
        self.metrics = metrics
    
    def evaluate(
        self,
        real_data: Union[str, Path, "ad.AnnData"],
        generated_data: Union[str, Path, "ad.AnnData"],
        condition_columns: List[str],
        split_column: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Evaluate in DEG space.
        
        Filters both real and generated data to DEG genes before evaluation.
        
        Parameters
        ----------
        real_data : str, Path, or AnnData
            Real gene expression data
        generated_data : str, Path, or AnnData
            Generated gene expression data
        condition_columns : List[str]
            Columns for condition matching
        split_column : str, optional
            Column for train/test split
        output_dir : str or Path, optional
            Directory to save results
        verbose : bool
            Print progress
        **kwargs
            Additional arguments for evaluate()
            
        Returns
        -------
        EvaluationResult
            Evaluation results computed in DEG space
        """
        from gge.evaluator import evaluate
        
        # Load and filter to DEGs
        if isinstance(real_data, (str, Path)):
            import scanpy as sc
            real_adata = sc.read_h5ad(real_data)
        else:
            real_adata = real_data
        
        if isinstance(generated_data, (str, Path)):
            import scanpy as sc
            gen_adata = sc.read_h5ad(generated_data)
        else:
            gen_adata = generated_data
        
        # Filter to DEG genes present in both datasets
        common_degs = [g for g in self.deg_genes 
                       if g in real_adata.var_names and g in gen_adata.var_names]
        
        if len(common_degs) == 0:
            raise ValueError("No DEG genes found in both datasets")
        
        if verbose:
            print(f"Evaluating in DEG space: {len(common_degs)} genes "
                  f"(of {len(self.deg_genes)} total DEGs)")
        
        real_filtered = filter_to_degs(real_adata, common_degs)
        gen_filtered = filter_to_degs(gen_adata, common_degs)
        
        # Run evaluation
        return evaluate(
            real_data=real_filtered,
            generated_data=gen_filtered,
            condition_columns=condition_columns,
            split_column=split_column,
            output_dir=output_dir,
            metrics=self.metrics,
            verbose=verbose,
            **kwargs,
        )


def evaluate_deg_space(
    real_data: Union[str, Path, "ad.AnnData"],
    generated_data: Union[str, Path, "ad.AnnData"],
    condition_columns: List[str],
    deg_condition_column: str,
    control_value: str,
    treatment_value: Optional[str] = None,
    log2fc_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    split_column: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    metrics: Optional[list] = None,
    verbose: bool = True,
    return_degs: bool = False,
    **kwargs,
):
    """
    Convenience function for DEG-space evaluation.
    
    Identifies DEGs from real data and evaluates both datasets in that space.
    
    Parameters
    ----------
    real_data : str, Path, or AnnData
        Real gene expression data
    generated_data : str, Path, or AnnData
        Generated gene expression data
    condition_columns : List[str]
        Columns for condition matching
    deg_condition_column : str
        Column containing perturbation labels for DEG identification
    control_value : str
        Value identifying control samples
    treatment_value : str, optional
        Value identifying treatment samples (if None, all non-control)
    log2fc_threshold : float
        Minimum absolute log2 fold change for DEGs
    pvalue_threshold : float
        Maximum p-value/FDR for DEGs
    split_column : str, optional
        Column for train/test split
    output_dir : str or Path, optional
        Directory to save results
    metrics : List, optional
        Metrics to compute
    verbose : bool
        Print progress
    return_degs : bool
        If True, return (results, deg_df) tuple
    **kwargs
        Additional arguments for evaluate()
        
    Returns
    -------
    EvaluationResult or Tuple[EvaluationResult, pd.DataFrame]
        Evaluation results (and optionally DEG identification results)
        
    Examples
    --------
    >>> results = evaluate_deg_space(
    ...     "real.h5ad",
    ...     "generated.h5ad",
    ...     condition_columns=["perturbation", "cell_type"],
    ...     deg_condition_column="perturbation",
    ...     control_value="control",
    ...     log2fc_threshold=1.0,
    ... )
    """
    # Load real data for DEG identification
    if isinstance(real_data, (str, Path)):
        import scanpy as sc
        real_adata = sc.read_h5ad(real_data)
    else:
        real_adata = real_data
    
    if verbose:
        print("Identifying differentially expressed genes...")
    
    # Identify DEGs
    deg_df = identify_degs(
        real_adata,
        condition_column=deg_condition_column,
        control_value=control_value,
        treatment_value=treatment_value,
        log2fc_threshold=log2fc_threshold,
        pvalue_threshold=pvalue_threshold,
    )
    
    deg_genes = deg_df[deg_df['is_deg']]['gene'].tolist()
    
    if verbose:
        print(f"Found {len(deg_genes)} DEGs ({len(deg_genes)/len(real_adata.var_names)*100:.1f}%)")
    
    if len(deg_genes) == 0:
        warnings.warn("No DEGs found with specified thresholds. Try relaxing thresholds.")
        if return_degs:
            return None, deg_df
        return None
    
    # Evaluate in DEG space
    evaluator = DEGSpaceEvaluator(deg_genes, metrics=metrics)
    results = evaluator.evaluate(
        real_data=real_data,
        generated_data=generated_data,
        condition_columns=condition_columns,
        split_column=split_column,
        output_dir=output_dir,
        verbose=verbose,
        **kwargs,
    )
    
    if return_degs:
        return results, deg_df
    return results
