"""
Correlation metrics for gene expression evaluation.

Provides Pearson and Spearman correlation metrics with per-gene computation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Optional

from .base_metric import CorrelationMetric


class PearsonCorrelation(CorrelationMetric):
    """
    Pearson correlation coefficient between real and generated gene expression.
    
    Computed per gene by correlating expression values across samples.
    Higher values (closer to 1) indicate better agreement.
    """
    
    def __init__(self):
        super().__init__(
            name="pearson",
            description="Pearson correlation coefficient (per gene across samples)"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Pearson correlation for each gene.
        
        For each gene, correlates expression values between:
        - Mean expression across real samples
        - Mean expression across generated samples
        
        Or if sample sizes match, computes correlation across paired samples.
        
        Parameters
        ----------
        real : np.ndarray
            Real data, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data, shape (n_samples_gen, n_genes)
            
        Returns
        -------
        np.ndarray
            Pearson correlation per gene
        """
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        correlations = np.zeros(n_genes)
        
        # If sample sizes match, compute correlation across samples
        if real.shape[0] == generated.shape[0]:
            for i in range(n_genes):
                r_vals = real[:, i]
                g_vals = generated[:, i]
                
                # Skip if constant values
                if np.std(r_vals) == 0 or np.std(g_vals) == 0:
                    correlations[i] = np.nan
                    continue
                    
                corr, _ = pearsonr(r_vals, g_vals)
                correlations[i] = corr
        else:
            # Use mean profiles when sample sizes differ
            real_mean = real.mean(axis=0)
            gen_mean = generated.mean(axis=0)
            
            # Compute single overall correlation
            if np.std(real_mean) == 0 or np.std(gen_mean) == 0:
                return np.full(n_genes, np.nan)
            
            overall_corr, _ = pearsonr(real_mean, gen_mean)
            # Return same value for all genes (overall correlation)
            correlations[:] = overall_corr
        
        return correlations


class SpearmanCorrelation(CorrelationMetric):
    """
    Spearman rank correlation between real and generated gene expression.
    
    More robust to outliers than Pearson. Measures monotonic relationship.
    """
    
    def __init__(self):
        super().__init__(
            name="spearman",
            description="Spearman rank correlation coefficient"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Spearman correlation for each gene.
        
        Parameters
        ----------
        real : np.ndarray
            Real data, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data, shape (n_samples_gen, n_genes)
            
        Returns
        -------
        np.ndarray
            Spearman correlation per gene
        """
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        correlations = np.zeros(n_genes)
        
        if real.shape[0] == generated.shape[0]:
            for i in range(n_genes):
                r_vals = real[:, i]
                g_vals = generated[:, i]
                
                if np.std(r_vals) == 0 or np.std(g_vals) == 0:
                    correlations[i] = np.nan
                    continue
                    
                corr, _ = spearmanr(r_vals, g_vals)
                correlations[i] = corr
        else:
            # Use mean profiles
            real_mean = real.mean(axis=0)
            gen_mean = generated.mean(axis=0)
            
            if np.std(real_mean) == 0 or np.std(gen_mean) == 0:
                return np.full(n_genes, np.nan)
            
            overall_corr, _ = spearmanr(real_mean, gen_mean)
            correlations[:] = overall_corr
        
        return correlations


class MeanPearsonCorrelation(CorrelationMetric):
    """
    Pearson correlation on mean expression profiles.
    
    Computes mean expression per gene, then correlates the profiles.
    Returns single value replicated across genes.
    """
    
    def __init__(self):
        super().__init__(
            name="mean_pearson",
            description="Pearson correlation on mean expression profiles"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute correlation between mean profiles.
        
        Parameters
        ----------
        real : np.ndarray
            Real data, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data, shape (n_samples_gen, n_genes)
            
        Returns
        -------
        np.ndarray
            Single correlation value replicated per gene
        """
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        real_mean = real.mean(axis=0)
        gen_mean = generated.mean(axis=0)
        
        if np.std(real_mean) == 0 or np.std(gen_mean) == 0:
            return np.full(n_genes, np.nan)
        
        corr, _ = pearsonr(real_mean, gen_mean)
        return np.full(n_genes, corr)


class MeanSpearmanCorrelation(CorrelationMetric):
    """
    Spearman correlation on mean expression profiles.
    """
    
    def __init__(self):
        super().__init__(
            name="mean_spearman",
            description="Spearman correlation on mean expression profiles"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Spearman correlation between mean profiles.
        """
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        real_mean = real.mean(axis=0)
        gen_mean = generated.mean(axis=0)
        
        if np.std(real_mean) == 0 or np.std(gen_mean) == 0:
            return np.full(n_genes, np.nan)
        
        corr, _ = spearmanr(real_mean, gen_mean)
        return np.full(n_genes, corr)


class RSquared(CorrelationMetric):
    """
    R-squared (coefficient of determination) between real and generated expression.
    
    Measures the proportion of variance in real data explained by generated data.
    R² = 1 - (SS_res / SS_tot), where:
    - SS_res = sum of squared residuals (real - generated)²
    - SS_tot = total sum of squares (real - mean(real))²
    
    Higher values (closer to 1) indicate better fit.
    Can be negative if generated data is worse than predicting the mean.
    
    Reference: GGE Paper Section 3.3 "The Space Question: A Theoretical Analysis"
    """
    
    def __init__(self):
        super().__init__(
            name="r_squared",
            description="Coefficient of determination (R²) measuring variance explained"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute R² for each gene.
        
        For each gene, compares mean expression values between real and generated.
        
        Parameters
        ----------
        real : np.ndarray
            Real data, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data, shape (n_samples_gen, n_genes)
            
        Returns
        -------
        np.ndarray
            R² per gene (can be negative if model is worse than mean)
        """
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        # Use mean profiles (aggregate samples to get single value per gene)
        real_mean = real.mean(axis=0)
        gen_mean = generated.mean(axis=0)
        
        r_squared = np.zeros(n_genes)
        
        for i in range(n_genes):
            y_real = real_mean[i]
            y_gen = gen_mean[i]
            
            # For single values, R² compares to overall mean
            ss_res = (y_real - y_gen) ** 2
            ss_tot = (y_real - real_mean.mean()) ** 2
            
            if ss_tot == 0:
                r_squared[i] = 1.0 if ss_res == 0 else np.nan
            else:
                r_squared[i] = 1 - (ss_res / ss_tot)
        
        return r_squared
    
    def compute(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        gene_names: Optional[list] = None,
        aggregate_method: str = "mean",
        condition: Optional[str] = None,
        split: Optional[str] = None,
    ):
        """
        Compute overall R² between mean expression profiles.
        
        This version computes a single R² value across all genes,
        measuring how well the generated mean profile matches the real one.
        """
        from .base_metric import MetricResult
        
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        # Mean expression per gene
        real_mean = real.mean(axis=0)
        gen_mean = generated.mean(axis=0)
        
        # Overall R²: how well does gen_mean predict real_mean?
        ss_res = np.sum((real_mean - gen_mean) ** 2)
        ss_tot = np.sum((real_mean - real_mean.mean()) ** 2)
        
        if ss_tot == 0:
            overall_r2 = 1.0 if ss_res == 0 else 0.0
        else:
            overall_r2 = 1 - (ss_res / ss_tot)
        
        # Per-gene values (for consistency with other metrics)
        per_gene = self.compute_per_gene(real, generated)
        
        return MetricResult(
            name=self.name,
            per_gene_values=per_gene,
            aggregate_value=float(overall_r2),
            gene_names=gene_names,
        )

class PerturbationEffectCorrelation(CorrelationMetric):
    """
    Perturbation-effect correlation metric.
    
    Measures whether models capture the direction and magnitude of perturbation
    effects, not just absolute expression levels. This is the key biologically-
    motivated metric from GGE Paper Equation 1.
    
    The metric computes:
        ρ_effect = corr(μ_real - μ_ctrl, μ_gen - μ_ctrl)
    
    Where:
        - μ_real = mean expression of real perturbed cells
        - μ_gen = mean expression of generated perturbed cells
        - μ_ctrl = mean expression of control cells
    
    This is crucial because computing correlation on raw expression means can
    be artificially high if control and perturbed conditions have similar
    expression, regardless of whether the model captures perturbation effects.
    
    Parameters
    ----------
    method : str
        Correlation method: 'pearson' or 'spearman'
    
    Examples
    --------
    >>> metric = PerturbationEffectCorrelation()
    >>> result = metric.compute(
    ...     real=perturbed_real,
    ...     generated=perturbed_generated,
    ...     control_mean=control_mean  # Mean expression of control cells
    ... )
    """
    
    def __init__(self, method: str = "pearson"):
        self.method = method
        super().__init__(
            name=f"perturbation_effect_{method}",
            description=f"{method.capitalize()} correlation on perturbation effects (Paper Eq. 1)"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        control_mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute perturbation-effect correlation per gene.
        
        Parameters
        ----------
        real : np.ndarray
            Real perturbed data, shape (n_samples, n_genes)
        generated : np.ndarray
            Generated perturbed data, shape (n_samples, n_genes)
        control_mean : np.ndarray, optional
            Mean expression of control cells, shape (n_genes,).
            If not provided, returns NaN array.
            
        Returns
        -------
        np.ndarray
            Per-gene correlation on effects. Shape: (n_genes,)
        """
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        if control_mean is None:
            # Cannot compute without control reference
            return np.full(n_genes, np.nan)
        
        control_mean = np.atleast_1d(control_mean).flatten()
        if len(control_mean) != n_genes:
            raise ValueError(
                f"control_mean has {len(control_mean)} genes but data has {n_genes}"
            )
        
        # Compute mean expression
        real_mean = real.mean(axis=0)
        gen_mean = generated.mean(axis=0)
        
        # Compute perturbation effects (difference from control)
        real_effect = real_mean - control_mean
        gen_effect = gen_mean - control_mean
        
        # Per-gene: not really meaningful for single values, but for completeness
        # We return the signed difference of effects as a proxy
        effect_diff = np.abs(real_effect - gen_effect)
        
        # Normalize to [0, 1] similarity score where 1 is best
        max_effect = np.maximum(np.abs(real_effect), np.abs(gen_effect))
        max_effect = np.where(max_effect == 0, 1, max_effect)
        per_gene_similarity = 1 - (effect_diff / max_effect)
        
        return per_gene_similarity
    
    def compute(
        self,
        real: np.ndarray,
        generated: np.ndarray,
        control_mean: Optional[np.ndarray] = None,
        gene_names: Optional[list] = None,
        aggregate_method: str = "mean",
        condition: Optional[str] = None,
        split: Optional[str] = None,
    ):
        """
        Compute overall perturbation-effect correlation.
        
        This is the key metric from GGE Paper Equation 1:
            ρ_effect = corr(μ_real - μ_ctrl, μ_gen - μ_ctrl)
        
        Parameters
        ----------
        real : np.ndarray
            Real perturbed data, shape (n_samples, n_genes)
        generated : np.ndarray
            Generated perturbed data, shape (n_samples, n_genes)
        control_mean : np.ndarray
            Mean expression of control cells, shape (n_genes,)
        gene_names : list, optional
            Gene names for results
            
        Returns
        -------
        MetricResult
            Result with aggregate correlation and per-gene values
        """
        from .base_metric import MetricResult
        
        real = np.atleast_2d(real)
        generated = np.atleast_2d(generated)
        n_genes = real.shape[1]
        
        if control_mean is None:
            # Return NaN result if no control provided
            return MetricResult(
                name=self.name,
                per_gene_values=np.full(n_genes, np.nan),
                aggregate_value=float('nan'),
                gene_names=gene_names or [f"gene_{i}" for i in range(n_genes)],
                metadata={"warning": "control_mean not provided"}
            )
        
        control_mean = np.atleast_1d(control_mean).flatten()
        
        # Compute mean expression
        real_mean = real.mean(axis=0)
        gen_mean = generated.mean(axis=0)
        
        # Compute perturbation effects (difference from control)
        real_effect = real_mean - control_mean
        gen_effect = gen_mean - control_mean
        
        # Compute correlation on effects
        if self.method == "pearson":
            if np.std(real_effect) == 0 or np.std(gen_effect) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(real_effect, gen_effect)
        else:  # spearman
            if np.std(real_effect) == 0 or np.std(gen_effect) == 0:
                corr = 0.0
            else:
                corr, _ = spearmanr(real_effect, gen_effect)
        
        per_gene = self.compute_per_gene(real, generated, control_mean)
        
        return MetricResult(
            name=self.name,
            per_gene_values=per_gene,
            aggregate_value=float(corr),
            gene_names=gene_names or [f"gene_{i}" for i in range(n_genes)],
            metadata={
                "method": self.method,
                "n_genes": n_genes,
            }
        )


def compute_perturbation_effect_correlation(
    real_perturbed: np.ndarray,
    generated_perturbed: np.ndarray,
    control_mean: np.ndarray,
    method: str = "pearson",
) -> float:
    """
    Convenience function for perturbation-effect correlation.
    
    Implements GGE Paper Equation 1:
        ρ_effect = corr(μ_real - μ_ctrl, μ_gen - μ_ctrl)
    
    Parameters
    ----------
    real_perturbed : np.ndarray
        Real perturbed expression, shape (n_samples, n_genes)
    generated_perturbed : np.ndarray
        Generated perturbed expression, shape (n_samples, n_genes)
    control_mean : np.ndarray
        Mean expression of control cells, shape (n_genes,)
    method : str
        'pearson' or 'spearman'
        
    Returns
    -------
    float
        Correlation value between -1 and 1
        
    Examples
    --------
    >>> # Get control mean from real data
    >>> control_mask = real_adata.obs['condition'] == 'control'
    >>> control_mean = real_adata[control_mask].X.mean(axis=0)
    >>> 
    >>> # Get perturbed data
    >>> perturbed_real = real_adata[~control_mask].X
    >>> perturbed_gen = gen_adata[~control_mask].X
    >>> 
    >>> # Compute perturbation-effect correlation
    >>> rho = compute_perturbation_effect_correlation(
    ...     perturbed_real, perturbed_gen, control_mean
    ... )
    """
    metric = PerturbationEffectCorrelation(method=method)
    result = metric.compute(real_perturbed, generated_perturbed, control_mean)
    return result.aggregate_value