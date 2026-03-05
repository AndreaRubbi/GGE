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
