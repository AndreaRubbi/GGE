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
