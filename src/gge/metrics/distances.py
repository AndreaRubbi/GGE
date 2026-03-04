"""
Distribution distance metrics for gene expression evaluation.

Provides Wasserstein, MMD, and Energy distance metrics with per-gene computation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from typing import Optional, Tuple
import warnings

from .base_metric import DistributionMetric


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (samples x genes)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class Wasserstein1Distance(DistributionMetric):
    """
    Wasserstein-1 (Earth Mover's) distance between distributions.
    
    Measures the minimum amount of work to transform one distribution
    into another. Computed per gene using 1D Wasserstein distance.
    
    Lower values indicate more similar distributions.
    """
    
    def __init__(self):
        super().__init__(
            name="wasserstein_1",
            description="Wasserstein-1 (Earth Mover's) distance per gene"
        )
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Wasserstein-1 distance for each gene.
        
        Parameters
        ----------
        real : np.ndarray
            Real data, shape (n_samples_real, n_genes)
        generated : np.ndarray
            Generated data, shape (n_samples_gen, n_genes)
            
        Returns
        -------
        np.ndarray
            W1 distance per gene
        """
        real = _ensure_2d(real)
        generated = _ensure_2d(generated)
        n_genes = real.shape[1]
        
        distances = np.zeros(n_genes)
        
        for i in range(n_genes):
            r_vals = real[:, i]
            g_vals = generated[:, i]
            
            # Filter NaN values
            r_vals = r_vals[~np.isnan(r_vals)]
            g_vals = g_vals[~np.isnan(g_vals)]
            
            if len(r_vals) == 0 or len(g_vals) == 0:
                distances[i] = np.nan
                continue
            
            distances[i] = wasserstein_distance(r_vals, g_vals)
        
        return distances


class Wasserstein2Distance(DistributionMetric):
    """
    Wasserstein-2 distance (quadratic cost) between distributions.
    
    Uses p=2 norm for transport cost. More sensitive to outliers than W1.
    Computed per gene.
    """
    
    def __init__(self, use_geomloss: bool = True):
        """
        Parameters
        ----------
        use_geomloss : bool
            If True, use geomloss for GPU-accelerated computation.
            Falls back to scipy otherwise.
        """
        super().__init__(
            name="wasserstein_2",
            description="Wasserstein-2 distance per gene"
        )
        self.use_geomloss = use_geomloss
        self._geomloss_available = False
        
        if use_geomloss:
            try:
                import torch
                from geomloss import SamplesLoss
                self._geomloss_available = True
            except ImportError:
                warnings.warn(
                    "geomloss not available, falling back to scipy implementation"
                )
    
    def _w2_scipy(self, r_vals: np.ndarray, g_vals: np.ndarray) -> float:
        """Compute W2 using scipy (approximation via sorted quantiles)."""
        # Sort values and compute quadratic Wasserstein
        r_sorted = np.sort(r_vals)
        g_sorted = np.sort(g_vals)
        
        # Resample to same length for comparison
        n = max(len(r_sorted), len(g_sorted))
        r_quantiles = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(r_sorted)),
            r_sorted
        )
        g_quantiles = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(g_sorted)),
            g_sorted
        )
        
        return np.sqrt(np.mean((r_quantiles - g_quantiles) ** 2))
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Wasserstein-2 distance for each gene.
        """
        real = _ensure_2d(real)
        generated = _ensure_2d(generated)
        n_genes = real.shape[1]
        
        distances = np.zeros(n_genes)
        
        if self._geomloss_available and self.use_geomloss:
            import torch
            from geomloss import SamplesLoss
            loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, backend="tensorized")
            
            for i in range(n_genes):
                r_vals = real[:, i]
                g_vals = generated[:, i]
                
                r_vals = r_vals[~np.isnan(r_vals)]
                g_vals = g_vals[~np.isnan(g_vals)]
                
                if len(r_vals) == 0 or len(g_vals) == 0:
                    distances[i] = np.nan
                    continue
                
                # Reshape for geomloss (N, D)
                r_tensor = torch.tensor(r_vals.reshape(-1, 1), dtype=torch.float32)
                g_tensor = torch.tensor(g_vals.reshape(-1, 1), dtype=torch.float32)
                
                distances[i] = loss_fn(r_tensor, g_tensor).item()
        else:
            for i in range(n_genes):
                r_vals = real[:, i]
                g_vals = generated[:, i]
                
                r_vals = r_vals[~np.isnan(r_vals)]
                g_vals = g_vals[~np.isnan(g_vals)]
                
                if len(r_vals) == 0 or len(g_vals) == 0:
                    distances[i] = np.nan
                    continue
                
                distances[i] = self._w2_scipy(r_vals, g_vals)
        
        return distances


class MMDDistance(DistributionMetric):
    """
    Maximum Mean Discrepancy (MMD) between distributions.
    
    Non-parametric distance based on kernel embeddings.
    Uses RBF (Gaussian) kernel. Computed per gene.
    """
    
    def __init__(self, kernel: str = "rbf", sigma: Optional[float] = None):
        """
        Parameters
        ----------
        kernel : str
            Kernel type ("rbf" for Gaussian)
        sigma : float, optional
            Kernel bandwidth. If None, uses median heuristic.
        """
        super().__init__(
            name="mmd",
            description="Maximum Mean Discrepancy with RBF kernel"
        )
        self.kernel = kernel
        self.sigma = sigma
    
    def _rbf_kernel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: float
    ) -> np.ndarray:
        """Compute RBF kernel matrix."""
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # Compute pairwise squared distances
        diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        
        return np.exp(-sq_dist / (2 * sigma ** 2))
    
    def _median_heuristic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute bandwidth using median heuristic."""
        combined = np.concatenate([x, y])
        pairwise = np.abs(combined[:, np.newaxis] - combined[np.newaxis, :])
        return float(np.median(pairwise[pairwise > 0]))
    
    def _compute_mmd_single(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: Optional[float] = None
    ) -> float:
        """Compute MMD for single gene."""
        if sigma is None:
            sigma = self._median_heuristic(x, y)
            if sigma == 0:
                sigma = 1.0
        
        K_xx = self._rbf_kernel(x, x, sigma)
        K_yy = self._rbf_kernel(y, y, sigma)
        K_xy = self._rbf_kernel(x, y, sigma)
        
        n_x = len(x)
        n_y = len(y)
        
        # Unbiased MMD estimator
        mmd = (
            (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1)) +
            (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1)) -
            2 * np.sum(K_xy) / (n_x * n_y)
        )
        
        return max(0, mmd)  # Ensure non-negative
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute MMD for each gene.
        """
        real = _ensure_2d(real)
        generated = _ensure_2d(generated)
        n_genes = real.shape[1]
        
        distances = np.zeros(n_genes)
        
        for i in range(n_genes):
            r_vals = real[:, i]
            g_vals = generated[:, i]
            
            r_vals = r_vals[~np.isnan(r_vals)]
            g_vals = g_vals[~np.isnan(g_vals)]
            
            if len(r_vals) < 2 or len(g_vals) < 2:
                distances[i] = np.nan
                continue
            
            distances[i] = self._compute_mmd_single(r_vals, g_vals, self.sigma)
        
        return distances


class EnergyDistance(DistributionMetric):
    """
    Energy distance between distributions.
    
    Based on statistical potential energy. Related to but different from
    Wasserstein distance. Computed per gene.
    """
    
    def __init__(self, use_geomloss: bool = True):
        super().__init__(
            name="energy",
            description="Energy distance per gene"
        )
        self.use_geomloss = use_geomloss
        self._geomloss_available = False
        
        if use_geomloss:
            try:
                import torch
                from geomloss import SamplesLoss
                self._geomloss_available = True
            except ImportError:
                pass
    
    def _energy_scipy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute energy distance using scipy."""
        n_x, n_y = len(x), len(y)
        
        # E[|X - Y|]
        xy_dist = np.mean(np.abs(x[:, np.newaxis] - y[np.newaxis, :]))
        
        # E[|X - X'|]
        xx_dist = np.mean(np.abs(x[:, np.newaxis] - x[np.newaxis, :]))
        
        # E[|Y - Y'|]
        yy_dist = np.mean(np.abs(y[:, np.newaxis] - y[np.newaxis, :]))
        
        energy = 2 * xy_dist - xx_dist - yy_dist
        return max(0, energy)
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute energy distance for each gene.
        """
        real = _ensure_2d(real)
        generated = _ensure_2d(generated)
        n_genes = real.shape[1]
        
        distances = np.zeros(n_genes)
        
        if self._geomloss_available and self.use_geomloss:
            import torch
            from geomloss import SamplesLoss
            loss_fn = SamplesLoss(loss="energy", blur=0.5, backend="tensorized")
            
            for i in range(n_genes):
                r_vals = real[:, i]
                g_vals = generated[:, i]
                
                r_vals = r_vals[~np.isnan(r_vals)]
                g_vals = g_vals[~np.isnan(g_vals)]
                
                if len(r_vals) == 0 or len(g_vals) == 0:
                    distances[i] = np.nan
                    continue
                
                r_tensor = torch.tensor(r_vals.reshape(-1, 1), dtype=torch.float32)
                g_tensor = torch.tensor(g_vals.reshape(-1, 1), dtype=torch.float32)
                
                distances[i] = loss_fn(r_tensor, g_tensor).item()
        else:
            for i in range(n_genes):
                r_vals = real[:, i]
                g_vals = generated[:, i]
                
                r_vals = r_vals[~np.isnan(r_vals)]
                g_vals = g_vals[~np.isnan(g_vals)]
                
                if len(r_vals) < 2 or len(g_vals) < 2:
                    distances[i] = np.nan
                    continue
                
                distances[i] = self._energy_scipy(r_vals, g_vals)
        
        return distances


# Multivariate distance metrics (computed on full gene space)

class MultivariateWasserstein(DistributionMetric):
    """
    Multivariate Wasserstein distance on full gene expression space.
    
    Unlike per-gene metrics, this computes distance in the joint space
    of all genes. Typically applied after PCA dimensionality reduction.
    """
    
    def __init__(self, p: int = 2, blur: float = 0.01):
        super().__init__(
            name="multivariate_wasserstein",
            description=f"Multivariate Wasserstein-{p} distance"
        )
        self.p = p
        self.blur = blur
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multivariate distance (returns same value for all genes).
        """
        real = _ensure_2d(real)
        generated = _ensure_2d(generated)
        n_genes = real.shape[1]
        
        try:
            import torch
            from geomloss import SamplesLoss
            
            loss_fn = SamplesLoss(
                loss="sinkhorn",
                p=self.p,
                blur=self.blur,
                backend="tensorized"
            )
            
            r_tensor = torch.tensor(real, dtype=torch.float32)
            g_tensor = torch.tensor(generated, dtype=torch.float32)
            
            distance = loss_fn(r_tensor, g_tensor).item()
        except ImportError:
            # Fallback: use sliced Wasserstein approximation
            warnings.warn("geomloss not available, using sliced Wasserstein approximation")
            distance = self._sliced_wasserstein(real, generated)
        
        return np.full(n_genes, distance)
    
    def _sliced_wasserstein(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_projections: int = 100
    ) -> float:
        """Compute sliced Wasserstein distance as fallback."""
        d = x.shape[1]
        
        # Random projections
        projections = np.random.randn(d, n_projections)
        projections /= np.linalg.norm(projections, axis=0)
        
        distances = []
        for i in range(n_projections):
            proj = projections[:, i]
            x_proj = x @ proj
            y_proj = y @ proj
            distances.append(wasserstein_distance(x_proj, y_proj))
        
        return float(np.mean(distances))


class MultivariateMMD(DistributionMetric):
    """
    Multivariate MMD on full gene expression space.
    """
    
    def __init__(self, sigma: Optional[float] = None):
        super().__init__(
            name="multivariate_mmd",
            description="Multivariate MMD with RBF kernel"
        )
        self.sigma = sigma
    
    def compute_per_gene(
        self,
        real: np.ndarray,
        generated: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multivariate MMD.
        """
        real = _ensure_2d(real)
        generated = _ensure_2d(generated)
        n_genes = real.shape[1]
        
        # Use median heuristic for bandwidth
        if self.sigma is None:
            combined = np.vstack([real, generated])
            pairwise_sq = np.sum(
                (combined[:, np.newaxis, :] - combined[np.newaxis, :, :]) ** 2,
                axis=-1
            )
            sigma = float(np.sqrt(np.median(pairwise_sq[pairwise_sq > 0])))
            if sigma == 0:
                sigma = 1.0
        else:
            sigma = self.sigma
        
        # Compute kernel matrices
        def rbf_kernel(x, y, sigma):
            pairwise_sq = np.sum(
                (x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2,
                axis=-1
            )
            return np.exp(-pairwise_sq / (2 * sigma ** 2))
        
        K_xx = rbf_kernel(real, real, sigma)
        K_yy = rbf_kernel(generated, generated, sigma)
        K_xy = rbf_kernel(real, generated, sigma)
        
        n_x, n_y = len(real), len(generated)
        
        mmd = (
            (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1)) +
            (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1)) -
            2 * np.sum(K_xy) / (n_x * n_y)
        )
        
        return np.full(n_genes, max(0, mmd))
