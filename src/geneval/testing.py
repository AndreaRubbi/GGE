"""
Testing utilities for GenEval.

This module provides mock data generators and testing helpers
that users can use to test their own integrations with GenEval.

Examples
--------
>>> from geneval.testing import MockDataGenerator
>>> 
>>> # Generate synthetic paired datasets
>>> generator = MockDataGenerator(n_samples=100, n_genes=50, seed=42)
>>> real, generated = generator.generate_paired_data(noise_level=0.3)
>>> 
>>> # Use with evaluation
>>> from geneval import evaluate
>>> results = evaluate(
...     real_data=real,
...     generated_data=generated,
...     condition_columns=["perturbation"],
... )
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


class MockDataGenerator:
    """
    Generator for synthetic gene expression data.
    
    Creates realistic-looking gene expression data with perturbation
    and cell type effects for testing evaluation pipelines.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_genes : int
        Number of genes.
    n_perturbations : int
        Number of different perturbation conditions.
    n_cell_types : int
        Number of different cell types.
    seed : int, optional
        Random seed for reproducibility.
    
    Examples
    --------
    >>> generator = MockDataGenerator(n_samples=100, n_genes=50, seed=42)
    >>> real = generator.generate_real_data()
    >>> generated = generator.generate_generated_data(real, noise_level=0.3)
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        n_genes: int = 50,
        n_perturbations: int = 3,
        n_cell_types: int = 2,
        seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.n_cell_types = n_cell_types
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate gene names
        self.gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Generate perturbation names
        self.perturbations = [f"perturbation_{i}" for i in range(n_perturbations)]
        
        # Generate cell type names
        self.cell_types = [f"cell_type_{i}" for i in range(n_cell_types)]
        
        # Generate random effects
        self._perturbation_effects = {
            p: np.random.randn(n_genes) * 0.5 for p in self.perturbations
        }
        self._cell_type_effects = {
            c: np.random.randn(n_genes) * 0.3 for c in self.cell_types
        }
    
    def generate_real_data(self) -> "ad.AnnData":
        """
        Generate realistic gene expression data.
        
        Returns
        -------
        AnnData
            Synthetic gene expression data with perturbation and cell type effects.
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata is required for AnnData generation")
        
        # Assign perturbations and cell types
        perturbations = np.random.choice(self.perturbations, self.n_samples)
        cell_types = np.random.choice(self.cell_types, self.n_samples)
        
        # Base expression (log-normal-like)
        base_expression = np.random.exponential(1.0, (self.n_samples, self.n_genes))
        
        # Add perturbation effects
        for i, pert in enumerate(perturbations):
            base_expression[i] += self._perturbation_effects[pert]
        
        # Add cell type effects
        for i, ct in enumerate(cell_types):
            base_expression[i] += self._cell_type_effects[ct]
        
        # Add noise
        base_expression += np.random.randn(self.n_samples, self.n_genes) * 0.2
        
        # Clip to realistic range
        base_expression = np.clip(base_expression, 0, None)
        
        # Create AnnData
        adata = ad.AnnData(X=base_expression)
        adata.var_names = self.gene_names
        adata.obs["perturbation"] = perturbations
        adata.obs["cell_type"] = cell_types
        adata.obs_names = [f"cell_{i}" for i in range(self.n_samples)]
        
        return adata
    
    def generate_generated_data(
        self,
        real_data: "ad.AnnData",
        noise_level: float = 0.3,
        quality: str = "good",
    ) -> "ad.AnnData":
        """
        Generate synthetic data matching real data structure.
        
        Parameters
        ----------
        real_data : AnnData
            Real data to match structure from.
        noise_level : float
            Amount of noise to add (0-1 scale).
        quality : str
            Quality level: "good", "medium", or "poor".
        
        Returns
        -------
        AnnData
            Generated data with same structure as real.
        """
        if not HAS_ANNDATA:
            raise ImportError("anndata is required for AnnData generation")
        
        # Copy structure
        X = real_data.X.copy()
        
        # Apply quality-based noise
        if quality == "good":
            noise_mult = noise_level
            bias = 0.0
        elif quality == "medium":
            noise_mult = noise_level * 1.5
            bias = 0.1
        else:  # poor
            noise_mult = noise_level * 2.0
            bias = 0.2
        
        # Add noise
        X = X + np.random.randn(*X.shape) * noise_mult
        
        # Add bias
        X = X + bias
        
        # Clip
        X = np.clip(X, 0, None)
        
        # Create AnnData
        generated = ad.AnnData(X=X)
        generated.var_names = list(real_data.var_names)
        generated.obs = real_data.obs.copy()
        generated.obs_names = [f"gen_cell_{i}" for i in range(len(X))]
        
        return generated
    
    def generate_paired_data(
        self,
        noise_level: float = 0.3,
        quality: str = "good",
        include_split: bool = False,
        train_fraction: float = 0.7,
    ) -> Tuple["ad.AnnData", "ad.AnnData"]:
        """
        Generate paired real and generated datasets.
        
        Parameters
        ----------
        noise_level : float
            Noise level for generated data.
        quality : str
            Quality of generated data.
        include_split : bool
            Whether to include train/test split column.
        train_fraction : float
            Fraction of samples in training set.
        
        Returns
        -------
        Tuple[AnnData, AnnData]
            (real_data, generated_data) tuple.
        """
        real = self.generate_real_data()
        generated = self.generate_generated_data(real, noise_level, quality)
        
        if include_split:
            n_train = int(self.n_samples * train_fraction)
            splits = np.array(["train"] * n_train + ["test"] * (self.n_samples - n_train))
            np.random.shuffle(splits)
            real.obs["split"] = splits
            generated.obs["split"] = splits
        
        return real, generated
    
    def save_paired_data(
        self,
        output_dir: Union[str, Path],
        noise_level: float = 0.3,
        quality: str = "good",
        include_split: bool = True,
    ) -> Tuple[Path, Path]:
        """
        Generate and save paired datasets to h5ad files.
        
        Parameters
        ----------
        output_dir : Path
            Directory to save files.
        noise_level : float
            Noise level for generated data.
        quality : str
            Quality of generated data.
        include_split : bool
            Whether to include train/test split column.
        
        Returns
        -------
        Tuple[Path, Path]
            (real_path, generated_path) tuple.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        real, generated = self.generate_paired_data(
            noise_level=noise_level,
            quality=quality,
            include_split=include_split,
        )
        
        real_path = output_dir / "real.h5ad"
        generated_path = output_dir / "generated.h5ad"
        
        real.write(real_path)
        generated.write(generated_path)
        
        return real_path, generated_path


class MockMetricData:
    """
    Generator for mock metric testing data.
    
    Creates numpy arrays with specific statistical properties
    for testing metric implementations.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def identical_distributions(
        self,
        n_samples: int = 100,
        n_features: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate identical distributions for testing zero distance."""
        data = np.random.randn(n_samples, n_features)
        return data.copy(), data.copy()
    
    def similar_distributions(
        self,
        n_samples: int = 100,
        n_features: int = 50,
        noise: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate similar but not identical distributions."""
        real = np.random.randn(n_samples, n_features)
        generated = real + np.random.randn(n_samples, n_features) * noise
        return real, generated
    
    def different_distributions(
        self,
        n_samples: int = 100,
        n_features: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate clearly different distributions."""
        real = np.random.randn(n_samples, n_features)
        generated = np.random.randn(n_samples, n_features) + 3.0  # Shifted mean
        return real, generated
    
    def with_outliers(
        self,
        n_samples: int = 100,
        n_features: int = 50,
        outlier_fraction: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with outliers in generated."""
        real = np.random.randn(n_samples, n_features)
        generated = real.copy()
        
        n_outliers = int(n_samples * outlier_fraction)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        generated[outlier_indices] = np.random.randn(n_outliers, n_features) * 10
        
        return real, generated
    
    def sparse_data(
        self,
        n_samples: int = 100,
        n_features: int = 50,
        sparsity: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sparse data (many zeros)."""
        real = np.random.randn(n_samples, n_features)
        generated = np.random.randn(n_samples, n_features)
        
        # Zero out based on sparsity
        mask_real = np.random.random((n_samples, n_features)) < sparsity
        mask_gen = np.random.random((n_samples, n_features)) < sparsity
        
        real[mask_real] = 0
        generated[mask_gen] = 0
        
        return real, generated


# Convenience functions
def create_test_data(
    n_samples: int = 100,
    n_genes: int = 50,
    noise_level: float = 0.3,
    seed: int = 42,
) -> Tuple["ad.AnnData", "ad.AnnData"]:
    """
    Create synthetic test data quickly.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_genes : int
        Number of genes.
    noise_level : float
        Noise level for generated data.
    seed : int
        Random seed.
    
    Returns
    -------
    Tuple[AnnData, AnnData]
        (real, generated) tuple.
    """
    generator = MockDataGenerator(
        n_samples=n_samples,
        n_genes=n_genes,
        seed=seed,
    )
    return generator.generate_paired_data(noise_level=noise_level)
