"""
Mock data generators for GenEval testing.

Provides utilities to create synthetic AnnData objects with realistic
gene expression patterns for testing evaluation pipelines.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False
    ad = None


class MockDataGenerator:
    """
    Generator for synthetic gene expression data.
    
    Creates realistic mock AnnData objects with controllable parameters
    for testing evaluation pipelines.
    
    Parameters
    ----------
    n_samples : int
        Number of samples (cells)
    n_genes : int
        Number of genes
    n_perturbations : int
        Number of unique perturbations
    n_cell_types : int
        Number of cell types
    seed : int
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_samples: int = 500,
        n_genes: int = 100,
        n_perturbations: int = 5,
        n_cell_types: int = 2,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.n_cell_types = n_cell_types
        self.seed = seed
        
        self._rng = np.random.RandomState(seed)
    
    def _check_anndata(self):
        """Check if anndata is available."""
        if not HAS_ANNDATA:
            raise ImportError(
                "anndata is required for mock data generation. "
                "Install with: pip install anndata"
            )
    
    def generate_gene_names(self) -> List[str]:
        """Generate realistic gene names."""
        prefixes = ["BRCA", "TP", "MYC", "EGFR", "KRAS", "PTEN", "AKT", "ERK", "JNK", "RAF"]
        genes = []
        for i in range(self.n_genes):
            prefix = prefixes[i % len(prefixes)]
            suffix = i // len(prefixes) + 1
            genes.append(f"{prefix}{suffix}")
        return genes
    
    def generate_perturbation_names(self) -> List[str]:
        """Generate perturbation names."""
        return [f"perturbation_{i}" for i in range(self.n_perturbations)]
    
    def generate_cell_type_names(self) -> List[str]:
        """Generate cell type names."""
        types = ["Neuron", "Astrocyte", "Microglia", "Oligodendrocyte", 
                 "Endothelial", "Fibroblast", "Epithelial", "Immune"]
        return types[:self.n_cell_types]
    
    def generate_base_expression(self) -> np.ndarray:
        """
        Generate base expression matrix with realistic patterns.
        
        Returns array of shape (n_samples, n_genes) with log-normalized values.
        """
        # Gene-specific baseline expression (some genes highly expressed)
        gene_baseline = self._rng.exponential(scale=2.0, size=self.n_genes)
        
        # Sample-specific scaling (library size variation)
        sample_scale = self._rng.lognormal(mean=0, sigma=0.3, size=(self.n_samples, 1))
        
        # Random noise
        noise = self._rng.randn(self.n_samples, self.n_genes) * 0.5
        
        # Combine
        X = gene_baseline * sample_scale + noise
        
        # Ensure non-negative and add small offset
        X = np.maximum(X, 0) + 0.1
        
        return X
    
    def add_perturbation_effects(
        self,
        X: np.ndarray,
        perturbations: np.ndarray,
        effect_size: float = 1.0,
        n_affected_genes: int = 20,
    ) -> np.ndarray:
        """
        Add perturbation-specific effects to expression matrix.
        
        Parameters
        ----------
        X : np.ndarray
            Base expression matrix
        perturbations : np.ndarray
            Perturbation labels per sample
        effect_size : float
            Magnitude of perturbation effects
        n_affected_genes : int
            Number of genes affected by each perturbation
            
        Returns
        -------
        np.ndarray
            Modified expression matrix
        """
        X = X.copy()
        unique_perts = np.unique(perturbations)
        
        for pert in unique_perts:
            mask = perturbations == pert
            
            # Select random genes to affect
            affected = self._rng.choice(
                self.n_genes, 
                size=min(n_affected_genes, self.n_genes),
                replace=False
            )
            
            # Add effect (some up, some down)
            effects = self._rng.randn(len(affected)) * effect_size
            X[mask][:, affected] += effects
        
        return np.maximum(X, 0)
    
    def add_cell_type_effects(
        self,
        X: np.ndarray,
        cell_types: np.ndarray,
        effect_size: float = 0.5,
    ) -> np.ndarray:
        """Add cell type-specific expression patterns."""
        X = X.copy()
        unique_types = np.unique(cell_types)
        
        # Each cell type has characteristic expression pattern
        for i, ct in enumerate(unique_types):
            mask = cell_types == ct
            
            # Shift expression of subset of genes
            n_marker_genes = self.n_genes // len(unique_types)
            start_idx = i * n_marker_genes
            end_idx = start_idx + n_marker_genes
            
            X[mask][:, start_idx:end_idx] += effect_size
        
        return np.maximum(X, 0)
    
    def generate_real_data(
        self,
        train_fraction: float = 0.7,
        include_control: bool = True,
    ) -> "ad.AnnData":
        """
        Generate mock "real" gene expression data.
        
        Parameters
        ----------
        train_fraction : float
            Fraction of samples in training set
        include_control : bool
            Whether to include a control perturbation
            
        Returns
        -------
        ad.AnnData
            Mock real data
        """
        self._check_anndata()
        
        # Generate base expression
        X = self.generate_base_expression()
        
        # Generate metadata
        perturbations = self._rng.choice(
            self.generate_perturbation_names(),
            size=self.n_samples
        )
        
        if include_control:
            # Make first perturbation the control
            control_mask = perturbations == self.generate_perturbation_names()[0]
            perturbations[control_mask] = "control"
        
        cell_types = self._rng.choice(
            self.generate_cell_type_names(),
            size=self.n_samples
        )
        
        # Add biological effects
        X = self.add_perturbation_effects(X, perturbations)
        X = self.add_cell_type_effects(X, cell_types)
        
        # Create splits
        n_train = int(self.n_samples * train_fraction)
        splits = np.array(["train"] * n_train + ["test"] * (self.n_samples - n_train))
        self._rng.shuffle(splits)
        
        # Create AnnData
        obs = pd.DataFrame({
            "perturbation": perturbations,
            "cell_type": cell_types,
            "split": splits,
            "batch": self._rng.choice(["batch1", "batch2"], self.n_samples),
        })
        
        var = pd.DataFrame(index=self.generate_gene_names())
        
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
        adata.var_names = self.generate_gene_names()
        
        return adata
    
    def generate_generated_data(
        self,
        real_data: "ad.AnnData",
        noise_level: float = 0.3,
        quality: str = "good",
    ) -> "ad.AnnData":
        """
        Generate mock "generated" data based on real data.
        
        Parameters
        ----------
        real_data : ad.AnnData
            Real data to base generation on
        noise_level : float
            Amount of noise to add (0 = identical, 1 = very noisy)
        quality : str
            Quality level: "good", "medium", "poor"
            
        Returns
        -------
        ad.AnnData
            Mock generated data
        """
        self._check_anndata()
        
        # Get real expression
        X_real = real_data.X.copy()
        
        # Quality affects noise and bias
        quality_params = {
            "good": {"noise_mult": 0.5, "bias": 0.0, "dropout": 0.0},
            "medium": {"noise_mult": 1.0, "bias": 0.2, "dropout": 0.1},
            "poor": {"noise_mult": 2.0, "bias": 0.5, "dropout": 0.2},
        }
        params = quality_params.get(quality, quality_params["medium"])
        
        # Add noise
        noise = self._rng.randn(*X_real.shape) * noise_level * params["noise_mult"]
        X_gen = X_real + noise
        
        # Add systematic bias
        if params["bias"] > 0:
            bias = self._rng.randn(X_real.shape[1]) * params["bias"]
            X_gen += bias
        
        # Simulate dropout (some values become lower)
        if params["dropout"] > 0:
            dropout_mask = self._rng.random(X_real.shape) < params["dropout"]
            X_gen[dropout_mask] *= 0.5
        
        X_gen = np.maximum(X_gen, 0).astype(np.float32)
        
        # Copy metadata (generated data matches real conditions)
        obs = real_data.obs.copy()
        
        adata = ad.AnnData(X=X_gen, obs=obs, var=real_data.var.copy())
        adata.var_names = real_data.var_names.tolist()
        
        return adata
    
    def generate_paired_data(
        self,
        noise_level: float = 0.3,
        quality: str = "good",
        train_fraction: float = 0.7,
    ) -> Tuple["ad.AnnData", "ad.AnnData"]:
        """
        Generate paired real and generated data.
        
        Parameters
        ----------
        noise_level : float
            Noise level for generated data
        quality : str
            Quality of generated data
        train_fraction : float
            Fraction of samples in training set
            
        Returns
        -------
        Tuple[ad.AnnData, ad.AnnData]
            (real_data, generated_data)
        """
        real = self.generate_real_data(train_fraction=train_fraction)
        generated = self.generate_generated_data(real, noise_level, quality)
        return real, generated
    
    def save_paired_data(
        self,
        output_dir: Union[str, Path],
        noise_level: float = 0.3,
        quality: str = "good",
    ) -> Tuple[Path, Path]:
        """
        Generate and save paired data to files.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory to save files
        noise_level : float
            Noise level for generated data
        quality : str
            Quality of generated data
            
        Returns
        -------
        Tuple[Path, Path]
            (real_path, generated_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        real, generated = self.generate_paired_data(noise_level, quality)
        
        real_path = output_dir / "real.h5ad"
        gen_path = output_dir / "generated.h5ad"
        
        real.write(real_path)
        generated.write(gen_path)
        
        return real_path, gen_path


class MockMetricData:
    """
    Generator for mock metric computation data.
    
    Creates numpy arrays suitable for testing metric computations
    without requiring full AnnData objects.
    """
    
    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)
    
    def identical_distributions(
        self,
        n_samples: int = 100,
        n_genes: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate identical real and generated data."""
        data = self._rng.randn(n_samples, n_genes)
        return data.copy(), data.copy()
    
    def similar_distributions(
        self,
        n_samples: int = 100,
        n_genes: int = 50,
        noise_level: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate similar but not identical data."""
        real = self._rng.randn(n_samples, n_genes)
        generated = real + self._rng.randn(n_samples, n_genes) * noise_level
        return real, generated
    
    def different_distributions(
        self,
        n_samples: int = 100,
        n_genes: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate very different distributions."""
        real = self._rng.randn(n_samples, n_genes)
        # Shift mean and scale
        generated = self._rng.randn(n_samples, n_genes) * 2 + 3
        return real, generated
    
    def with_outliers(
        self,
        n_samples: int = 100,
        n_genes: int = 50,
        outlier_fraction: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with outliers."""
        real = self._rng.randn(n_samples, n_genes)
        generated = real + self._rng.randn(n_samples, n_genes) * 0.3
        
        # Add outliers
        n_outliers = int(n_samples * outlier_fraction)
        outlier_idx = self._rng.choice(n_samples, n_outliers, replace=False)
        generated[outlier_idx] += self._rng.randn(n_outliers, n_genes) * 5
        
        return real, generated
    
    def sparse_data(
        self,
        n_samples: int = 100,
        n_genes: int = 50,
        sparsity: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sparse data (many zeros)."""
        real = self._rng.randn(n_samples, n_genes)
        generated = real + self._rng.randn(n_samples, n_genes) * 0.3
        
        # Zero out values
        mask = self._rng.random((n_samples, n_genes)) < sparsity
        real[mask] = 0
        generated[mask] = 0
        
        return real, generated


def create_test_anndata(
    n_samples: int = 100,
    n_genes: int = 50,
    n_perturbations: int = 3,
    seed: int = 42,
) -> "ad.AnnData":
    """
    Convenience function to create a simple test AnnData.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_genes : int
        Number of genes
    n_perturbations : int
        Number of perturbations
    seed : int
        Random seed
        
    Returns
    -------
    ad.AnnData
        Test AnnData object
    """
    generator = MockDataGenerator(
        n_samples=n_samples,
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        seed=seed,
    )
    return generator.generate_real_data()


def create_test_pair(
    n_samples: int = 100,
    n_genes: int = 50,
    noise_level: float = 0.3,
    seed: int = 42,
) -> Tuple["ad.AnnData", "ad.AnnData"]:
    """
    Convenience function to create paired test data.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_genes : int
        Number of genes  
    noise_level : float
        Noise level for generated data
    seed : int
        Random seed
        
    Returns
    -------
    Tuple[ad.AnnData, ad.AnnData]
        (real, generated) AnnData objects
    """
    generator = MockDataGenerator(
        n_samples=n_samples,
        n_genes=n_genes,
        seed=seed,
    )
    return generator.generate_paired_data(noise_level=noise_level)
