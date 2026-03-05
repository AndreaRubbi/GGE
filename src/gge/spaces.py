"""
Space transformations for GGE metrics.

Provides the SpaceTransformer class and utilities for computing metrics
in different representation spaces: raw, PCA, and DEG.

Reference: GGE Paper Section 3.3 "The Space Question: A Theoretical Analysis"

Spaces:
- raw: Full gene space (G ~ 5,000-20,000 dimensions)
- pca: Reduced PCA space (k dimensions, default k=50)
- deg: Restricted to differentially expressed genes
"""
from __future__ import annotations

from typing import Optional, Union, Dict, Tuple, List, Literal, TYPE_CHECKING
import numpy as np
import warnings
from dataclasses import dataclass

if TYPE_CHECKING:
    import anndata as ad

SpaceType = Literal["raw", "pca", "deg"]


@dataclass
class SpaceConfig:
    """Configuration for a computation space."""
    
    space: SpaceType = "raw"
    
    # PCA parameters
    n_components: int = 50
    use_highly_variable: bool = True
    n_top_genes: int = 2000
    
    # DEG parameters
    deg_lfc: float = 1.0  # log2 fold change threshold
    deg_pval: float = 0.05  # p-value threshold
    n_top_degs: Optional[int] = None  # Optional: select top N DEGs
    
    def __post_init__(self):
        if self.space not in ("raw", "pca", "deg"):
            raise ValueError(f"Invalid space: {self.space}. Must be 'raw', 'pca', or 'deg'")
    
    @property
    def suffix(self) -> str:
        """Get suffix for metric names."""
        if self.space == "raw":
            return ""
        elif self.space == "pca":
            return f"_pca{self.n_components}"
        else:  # deg
            return f"_deg"
    
    def __repr__(self) -> str:
        if self.space == "raw":
            return "SpaceConfig(raw)"
        elif self.space == "pca":
            return f"SpaceConfig(pca, n_components={self.n_components})"
        else:
            params = [f"deg_lfc={self.deg_lfc}", f"deg_pval={self.deg_pval}"]
            if self.n_top_degs:
                params.append(f"n_top_degs={self.n_top_degs}")
            return f"SpaceConfig(deg, {', '.join(params)})"


class SpaceTransformer:
    """
    Transforms data between computation spaces.
    
    Handles the three computation spaces from the GGE paper:
    - raw: Full gene space
    - pca: Principal component space
    - deg: Differentially expressed gene space
    
    Parameters
    ----------
    config : SpaceConfig
        Configuration for the target space
    control_data : np.ndarray, optional
        Control data for DEG computation (required for space="deg")
    gene_names : list, optional
        Gene names for tracking DEGs
    
    Examples
    --------
    >>> # PCA space
    >>> config = SpaceConfig(space="pca", n_components=50)
    >>> transformer = SpaceTransformer(config)
    >>> real_pca, gen_pca, names = transformer.transform(real_data, gen_data, gene_names)
    
    >>> # DEG space
    >>> config = SpaceConfig(space="deg", deg_lfc=0.25, deg_pval=0.1)
    >>> transformer = SpaceTransformer(config, control_data=ctrl)
    >>> real_deg, gen_deg, deg_names = transformer.transform(real_data, gen_data, gene_names)
    """
    
    def __init__(
        self,
        config: SpaceConfig,
        control_data: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
    ):
        self.config = config
        self.control_data = control_data
        self.gene_names = gene_names
        
        # Fitted state
        self._pca_model = None
        self._deg_mask = None
        self._deg_genes = None
        self._is_fitted = False
    
    def fit(
        self,
        real_data: np.ndarray,
        control_data: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
    ) -> "SpaceTransformer":
        """
        Fit the transformer on reference data.
        
        For PCA: fits PCA on real_data
        For DEG: identifies DEGs comparing real_data to control_data
        
        Parameters
        ----------
        real_data : np.ndarray
            Real data matrix, shape (n_samples, n_genes)
        control_data : np.ndarray, optional
            Control data for DEG identification
        gene_names : list, optional
            Gene names
            
        Returns
        -------
        self
        """
        if control_data is not None:
            self.control_data = control_data
        if gene_names is not None:
            self.gene_names = gene_names
        
        n_genes = real_data.shape[1]
        if self.gene_names is None:
            self.gene_names = [f"gene_{i}" for i in range(n_genes)]
        
        if self.config.space == "pca":
            self._fit_pca(real_data)
        elif self.config.space == "deg":
            self._fit_deg(real_data)
        
        self._is_fitted = True
        return self
    
    def _fit_pca(self, data: np.ndarray):
        """Fit PCA model."""
        from sklearn.decomposition import PCA
        
        n_components = min(
            self.config.n_components,
            data.shape[0] - 1,
            data.shape[1] - 1
        )
        
        self._pca_model = PCA(n_components=n_components)
        self._pca_model.fit(data)
    
    def _fit_deg(self, real_data: np.ndarray):
        """Identify DEGs for filtering."""
        if self.control_data is None:
            raise ValueError("control_data required for DEG space. "
                           "Provide control_data when creating SpaceTransformer.")
        
        n_genes = real_data.shape[1]
        
        # Compute mean expression
        real_mean = real_data.mean(axis=0)
        ctrl_mean = self.control_data.mean(axis=0)
        
        # Avoid division by zero
        ctrl_mean_safe = np.where(ctrl_mean == 0, 1e-10, ctrl_mean)
        
        # Compute log2 fold change
        # log2(real / ctrl) = log2(real) - log2(ctrl)
        with np.errstate(divide='ignore', invalid='ignore'):
            log2fc = np.log2(real_mean + 1) - np.log2(ctrl_mean + 1)
        
        # Simple statistical test (t-test)
        pvalues = np.ones(n_genes)
        from scipy import stats
        for i in range(n_genes):
            if real_data.shape[0] > 1 and self.control_data.shape[0] > 1:
                try:
                    _, pval = stats.ttest_ind(
                        real_data[:, i], 
                        self.control_data[:, i],
                        equal_var=False
                    )
                    pvalues[i] = pval if not np.isnan(pval) else 1.0
                except:
                    pvalues[i] = 1.0
        
        # Identify DEGs
        is_deg = (np.abs(log2fc) >= self.config.deg_lfc) & (pvalues <= self.config.deg_pval)
        
        # Handle n_top_degs if specified
        if self.config.n_top_degs is not None and is_deg.sum() > self.config.n_top_degs:
            # Rank by absolute log2fc
            deg_indices = np.where(is_deg)[0]
            abs_lfc = np.abs(log2fc[deg_indices])
            top_indices = deg_indices[np.argsort(abs_lfc)[::-1][:self.config.n_top_degs]]
            is_deg = np.zeros(n_genes, dtype=bool)
            is_deg[top_indices] = True
        
        if is_deg.sum() == 0:
            warnings.warn(
                f"No DEGs found with lfc>={self.config.deg_lfc} and pval<={self.config.deg_pval}. "
                "Using all genes instead."
            )
            is_deg = np.ones(n_genes, dtype=bool)
        
        self._deg_mask = is_deg
        self._deg_genes = [g for g, m in zip(self.gene_names, is_deg) if m]
    
    def transform(
        self,
        real_data: np.ndarray,
        generated_data: np.ndarray,
        gene_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transform data to the configured space.
        
        Parameters
        ----------
        real_data : np.ndarray
            Real data matrix, shape (n_samples, n_genes)
        generated_data : np.ndarray
            Generated data matrix, shape (n_samples, n_genes)
        gene_names : list, optional
            Gene names (used if not fitted)
            
        Returns
        -------
        real_transformed : np.ndarray
            Transformed real data
        gen_transformed : np.ndarray
            Transformed generated data
        feature_names : list
            Names of features in transformed space
        """
        if self.config.space == "raw":
            names = gene_names or self.gene_names or [f"gene_{i}" for i in range(real_data.shape[1])]
            return real_data, generated_data, names
        
        if not self._is_fitted:
            self.fit(real_data, gene_names=gene_names)
        
        if self.config.space == "pca":
            return self._transform_pca(real_data, generated_data)
        else:  # deg
            return self._transform_deg(real_data, generated_data)
    
    def _transform_pca(
        self, 
        real_data: np.ndarray, 
        generated_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Transform to PCA space."""
        real_pca = self._pca_model.transform(real_data)
        gen_pca = self._pca_model.transform(generated_data)
        names = [f"PC{i+1}" for i in range(real_pca.shape[1])]
        return real_pca, gen_pca, names
    
    def _transform_deg(
        self, 
        real_data: np.ndarray, 
        generated_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Transform to DEG space (filter to DEG genes)."""
        real_deg = real_data[:, self._deg_mask]
        gen_deg = generated_data[:, self._deg_mask]
        return real_deg, gen_deg, self._deg_genes
    
    @property
    def n_features(self) -> Optional[int]:
        """Number of features in transformed space."""
        if self.config.space == "raw":
            return None  # Unknown until transform
        elif self.config.space == "pca":
            return self._pca_model.n_components_ if self._pca_model else self.config.n_components
        else:  # deg
            return len(self._deg_genes) if self._deg_genes else None


def get_space_config(
    space: SpaceType = "raw",
    n_components: int = 50,
    deg_lfc: float = 1.0,
    deg_pval: float = 0.05,
    n_top_degs: Optional[int] = None,
    **kwargs
) -> SpaceConfig:
    """
    Create a SpaceConfig from parameters.
    
    Parameters
    ----------
    space : str
        Space type: "raw", "pca", or "deg"
    n_components : int
        PCA components (for space="pca")
    deg_lfc : float
        log2 fold change threshold (for space="deg")
    deg_pval : float
        p-value threshold (for space="deg")
    n_top_degs : int, optional
        Select top N DEGs (for space="deg")
        
    Returns
    -------
    SpaceConfig
    """
    return SpaceConfig(
        space=space,
        n_components=n_components,
        deg_lfc=deg_lfc,
        deg_pval=deg_pval,
        n_top_degs=n_top_degs,
    )
