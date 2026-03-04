"""
Data loader module for paired real and generated datasets.

Provides loading, validation, and alignment of AnnData objects for evaluation.
"""
from __future__ import annotations

from typing import Optional, List, Union, Dict, Tuple, Iterator
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
    import scanpy as sc
except ImportError:
    raise ImportError("anndata and scanpy are required. Install with: pip install anndata scanpy")


class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass


class GeneExpressionDataLoader:
    """
    Data loader for paired real and generated gene expression datasets.
    
    Handles:
    - Loading AnnData files (h5ad format) or accepting AnnData objects directly
    - Validation of required columns
    - Alignment of gene names between datasets
    - Matching samples by condition columns
    - Split handling (train/test/all)
    
    Parameters
    ----------
    real_data : str, Path, or AnnData
        Path to real data h5ad file or AnnData object
    generated_data : str, Path, or AnnData
        Path to generated data h5ad file or AnnData object
    condition_columns : List[str]
        Columns to match between datasets (e.g., ['perturbation', 'cell_type'])
    split_column : str, optional
        Column indicating train/test split. If None, all data treated as single split.
    min_samples_per_condition : int
        Minimum samples required per condition to include
        
    Examples
    --------
    >>> # From paths
    >>> loader = GeneExpressionDataLoader("real.h5ad", "gen.h5ad", ["perturbation"])
    >>> 
    >>> # From AnnData objects
    >>> loader = GeneExpressionDataLoader(real_adata, gen_adata, ["perturbation"])
    >>> 
    >>> # Mixed
    >>> loader = GeneExpressionDataLoader("real.h5ad", gen_adata, ["perturbation"])
    """
    
    def __init__(
        self,
        real_data: Union[str, Path, "ad.AnnData"],
        generated_data: Union[str, Path, "ad.AnnData"],
        condition_columns: List[str],
        split_column: Optional[str] = None,
        min_samples_per_condition: int = 2,
    ):
        # Handle input types
        self._real_input = real_data
        self._generated_input = generated_data
        self._real_is_path = isinstance(real_data, (str, Path))
        self._generated_is_path = isinstance(generated_data, (str, Path))
        
        # Store paths for metadata (use placeholder for AnnData inputs)
        self.real_path = Path(real_data) if self._real_is_path else Path("<AnnData>")
        self.generated_path = Path(generated_data) if self._generated_is_path else Path("<AnnData>")
        
        self.condition_columns = condition_columns
        self.split_column = split_column
        self.min_samples_per_condition = min_samples_per_condition
        
        # Loaded data
        self._real: Optional[ad.AnnData] = None
        self._generated: Optional[ad.AnnData] = None
        
        # Aligned data
        self._real_aligned: Optional[ad.AnnData] = None
        self._generated_aligned: Optional[ad.AnnData] = None
        
        # Common genes and conditions
        self._common_genes: Optional[List[str]] = None
        self._common_conditions: Optional[Dict[str, List[str]]] = None
        
        # Cached condition masks
        self._condition_cache: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Validation state
        self._is_loaded = False
        self._is_aligned = False
    
    def load(self) -> "GeneExpressionDataLoader":
        """
        Load both datasets from disk or use provided AnnData objects.
        
        Returns
        -------
        self
            For method chaining
        """
        # Load/assign real data
        if self._real_is_path:
            real_path = Path(self._real_input)
            if not real_path.exists():
                raise DataLoaderError(f"Real data file not found: {real_path}")
            try:
                self._real = sc.read_h5ad(real_path)
            except Exception as e:
                raise DataLoaderError(f"Failed to load real data: {e}")
        else:
            # AnnData object provided directly
            if not isinstance(self._real_input, ad.AnnData):
                raise DataLoaderError(
                    f"real_data must be a path or AnnData object, got {type(self._real_input)}"
                )
            self._real = self._real_input.copy()
        
        # Load/assign generated data
        if self._generated_is_path:
            gen_path = Path(self._generated_input)
            if not gen_path.exists():
                raise DataLoaderError(f"Generated data file not found: {gen_path}")
            try:
                self._generated = sc.read_h5ad(gen_path)
            except Exception as e:
                raise DataLoaderError(f"Failed to load generated data: {e}")
        else:
            # AnnData object provided directly
            if not isinstance(self._generated_input, ad.AnnData):
                raise DataLoaderError(
                    f"generated_data must be a path or AnnData object, got {type(self._generated_input)}"
                )
            self._generated = self._generated_input.copy()
        
        # Validate columns
        self._validate_columns()
        
        self._is_loaded = True
        return self
    
    def _validate_columns(self):
        """Validate that required columns exist in both datasets."""
        for col in self.condition_columns:
            if col not in self._real.obs.columns:
                raise DataLoaderError(
                    f"Condition column '{col}' not found in real data. "
                    f"Available columns: {list(self._real.obs.columns)}"
                )
            if col not in self._generated.obs.columns:
                raise DataLoaderError(
                    f"Condition column '{col}' not found in generated data. "
                    f"Available columns: {list(self._generated.obs.columns)}"
                )
        
        if self.split_column is not None:
            if self.split_column not in self._real.obs.columns:
                raise DataLoaderError(
                    f"Split column '{self.split_column}' not found in real data."
                )
            # Generated data may not have split column - that's OK
            if self.split_column not in self._generated.obs.columns:
                warnings.warn(
                    f"Split column '{self.split_column}' not in generated data. "
                    "Generated data will be matched to real data by conditions only."
                )
    
    def align_genes(self) -> "GeneExpressionDataLoader":
        """
        Align gene names between real and generated datasets.
        
        Keeps only genes present in both datasets in the same order.
        
        Returns
        -------
        self
            For method chaining
        """
        if not self._is_loaded:
            raise DataLoaderError("Data not loaded. Call load() first.")
        
        real_genes = pd.Index(self._real.var_names.astype(str))
        gen_genes = pd.Index(self._generated.var_names.astype(str))
        
        # Find common genes
        common = real_genes.intersection(gen_genes)
        
        if len(common) == 0:
            raise DataLoaderError(
                "No overlapping genes between real and generated data."
            )
        
        # Warn about dropped genes
        n_real_only = len(real_genes) - len(common)
        n_gen_only = len(gen_genes) - len(common)
        
        if n_real_only > 0 or n_gen_only > 0:
            warnings.warn(
                f"Gene alignment: keeping {len(common)} common genes. "
                f"Dropped {n_real_only} from real, {n_gen_only} from generated."
            )
        
        # Subset and order genes
        self._common_genes = common.tolist()
        
        # Create aligned copies
        real_idx = real_genes.get_indexer(common)
        gen_idx = gen_genes.get_indexer(common)
        
        self._real_aligned = self._real[:, real_idx].copy()
        self._generated_aligned = self._generated[:, gen_idx].copy()
        
        # Ensure var_names match
        self._real_aligned.var_names = common
        self._generated_aligned.var_names = common
        
        self._is_aligned = True
        return self
    
    def _get_condition_key(self, row: pd.Series) -> str:
        """Generate unique key for a condition combination."""
        return "####".join([str(row[c]) for c in self.condition_columns])
    
    def _build_condition_masks(
        self,
        adata: ad.AnnData,
        split: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Build boolean masks for each unique condition."""
        obs = adata.obs.copy()
        
        # Apply split filter if specified
        if split is not None and self.split_column is not None:
            if self.split_column in obs.columns:
                split_mask = obs[self.split_column].astype(str) == split
                obs = obs[split_mask]
        
        # Get unique condition combinations
        conditions = obs[self.condition_columns].astype(str).drop_duplicates()
        
        masks = {}
        for _, row in conditions.iterrows():
            key = self._get_condition_key(row)
            
            # Build mask
            mask = np.ones(adata.n_obs, dtype=bool)
            for col in self.condition_columns:
                mask &= (adata.obs[col].astype(str) == str(row[col])).values
            
            if split is not None and self.split_column is not None:
                if self.split_column in adata.obs.columns:
                    mask &= (adata.obs[self.split_column].astype(str) == split).values
            
            if mask.sum() >= self.min_samples_per_condition:
                masks[key] = mask
        
        return masks
    
    def get_splits(self) -> List[str]:
        """
        Get list of available splits.
        
        Returns
        -------
        List[str]
            Split names (e.g., ['train', 'test'] or ['all'])
        """
        if not self._is_loaded:
            raise DataLoaderError("Data not loaded. Call load() first.")
        
        if self.split_column is None:
            return ["all"]
        
        if self.split_column not in self._real.obs.columns:
            return ["all"]
        
        return list(self._real.obs[self.split_column].astype(str).unique())
    
    def get_common_conditions(
        self,
        split: Optional[str] = None
    ) -> List[str]:
        """
        Get conditions present in both real and generated data.
        
        Parameters
        ----------
        split : str, optional
            If specified, only return conditions in this split
            
        Returns
        -------
        List[str]
            Condition keys present in both datasets
        """
        if not self._is_aligned:
            self.align_genes()
        
        real_masks = self._build_condition_masks(self._real_aligned, split)
        gen_masks = self._build_condition_masks(self._generated_aligned, None)
        
        # Find intersection
        common = sorted(set(real_masks.keys()) & set(gen_masks.keys()))
        
        return common
    
    def iterate_conditions(
        self,
        split: Optional[str] = None
    ) -> Iterator[Tuple[str, np.ndarray, np.ndarray, Dict[str, str]]]:
        """
        Iterate over matched conditions yielding aligned data.
        
        Parameters
        ----------
        split : str, optional
            If specified, only iterate conditions in this split
            
        Yields
        ------
        Tuple[str, np.ndarray, np.ndarray, Dict[str, str]]
            (condition_key, real_data, generated_data, condition_info)
            where condition_info contains the parsed condition values
        """
        if not self._is_aligned:
            self.align_genes()
        
        real_masks = self._build_condition_masks(self._real_aligned, split)
        gen_masks = self._build_condition_masks(self._generated_aligned, None)
        
        common = sorted(set(real_masks.keys()) & set(gen_masks.keys()))
        
        for key in common:
            real_mask = real_masks[key]
            gen_mask = gen_masks[key]
            
            # Extract data matrices
            real_data = self._to_dense(self._real_aligned.X[real_mask])
            gen_data = self._to_dense(self._generated_aligned.X[gen_mask])
            
            # Parse condition info
            parts = key.split("####")
            condition_info = dict(zip(self.condition_columns, parts))
            
            yield key, real_data, gen_data, condition_info
    
    @staticmethod
    def _to_dense(X) -> np.ndarray:
        """Convert matrix to dense numpy array."""
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)
    
    @property
    def real(self) -> ad.AnnData:
        """Get aligned real data."""
        if not self._is_aligned:
            self.align_genes()
        return self._real_aligned
    
    @property
    def generated(self) -> ad.AnnData:
        """Get aligned generated data."""
        if not self._is_aligned:
            self.align_genes()
        return self._generated_aligned
    
    @property
    def gene_names(self) -> List[str]:
        """Get common gene names."""
        if not self._is_aligned:
            self.align_genes()
        return self._common_genes
    
    @property
    def n_genes(self) -> int:
        """Number of common genes."""
        return len(self.gene_names)
    
    def summary(self) -> Dict[str, any]:
        """Get summary of loaded data."""
        if not self._is_loaded:
            return {"loaded": False}
        
        result = {
            "loaded": True,
            "aligned": self._is_aligned,
            "real": {
                "n_samples": self._real.n_obs,
                "n_genes": self._real.n_vars,
                "path": str(self.real_path),
            },
            "generated": {
                "n_samples": self._generated.n_obs,
                "n_genes": self._generated.n_vars,
                "path": str(self.generated_path),
            },
            "condition_columns": self.condition_columns,
            "split_column": self.split_column,
        }
        
        if self._is_aligned:
            result["n_common_genes"] = len(self._common_genes)
            result["splits"] = self.get_splits()
            
            for split in result["splits"]:
                s = split if split != "all" else None
                result[f"n_conditions_{split}"] = len(self.get_common_conditions(s))
        
        return result
    
    def __repr__(self) -> str:
        if not self._is_loaded:
            return "GeneExpressionDataLoader(not loaded)"
        
        return (
            f"GeneExpressionDataLoader("
            f"real={self._real.n_obs}x{self._real.n_vars}, "
            f"gen={self._generated.n_obs}x{self._generated.n_vars}, "
            f"aligned={self._is_aligned})"
        )


def load_data(
    real_data: Union[str, Path, "ad.AnnData"],
    generated_data: Union[str, Path, "ad.AnnData"],
    condition_columns: List[str],
    split_column: Optional[str] = None,
    **kwargs
) -> GeneExpressionDataLoader:
    """
    Convenience function to load and align data.
    
    Parameters
    ----------
    real_data : str, Path, or AnnData
        Path to real data h5ad file or AnnData object
    generated_data : str, Path, or AnnData
        Path to generated data h5ad file or AnnData object
    condition_columns : List[str]
        Columns to match between datasets
    split_column : str, optional
        Column indicating train/test split
    **kwargs
        Additional arguments for GeneExpressionDataLoader
        
    Returns
    -------
    GeneExpressionDataLoader
        Loaded and aligned data loader
        
    Examples
    --------
    >>> # From paths
    >>> loader = load_data("real.h5ad", "gen.h5ad", ["perturbation"])
    >>> 
    >>> # From AnnData objects
    >>> loader = load_data(real_adata, gen_adata, ["perturbation"])
    """
    loader = GeneExpressionDataLoader(
        real_data=real_data,
        generated_data=generated_data,
        condition_columns=condition_columns,
        split_column=split_column,
        **kwargs
    )
    loader.load()
    loader.align_genes()
    return loader
