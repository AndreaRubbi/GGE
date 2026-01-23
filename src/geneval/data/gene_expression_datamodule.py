from typing import Optional, List, Union
import warnings
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc


class DataModuleError(Exception):
    """Custom exception for data module validation errors."""


class GeneExpressionDataModule:
    """
    Safe data module for gene expression datasets.

    Adds robust validation on construction:
    - Checks required obs columns (perturbation_key, split_key)
    - Validates control value presence (if provided)
    - Ensures minimum cells/genes
    - Detects duplicate gene names
    - Flags sparsity, normalization, log state
    - Prevents negative counts before preprocessing
    """

    def __init__(
        self,
        adata: ad.AnnData,
        perturbation_key: str,
        split_key: str,
        control: Optional[str] = None,
        condition_keys: Optional[List[str]] = None,
        min_cells: int = 10,
        min_genes: int = 50,
        allow_float_counts: bool = True,
        enforce_unique_var_names: bool = True,
    ):
        if adata is None:
            raise DataModuleError("AnnData object cannot be None.")
        self.adata = adata
        self.perturbation_key = perturbation_key
        self.split_key = split_key
        self.control = control
        self.condition_keys = condition_keys or []
        self.min_cells = int(min_cells)
        self.min_genes = int(min_genes)
        self.allow_float_counts = allow_float_counts
        self.enforce_unique_var_names = enforce_unique_var_names

        # State flags
        self.is_normalized: bool = False
        self.is_logged: bool = False
        self.is_sparse: bool = self._is_sparse(self.adata.X)

        self._validate_adata()

    # ----------------- validation helpers -----------------

    @staticmethod
    def _is_sparse(X) -> bool:
        try:
            from scipy import sparse
            return sparse.issparse(X)
        except ImportError:
            return False

    def _validate_obs_column(self, key: str):
        if key not in self.adata.obs.columns:
            raise DataModuleError(f"Required obs column '{key}' not found in AnnData.obs.")

    def _validate_control(self):
        if self.control is None:
            return
        col = self.perturbation_key
        vals = self.adata.obs[col].astype(str)
        if str(self.control) not in set(vals):
            raise DataModuleError(f"Control value '{self.control}' not present in '{col}' column.")

    def _validate_sizes(self):
        if self.adata.n_obs < self.min_cells:
            raise DataModuleError(
                f"Too few cells ({self.adata.n_obs}). Minimum required: {self.min_cells}."
            )
        if self.adata.n_vars < self.min_genes:
            warnings.warn(
                f"Low gene count ({self.adata.n_vars} < {self.min_genes}). Evaluation may be unstable.",
                RuntimeWarning,
            )

    def _validate_var_names(self):
        v = pd.Index(self.adata.var_names.astype(str))
        if v.has_duplicates:
            if self.enforce_unique_var_names:
                raise DataModuleError("Duplicate gene names detected in var_names.")
            else:
                warnings.warn("Duplicate gene names detected; downstream alignment may fail.", RuntimeWarning)

    def _detect_logged(self):
        X = self.adata.X
        # Heuristic: if many values < 0 or max < 50 maybe already logged.
        arr = X.toarray() if self._is_sparse(X) else np.asarray(X)
        if np.any(arr < 0):
            warnings.warn("Negative values detected in expression matrix.", RuntimeWarning)
        # Fraction of integer entries
        finite = np.isfinite(arr)
        sample = arr[finite]
        if sample.size == 0:
            return
        frac_int = np.mean(np.isclose(sample, np.round(sample)))
        if frac_int < 0.7:
            # likely normalized/logged
            self.is_normalized = True
            # check for log transform: typical upper bound after log1p ~ ~15
            if np.nanmax(sample) < 25:
                self.is_logged = True

    def _validate_preprocessing_state(self):
        # If counts are integers and large, warn if not normalized/logged
        X = self.adata.X
        arr = X.toarray() if self._is_sparse(X) else np.asarray(X)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise DataModuleError("Expression matrix contains no finite values.")
        frac_int = np.mean(np.isclose(finite, np.round(finite)))
        if frac_int > 0.95 and np.nanmax(finite) > 50 and not self.is_normalized:
            warnings.warn(
                "Data appears to be raw counts (mostly integers, high max). "
                "Run preprocess_data() before evaluation.",
                RuntimeWarning,
            )
        if np.nanmin(finite) < 0:
            raise DataModuleError("Negative values found in raw counts; data corruption suspected.")

    def _validate_condition_keys(self):
        for c in self.condition_keys:
            if c not in self.adata.obs.columns:
                warnings.warn(f"Condition key '{c}' not found in obs; it will be ignored.", RuntimeWarning)

    def _validate_split_column(self):
        self._validate_obs_column(self.split_key)
        splits = set(self.adata.obs[self.split_key].astype(str))
        if not splits.intersection({"test", "train", "val", "validation"}):
            warnings.warn(
                f"Split column '{self.split_key}' lacks standard split labels (e.g., 'test').",
                RuntimeWarning,
            )

    def _validate_adata(self):
        self._validate_obs_column(self.perturbation_key)
        self._validate_split_column()
        self._validate_control()
        self._validate_sizes()
        self._validate_var_names()
        self._detect_logged()
        self._validate_preprocessing_state()
        self._validate_condition_keys()

    # ----------------- public API -----------------

    def load_data(self, filepath: str):
        """Load AnnData from file and re-run validation."""
        self.adata = sc.read(filepath)
        self.is_sparse = self._is_sparse(self.adata.X)
        self.is_normalized = False
        self.is_logged = False
        self._validate_adata()

    def preprocess_data(
        self,
        filter_min_cells: int = 1,
        target_sum: float = 1e4,
        log_base: Union[int, float] = np.e,
    ):
        """
        Apply basic preprocessing: gene filtering, total count normalization, log1p.
        Sets flags accordingly.
        """
        sc.pp.filter_genes(self.adata, min_cells=filter_min_cells)
        sc.pp.normalize_total(self.adata, target_sum=target_sum)
        sc.pp.log1p(self.adata)
        self.is_normalized = True
        self.is_logged = True

    def get_data(self) -> ad.AnnData:
        """Return AnnData (post any preprocessing)."""
        return self.adata

    def get_conditions(self) -> pd.Series:
        """Return unique perturbation conditions."""
        return pd.Series(self.adata.obs[self.perturbation_key].unique(), name="condition")

    def summary(self) -> dict:
        """Structured summary of current dataset and preprocessing state."""
        return {
            "n_cells": int(self.adata.n_obs),
            "n_genes": int(self.adata.n_vars),
            "is_sparse": bool(self.is_sparse),
            "is_normalized": bool(self.is_normalized),
            "is_logged": bool(self.is_logged),
            "perturbation_key": self.perturbation_key,
            "split_key": self.split_key,
            "control": self.control,
            "condition_keys_present": [c for c in self.condition_keys if c in self.adata.obs.columns],
        }

    def assert_ready_for_evaluation(self):
        """Raise error if dataset appears unprocessed."""
        if not self.is_logged or not self.is_normalized:
            raise DataModuleError(
                "Dataset not preprocessed (normalization/log). Call preprocess_data() before evaluation."
            )