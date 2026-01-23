from __future__ import annotations

from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from ..utils.preprocessing import to_dense


class BaseEvaluator(ABC):
    """
    Base class for evaluation of generated data against real datasets.

    Provides:
    - Variable/gene alignment between real and generated AnnData objects
    - Computation and application of control baselines per strata
    """

    def __init__(self, data, output: AnnData):
        """
        Parameters
        ----------
        data : object
            An object providing at least:
              - gene_expression_dataset.adata: AnnData
              - perturbation_key: str
              - split_key: str
              - control: str
              - condition_keys: Optional[List[str]]
        output : AnnData
            Generated data to evaluate.
        """
        self.data = data
        self.output = output

    # ---------- alignment utilities ----------

    def _align_varnames_like(self, real: AnnData, generated: AnnData) -> Tuple[AnnData, AnnData]:
        """
        Align real and generated AnnData to the common set of var_names (genes),
        preserving order based on the real AnnData.
        """
        real_genes = pd.Index(real.var_names.astype(str))
        gen_genes = pd.Index(generated.var_names.astype(str))
        common = real_genes.intersection(gen_genes)
        if len(common) == 0:
            raise ValueError("No overlapping genes between real and generated AnnData.")

        # Reindex both adatas to the common genes in the order of real
        real = real[:, real_genes.get_indexer(common)].copy()
        generated = generated[:, generated.var_names.astype(str).isin(common)].copy()
        # Reorder generated to match real
        generated = generated[:, pd.Index(generated.var_names.astype(str)).get_indexer(common)].copy()

        real.var_names = common
        generated.var_names = common
        return real, generated

    # ---------- baseline utilities ----------

    @staticmethod
    def _key_from_values(values: Iterable[object]) -> str:
        # stable string key for strata-tuples
        return "####".join([str(v) for v in values])

    def _compute_control_means(
        self,
        adata: AnnData,
        perturbation_col: str,
        control_value: str,
        strata_cols: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-strata control means across genes.

        Returns a dict mapping a strata-key -> mean vector (n_genes,).
        """
        strata_cols = strata_cols or []
        obs = adata.obs

        if perturbation_col not in obs.columns:
            raise KeyError(f"'{perturbation_col}' not found in adata.obs.")

        is_control = (obs[perturbation_col].astype(str) == str(control_value)).to_numpy()
        if not is_control.any():
            # no controls; return empty means map
            return {}

        ctrl = adata[is_control]
        if not strata_cols:
            return {self._key_from_values([]): to_dense(ctrl.X).mean(axis=0)}

        # group by strata columns (as strings to be robust)
        df = ctrl.obs[strata_cols].astype(str)
        means: Dict[str, np.ndarray] = {}
        # compute mean per unique strata combination
        for _, row in df.drop_duplicates().iterrows():
            mask = np.ones(ctrl.n_obs, dtype=bool)
            for c in strata_cols:
                mask &= (df[c].to_numpy() == str(row[c]))
            if not mask.any():
                continue
            key = self._key_from_values([row[c] for c in strata_cols])
            means[key] = to_dense(ctrl.X[mask]).mean(axis=0)
        return means

    def _apply_baseline_per_strata(
        self,
        X,
        obs: pd.DataFrame,
        baseline: Dict[str, np.ndarray],
        strata_cols: Optional[List[str]] = None,
        mode: str = "subtract",
    ):
        """
        Apply per-strata baseline vectors to rows in X based on obs[strata_cols].

        mode: 'subtract' or 'add'
        """
        strata_cols = strata_cols or []
        if mode not in ("subtract", "add"):
            raise ValueError("mode must be 'subtract' or 'add'.")

        if sparse.issparse(X):
            X = X.tocsr(copy=True)
            to_dense_first = False
        else:
            X = np.array(X, copy=True)
            to_dense_first = True  # already dense

        if not strata_cols:
            key = self._key_from_values([])
            b = baseline.get(key, None)
            if b is None:
                return X
            if sparse.issparse(X):
                # operate dense for simplicity
                X = X.toarray()
            if mode == "subtract":
                X -= b
            else:
                X += b
            return X

        # Apply per group
        df = obs[strata_cols].astype(str)
        # iterate groups in baseline for efficiency
        for key, b in baseline.items():
            # decode key into tuple of values
            parts = key.split("####") if key else []
            if len(parts) != len(strata_cols):
                # skip mismatched key
                continue
            mask = np.ones(df.shape[0], dtype=bool)
            for col, val in zip(strata_cols, parts):
                mask &= (df[col].to_numpy() == val)
            if not mask.any():
                continue

            if sparse.issparse(X):
                # operate in dense block then write back
                block = X[mask].toarray()
                if mode == "subtract":
                    block -= b
                else:
                    block += b
                X[mask] = sparse.csr_matrix(block)
            else:
                if mode == "subtract":
                    X[mask] -= b
                else:
                    X[mask] += b

        return X