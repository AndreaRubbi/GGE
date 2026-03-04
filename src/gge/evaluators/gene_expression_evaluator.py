from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any

import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.stats as sstats
from sklearn.metrics import mean_squared_error

from ..metrics.metrics import compute_metrics
from ..utils.preprocessing import to_dense as _to_dense
from .base_evaluator import BaseEvaluator
from ..visualization import EvaluationPlotter

if TYPE_CHECKING:
    from ..data.gene_expression_datamodule import GeneExpressionDataModule


class GeneExpressionEvaluator(BaseEvaluator):
    """
    Evaluator for gene expression data.
    """

    def __init__(self, data: "GeneExpressionDataModule", output: AnnData):
        super().__init__(data, output)

    def evaluate(
        self,
        delta: bool = False,
        plot: bool = False,
        DEG: Optional[Dict[str, Any]] = None,
        save_dir: Optional[str] = None,
        max_panels: int = 12,
        dpi: int = 150,
    ):
        """
        Run evaluation. If plot=True, returns and optionally saves figures.
        """
        data = self.data.gene_expression_dataset.adata.copy()
        generated = self.output.copy()
        data, generated = self._align_varnames_like(data, generated)

        pert_col = self.data.perturbation_key
        split_key = self.data.split_key
        control = self.data.control

        order_cols = []
        if "cell_type" in data.obs.columns and "cell_type" in generated.obs.columns:
            order_cols.append("cell_type")
        for c in (getattr(self.data, "condition_keys", None) or []):
            if c in data.obs.columns and c in generated.obs.columns:
                order_cols.append(c)

        # Baseline handling
        if delta:
            b = self._compute_control_means(data, pert_col, control, strata_cols=order_cols)
            data.X = self._apply_baseline_per_strata(data.X, data.obs, b, strata_cols=order_cols, mode="subtract")
        else:
            b = self._compute_control_means(data, pert_col, control, strata_cols=order_cols)
            generated.X = self._apply_baseline_per_strata(
                generated.X, generated.obs, b, strata_cols=order_cols, mode="add"
            )

        is_test = (data.obs[split_key].astype(str) == "test").to_numpy()
        test_data = data[is_test].copy()

        if "perturbation" not in generated.obs.columns and pert_col not in generated.obs.columns:
            raise KeyError("'perturbation' column not found in generated data.")
        if pert_col not in generated.obs.columns and "perturbation" in generated.obs.columns:
            generated.obs[pert_col] = generated.obs["perturbation"].astype(test_data.obs[pert_col].dtype)

        def _means_masks(adata, cols):
            means, masks = {}, {}
            df = adata.obs[[pert_col] + cols].astype(str)
            for _, row in df.drop_duplicates().iterrows():
                pert = row[pert_col]
                key = "####".join([pert] + [row[c] for c in cols])
                mask = (adata.obs[pert_col].astype(str) == pert).to_numpy()
                for c in cols:
                    mask &= (adata.obs[c].astype(str) == str(row[c])).to_numpy()
                if mask.any():
                    masks[key] = mask
                    means[key] = _to_dense(adata[mask].X).mean(axis=0)
            return means, masks

        real_means, real_masks = _means_masks(test_data, order_cols)
        gen_means, gen_masks = _means_masks(generated, order_cols)
        common = sorted(set(real_means).intersection(gen_means))
        if not common:
            raise ValueError("No common (pert + covariates) between real TEST and generated.")

        # Metric accumulators
        w1 = []; w2 = []; mmd = []; energy = []
        pearson_corr = []; pearson_p = []
        spearman_corr = []; spearman_p = []
        mse_val = []

        vnames = pd.Index(test_data.var_names.astype(str))

        # For plotting
        plot_means = {}
        residuals_per_key = {}
        stats_per_key = {}
        deg_map = {}

        def maybe_filter(om, gm, td, gd, key):
            if DEG is None:
                return om, gm, td, gd
            deg = DEG.get(key) or DEG.get(key.split("####", 1)[0])
            if deg is None:
                return om, gm, td, gd
            names = None
            if isinstance(deg, dict):
                names = deg.get("names", None)
            elif hasattr(deg, "columns") and "names" in deg.columns:
                names = deg["names"]
            else:
                names = deg
            if hasattr(names, "tolist"):
                names = names.tolist()
            if not names:
                return om, gm, td, gd
            mask = np.asarray(vnames.isin([str(x) for x in names]), dtype=bool)
            if not mask.any():
                return om, gm, td, gd
            return om[mask], gm[mask], td[:, mask], gd[:, mask]

        for key in common:
            td = _to_dense(test_data.X[real_masks[key], :])
            gd = _to_dense(generated.X[gen_masks[key], :])
            om = real_means[key]; gm = gen_means[key]
            om_f, gm_f, td_f, gd_f = maybe_filter(om, gm, td, gd, key)

            # distributional metrics
            w1.append({key: compute_metrics(td_f, gd_f, 'w1')})
            w2.append({key: compute_metrics(td_f, gd_f, 'w2')})
            mmd.append({key: compute_metrics(td_f, gd_f, 'mmd')})
            energy.append({key: compute_metrics(td_f, gd_f, 'energy')})

            # mean-wise metrics
            pc, pcp = sstats.pearsonr(om_f, gm_f)
            sc, scp = sstats.spearmanr(om_f, gm_f)
            pearson_corr.append({key: pc}); pearson_p.append({key: pcp})
            spearman_corr.append({key: sc}); spearman_p.append({key: scp})
            mse = mean_squared_error(om_f, gm_f)
            mse_val.append({key: mse})

            # for plots
            plot_means[key] = (om, gm, vnames.tolist())
            residuals_per_key[key] = (gm - om)
            stats_per_key[key] = {"pearson": float(pc), "spearman": float(sc), "mse": float(mse)}
            if DEG is not None:
                deg_map[key] = DEG.get(key) or DEG.get(key.split("####", 1)[0])

        def _m(lst):
            return float("nan") if not lst else float(np.mean([list(d.values())[0] for d in lst]))

        print(f"Mean Pearson: {_m(pearson_corr):.4f} (p={_m(pearson_p):.4g})")
        print(f"Mean Spearman: {_m(spearman_corr):.4f} (p={_m(spearman_p):.4g})")
        print(f"Mean MSE: {_m(mse_val):.4f}")
        print(f"Wasserstein-1: {_m(w1):.4f}")
        print(f"Wasserstein-2: {_m(w2):.4f}")
        print(f"MMD: {_m(mmd):.4f}")
        print(f"Energy: {_m(energy):.4f}")

        results = dict(
            pearson_corr=pearson_corr,
            spearman_corr=spearman_corr,
            mse_val=mse_val,
            w1=w1,
            w2=w2,
            mmd=mmd,
            energy=energy,
        )

        # Plotting
        figures = {}
        if plot:
            plotter = EvaluationPlotter()
            # scatter grid
            fig_scatter = plotter.scatter_means_grid(
                data=plot_means,
                stats=stats_per_key,
                deg_map=deg_map if deg_map else None,
                max_panels=max_panels,
            )
            figures["scatter_means"] = fig_scatter

            # residual distributions
            fig_residuals = plotter.residuals_violin(residuals=residuals_per_key)
            figures["residuals"] = fig_residuals

            # metrics bar: combine main metrics
            metrics_pk = {}
            for k in common:
                metrics_pk[k] = {
                    "pearson": stats_per_key[k]["pearson"],
                    "spearman": stats_per_key[k]["spearman"],
                    "MSE": stats_per_key[k]["mse"],
                    "W1": float([d[k] for d in w1 if k in d][0]),
                    "W2": float([d[k] for d in w2 if k in d][0]),
                    "MMD": float([d[k] for d in mmd if k in d][0]),
                    "Energy": float([d[k] for d in energy if k in d][0]),
                }
            fig_metrics = plotter.metrics_bar(metrics_per_key=metrics_pk)
            figures["metrics_bar"] = fig_metrics

            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                fig_scatter.savefig(os.path.join(save_dir, "scatter_means.png"), dpi=dpi, bbox_inches="tight")
                fig_residuals.savefig(os.path.join(save_dir, "residuals.png"), dpi=dpi, bbox_inches="tight")
                fig_metrics.savefig(os.path.join(save_dir, "metrics_bar.png"), dpi=dpi, bbox_inches="tight")

            results["figures"] = figures

        return results