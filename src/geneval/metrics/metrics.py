from geomloss import SamplesLoss
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from . import metric_MMD

class Metric():
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn

    def compute(self, x, y):
        return self.fn(x, y)

class PerturbationMetric():
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn

    def compute(self, adata_true: ad.AnnData, adata_generated: ad.AnnData, groupby: str):
        return self.fn(adata_true, adata_generated, groupby)

def compute_metrics(original_data, generated_data, metric_fn):
    metric_funcs = {
        'w1': SamplesLoss(loss="sinkhorn", p=1, blur=0.01),
        'w2': SamplesLoss(loss="sinkhorn", p=2, blur=0.01),
        'mmd': metric_MMD.iface_compute_MMD, 
        'energy': SamplesLoss(loss="energy", blur=0.5),
    }
    metric_fn = metric_funcs[metric_fn]
    original_data = torch.tensor(original_data)
    generated_data = torch.tensor(generated_data)
    metric = metric_fn(generated_data, original_data)
    return metric.item()

def W1(x, y):
    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, backend="tensorized")
    return loss_fn(x, y).item()

def W2(x, y):
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, backend="tensorized")
    return loss_fn(x, y).item()

def W1_complete(x, y, preprocess=False):
    if preprocess:
        x = scanpy_preprocessing(x)
        y = scanpy_preprocessing(y)

    x_reduced = scanpy_pca(x)
    y_reduced = scanpy_pca(y)

    x_pca = torch.tensor(x_reduced.obsm['X_pca'], dtype=torch.float32)
    y_pca = torch.tensor(y_reduced.obsm['X_pca'], dtype=torch.float32)
    
    return W1(x_pca, y_pca)

def get_deg_genes(adata: ad.AnnData, groupby: str = "condition_ID", method: str = "wilcoxon", alpha: float = 0.05):
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False, n_genes=adata.shape[1])
    
    degs = set()
    rg_results = adata.uns["rank_genes_groups"]
    
    for group in rg_results["names"].dtype.names:
        pvals_adj = rg_results["pvals_adj"][group]
        genes = rg_results["names"][group]
        
        for gene, pval in zip(genes, pvals_adj):
            if pval < alpha:
                degs.add(gene)
    
    return degs

def get_avg_expression(adata: ad.AnnData, genes: set) -> pd.Series:
    common_genes = list(set(adata.var_names).intersection(genes))
    if len(common_genes) == 0:
        return pd.Series(dtype=float)
    
    sub_adata = adata[:, common_genes]
    avg_exp = np.array(sub_adata.X.mean(axis=0)).ravel()
    
    return pd.Series(data=avg_exp, index=common_genes)

def pearson_dict(x, y):
    common_keys = set(x.keys()).intersection(y.keys())
    true_values = [x[key] for key in common_keys]
    calculated_values = [y[key] for key in common_keys]
    correlation, _ = pearsonr(true_values, calculated_values)
    
    return correlation

def spearman_dict(x, y):
    common_keys = set(x.keys()).intersection(y.keys())
    true_values = [x[key] for key in common_keys]
    calculated_values = [y[key] for key in common_keys]
    correlation, _ = spearmanr(true_values, calculated_values)
    
    return correlation

def mse_dict(x, y):
    common_keys = set(x.keys()).intersection(y.keys())
    true_values = np.array([x[key] for key in common_keys])
    calculated_values = np.array([y[key] for key in common_keys])
    mse = np.mean((true_values - calculated_values) ** 2)

    return mse

def compute_pearson(x, y):
    common_genes = x.index.intersection(y.index)
    
    if len(common_genes) == 0:
        return float('nan')
    
    x_vals = x.loc[common_genes].values
    y_vals = y.loc[common_genes].values
    
    pearson_corr, _ = pearsonr(x_vals, y_vals)
    
    return pearson_corr

def compute_spearman(x, y):
    common_genes = x.index.intersection(y.index)
    
    if len(common_genes) == 0:
        return float('nan')
    
    x_vals = x.loc[common_genes].values
    y_vals = y.loc[common_genes].values
    
    spearman_corr, _ = spearmanr(x_vals, y_vals)
    
    return spearman_corr