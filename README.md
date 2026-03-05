# GGE: A Standardized Framework for Evaluating Gene Expression Generative Models

[![PyPI version](https://badge.fury.io/py/gge-eval.svg)](https://badge.fury.io/py/gge-eval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AndreaRubbi/GGE/actions/workflows/test.yml/badge.svg)](https://github.com/AndreaRubbi/GGE/actions)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://andrearubbi.github.io/GGE/)

> **Paper**: Accepted at the **Gen2 Workshop at ICLR 2026**

**Comprehensive, standardized evaluation of generated gene expression data.**

GGE (Generated Genetic Expression Evaluator) addresses the urgent need for standardized evaluation in single-cell gene expression generative models. Current practices suffer from inconsistent metric implementations, incomparable hyperparameter choices, and lack of biologically-grounded metrics. GGE provides:

- **Comprehensive suite of distributional metrics** with explicit computation space options
- **Biologically-motivated evaluation** through DEG-focused analysis with perturbation-effect correlation
- **Standardized reporting** for reproducible benchmarking

## Key Features

- Explicit space control (raw, PCA, DEG)
- Perturbation-effect correlation (Paper Eq. 1)
- Configurable DEG thresholds
- GPU (CUDA) and Apple MPS acceleration
- Per-gene and aggregate metrics
- Publication-quality visualizations (static and interactive)
- Simple Python API and CLI

## Metrics
All metrics are computed **per-gene** (returning a vector) and **aggregated**:

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Pearson Correlation** | Linear correlation between expression profiles | Higher is better |
| **Spearman Correlation** | Rank correlation (robust to outliers) | Higher is better |
| **R²** | Coefficient of determination | Higher is better |
| **Perturbation-Effect Correlation** | Correlation on (real - ctrl) vs (gen - ctrl) | Higher is better |
| **MSE** | Mean Squared Error | Lower is better |
| **Wasserstein-1** | Earth Mover's Distance (L1) | Lower is better |
| **Wasserstein-2** | Sinkhorn-regularized OT | Lower is better |
| **MMD** | Maximum Mean Discrepancy (RBF kernel) | Lower is better |
| **Energy Distance** | Statistical potential energy | Lower is better |

## Visualizations

- Boxplots and violin plots for metric distributions
- Radar plots for multi-metric comparison
- Scatter plots for real vs generated expression
- Embedding plots (PCA/UMAP) for real vs generated data
- Heatmaps for per-gene metric values
- Interactive Plotly plots with density overlays and metadata coloring

## Computation Spaces

GGE treats computation space as a **first-class parameter** (see Paper Section 3.3):

| Space | Description | When to Use |
|-------|-------------|-------------|
| **Raw Gene Space** | Full ~5,000–20,000 gene dimensions | Gene-level interpretability needed |
| **PCA Space** | Reduced k-dimensional space (default: 50) | Primary distributional metrics |
| **DEG Space** | Restricted to differentially expressed genes | Biologically-targeted evaluation |

**Recommendation**: Use multi-space evaluation—PCA-50 for distributional metrics, DEG for biological focus.

## Installation

```bash
pip install gge-eval
```

The package includes GPU-accelerated metrics via geomloss, which automatically falls back to CPU if no GPU is available.

## Quick Start

### Python API

```python
from gge import evaluate

# From file paths
results = evaluate(
    real_data="real_data.h5ad",
    generated_data="generated_data.h5ad",
    condition_columns=["perturbation", "cell_type"],
    split_column="split",  # Optional: for train/test
    output_dir="evaluation_output/"
)

# From AnnData objects
import scanpy as sc
real_adata = sc.read_h5ad("real_data.h5ad")
generated_adata = sc.read_h5ad("generated_data.h5ad")

results = evaluate(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
)

# Mixed (path + AnnData)
results = evaluate(
    real_data="real_data.h5ad",
    generated_data=generated_adata,
    condition_columns=["perturbation"],
)

# Access results
print(results.summary())

# Get metric for specific split
test_results = results.get_split("test")
for condition, cond_result in test_results.conditions.items():
    print(f"{condition}: Pearson={cond_result.get_metric_value('pearson'):.3f}")
```

### Command Line

```bash
# Basic usage
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation cell_type \
    --output results/

# With split column
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --split-column split \
    --splits test \
    --output results/

# Specify metrics
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --metrics pearson spearman wasserstein_1 mmd r_squared \
    --output results/
```

### DEG-Space Evaluation

GGE supports evaluating generative models specifically on differentially expressed genes (DEGs), which focuses the evaluation on the genes that matter most for capturing perturbation effects (Paper Section 4.3).

**The Problem**: Computing correlation on raw expression means can be artificially high if control and perturbed conditions have similar expression—dominated by genes similarly expressed across conditions.

**The Solution**: Perturbation-Effect Correlation (Paper Equation 1):
```
ρ_effect = corr(μ_real - μ_ctrl, μ_gen - μ_ctrl)
```

This measures whether models capture the **direction and magnitude** of perturbation effects, not just absolute expression levels.

```python
from gge import (
    evaluate_deg_space, 
    identify_degs, 
    compute_perturbation_effects,
    compute_perturbation_effect_correlation,
)
import scanpy as sc

real_adata = sc.read_h5ad("real_data.h5ad")
generated_adata = sc.read_h5ad("generated_data.h5ad")

# Evaluate in DEG space (automatically identifies DEGs)
results, deg_info = evaluate_deg_space(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
    deg_condition_column="perturbation",  # Column for DEG identification
    control_value="control",               # Control condition label
    log2fc_threshold=1.0,                  # |log2FC| > 1
    pvalue_threshold=0.05,                 # Adjusted p-value < 0.05
    return_degs=True,
)

# View identified DEGs
print(f"Found {deg_info['is_deg'].sum()} DEGs")
deg_genes = deg_info[deg_info['is_deg']]['gene'].tolist()

# Compute perturbation-effect correlation (Paper Eq. 1)
control_mask = real_adata.obs['perturbation'] == 'control'
control_mean = real_adata[control_mask].X.mean(axis=0)

perturbed_mask = real_adata.obs['perturbation'] != 'control'
rho_effect = compute_perturbation_effect_correlation(
    real_perturbed=real_adata[perturbed_mask].X,
    generated_perturbed=generated_adata[perturbed_mask].X,
    control_mean=control_mean,
    method="pearson",  # or "spearman"
)
print(f"Perturbation-effect correlation: {rho_effect:.3f}")

# Compute fold changes for analysis
effects = compute_perturbation_effects(
    real_adata,
    condition_column="perturbation",
    control_value="control",
)
```

### PC-Space Evaluation

For comparing global structure efficiently, GGE provides PC-space (principal component) evaluation (see Paper Section 3.3):

```python
from gge import evaluate_pc_space, compute_pca, PCSpaceEvaluator

# Quick evaluation in PC space
results = evaluate_pc_space(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
    n_components=50,              # Number of PCs
    use_highly_variable=True,     # Filter to HVGs first
    n_top_genes=2000,             # Number of HVGs
)
print(results.summary())

# Or use the evaluator class for more control
evaluator = PCSpaceEvaluator(n_components=50)
real_pc, gen_pc = evaluator.transform_to_pc_space(real_adata, generated_adata)

# Access PC coordinates
real_coords = real_pc.obsm['X_pca']  # shape: (n_samples, n_components)
gen_coords = gen_pc.obsm['X_pca']

# Compute PCA on a single dataset
adata_pca = compute_pca(real_adata, n_components=50)
```

### Combined Evaluation Strategy

For comprehensive evaluation, combine gene-space, DEG-space, and PC-space metrics:

```python
from gge import evaluate, evaluate_deg_space, evaluate_pc_space

# 1. Full gene-space evaluation
gene_results = evaluate(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
    metrics=["pearson", "spearman", "r_squared", "wasserstein_1", "mmd"],
)

# 2. DEG-space evaluation (perturbation-focused)
deg_results, degs = evaluate_deg_space(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
    deg_condition_column="perturbation",
    control_value="control",
    return_degs=True,
)

# 3. PC-space evaluation (global structure)
pc_results = evaluate_pc_space(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
    n_components=50,
)

print("Gene-space:", gene_results.summary())
print("DEG-space:", deg_results.summary())
print("PC-space:", pc_results.summary())
```

## Expected Data Format

GGE expects AnnData (h5ad) files with:

### Required
- `adata.X`: Gene expression matrix (samples × genes)
- `adata.var_names`: Gene identifiers (must overlap between datasets)
- `adata.obs[condition_columns]`: Columns for matching conditions

### Optional
- `adata.obs[split_column]`: Train/test split indicator

## Output Structure

```
output/
├── summary.json          # Aggregate metrics and metadata
├── results.csv           # Per-condition metrics table
├── per_gene_*.csv        # Per-gene metric values
└── plots/
    ├── boxplot_metrics.png
    ├── violin_metrics.png
    ├── radar_split.png
    ├── scatter_grid.png
    └── embedding_pca.png
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Citation

If you use GGE in your research, please cite our paper:

```bibtex
@inproceedings{rubbi2026gge,
  title = {A Standardized Framework for Evaluating Gene Expression Generative Models},
  author = {Rubbi, Andrea and [CO-AUTHORS]},
  booktitle = {Gen2 Workshop at the International Conference on Learning Representations (ICLR)},
  year = {2026},
  note = {[PROCEEDINGS DETAILS TO BE ADDED]},
  url = {https://github.com/AndreaRubbi/GGE}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.