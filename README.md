# GGE: Generated Genetic Expression Evaluator

[![PyPI version](https://badge.fury.io/py/gge-eval.svg)](https://badge.fury.io/py/gge-eval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AndreaRubbi/GGE/actions/workflows/test.yml/badge.svg)](https://github.com/AndreaRubbi/GGE/actions)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://andrearubbi.github.io/GGE/)

**Comprehensive evaluation of generated gene expression data against real datasets.**

GGE is a modular, object-oriented Python framework for computing metrics between real and generated gene expression datasets stored in AnnData (h5ad) format. It supports condition-based matching, train/test splits, and generates publication-quality visualizations.

## Features

### Metrics
All metrics are computed **per-gene** (returning a vector) and **aggregated**:

| Metric | Description | Direction |
|--------|-------------|-----------|
| **Pearson Correlation** | Linear correlation between expression profiles | Higher is better |
| **Spearman Correlation** | Rank correlation (robust to outliers) | Higher is better |
| **Wasserstein-1** | Earth Mover's Distance (L1) | Lower is better |
| **Wasserstein-2** | Quadratic optimal transport | Lower is better |
| **MMD** | Maximum Mean Discrepancy (kernel-based) | Lower is better |
| **Energy Distance** | Statistical potential energy | Lower is better |

### Visualizations
- **Boxplots & Violin plots**: Metric distributions across conditions
- **Radar plots**: Multi-metric comparison
- **Scatter plots**: Real vs generated expression
- **Embedding plots**: PCA/UMAP of real vs generated data
- **Heatmaps**: Per-gene metric values
- **Interactive Plotly plots**: Density overlays, embeddings with metadata coloring

### Key Features
- ✅ Condition-based matching (perturbation, cell type, etc.)
- ✅ Train/test split support
- ✅ Per-gene and aggregate metrics
- ✅ Modular, extensible architecture
- ✅ Command-line interface
- ✅ Publication-quality visualizations

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
    --metrics pearson spearman wasserstein_1 mmd \
    --output results/
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

## License

This project is licensed under the MIT License. See the LICENSE file for details.