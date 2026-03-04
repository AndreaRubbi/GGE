# Quick Start

This guide will get you up and running with GGE in minutes.

## Basic Usage

### Python API

```python
from gge import evaluate

# Run evaluation
results = evaluate(
    real_path="real_data.h5ad",
    generated_path="generated_data.h5ad",
    condition_columns=["perturbation", "cell_type"],
    split_column="split",  # Optional: for train/test evaluation
    output_dir="evaluation_output/"
)

# View summary
print(results.summary())

# Access specific results
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

# With train/test split
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

## Input Data Format

GGE expects AnnData (`.h5ad`) files with:

### Required

| Component | Description |
|-----------|-------------|
| `adata.X` | Gene expression matrix (samples × genes) |
| `adata.var_names` | Gene identifiers (must overlap between datasets) |
| `adata.obs[condition_columns]` | Columns for matching conditions |

### Optional

| Component | Description |
|-----------|-------------|
| `adata.obs[split_column]` | Train/test split indicator |

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

## Available Metrics

| Metric | Key | Direction |
|--------|-----|-----------|
| Pearson Correlation | `pearson` | Higher is better |
| Spearman Correlation | `spearman` | Higher is better |
| Wasserstein-1 Distance | `wasserstein_1` | Lower is better |
| Wasserstein-2 Distance | `wasserstein_2` | Lower is better |
| MMD | `mmd` | Lower is better |
| Energy Distance | `energy` | Lower is better |

## Next Steps

- Learn about [available metrics](guide/metrics.md)
- Explore [visualization options](guide/visualizations.md) 
- See the [CLI reference](guide/cli.md)
- Check the [API reference](api.md)
