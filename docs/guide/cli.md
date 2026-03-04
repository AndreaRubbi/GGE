# CLI Reference

GGE provides a comprehensive command-line interface for running evaluations.

## Basic Usage

```bash
gge --real <real_data.h5ad> --generated <generated_data.h5ad> --conditions <columns> --output <dir>
```

## Arguments

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--real` | `-r` | Path to real data file (h5ad format) |
| `--generated` | `-g` | Path to generated data file (h5ad format) |
| `--conditions` | `-c` | One or more condition columns for matching |
| `--output` | `-o` | Output directory for results |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--split-column` | None | Column name for train/test split |
| `--splits` | all | Which splits to evaluate (e.g., "test") |
| `--metrics` | all | Which metrics to compute |
| `--n-genes` | all | Number of genes to evaluate (random subset) |
| `--seed` | 42 | Random seed for reproducibility |
| `--verbose` | False | Print detailed progress |
| `--version` | - | Show version and exit |

## Examples

### Basic Evaluation

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --output results/
```

### Multiple Conditions

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation cell_type dose \
    --output results/
```

### Test Set Only

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --split-column split \
    --splits test \
    --output results/
```

### Specific Metrics

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --metrics pearson spearman wasserstein_1 \
    --output results/
```

### Subset of Genes

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --n-genes 1000 \
    --seed 42 \
    --output results/
```

### Verbose Output

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --verbose \
    --output results/
```

## Available Metrics

Use these names with `--metrics`:

- `pearson` - Pearson correlation
- `spearman` - Spearman correlation  
- `wasserstein_1` - Wasserstein-1 distance
- `wasserstein_2` - Wasserstein-2 distance
- `mmd` - Maximum Mean Discrepancy
- `energy` - Energy distance

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid arguments, file not found, etc.) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GGE_SEED` | Default random seed |
| `GGE_VERBOSE` | Default verbosity (0 or 1) |
