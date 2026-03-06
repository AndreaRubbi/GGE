# API Reference

This page provides detailed API documentation for GGE's main functions and classes.

---

## Main Evaluation Functions

### evaluate

```python
gge.evaluate(
    real_data,
    generated_data,
    condition_columns,
    split_column=None,
    output_dir=None,
    metrics=None,
    include_multivariate=False,
    control_key=None,
    control_column=None,
    verbose=True,
    **loader_kwargs
) -> EvaluationResult
```

Convenience function to run full evaluation.

All metrics support computation in different spaces (raw, pca, deg) through their space parameter.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_data` | str, Path, or AnnData | Path to real data h5ad file or AnnData object |
| `generated_data` | str, Path, or AnnData | Path to generated data h5ad file or AnnData object |
| `condition_columns` | List[str] | Columns to match between datasets |
| `split_column` | str, optional | Column indicating train/test split |
| `output_dir` | str or Path, optional | Directory to save results |
| `metrics` | List, optional | Metrics to compute (default: all paper metrics) |
| `include_multivariate` | bool | Whether to include multivariate metrics |
| `control_key` | str, optional | Value identifying control samples for DEG space |
| `control_column` | str, optional | Column containing control identifier |
| `verbose` | bool | Print progress |

**Returns:** `EvaluationResult` - Complete evaluation results

**Example:**

```python
from gge import evaluate

# From paths
results = evaluate(
    "real.h5ad",
    "generated.h5ad",
    condition_columns=["perturbation", "cell_type"],
    output_dir="evaluation_output/"
)

# From AnnData objects
results = evaluate(
    real_adata,
    generated_adata,
    condition_columns=["perturbation"],
)
```

---

### evaluate_lazy

```python
gge.evaluate_lazy(
    real_path,
    generated_path,
    condition_columns,
    metrics,
    control_key=None,
    control_column=None,
    split_column=None,
    output_dir=None,
    verbose=True,
    **loader_kwargs
) -> EvaluationResult
```

Lazy-loading evaluation with explicit metric configuration. Each metric can specify its own space (raw, pca, or deg).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_path` | str | Path to real gene expression data |
| `generated_path` | str | Path to generated gene expression data |
| `condition_columns` | str or List[str] | Column(s) for condition-wise stratification |
| `metrics` | List[BaseMetric] | List of metric instances with space configurations |
| `control_key` | str, optional | Value identifying control samples (required for DEG space) |
| `control_column` | str, optional | Column containing control identifier |
| `split_column` | str, optional | Column for train/test split evaluation |
| `output_dir` | str, optional | Directory to save results |

**Returns:** `EvaluationResult`

**Example:**

```python
from gge import evaluate_lazy
from gge.metrics import PearsonCorrelation, Wasserstein2Distance, MMDDistance

metrics = [
    PearsonCorrelation(space="deg", deg_lfc=0.25, deg_pval=0.1),
    Wasserstein2Distance(space="pca", n_components=50),
    MMDDistance(space="pca", n_components=50),
]

results = evaluate_lazy(
    "real_data.h5ad",
    "generated_data.h5ad",
    condition_columns="perturbation",
    control_key="ctrl",
    metrics=metrics
)
```

---

### evaluate_deg_space

```python
gge.evaluate_deg_space(
    real_data,
    generated_data,
    condition_columns,
    deg_condition_column,
    control_value,
    treatment_value=None,
    log2fc_threshold=1.0,
    pvalue_threshold=0.05,
    split_column=None,
    output_dir=None,
    metrics=None,
    verbose=True,
    return_degs=False,
    **kwargs
)
```

Evaluate in DEG (differentially expressed genes) space.

Identifies DEGs from real data and evaluates both datasets restricted to those genes.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_data` | str, Path, or AnnData | Real gene expression data |
| `generated_data` | str, Path, or AnnData | Generated gene expression data |
| `condition_columns` | List[str] | Columns for condition matching |
| `deg_condition_column` | str | Column containing perturbation labels |
| `control_value` | str | Value identifying control samples |
| `treatment_value` | str, optional | Value identifying treatment (if None, all non-control) |
| `log2fc_threshold` | float | Minimum absolute log2 fold change for DEGs |
| `pvalue_threshold` | float | Maximum p-value/FDR for DEGs |
| `return_degs` | bool | If True, return (results, deg_df) tuple |

**Returns:** `EvaluationResult` or `Tuple[EvaluationResult, pd.DataFrame]`

**Example:**

```python
from gge import evaluate_deg_space

results, deg_info = evaluate_deg_space(
    real_data=real_adata,
    generated_data=generated_adata,
    condition_columns=["perturbation"],
    deg_condition_column="perturbation",
    control_value="control",
    log2fc_threshold=1.0,
    pvalue_threshold=0.05,
    return_degs=True,
)

print(f"Found {deg_info['is_deg'].sum()} DEGs")
```

---

### evaluate_pc_space

```python
gge.evaluate_pc_space(
    real_data,
    generated_data,
    condition_columns=None,
    n_components=50,
    use_highly_variable=True,
    n_top_genes=2000,
    metrics=None,
    verbose=True
) -> EvaluationResult
```

Evaluate in PC (principal component) space.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_data` | AnnData, str, or Path | Real/reference data |
| `generated_data` | AnnData, str, or Path | Generated data |
| `condition_columns` | List[str], optional | Columns for condition matching |
| `n_components` | int | Number of principal components (default: 50) |
| `use_highly_variable` | bool | Filter to HVGs before PCA |
| `n_top_genes` | int | Number of HVGs to use |

**Returns:** `EvaluationResult`

**Example:**

```python
from gge import evaluate_pc_space

results = evaluate_pc_space(
    real_data="real.h5ad",
    generated_data="generated.h5ad",
    condition_columns=["perturbation"],
    n_components=50,
)
print(results.summary())
```

---

## DEG Utilities

### identify_degs

```python
gge.identify_degs(
    adata,
    condition_column,
    control_value,
    treatment_value=None,
    log2fc_threshold=1.0,
    pvalue_threshold=0.05,
    method="ttest",
    use_fdr=True
) -> pd.DataFrame
```

Identify differentially expressed genes between conditions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `adata` | AnnData | Gene expression data with condition annotations |
| `condition_column` | str | Column containing condition labels |
| `control_value` | str | Value identifying control/baseline samples |
| `treatment_value` | str, optional | Value identifying treatment samples |
| `log2fc_threshold` | float | Minimum absolute log2 fold change |
| `pvalue_threshold` | float | Maximum p-value (or FDR) threshold |
| `method` | str | Statistical test: 'ttest' or 'wilcoxon' |
| `use_fdr` | bool | Apply Benjamini-Hochberg FDR correction |

**Returns:** `pd.DataFrame` with columns: `gene`, `log2fc`, `pvalue`, `fdr`, `is_deg`

**Example:**

```python
from gge import identify_degs

degs = identify_degs(
    adata,
    condition_column="perturbation",
    control_value="control",
    log2fc_threshold=1.0
)
deg_genes = degs[degs['is_deg']]['gene'].tolist()
```

---

### compute_perturbation_effects

```python
gge.compute_perturbation_effects(
    adata,
    condition_column,
    control_value
) -> pd.DataFrame
```

Compute log2 fold changes for all perturbations vs control.

**Returns:** DataFrame with genes as rows and perturbations as columns, values are log2 fold changes.

---

### compute_perturbation_effect_correlation

```python
gge.compute_perturbation_effect_correlation(
    real_perturbed,
    generated_perturbed,
    control_mean,
    method="pearson"
) -> float
```

Compute perturbation-effect correlation (Paper Equation 1):

$$\rho_{effect} = \text{corr}(\mu_{real} - \mu_{ctrl}, \mu_{gen} - \mu_{ctrl})$$

Measures whether models capture the **direction and magnitude** of perturbation effects.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `real_perturbed` | ndarray | Real perturbed expression (samples × genes) |
| `generated_perturbed` | ndarray | Generated perturbed expression |
| `control_mean` | ndarray | Mean expression of control samples |
| `method` | str | 'pearson' or 'spearman' |

**Returns:** Correlation coefficient (float)

---

## PC-Space Utilities

### compute_pca

```python
gge.compute_pca(adata, n_components=50) -> AnnData
```

Compute PCA on a single dataset.

**Returns:** AnnData with `obsm['X_pca']` containing PC coordinates.

---

### PCSpaceEvaluator

```python
class gge.PCSpaceEvaluator(n_components=50, use_highly_variable=True, n_top_genes=2000)
```

Evaluator for PC-space transformations.

**Methods:**

- `transform_to_pc_space(real_data, generated_data) -> Tuple[AnnData, AnnData]`
  
  Transform both datasets to shared PC space. Returns (real_pc, gen_pc) tuple.

**Example:**

```python
from gge import PCSpaceEvaluator

evaluator = PCSpaceEvaluator(n_components=50)
real_pc, gen_pc = evaluator.transform_to_pc_space(real_adata, generated_adata)

# Access PC coordinates
real_coords = real_pc.obsm['X_pca']  # shape: (n_samples, 50)
```

---

## Metrics

All metrics inherit from `BaseMetric` and support the `space` parameter.

### Correlation Metrics

| Class | Description | Direction |
|-------|-------------|-----------|
| `PearsonCorrelation(space="raw")` | Linear correlation | Higher is better |
| `SpearmanCorrelation(space="raw")` | Rank correlation | Higher is better |
| `RSquared(space="raw")` | Coefficient of determination | Higher is better |

### Distance Metrics

| Class | Description | Direction |
|-------|-------------|-----------|
| `Wasserstein1Distance(space="raw")` | Earth Mover's Distance (L1) | Lower is better |
| `Wasserstein2Distance(space="raw")` | Sinkhorn-regularized OT | Lower is better |
| `MMDDistance(space="raw")` | Maximum Mean Discrepancy | Lower is better |
| `EnergyDistance(space="raw")` | Statistical potential energy | Lower is better |

**Space Parameter:**

```python
from gge.metrics import PearsonCorrelation, MMDDistance

# Correlation in DEG space
pearson_deg = PearsonCorrelation(space="deg", deg_lfc=0.25, deg_pval=0.1)

# MMD in PCA space with 50 components
mmd_pca = MMDDistance(space="pca", n_components=50)
```

---

## Results Classes

### EvaluationResult

Main results container.

**Methods:**

| Method | Description |
|--------|-------------|
| `summary()` | Human-readable summary string |
| `get_split(name)` | Get results for a specific split |
| `save(output_dir)` | Save results to directory |
| `to_dataframe()` | Convert to pandas DataFrame |

### ConditionResult

Results for a single condition.

**Methods:**

| Method | Description |
|--------|-------------|
| `get_metric_value(name)` | Get aggregate value for metric |
| `get_per_gene_values(name)` | Get per-gene metric values |

---

## Visualization

### visualize

```python
gge.visualize(results, output_dir, **kwargs)
```

Generate all default visualizations for evaluation results.

### EvaluationVisualizer

```python
class gge.EvaluationVisualizer(results, output_dir=None)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `boxplot_metrics()` | Boxplots of metric distributions |
| `violin_metrics()` | Violin plots of metrics |
| `radar_plot()` | Radar/spider plot for multi-metric comparison |
| `scatter_grid()` | Scatter plots of real vs generated |
| `embedding_plot()` | PCA/UMAP embedding visualization |
| `heatmap()` | Heatmap of per-gene metrics |

---

## Data Loading

### load_data

```python
gge.load_data(
    real_data,
    generated_data,
    condition_columns,
    split_column=None,
    **kwargs
) -> GeneExpressionDataLoader
```

Load and align gene expression datasets.

**Returns:** `GeneExpressionDataLoader` instance with aligned data.
