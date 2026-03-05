# Metrics

GGE provides a comprehensive suite of metrics for evaluating generated gene expression data. All metrics are computed **per-gene** and then aggregated.

## Computation Spaces

GGE treats computation space as a **first-class parameter** (Paper Section 3.3):

| Space | Description | When to Use |
|-------|-------------|-------------|
| **Raw Gene Space** | Full ~5,000–20,000 dimensions | Gene-level interpretability needed |
| **PCA Space** | Reduced k-dimensional space | Primary distributional metrics (default: 50 PCs) |
| **DEG Space** | Restricted to differentially expressed genes | Biologically-targeted evaluation |

**Recommendation**: Use multi-space evaluation strategy—PCA-50 for distributional metrics, DEG for biological focus.

## Correlation Metrics

### Pearson Correlation

Measures linear correlation between real and generated expression profiles.

```python
from gge.metrics import PearsonCorrelation

metric = PearsonCorrelation()
result = metric.compute(real_data, generated_data)
print(f"Mean Pearson: {result.aggregate_value:.3f}")
```

**Interpretation**: Values range from -1 to 1. Higher values indicate better agreement.

### Spearman Correlation

Rank-based correlation, robust to outliers and non-linear relationships.

```python
from gge.metrics import SpearmanCorrelation

metric = SpearmanCorrelation()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Values range from -1 to 1. Higher values indicate better agreement.

### R² (Coefficient of Determination)

Measures the proportion of variance in the real data explained by the generated data.

```python
from gge.metrics import RSquared

metric = RSquared()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Values typically 0 to 1 (can be negative for poor fits). Higher is better.

### Perturbation-Effect Correlation

**(Paper Equation 1)**: Measures whether models capture the **direction and magnitude** of perturbation effects.

```python
from gge import compute_perturbation_effect_correlation

# Compute control mean
control_mean = real_adata[real_adata.obs['condition'] == 'control'].X.mean(axis=0)

# Get perturbed samples
mask = real_adata.obs['condition'] != 'control'
rho = compute_perturbation_effect_correlation(
    real_perturbed=real_adata[mask].X,
    generated_perturbed=gen_adata[mask].X,
    control_mean=control_mean,
    method="pearson",  # or "spearman"
)
print(f"Perturbation-effect correlation: {rho:.3f}")
```

The formula is:
```
ρ_effect = corr(μ_real - μ_ctrl, μ_gen - μ_ctrl)
```

**Why this matters**: Computing correlation on raw expression means can be artificially high if control and perturbed conditions have similar expression. This metric focuses on whether the perturbation *effect* is captured.

**Interpretation**: Values range from -1 to 1. Higher values indicate the model better captures perturbation effects.

## Distribution Metrics

### Wasserstein-1 Distance (Earth Mover's Distance)

Measures the minimum "work" required to transform one distribution into another using L1 ground distance.

```python
from gge.metrics import Wasserstein1Distance

metric = Wasserstein1Distance()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Non-negative values. Lower is better.

### Wasserstein-2 Distance

Sinkhorn-regularized optimal transport distance with quadratic cost.

```python
from gge.metrics import Wasserstein2Distance

metric = Wasserstein2Distance()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Non-negative values. Lower is better.

### Maximum Mean Discrepancy (MMD)

Kernel-based two-sample test statistic using RBF kernel with median heuristic bandwidth.

```python
from gge.metrics import MMDDistance

metric = MMDDistance()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Non-negative values. Lower indicates distributions are more similar.

### Energy Distance

Statistical distance based on potential energy concepts.

```python
from gge.metrics import EnergyDistance

metric = EnergyDistance()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Non-negative values. Lower is better.

### Mean Squared Error (MSE)

Simple reconstruction metric measuring average squared difference.

```python
from gge.metrics import MSEDistance

metric = MSEDistance()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Non-negative values. Lower is better.

## Metric Summary Table

| Metric | Type | Range | Optimal |
|--------|------|-------|---------|
| Pearson | Correlation | [-1, 1] | Higher |
| Spearman | Correlation | [-1, 1] | Higher |
| R² | Reconstruction | (-∞, 1] | Higher |
| Perturbation-Effect | Correlation | [-1, 1] | Higher |
| MSE | Reconstruction | [0, ∞) | Lower |
| Wasserstein-1 | Distribution | [0, ∞) | Lower |
| Wasserstein-2 | Distribution | [0, ∞) | Lower |
| MMD | Distribution | [0, ∞) | Lower |
| Energy | Distribution | [0, ∞) | Lower |

## Using Multiple Metrics

```python
from gge import evaluate

results = evaluate(
    real_path="real.h5ad",
    generated_path="generated.h5ad",
    condition_columns=["perturbation"],
    metrics=["pearson", "spearman", "wasserstein_1", "mmd"]
)
```

## Custom Metrics

You can implement custom metrics by extending `BaseMetric`:

```python
from gge.metrics.base_metric import BaseMetric, MetricResult

class MyCustomMetric(BaseMetric):
    name = "custom"
    higher_is_better = True
    
    def _compute_per_gene(self, real, generated):
        # Your computation here
        return per_gene_values
```
