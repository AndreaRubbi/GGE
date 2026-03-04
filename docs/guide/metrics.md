# Metrics

GGE provides a comprehensive suite of metrics for evaluating generated gene expression data. All metrics are computed **per-gene** and then aggregated.

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

Quadratic optimal transport distance, more sensitive to distribution shape.

```python
from gge.metrics import Wasserstein2Distance

metric = Wasserstein2Distance()
result = metric.compute(real_data, generated_data)
```

**Interpretation**: Non-negative values. Lower is better.

### Maximum Mean Discrepancy (MMD)

Kernel-based two-sample test statistic using RBF kernel.

```python
from gge.metrics import MMDMetric

metric = MMDMetric(kernel="rbf")
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

## Metric Summary Table

| Metric | Type | Range | Optimal |
|--------|------|-------|---------|
| Pearson | Correlation | [-1, 1] | Higher |
| Spearman | Correlation | [-1, 1] | Higher |
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
