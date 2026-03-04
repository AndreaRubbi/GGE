# API Reference

## Main Functions

### evaluate

The main entry point for running evaluations.

```python
from gge import evaluate

results = evaluate(
    real_path: str | Path | AnnData,
    generated_path: str | Path | AnnData,
    condition_columns: List[str],
    split_column: Optional[str] = None,
    splits: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    output_dir: Optional[str | Path] = None,
    n_genes: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> EvaluationResult
```

**Parameters:**

- `real_path`: Path to real data h5ad file or AnnData object
- `generated_path`: Path to generated data h5ad file or AnnData object
- `condition_columns`: Column names to use for condition matching
- `split_column`: Optional column for train/test split
- `splits`: Which splits to evaluate (default: all)
- `metrics`: Which metrics to compute (default: all)
- `output_dir`: Directory to save results and plots
- `n_genes`: Evaluate subset of genes (default: all)
- `seed`: Random seed for reproducibility
- `verbose`: Print progress information

**Returns:** `EvaluationResult` object containing all computed metrics

## Result Classes

### EvaluationResult

```python
class EvaluationResult:
    splits: Dict[str, SplitResult]
    metadata: Dict[str, Any]
    
    def summary(self) -> str: ...
    def get_split(self, name: str) -> SplitResult: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def save(self, path: str | Path) -> None: ...
```

### SplitResult

```python
class SplitResult:
    name: str
    conditions: Dict[str, ConditionResult]
    
    def get_condition(self, name: str) -> ConditionResult: ...
```

### ConditionResult

```python
class ConditionResult:
    condition: str
    metrics: Dict[str, MetricResult]
    
    def get_metric_value(self, name: str) -> float: ...
    def get_per_gene_values(self, name: str) -> np.ndarray: ...
```

## Metrics

### Base Classes

```python
from gge.metrics.base_metric import BaseMetric, MetricResult

class BaseMetric:
    name: str
    higher_is_better: bool
    
    def compute(self, real: np.ndarray, generated: np.ndarray) -> MetricResult: ...
```

### Available Metrics

```python
from gge.metrics import (
    PearsonCorrelation,
    SpearmanCorrelation,
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDMetric,
    EnergyDistance,
)
```

## Data Loading

```python
from gge.data import GeneExpressionDataLoader, load_data

# Load single file
adata = load_data("data.h5ad")

# Use loader for paired data
loader = GeneExpressionDataLoader(
    real_path="real.h5ad",
    generated_path="generated.h5ad",
    condition_columns=["perturbation"],
)
```

## Visualization

```python
from gge.visualization import EvaluationVisualizer

visualizer = EvaluationVisualizer(results)
visualizer.plot_boxplot(metric="pearson")
visualizer.plot_violin(metric="pearson")
visualizer.plot_radar(metrics=["pearson", "spearman"])
visualizer.plot_embedding(method="pca")
```

## Testing Utilities

```python
from gge.testing import MockDataGenerator

# Generate test data
generator = MockDataGenerator(n_samples=100, n_genes=50, seed=42)
real, generated = generator.generate_paired_data(noise_level=0.3)
```
