# Visualizations

GGE generates publication-quality visualizations to help interpret evaluation results.

## Available Plots

### Boxplots

Distribution of metric values across conditions.

```python
from gge.visualization import EvaluationVisualizer

visualizer = EvaluationVisualizer(results)
visualizer.plot_boxplot(metric="pearson", save_path="boxplot.png")
```

### Violin Plots

Similar to boxplots but show full distribution shape.

```python
visualizer.plot_violin(metric="pearson", save_path="violin.png")
```

### Radar Plots

Multi-metric comparison in a single view.

```python
visualizer.plot_radar(metrics=["pearson", "spearman", "wasserstein_1"], save_path="radar.png")
```

### Scatter Plots

Real vs generated expression for selected genes.

```python
visualizer.plot_scatter_grid(genes=["GAPDH", "ACTB", "TP53"], save_path="scatter.png")
```

### Embedding Plots

PCA or UMAP visualization of real vs generated data.

```python
visualizer.plot_embedding(method="pca", save_path="embedding_pca.png")
visualizer.plot_embedding(method="umap", save_path="embedding_umap.png")
```

### Heatmaps

Per-gene metric values across conditions.

```python
visualizer.plot_heatmap(metric="pearson", save_path="heatmap.png")
```

## Automatic Visualization

When using the `evaluate()` function with `output_dir`, all standard plots are automatically generated:

```python
from gge import evaluate

results = evaluate(
    real_path="real.h5ad",
    generated_path="generated.h5ad",
    condition_columns=["perturbation"],
    output_dir="output/"  # Plots saved to output/plots/
)
```

## Customization

### Figure Size and DPI

```python
visualizer.plot_boxplot(
    metric="pearson",
    figsize=(12, 8),
    dpi=300,
    save_path="boxplot_hires.png"
)
```

### Color Palettes

```python
visualizer.plot_violin(
    metric="pearson",
    palette="Set2",
    save_path="violin_custom.png"
)
```

### Styling

All plots use seaborn styling. You can customize globally:

```python
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
```

## CLI Visualization Options

```bash
gge --real real.h5ad --generated generated.h5ad \
    --conditions perturbation \
    --output results/ \
    --plots boxplot violin radar embedding
```
