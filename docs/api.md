# API Reference

This page provides detailed API documentation for GGE's main functions and classes.

## Main Evaluation Functions

### evaluate

::: gge.evaluate
    options:
        show_root_heading: true
        show_source: false

### evaluate_lazy

::: gge.evaluate_lazy
    options:
        show_root_heading: true
        show_source: false

### evaluate_deg_space

::: gge.evaluate_deg_space
    options:
        show_root_heading: true
        show_source: false

### evaluate_pc_space

::: gge.evaluate_pc_space
    options:
        show_root_heading: true
        show_source: false

---

## DEG Utilities

### identify_degs

::: gge.identify_degs
    options:
        show_root_heading: true
        show_source: false

### compute_perturbation_effects

::: gge.compute_perturbation_effects
    options:
        show_root_heading: true
        show_source: false

### compute_perturbation_effect_correlation

::: gge.compute_perturbation_effect_correlation
    options:
        show_root_heading: true
        show_source: false

---

## PC-Space Utilities

### compute_pca

::: gge.compute_pca
    options:
        show_root_heading: true
        show_source: false

### PCSpaceEvaluator

::: gge.PCSpaceEvaluator
    options:
        show_root_heading: true
        show_source: false
        members:
            - transform_to_pc_space
            - fit
            - transform

---

## Metrics

All metrics inherit from `BaseMetric` and support the `space` parameter for computing in different spaces (`raw`, `pca`, `deg`).

### Correlation Metrics

::: gge.metrics.PearsonCorrelation
    options:
        show_root_heading: true
        show_source: false

::: gge.metrics.SpearmanCorrelation
    options:
        show_root_heading: true
        show_source: false

::: gge.metrics.RSquared
    options:
        show_root_heading: true
        show_source: false

### Distance Metrics

::: gge.metrics.Wasserstein1Distance
    options:
        show_root_heading: true
        show_source: false

::: gge.metrics.Wasserstein2Distance
    options:
        show_root_heading: true
        show_source: false

::: gge.metrics.MMDDistance
    options:
        show_root_heading: true
        show_source: false

::: gge.metrics.EnergyDistance
    options:
        show_root_heading: true
        show_source: false

---

## Results Classes

### EvaluationResult

::: gge.EvaluationResult
    options:
        show_root_heading: true
        show_source: false
        members:
            - summary
            - get_split
            - save
            - to_dataframe

### SplitResult

::: gge.SplitResult
    options:
        show_root_heading: true
        show_source: false

### ConditionResult

::: gge.ConditionResult
    options:
        show_root_heading: true
        show_source: false
        members:
            - get_metric_value
            - get_per_gene_values

---

## Data Loading

### load_data

::: gge.load_data
    options:
        show_root_heading: true
        show_source: false

### GeneExpressionDataLoader

::: gge.GeneExpressionDataLoader
    options:
        show_root_heading: true
        show_source: false

---

## Visualization

### visualize

::: gge.visualize
    options:
        show_root_heading: true
        show_source: false

### EvaluationVisualizer

::: gge.EvaluationVisualizer
    options:
        show_root_heading: true
        show_source: false
        members:
            - boxplot_metrics
            - violin_metrics
            - radar_plot
            - scatter_grid
            - embedding_plot
            - heatmap
