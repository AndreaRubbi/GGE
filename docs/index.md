# GGE: A Standardized Framework for Evaluating Gene Expression Generative Models

[![PyPI version](https://badge.fury.io/py/gge-eval.svg)](https://badge.fury.io/py/gge-eval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AndreaRubbi/GGE/actions/workflows/test.yml/badge.svg)](https://github.com/AndreaRubbi/GGE/actions)

> **Paper**: Accepted at the **Gen2 Workshop at ICLR 2026**

**Comprehensive, standardized evaluation of generated gene expression data.**

## Overview

GGE (Generated Genetic Expression Evaluator) addresses the urgent need for standardized evaluation in single-cell gene expression generative models. Current practices suffer from:

- Inconsistent metric implementations
- Incomparable hyperparameter choices
- Lack of biologically-grounded metrics

GGE provides:

- **Comprehensive suite of distributional metrics** with explicit computation space options (raw, PCA, DEG)
- **Biologically-motivated evaluation** through DEG-focused analysis with perturbation-effect correlation
- **Standardized reporting** for reproducible benchmarking
- **GPU (CUDA) and Apple MPS acceleration** for efficient computation

## Key Features

- **Explicit Space Control**: Compute metrics in raw gene space, PCA space, or DEG-restricted space
- **Perturbation-Effect Correlation**: Paper Equation 1: ρ_effect = corr(μ_real - μ_ctrl, μ_gen - μ_ctrl)
- **Multiple Metrics**: Pearson, Spearman, R², MSE, Wasserstein, MMD, Energy distance
- **Per-gene Analysis**: All metrics computed per-gene with aggregation options
- **Condition Matching**: Match samples by perturbation, cell type, or other metadata
- **Train/Test Splits**: Evaluate on held-out data
- **Visualizations**: Boxplots, violin plots, radar charts, scatter plots, embeddings, interactive Plotly
- **CLI & API**: Use from command line or Python

## Quick Installation

```bash
pip install gge-eval
```

## Quick Example

```python
from gge import evaluate

results = evaluate(
    real_path="real_data.h5ad",
    generated_path="generated_data.h5ad",
    condition_columns=["perturbation"],
    output_dir="output/"
)

print(results.summary())
```

## Citation

If you use GGE in your research, please cite our paper:

```bibtex
@inproceedings{rubbi2026gge,
  title = {A Standardized Framework for Evaluating Gene Expression Generative Models},
  author = {Rubbi, Andrea},
  booktitle = {Gen2 Workshop at the International Conference on Learning Representations (ICLR)},
  year = {2026},
  url = {https://github.com/AndreaRubbi/GGE}
}
```

## License

This project is licensed under the MIT License.

We would like to thank the contributors and the community for their support in developing this project.