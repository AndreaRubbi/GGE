# GGE: Generated Genetic Expression Evaluator

[![PyPI version](https://badge.fury.io/py/gge-eval.svg)](https://badge.fury.io/py/gge-eval)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/AndreaRubbi/GGE/actions/workflows/test.yml/badge.svg)](https://github.com/AndreaRubbi/GGE/actions)

**Comprehensive evaluation of generated gene expression data against real datasets.**

## Overview

GGE is a modular, object-oriented Python framework for computing metrics between real and generated gene expression datasets stored in AnnData (h5ad) format. It supports condition-based matching, train/test splits, and generates publication-quality visualizations.

## Key Features

- **Multiple Metrics**: Pearson/Spearman correlation, Wasserstein distances, MMD, Energy distance
- **Per-gene Analysis**: All metrics computed per-gene with aggregation options
- **Condition Matching**: Match samples by perturbation, cell type, or other metadata
- **Train/Test Splits**: Evaluate on held-out data
- **Visualizations**: Boxplots, violin plots, radar charts, scatter plots, embeddings
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

If you use GGE in your research, please cite:

```bibtex
@software{gge2026,
  title = {GGE: Generated Genetic Expression Evaluator},
  author = {Rubbi, Andrea},
  year = {2026},
  url = {https://github.com/AndreaRubbi/GGE}
}
```

## License

This project is licensed under the MIT License.

We would like to thank the contributors and the community for their support in developing this project.