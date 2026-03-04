# GGE: Generated Genetic Expression Evaluator

Welcome to the Generated Genetic Expression Evaluator (GGE)! This project provides a comprehensive framework for evaluating gene expression data through various metrics and methodologies. 

## Overview

GGE is designed to facilitate the comparison of real and generated gene expression profiles. It includes functionalities for loading datasets, computing metrics, and visualizing results. The system is built with modularity in mind, allowing for easy extension and integration with other projects.

## Features

- **Modular Design**: The project is structured into distinct modules for data handling, evaluation metrics, and model definitions.
- **Metrics Computation**: Includes various metrics such as Wasserstein distances, Pearson and Spearman correlations, and Mean Squared Error (MSE) for evaluating gene expression profiles.
- **Command-Line Interface**: Users can run evaluations directly from the terminal, making it easy to integrate into workflows.
- **Documentation**: Comprehensive documentation is provided to guide users through installation, usage, and examples.

## Installation

To install GGE from PyPI:

```
pip install gge-eval
```

Or install from source:

```
pip install -e .
```

## Usage

To run an evaluation, you can use the command-line interface or execute the example script provided in the `examples` directory. 

### Example

```python
from gge import evaluate

# Initialize the evaluator with your data
results = evaluate(
    real_path="real_data.h5ad",
    generated_path="generated_data.h5ad",
    condition_columns=["perturbation"],
    output_dir="output/"
)

# Access results
print(results.summary())
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

We would like to thank the contributors and the community for their support in developing this project.