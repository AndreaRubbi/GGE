# Gene Expression Evaluation System

Welcome to the Gene Expression Evaluation System! This project provides a comprehensive framework for evaluating gene expression data through various metrics and methodologies. 

## Overview

The Gene Expression Evaluation System is designed to facilitate the comparison of real and generated gene expression profiles. It includes functionalities for loading datasets, computing metrics, and visualizing results. The system is built with modularity in mind, allowing for easy extension and integration with other projects.

## Features

- **Modular Design**: The project is structured into distinct modules for data handling, evaluation metrics, and model definitions.
- **Metrics Computation**: Includes various metrics such as Wasserstein distances, Pearson and Spearman correlations, and Mean Squared Error (MSE) for evaluating gene expression profiles.
- **Command-Line Interface**: Users can run evaluations directly from the terminal, making it easy to integrate into workflows.
- **Documentation**: Comprehensive documentation is provided to guide users through installation, usage, and examples.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To run an evaluation, you can use the command-line interface or execute the example script provided in the `examples` directory. 

### Example

```python
from geneval import GeneExpressionEvaluator

# Initialize the evaluator with your data
evaluator = GeneExpressionEvaluator(data, generated_output)

# Run the evaluation
results = evaluator.evaluate()
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

We would like to thank the contributors and the community for their support in developing this project.