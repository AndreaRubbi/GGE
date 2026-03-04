# Contributing

Thank you for your interest in contributing to GGE!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/AndreaRubbi/GGE.git
cd GGE
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gge

# Run specific test file
pytest tests/test_metrics.py

# Run with verbose output
pytest -v
```

## Code Style

We use:
- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type check
mypy src/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Adding New Metrics

1. Create a new metric class in `src/gge/metrics/`:

```python
from gge.metrics.base_metric import BaseMetric, MetricResult

class MyNewMetric(BaseMetric):
    name = "my_metric"
    higher_is_better = True  # or False
    
    def _compute_per_gene(self, real, generated):
        # Implement per-gene computation
        return per_gene_values
```

2. Add to `src/gge/metrics/__init__.py`
3. Add tests in `tests/test_metrics.py`
4. Update documentation

## Adding New Visualizations

1. Add method to `src/gge/visualization/visualizer.py`
2. Add tests in `tests/test_visualization.py`
3. Update documentation

## Reporting Issues

When reporting issues, please include:
- GGE version (`gge --version`)
- Python version
- Operating system
- Minimal reproducible example
- Full error traceback

## Questions?

Feel free to open an issue or discussion on GitHub!
