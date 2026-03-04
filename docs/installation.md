# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 1.9+
- AnnData 0.8+

## Install from PyPI

```bash
pip install gge-eval
```

This installs GGE with all dependencies including `geomloss` for optimal transport metrics. GPU acceleration is automatic when available, and falls back to CPU otherwise.

## Verify Installation

```python
import gge
print(gge.__version__)

# Test basic import
from gge import evaluate
print("GGE installed successfully!")
```

## Troubleshooting

### Common Issues

**ImportError: anndata/scanpy not found**

```bash
pip install anndata scanpy
```

**CUDA/GPU issues with geomloss**

If you don't have a GPU, geomloss will automatically fall back to CPU computation. No action needed.

**PyKeOps compilation errors**

Ensure you have:
- CUDA toolkit installed
- Compatible C++ compiler
- Correct CUDA_HOME environment variable
