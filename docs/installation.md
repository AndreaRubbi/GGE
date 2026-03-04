# Installation

## Requirements

- Python 3.8 or higher
- PyTorch 1.9+
- AnnData 0.8+

## Install from PyPI

The recommended way to install GGE:

```bash
pip install gge-eval
```

This installs GGE with all core dependencies including `geomloss` for optimal transport metrics.

## GPU Acceleration (Optional)

For faster distance metric computation on GPU, install with pykeops:

```bash
pip install "gge-eval[gpu]"
```

!!! note "PyKeOps Requirements"
    PyKeOps requires a CUDA-capable GPU and appropriate CUDA toolkit installation.

## Install from Source

For development or to get the latest changes:

```bash
git clone https://github.com/AndreaRubbi/GGE.git
cd GGE
pip install -e .
```

## Full Installation

Install with all optional dependencies (GPU + UMAP):

```bash
pip install "gge-eval[full]"
```

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
