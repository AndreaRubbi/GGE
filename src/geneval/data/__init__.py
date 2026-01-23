"""
Data loading module for gene expression evaluation.

Provides data loaders for paired real and generated datasets.
"""

from .loader import (
    GeneExpressionDataLoader,
    load_data,
    DataLoaderError,
)
from .gene_expression_datamodule import (
    GeneExpressionDataModule,
    DataModuleError,
)

__all__ = [
    "GeneExpressionDataLoader",
    "load_data",
    "DataLoaderError",
    "GeneExpressionDataModule",
    "DataModuleError",
]