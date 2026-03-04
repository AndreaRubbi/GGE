"""
GGE Test Suite.

Provides comprehensive tests for the gene expression evaluation framework
along with mock data generators for testing.

Mock Data Generators
--------------------
- MockDataGenerator: Generates synthetic AnnData objects for testing
- MockMetricData: Generates numpy arrays for metric testing

Usage
-----
>>> from gge.tests.fixtures import MockDataGenerator
>>> generator = MockDataGenerator(n_samples=100, n_genes=50)
>>> real, generated = generator.generate_paired_data(noise_level=0.3)
"""

from .fixtures import (
    MockDataGenerator,
    MockMetricData,
    create_test_anndata,
    create_test_pair,
)

__all__ = [
    "MockDataGenerator",
    "MockMetricData",
    "create_test_anndata",
    "create_test_pair",
]