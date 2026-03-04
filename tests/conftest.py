"""
Pytest configuration and shared fixtures for GGE tests.
"""
import pytest
import sys
import tempfile
from pathlib import Path
import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import fixtures module
from .fixtures import (
    MockDataGenerator,
    MockMetricData,
    create_test_anndata,
    create_test_pair,
)

# Check for optional dependencies
try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False


# Skip markers
requires_anndata = pytest.mark.skipif(
    not HAS_ANNDATA, reason="anndata not installed"
)
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not installed"
)
requires_geomloss = pytest.mark.skipif(
    not HAS_GEOMLOSS, reason="geomloss not installed"
)


# ==================== FIXTURES ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_generator():
    """Create a MockDataGenerator instance."""
    return MockDataGenerator(
        n_samples=100,
        n_genes=50,
        n_perturbations=3,
        n_cell_types=2,
        seed=42,
    )


@pytest.fixture
def mock_metric_data():
    """Create a MockMetricData instance."""
    return MockMetricData(seed=42)


@pytest.fixture
def sample_arrays():
    """Create simple numpy arrays for metric testing."""
    np.random.seed(42)
    real = np.random.randn(100, 50)
    generated = real + np.random.randn(100, 50) * 0.3
    return real, generated


@pytest.fixture
def identical_arrays():
    """Create identical arrays (for testing zero distance)."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    return data.copy(), data.copy()


@pytest.fixture
def gene_names():
    """Sample gene names."""
    return [f"gene_{i}" for i in range(50)]


@pytest.fixture
@requires_anndata
def sample_anndata(mock_generator):
    """Create sample AnnData objects for testing."""
    return mock_generator.generate_paired_data(noise_level=0.3, quality="good")


@pytest.fixture
@requires_anndata
def saved_anndata(sample_anndata, temp_dir):
    """Save sample AnnData to files and return paths."""
    real, gen = sample_anndata
    
    real_path = temp_dir / "real.h5ad"
    gen_path = temp_dir / "generated.h5ad"
    
    real.write(real_path)
    gen.write(gen_path)
    
    return real_path, gen_path


@pytest.fixture
@requires_anndata
def large_anndata():
    """Create larger dataset for performance testing."""
    generator = MockDataGenerator(
        n_samples=1000,
        n_genes=500,
        n_perturbations=10,
        n_cell_types=4,
        seed=42,
    )
    return generator.generate_paired_data(noise_level=0.3)


@pytest.fixture
@requires_anndata
def poor_quality_anndata(mock_generator):
    """Create poor quality generated data for testing edge cases."""
    real = mock_generator.generate_real_data()
    generated = mock_generator.generate_generated_data(real, noise_level=0.5, quality="poor")
    return real, generated


# ==================== FIXTURE FUNCTIONS ====================

@pytest.fixture
def create_mock_loader(saved_anndata):
    """Factory fixture to create data loaders."""
    def _create_loader(**kwargs):
        from gge.data.loader import GeneExpressionDataLoader
        
        real_path, gen_path = saved_anndata
        
        default_kwargs = {
            "real_path": real_path,
            "generated_path": gen_path,
            "condition_columns": ["perturbation"],
        }
        default_kwargs.update(kwargs)
        
        loader = GeneExpressionDataLoader(**default_kwargs)
        loader.load()
        loader.align_genes()
        return loader
    
    return _create_loader


@pytest.fixture
def create_mock_evaluator(create_mock_loader):
    """Factory fixture to create evaluators."""
    def _create_evaluator(metrics=None, **kwargs):
        from gge.evaluator import GeneEvalEvaluator
        from gge.metrics import PearsonCorrelation, Wasserstein1Distance
        
        loader = create_mock_loader()
        
        if metrics is None:
            metrics = [PearsonCorrelation(), Wasserstein1Distance()]
        
        default_kwargs = {
            "data_loader": loader,
            "metrics": metrics,
            "include_multivariate": False,
            "verbose": False,
        }
        default_kwargs.update(kwargs)
        
        return GeneEvalEvaluator(**default_kwargs)
    
    return _create_evaluator

