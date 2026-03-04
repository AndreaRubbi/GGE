"""
Tests for GGE metrics module.
"""
import numpy as np
import pytest

from gge.metrics import (
    PearsonCorrelation,
    SpearmanCorrelation,
    MeanPearsonCorrelation,
    MeanSpearmanCorrelation,
    Wasserstein1Distance,
    Wasserstein2Distance,
    MMDDistance,
    EnergyDistance,
    MultivariateWasserstein,
    MultivariateMMD,
)
from gge.metrics.base_metric import MetricResult


@pytest.fixture
def sample_data():
    """Create sample real and generated data."""
    np.random.seed(42)
    n_samples = 100
    n_genes = 50
    
    # Real data
    real = np.random.randn(n_samples, n_genes)
    
    # Generated data (similar to real with some noise)
    generated = real + np.random.randn(n_samples, n_genes) * 0.5
    
    return real, generated


@pytest.fixture
def identical_data():
    """Create identical real and generated data."""
    np.random.seed(42)
    data = np.random.randn(100, 50)
    return data.copy(), data.copy()


@pytest.fixture
def gene_names():
    """Sample gene names."""
    return [f"gene_{i}" for i in range(50)]


class TestPearsonCorrelation:
    """Tests for Pearson correlation metric."""
    
    def test_compute_per_gene(self, sample_data):
        """Test per-gene computation."""
        real, generated = sample_data
        metric = PearsonCorrelation()
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real.shape[1],)
        assert all(-1 <= r <= 1 for r in result if not np.isnan(r))
    
    def test_identical_data(self, identical_data):
        """Test with identical data should give high correlation."""
        real, generated = identical_data
        metric = PearsonCorrelation()
        
        result = metric.compute_per_gene(real, generated)
        
        # Identical data should have perfect correlation
        assert np.allclose(result, 1.0, atol=1e-10)
    
    def test_full_compute(self, sample_data, gene_names):
        """Test full compute with MetricResult."""
        real, generated = sample_data
        metric = PearsonCorrelation()
        
        result = metric.compute(
            real, generated,
            gene_names=gene_names,
            condition="test_condition",
            split="test"
        )
        
        assert isinstance(result, MetricResult)
        assert result.name == "pearson"
        assert len(result.per_gene_values) == len(gene_names)
        assert -1 <= result.aggregate_value <= 1


class TestSpearmanCorrelation:
    """Tests for Spearman correlation metric."""
    
    def test_compute_per_gene(self, sample_data):
        """Test per-gene computation."""
        real, generated = sample_data
        metric = SpearmanCorrelation()
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real.shape[1],)


class TestWasserstein1Distance:
    """Tests for Wasserstein-1 distance."""
    
    def test_compute_per_gene(self, sample_data):
        """Test per-gene computation."""
        real, generated = sample_data
        metric = Wasserstein1Distance()
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real.shape[1],)
        assert all(r >= 0 for r in result if not np.isnan(r))
    
    def test_identical_data(self, identical_data):
        """Test with identical data should give zero distance."""
        real, generated = identical_data
        metric = Wasserstein1Distance()
        
        result = metric.compute_per_gene(real, generated)
        
        # Identical data should have zero distance
        assert np.allclose(result, 0.0, atol=1e-10)
    
    def test_full_compute(self, sample_data, gene_names):
        """Test full compute."""
        real, generated = sample_data
        metric = Wasserstein1Distance()
        
        result = metric.compute(real, generated, gene_names=gene_names)
        
        assert isinstance(result, MetricResult)
        assert result.name == "wasserstein_1"
        assert result.aggregate_value >= 0


class TestWasserstein2Distance:
    """Tests for Wasserstein-2 distance."""
    
    def test_compute_per_gene(self, sample_data):
        """Test per-gene computation."""
        real, generated = sample_data
        metric = Wasserstein2Distance(use_geomloss=False)  # Use scipy fallback
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real.shape[1],)
        assert all(r >= 0 for r in result if not np.isnan(r))


class TestMMDDistance:
    """Tests for MMD distance."""
    
    def test_compute_per_gene(self, sample_data):
        """Test per-gene computation."""
        real, generated = sample_data
        metric = MMDDistance()
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real.shape[1],)
        assert all(r >= 0 for r in result if not np.isnan(r))
    
    def test_identical_data(self, identical_data):
        """Test with identical data should give near-zero MMD."""
        real, generated = identical_data
        metric = MMDDistance()
        
        result = metric.compute_per_gene(real, generated)
        
        # Identical data should have very low MMD
        assert np.mean(result) < 0.01
    
    def test_custom_sigma(self, sample_data):
        """Test with custom bandwidth."""
        real, generated = sample_data
        metric = MMDDistance(sigma=1.0)
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)


class TestEnergyDistance:
    """Tests for Energy distance."""
    
    def test_compute_per_gene(self, sample_data):
        """Test per-gene computation."""
        real, generated = sample_data
        metric = EnergyDistance(use_geomloss=False)  # Use scipy fallback
        
        result = metric.compute_per_gene(real, generated)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (real.shape[1],)
        assert all(r >= 0 for r in result if not np.isnan(r))


class TestMultivariateMetrics:
    """Tests for multivariate metrics."""
    
    def test_multivariate_mmd(self, sample_data):
        """Test multivariate MMD."""
        real, generated = sample_data
        metric = MultivariateMMD()
        
        result = metric.compute_per_gene(real, generated)
        
        # Multivariate returns same value for all genes
        assert len(np.unique(result)) == 1
        assert result[0] >= 0


class TestMetricResult:
    """Tests for MetricResult dataclass."""
    
    def test_to_dict(self, sample_data, gene_names):
        """Test conversion to dictionary."""
        real, generated = sample_data
        metric = PearsonCorrelation()
        
        result = metric.compute(real, generated, gene_names=gene_names)
        result_dict = result.as_dict
        
        assert "name" in result_dict
        assert "aggregate_value" in result_dict
        assert "per_gene_mean" in result_dict
        assert "n_genes" in result_dict
    
    def test_top_genes(self, sample_data, gene_names):
        """Test getting top genes."""
        real, generated = sample_data
        metric = PearsonCorrelation()
        
        result = metric.compute(real, generated, gene_names=gene_names)
        top = result.top_genes(n=5, ascending=True)
        
        assert len(top) == 5
        assert all(isinstance(g, str) for g in top.keys())
        assert all(isinstance(v, float) for v in top.values())


class TestAggregationMethods:
    """Tests for different aggregation methods."""
    
    def test_mean_aggregation(self, sample_data):
        """Test mean aggregation."""
        real, generated = sample_data
        metric = PearsonCorrelation()
        
        result = metric.compute(real, generated, aggregate_method="mean")
        per_gene = metric.compute_per_gene(real, generated)
        
        assert np.isclose(result.aggregate_value, np.nanmean(per_gene))
    
    def test_median_aggregation(self, sample_data):
        """Test median aggregation."""
        real, generated = sample_data
        metric = PearsonCorrelation()
        
        result = metric.compute(real, generated, aggregate_method="median")
        per_gene = metric.compute_per_gene(real, generated)
        
        assert np.isclose(result.aggregate_value, np.nanmedian(per_gene))


# Legacy tests for backwards compatibility
def test_wasserstein_1_legacy(sample_data):
    """Legacy test for W1."""
    real, generated = sample_data
    metric = Wasserstein1Distance()
    result = metric.compute(real, generated)
    assert isinstance(result.aggregate_value, float)


def test_wasserstein_2_legacy(sample_data):
    """Legacy test for W2."""
    real, generated = sample_data
    metric = Wasserstein2Distance(use_geomloss=False)
    result = metric.compute(real, generated)
    assert isinstance(result.aggregate_value, float)


def test_mmd_legacy(sample_data):
    """Legacy test for MMD."""
    real, generated = sample_data
    metric = MMDDistance()
    result = metric.compute(real, generated)
    assert isinstance(result.aggregate_value, float)


def test_energy_legacy(sample_data):
    """Legacy test for Energy."""
    real, generated = sample_data
    metric = EnergyDistance(use_geomloss=False)
    result = metric.compute(real, generated)
    assert isinstance(result.aggregate_value, float)