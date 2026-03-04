"""
Tests for data loading and processing utilities.
"""
import pytest
import numpy as np
from pathlib import Path

from .conftest import requires_anndata


class TestMockDataGenerator:
    """Test the mock data generator itself."""
    
    @requires_anndata
    def test_generate_real_data(self, mock_generator):
        """Test generating real data."""
        real = mock_generator.generate_real_data()
        
        assert real.shape[0] == 100  # n_samples
        assert real.shape[1] == 50   # n_genes
        assert "perturbation" in real.obs.columns
        assert "cell_type" in real.obs.columns
    
    @requires_anndata
    def test_generate_generated_data(self, mock_generator):
        """Test generating synthetic data matching real."""
        real = mock_generator.generate_real_data()
        generated = mock_generator.generate_generated_data(real, noise_level=0.3)
        
        assert generated.shape == real.shape
        assert list(generated.var_names) == list(real.var_names)
        assert "perturbation" in generated.obs.columns
    
    @requires_anndata
    def test_generate_paired_data(self, mock_generator):
        """Test generating paired datasets."""
        real, generated = mock_generator.generate_paired_data(noise_level=0.3)
        
        assert real.shape == generated.shape
        assert list(real.var_names) == list(generated.var_names)
    
    @requires_anndata
    def test_quality_levels(self, mock_generator):
        """Test different quality levels."""
        real = mock_generator.generate_real_data()
        
        # Test each quality level
        good = mock_generator.generate_generated_data(real, quality="good")
        medium = mock_generator.generate_generated_data(real, quality="medium")
        poor = mock_generator.generate_generated_data(real, quality="poor")
        
        # All should have same shape
        assert good.shape == real.shape
        assert medium.shape == real.shape
        assert poor.shape == real.shape
    
    @requires_anndata
    def test_with_split(self, mock_generator):
        """Test generating data with train/test split."""
        # generate_paired_data always includes split column
        real, generated = mock_generator.generate_paired_data(
            noise_level=0.3,
        )
        
        assert "split" in real.obs.columns
        assert "split" in generated.obs.columns
        
        # Check split values
        assert set(real.obs["split"].unique()) == {"train", "test"}
    
    @requires_anndata
    def test_save_and_load(self, mock_generator, temp_dir):
        """Test saving data to files."""
        real_path, gen_path = mock_generator.save_paired_data(temp_dir)
        
        assert real_path.exists()
        assert gen_path.exists()
        
        # Load and verify
        import anndata as ad
        loaded_real = ad.read_h5ad(real_path)
        loaded_gen = ad.read_h5ad(gen_path)
        
        assert loaded_real.shape[0] == 100
        assert loaded_gen.shape[0] == 100


class TestMockMetricData:
    """Test the mock metric data generator."""
    
    def test_identical_distributions(self, mock_metric_data):
        """Test generating identical distributions."""
        real, gen = mock_metric_data.identical_distributions(100, 50)
        
        assert real.shape == (100, 50)
        assert gen.shape == (100, 50)
        np.testing.assert_allclose(real, gen, rtol=1e-5)
    
    def test_similar_distributions(self, mock_metric_data):
        """Test generating similar distributions."""
        real, gen = mock_metric_data.similar_distributions(100, 50, noise_level=0.3)
        
        assert real.shape == (100, 50)
        assert gen.shape == (100, 50)
        
        # Should be correlated but not identical
        corr = np.corrcoef(real.flatten(), gen.flatten())[0, 1]
        assert 0.5 < corr < 1.0
    
    def test_different_distributions(self, mock_metric_data):
        """Test generating different distributions."""
        real, gen = mock_metric_data.different_distributions(100, 50)
        
        assert real.shape == (100, 50)
        assert gen.shape == (100, 50)
        
        # Means should be different
        assert abs(real.mean() - gen.mean()) > 1.0
    
    def test_with_outliers(self, mock_metric_data):
        """Test generating data with outliers."""
        real, gen = mock_metric_data.with_outliers(100, 50, outlier_fraction=0.1)
        
        assert real.shape == (100, 50)
        assert gen.shape == (100, 50)
        
        # Check for extreme values in generated
        assert np.max(np.abs(gen)) > 5  # Outliers should be large
    
    def test_sparse_data(self, mock_metric_data):
        """Test generating sparse data."""
        real, gen = mock_metric_data.sparse_data(100, 50, sparsity=0.8)
        
        assert real.shape == (100, 50)
        assert gen.shape == (100, 50)
        
        # Check sparsity
        real_sparsity = np.mean(real == 0)
        gen_sparsity = np.mean(gen == 0)
        
        assert real_sparsity > 0.7
        assert gen_sparsity > 0.7


class TestDataLoaderBasics:
    """Test data loader basic operations."""
    
    @requires_anndata
    def test_load_files(self, saved_anndata):
        """Test loading h5ad files."""
        from gge.data.loader import GeneExpressionDataLoader
        
        real_path, gen_path = saved_anndata
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
        )
        
        loader.load()
        loader.align_genes()
        
        assert loader.real is not None
        assert loader.generated is not None
    
    @requires_anndata
    def test_align_genes(self, saved_anndata):
        """Test gene alignment."""
        from gge.data.loader import GeneExpressionDataLoader
        
        real_path, gen_path = saved_anndata
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
        )
        
        loader.load()
        loader.align_genes()
        
        # Gene names should be identical
        assert list(loader.real.var_names) == list(loader.generated.var_names)
    
    @requires_anndata
    def test_get_condition_groups(self, saved_anndata):
        """Test getting condition groups."""
        from gge.data.loader import GeneExpressionDataLoader
        
        real_path, gen_path = saved_anndata
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
        )
        
        loader.load()
        loader.align_genes()
        
        conditions = list(loader.iterate_conditions())
        
        assert len(conditions) > 0
        
        # Each condition should have condition key, real data, generated data, condition_info
        for cond_key, real, gen, cond_info in conditions:
            assert isinstance(cond_key, str)
            assert real.shape[1] == gen.shape[1]  # Same number of genes


class TestDataLoaderSplits:
    """Test data loader with train/test splits."""
    
    @requires_anndata
    def test_data_has_splits(self, mock_generator, temp_dir):
        """Test that generated data has split column."""
        from gge.data.loader import GeneExpressionDataLoader
        
        # Generate data (split is included by default)
        real, gen = mock_generator.generate_paired_data(
            noise_level=0.3,
        )
        
        # Verify split column exists
        assert "split" in real.obs.columns
        assert "split" in gen.obs.columns
        
        # Verify split values
        assert set(real.obs["split"].unique()) == {"train", "test"}
    
    @requires_anndata  
    def test_filter_by_split_column(self, mock_generator, temp_dir):
        """Test that split column is passed to loader."""
        from gge.data.loader import GeneExpressionDataLoader
        
        # Generate data with splits (split is included by default)
        real, gen = mock_generator.generate_paired_data(
            noise_level=0.3,
        )
        
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        gen.write(gen_path)
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
            split_column="split",
        )
        
        loader.load()
        loader.align_genes()
        
        # Split column should be set
        assert loader.split_column == "split"


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_missing_file(self, temp_dir):
        """Test error on missing file."""
        from gge.data.loader import GeneExpressionDataLoader
        
        loader = GeneExpressionDataLoader(
            real_path=temp_dir / "nonexistent.h5ad",
            generated_path=temp_dir / "also_nonexistent.h5ad",
            condition_columns=["perturbation"],
        )
        
        with pytest.raises(Exception):
            loader.load()
    
    @requires_anndata
    def test_missing_condition_column(self, temp_dir):
        """Test error on missing condition column."""
        from gge.data.loader import GeneExpressionDataLoader
        import anndata as ad
        
        # Create data without required column
        real = ad.AnnData(X=np.random.randn(10, 5))
        real.write(temp_dir / "real.h5ad")
        
        gen = ad.AnnData(X=np.random.randn(10, 5))
        gen.write(temp_dir / "gen.h5ad")
        
        loader = GeneExpressionDataLoader(
            real_path=temp_dir / "real.h5ad",
            generated_path=temp_dir / "gen.h5ad",
            condition_columns=["nonexistent_column"],
        )
        
        # Should raise DataLoaderError during load when validating columns
        with pytest.raises(Exception):  # DataLoaderError
            loader.load()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
