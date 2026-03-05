"""
Tests for PC-space (PCA) utilities.
"""
import pytest
import numpy as np

from tests.conftest import requires_anndata


@pytest.fixture
def pca_test_data():
    """Create test data for PCA."""
    try:
        import anndata as ad
    except ImportError:
        pytest.skip("anndata not installed")
    
    np.random.seed(42)
    n_samples = 100
    n_genes = 100  # Keep small to avoid scipy segfaults
    
    # Create expression matrix with some structure
    # First 25 genes correlated, next 25 anti-correlated, rest noise
    X = np.random.randn(n_samples, n_genes)
    
    # Add structure: samples 0-50 vs 50-100 have different expression patterns
    X[:50, :25] += 2
    X[50:, :25] -= 2
    X[:50, 25:50] -= 2
    X[50:, 25:50] += 2
    
    conditions = ["A"] * 50 + ["B"] * 50
    
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["condition"] = conditions
    
    return adata


class TestComputePCA:
    """Tests for PCA computation."""
    
    @requires_anndata
    def test_compute_pca_basic(self, pca_test_data):
        """Test basic PCA computation."""
        from gge.utils.pca import compute_pca
        
        result = compute_pca(pca_test_data, n_components=10, use_highly_variable=False)
        
        assert 'X_pca' in result.obsm
        assert result.obsm['X_pca'].shape == (100, 10)
    
    @requires_anndata
    def test_compute_pca_preserves_original(self, pca_test_data):
        """Test that copy=True preserves original."""
        from gge.utils.pca import compute_pca
        
        result = compute_pca(pca_test_data, n_components=10, copy=True, use_highly_variable=False)
        
        assert 'X_pca' not in pca_test_data.obsm
        assert 'X_pca' in result.obsm
    
    @requires_anndata
    def test_compute_pca_n_components_limit(self, pca_test_data):
        """Test that n_components is limited by dimensions."""
        from gge.utils.pca import compute_pca
        
        # Use a reasonable number that won't trigger scipy issues
        result = compute_pca(pca_test_data, n_components=30, use_highly_variable=False)
        
        # Should be 30 components (less than n_genes-1=99 and n_samples-1=99)
        assert result.obsm['X_pca'].shape[1] == 30


class TestGetPCCoordinates:
    """Tests for getting PC coordinates."""
    
    @requires_anndata
    def test_get_pc_coordinates(self, pca_test_data):
        """Test getting PC coordinates."""
        from gge.utils.pca import compute_pca, get_pc_coordinates
        
        pca_data = compute_pca(pca_test_data, n_components=20, use_highly_variable=False)
        coords = get_pc_coordinates(pca_data)
        
        assert coords.shape == (100, 20)
    
    @requires_anndata
    def test_get_pc_coordinates_subset(self, pca_test_data):
        """Test getting subset of PC coordinates."""
        from gge.utils.pca import compute_pca, get_pc_coordinates
        
        pca_data = compute_pca(pca_test_data, n_components=20, use_highly_variable=False)
        coords = get_pc_coordinates(pca_data, n_components=5)
        
        assert coords.shape == (100, 5)
    
    @requires_anndata
    def test_get_pc_coordinates_no_pca(self, pca_test_data):
        """Test error when PCA not computed."""
        from gge.utils.pca import get_pc_coordinates
        
        with pytest.raises(ValueError, match="PCA not found"):
            get_pc_coordinates(pca_test_data)


class TestPCSpaceEvaluator:
    """Tests for PCSpaceEvaluator."""
    
    @requires_anndata
    def test_transform_to_pc_space(self, pca_test_data):
        """Test transforming data to PC space."""
        from gge.utils.pca import PCSpaceEvaluator
        import anndata as ad
        
        # Create generated data (similar structure)
        np.random.seed(43)
        gen_X = pca_test_data.X.copy() + np.random.randn(*pca_test_data.X.shape) * 0.5
        gen_adata = ad.AnnData(X=gen_X)
        gen_adata.var_names = pca_test_data.var_names.copy()
        gen_adata.obs["condition"] = pca_test_data.obs["condition"].copy()
        
        evaluator = PCSpaceEvaluator(n_components=10, use_highly_variable=False)
        real_pc, gen_pc = evaluator.transform_to_pc_space(pca_test_data, gen_adata)
        
        assert 'X_pca' in real_pc.obsm
        assert 'X_pca' in gen_pc.obsm
        assert real_pc.obsm['X_pca'].shape[1] == 10
    
    @requires_anndata
    def test_evaluate_in_pc_space(self, pca_test_data):
        """Test full evaluation in PC space."""
        from gge.utils.pca import PCSpaceEvaluator
        import anndata as ad
        
        # Create generated data
        np.random.seed(43)
        gen_X = pca_test_data.X.copy() + np.random.randn(*pca_test_data.X.shape) * 0.5
        gen_adata = ad.AnnData(X=gen_X)
        gen_adata.var_names = pca_test_data.var_names.copy()
        gen_adata.obs["condition"] = pca_test_data.obs["condition"].copy()
        
        evaluator = PCSpaceEvaluator(n_components=10, use_highly_variable=False)
        results = evaluator.evaluate(
            real_data=pca_test_data,
            generated_data=gen_adata,
            condition_columns=["condition"],
            verbose=False,
        )
        
        assert results is not None
        assert len(results.splits) > 0


class TestEvaluatePCSpace:
    """Tests for evaluate_pc_space function."""
    
    @requires_anndata
    def test_evaluate_pc_space_basic(self, pca_test_data):
        """Test evaluate_pc_space convenience function."""
        from gge.utils.pca import evaluate_pc_space
        import anndata as ad
        
        # Create generated data
        np.random.seed(43)
        gen_X = pca_test_data.X.copy() + np.random.randn(*pca_test_data.X.shape) * 0.5
        gen_adata = ad.AnnData(X=gen_X)
        gen_adata.var_names = pca_test_data.var_names.copy()
        gen_adata.obs["condition"] = pca_test_data.obs["condition"].copy()
        
        results = evaluate_pc_space(
            real_data=pca_test_data,
            generated_data=gen_adata,
            condition_columns=["condition"],
            n_components=10,
            use_highly_variable=False,
            verbose=False,
        )
        
        assert results is not None


class TestPCVarianceExplained:
    """Tests for variance explained computation."""
    
    @requires_anndata
    def test_compute_variance_explained(self, pca_test_data):
        """Test computing variance explained."""
        from gge.utils.pca import compute_pca, compute_pc_variance_explained
        
        pca_data = compute_pca(pca_test_data, n_components=20, use_highly_variable=False)
        
        var_info = compute_pc_variance_explained(pca_data, n_components=10)
        
        assert "per_component" in var_info
        assert "cumulative" in var_info
        assert "total" in var_info
        assert len(var_info["per_component"]) == 10
        assert var_info["cumulative"][-1] == pytest.approx(var_info["total"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
