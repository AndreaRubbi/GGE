"""
Tests for DEG (Differentially Expressed Gene) utilities.
"""
import pytest
import numpy as np
import pandas as pd

from tests.conftest import requires_anndata


@pytest.fixture
def deg_test_data():
    """Create test data with known differential expression."""
    try:
        import anndata as ad
    except ImportError:
        pytest.skip("anndata not installed")
    
    np.random.seed(42)
    n_samples = 100
    n_genes = 50
    
    # Create positive expression matrix (log-scale data, typical for gene expression)
    # Use lognormal to ensure positive values
    X = np.abs(np.random.randn(n_samples, n_genes)) + 1  # Base expression level > 1
    
    # Create conditions: control vs treatment
    conditions = ["control"] * 50 + ["treatment"] * 50
    
    # Make some genes differentially expressed between conditions
    # Genes 0-9: upregulated in treatment (log2fc > 1, so multiply by > 2)
    X[50:, :10] *= 4  # Treatment samples have ~4x higher expression for genes 0-9
    
    # Genes 10-19: downregulated in treatment (log2fc < -1, so divide by > 2)
    X[50:, 10:20] /= 4  # Treatment samples have ~0.25x expression for genes 10-19
    
    # Genes 20-49: no significant change (noise only)
    
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs["perturbation"] = conditions
    
    return adata


class TestIdentifyDEGs:
    """Tests for DEG identification."""
    
    @requires_anndata
    def test_identify_degs_basic(self, deg_test_data):
        """Test basic DEG identification."""
        from gge.utils.deg import identify_degs
        
        result = identify_degs(
            deg_test_data,
            condition_column="perturbation",
            control_value="control",
            treatment_value="treatment",
            log2fc_threshold=1.0,
            pvalue_threshold=0.05,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "log2fc" in result.columns
        assert "pvalue" in result.columns
        assert "is_deg" in result.columns
        assert len(result) == 50  # All genes
    
    @requires_anndata
    def test_identify_degs_finds_upregulated(self, deg_test_data):
        """Test that upregulated genes are identified."""
        from gge.utils.deg import identify_degs
        
        result = identify_degs(
            deg_test_data,
            condition_column="perturbation",
            control_value="control",
            log2fc_threshold=1.0,
        )
        
        # Genes 0-9 should be identified as DEGs (upregulated)
        upregulated_degs = result[(result['is_deg']) & (result['log2fc'] > 0)]
        upregulated_genes = set(upregulated_degs['gene'].tolist())
        
        expected_up = {f"gene_{i}" for i in range(10)}
        assert len(upregulated_genes.intersection(expected_up)) >= 5  # At least 5/10 detected
    
    @requires_anndata
    def test_identify_degs_finds_downregulated(self, deg_test_data):
        """Test that downregulated genes are identified."""
        from gge.utils.deg import identify_degs
        
        result = identify_degs(
            deg_test_data,
            condition_column="perturbation",
            control_value="control",
            log2fc_threshold=1.0,
        )
        
        # Genes 10-19 should be identified as DEGs (downregulated)
        downregulated_degs = result[(result['is_deg']) & (result['log2fc'] < 0)]
        downregulated_genes = set(downregulated_degs['gene'].tolist())
        
        expected_down = {f"gene_{i}" for i in range(10, 20)}
        assert len(downregulated_genes.intersection(expected_down)) >= 5  # At least 5/10 detected
    
    @requires_anndata
    def test_identify_degs_wilcoxon(self, deg_test_data):
        """Test DEG identification with Wilcoxon test."""
        from gge.utils.deg import identify_degs
        
        result = identify_degs(
            deg_test_data,
            condition_column="perturbation",
            control_value="control",
            method="wilcoxon",
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result['is_deg'].sum() > 0  # Should find some DEGs
    
    @requires_anndata
    def test_identify_degs_missing_column(self, deg_test_data):
        """Test error on missing column."""
        from gge.utils.deg import identify_degs
        
        with pytest.raises(ValueError, match="not found"):
            identify_degs(
                deg_test_data,
                condition_column="nonexistent",
                control_value="control",
            )


class TestFilterToDEGs:
    """Tests for DEG filtering."""
    
    @requires_anndata
    def test_filter_to_degs(self, deg_test_data):
        """Test filtering AnnData to DEGs."""
        from gge.utils.deg import filter_to_degs
        
        deg_genes = ["gene_0", "gene_1", "gene_10", "gene_11"]
        
        filtered = filter_to_degs(deg_test_data, deg_genes)
        
        assert filtered.shape[1] == 4
        assert all(g in filtered.var_names for g in deg_genes)
    
    @requires_anndata
    def test_get_deg_mask(self, deg_test_data):
        """Test getting DEG mask."""
        from gge.utils.deg import get_deg_mask
        
        deg_genes = ["gene_0", "gene_5", "gene_10"]
        
        mask = get_deg_mask(deg_test_data, deg_genes)
        
        assert mask.sum() == 3
        assert mask[0] == True
        assert mask[5] == True
        assert mask[10] == True


class TestPerturbationEffects:
    """Tests for perturbation effect computation."""
    
    @requires_anndata
    def test_compute_perturbation_effects(self, deg_test_data):
        """Test computing perturbation effects."""
        from gge.utils.deg import compute_perturbation_effects
        
        effects = compute_perturbation_effects(
            deg_test_data,
            condition_column="perturbation",
            control_value="control",
        )
        
        assert isinstance(effects, pd.DataFrame)
        assert "treatment" in effects.columns
        assert len(effects) == 50  # All genes
        
        # Genes 0-9 should have positive log2fc
        for i in range(10):
            assert effects.loc[f"gene_{i}", "treatment"] > 0
        
        # Genes 10-19 should have negative log2fc
        for i in range(10, 20):
            assert effects.loc[f"gene_{i}", "treatment"] < 0


class TestDEGSpaceEvaluation:
    """Tests for DEG-space evaluation."""
    
    @requires_anndata
    def test_deg_space_evaluator(self, deg_test_data, temp_dir):
        """Test DEGSpaceEvaluator."""
        from gge.utils.deg import DEGSpaceEvaluator, identify_degs
        import anndata as ad
        
        # Create generated data (similar to deg_test_data)
        np.random.seed(43)
        gen_X = deg_test_data.X.copy() + np.random.randn(*deg_test_data.X.shape) * 0.5
        gen_adata = ad.AnnData(X=gen_X)
        gen_adata.var_names = deg_test_data.var_names.copy()
        gen_adata.obs["perturbation"] = deg_test_data.obs["perturbation"].copy()
        
        # Identify DEGs
        degs = identify_degs(
            deg_test_data,
            condition_column="perturbation",
            control_value="control",
            log2fc_threshold=0.5,
        )
        deg_genes = degs[degs['is_deg']]['gene'].tolist()
        
        if len(deg_genes) == 0:
            pytest.skip("No DEGs found")
        
        # Evaluate in DEG space
        evaluator = DEGSpaceEvaluator(deg_genes)
        results = evaluator.evaluate(
            real_data=deg_test_data,
            generated_data=gen_adata,
            condition_columns=["perturbation"],
            verbose=False,
        )
        
        assert results is not None
        assert len(results.splits) > 0
    
    @requires_anndata
    def test_evaluate_deg_space_function(self, deg_test_data, temp_dir):
        """Test evaluate_deg_space convenience function."""
        from gge.utils.deg import evaluate_deg_space
        import anndata as ad
        
        # Create generated data
        np.random.seed(43)
        gen_X = deg_test_data.X.copy() + np.random.randn(*deg_test_data.X.shape) * 0.5
        gen_adata = ad.AnnData(X=gen_X)
        gen_adata.var_names = deg_test_data.var_names.copy()
        gen_adata.obs["perturbation"] = deg_test_data.obs["perturbation"].copy()
        
        # Evaluate in DEG space
        results, deg_df = evaluate_deg_space(
            real_data=deg_test_data,
            generated_data=gen_adata,
            condition_columns=["perturbation"],
            deg_condition_column="perturbation",
            control_value="control",
            log2fc_threshold=0.5,
            verbose=False,
            return_degs=True,
        )
        
        assert results is not None
        assert isinstance(deg_df, pd.DataFrame)
        assert len(results.splits) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
