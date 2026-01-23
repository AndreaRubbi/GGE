"""
Tests for GenEval evaluator and data loader.
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

from geneval.evaluator import GeneEvalEvaluator, evaluate
from geneval.data.loader import GeneExpressionDataLoader, load_data
from geneval.results import EvaluationResult, SplitResult, ConditionResult
from geneval.metrics import PearsonCorrelation, Wasserstein1Distance


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_anndata():
    """Create sample AnnData objects for testing."""
    if not HAS_ANNDATA:
        pytest.skip("anndata not installed")
    
    np.random.seed(42)
    n_samples = 100
    n_genes = 50
    n_perturbations = 3
    
    # Create real data
    real_X = np.random.randn(n_samples, n_genes)
    real_obs = pd.DataFrame({
        "perturbation": np.random.choice([f"pert_{i}" for i in range(n_perturbations)], n_samples),
        "cell_type": np.random.choice(["TypeA", "TypeB"], n_samples),
        "split": np.random.choice(["train", "test"], n_samples, p=[0.7, 0.3]),
    })
    real = ad.AnnData(X=real_X, obs=real_obs)
    real.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Create generated data (slightly noisy version of real)
    gen_X = real_X + np.random.randn(n_samples, n_genes) * 0.3
    gen_obs = real_obs.copy()
    gen = ad.AnnData(X=gen_X, obs=gen_obs)
    gen.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    return real, gen


@pytest.fixture
def saved_anndata(sample_anndata, temp_dir):
    """Save sample AnnData to files."""
    real, gen = sample_anndata
    
    real_path = temp_dir / "real.h5ad"
    gen_path = temp_dir / "generated.h5ad"
    
    real.write(real_path)
    gen.write(gen_path)
    
    return real_path, gen_path


class TestGeneExpressionDataLoader:
    """Tests for data loader."""
    
    def test_load(self, saved_anndata):
        """Test loading data."""
        real_path, gen_path = saved_anndata
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
        )
        loader.load()
        
        assert loader._is_loaded
        assert loader._real is not None
        assert loader._generated is not None
    
    def test_align_genes(self, saved_anndata):
        """Test gene alignment."""
        real_path, gen_path = saved_anndata
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
        )
        loader.load()
        loader.align_genes()
        
        assert loader._is_aligned
        assert len(loader.gene_names) == 50
        assert loader.real.var_names.tolist() == loader.generated.var_names.tolist()
    
    def test_get_splits(self, saved_anndata):
        """Test getting available splits."""
        real_path, gen_path = saved_anndata
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
            split_column="split",
        )
        
        splits = loader.get_splits()
        assert "train" in splits or "test" in splits
    
    def test_get_common_conditions(self, saved_anndata):
        """Test getting common conditions."""
        real_path, gen_path = saved_anndata
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
        )
        
        conditions = loader.get_common_conditions()
        assert len(conditions) > 0
    
    def test_iterate_conditions(self, saved_anndata):
        """Test iterating over conditions."""
        real_path, gen_path = saved_anndata
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
        )
        
        count = 0
        for key, real_data, gen_data, info in loader.iterate_conditions():
            assert real_data.shape[1] == gen_data.shape[1]
            assert "perturbation" in info
            count += 1
        
        assert count > 0
    
    def test_summary(self, saved_anndata):
        """Test summary method."""
        real_path, gen_path = saved_anndata
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
        )
        
        summary = loader.summary()
        
        assert "loaded" in summary
        assert summary["loaded"] is True
        assert "n_common_genes" in summary


class TestGeneEvalEvaluator:
    """Tests for main evaluator."""
    
    def test_evaluate_basic(self, saved_anndata, temp_dir):
        """Test basic evaluation."""
        real_path, gen_path = saved_anndata
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
        )
        
        evaluator = GeneEvalEvaluator(
            data_loader=loader,
            metrics=[PearsonCorrelation(), Wasserstein1Distance()],
            include_multivariate=False,
            verbose=False,
        )
        
        results = evaluator.evaluate()
        
        assert isinstance(results, EvaluationResult)
        assert len(results.splits) > 0
    
    def test_evaluate_with_splits(self, saved_anndata, temp_dir):
        """Test evaluation with split column."""
        real_path, gen_path = saved_anndata
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
            split_column="split",
        )
        
        evaluator = GeneEvalEvaluator(
            data_loader=loader,
            include_multivariate=False,
            verbose=False,
        )
        
        results = evaluator.evaluate(splits=["test"])
        
        assert "test" in results.splits
    
    def test_evaluate_and_save(self, saved_anndata, temp_dir):
        """Test evaluation with saving."""
        real_path, gen_path = saved_anndata
        output_dir = temp_dir / "output"
        
        loader = load_data(
            real_path, gen_path,
            condition_columns=["perturbation"],
        )
        
        evaluator = GeneEvalEvaluator(
            data_loader=loader,
            metrics=[PearsonCorrelation()],
            include_multivariate=False,
            verbose=False,
        )
        
        results = evaluator.evaluate(save_dir=output_dir)
        
        assert output_dir.exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "results.csv").exists()


class TestEvaluateFunction:
    """Tests for convenience evaluate function."""
    
    def test_evaluate_function(self, saved_anndata, temp_dir):
        """Test the evaluate() convenience function."""
        real_path, gen_path = saved_anndata
        
        results = evaluate(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["perturbation"],
            output_dir=temp_dir / "results",
            verbose=False,
        )
        
        assert isinstance(results, EvaluationResult)


class TestEvaluationResult:
    """Tests for result containers."""
    
    def test_condition_result(self):
        """Test ConditionResult."""
        cond = ConditionResult(
            condition_key="test_cond",
            split="test",
            n_real_samples=100,
            n_generated_samples=100,
            n_genes=50,
            gene_names=[f"gene_{i}" for i in range(50)],
        )
        
        assert cond.condition_key == "test_cond"
        summary = cond.summary
        assert "n_real_samples" in summary
    
    def test_split_result(self):
        """Test SplitResult."""
        split = SplitResult(split_name="test")
        
        cond = ConditionResult(
            condition_key="cond1",
            split="test",
            n_real_samples=100,
            n_generated_samples=100,
            n_genes=50,
            gene_names=[],
        )
        split.add_condition(cond)
        
        assert split.n_conditions == 1
    
    def test_evaluation_result_to_dataframe(self):
        """Test DataFrame conversion."""
        result = EvaluationResult(gene_names=[f"gene_{i}" for i in range(10)])
        
        split = SplitResult(split_name="test")
        cond = ConditionResult(
            condition_key="cond1",
            split="test",
            n_real_samples=100,
            n_generated_samples=100,
            n_genes=10,
            gene_names=[f"gene_{i}" for i in range(10)],
        )
        split.add_condition(cond)
        result.add_split(split)
        
        df = result.to_dataframe()
        assert len(df) == 1


class TestDataLoaderError:
    """Tests for error handling."""
    
    def test_missing_file(self, temp_dir):
        """Test error on missing file."""
        from geneval.data.loader import DataLoaderError
        
        loader = GeneExpressionDataLoader(
            real_path=temp_dir / "nonexistent.h5ad",
            generated_path=temp_dir / "also_nonexistent.h5ad",
            condition_columns=["perturbation"],
        )
        
        with pytest.raises(DataLoaderError):
            loader.load()
    
    def test_missing_column(self, saved_anndata):
        """Test error on missing condition column."""
        from geneval.data.loader import DataLoaderError
        
        real_path, gen_path = saved_anndata
        
        loader = GeneExpressionDataLoader(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=["nonexistent_column"],
        )
        
        with pytest.raises(DataLoaderError):
            loader.load()


# Legacy test for backwards compatibility
def test_gene_expression_evaluator_legacy(sample_anndata, temp_dir):
    """Legacy-style test."""
    real, gen = sample_anndata
    
    # Save to disk
    real_path = temp_dir / "real.h5ad"
    gen_path = temp_dir / "gen.h5ad"
    real.write(real_path)
    gen.write(gen_path)
    
    # Use new API
    results = evaluate(
        real_path=real_path,
        generated_path=gen_path,
        condition_columns=["perturbation"],
        verbose=False,
    )
    
    # Check results have expected metrics
    for split in results.splits.values():
        for cond in split.conditions.values():
            assert "pearson" in cond.metrics or len(cond.metrics) > 0