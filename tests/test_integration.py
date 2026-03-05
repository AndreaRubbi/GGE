"""
Integration tests for the complete GGE pipeline.

These tests verify end-to-end functionality from data loading
through evaluation to visualization.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path

from .conftest import requires_anndata, requires_torch


class TestEndToEndPipeline:
    """Test complete evaluation pipeline."""
    
    @requires_anndata
    def test_basic_evaluation_pipeline(self, sample_anndata, temp_dir):
        """Test basic evaluation from AnnData to results."""
        from gge import evaluate
        from gge.metrics import PearsonCorrelation, Wasserstein1Distance
        
        real, generated = sample_anndata
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run evaluation
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
            metrics=[PearsonCorrelation(), Wasserstein1Distance()],
            include_multivariate=False,
            verbose=False,
        )
        
        # Verify results structure
        assert results is not None
        assert hasattr(results, "splits")
        assert len(results.splits) > 0
        
        # Check aggregate metrics via summary
        summary = results.summary()
        assert "n_splits" in summary
    
    @requires_anndata
    def test_evaluation_with_splits(self, mock_generator, temp_dir):
        """Test evaluation with train/test split."""
        from gge import evaluate
        from gge.metrics import PearsonCorrelation
        
        # Generate data (split is included by default)
        real, generated = mock_generator.generate_paired_data(
            noise_level=0.3,
        )
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run evaluation with split
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
            split_column="split",
            metrics=[PearsonCorrelation()],
            verbose=False,
        )
        
        # Verify split results
        assert hasattr(results, "splits")
        assert "train" in results.splits or "test" in results.splits
    
    @requires_anndata
    def test_evaluation_with_multiple_conditions(self, mock_generator, temp_dir):
        """Test evaluation with multiple condition columns."""
        from gge import evaluate
        from gge.metrics import SpearmanCorrelation
        
        real, generated = mock_generator.generate_paired_data(noise_level=0.3)
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run with multiple conditions
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation", "cell_type"],
            metrics=[SpearmanCorrelation()],
            verbose=False,
        )
        
        assert results is not None
        assert len(results.get_all_conditions()) > 0
    
    @requires_anndata
    def test_results_serialization(self, sample_anndata, temp_dir):
        """Test saving and loading results."""
        from gge import evaluate
        from gge.metrics import PearsonCorrelation
        import json
        
        real, generated = sample_anndata
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run evaluation
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
            metrics=[PearsonCorrelation()],
            verbose=False,
        )
        
        # Save results using save method
        output_dir = temp_dir / "results"
        results.save(output_dir)
        
        assert (output_dir / "summary.json").exists()
        
        # Verify JSON is valid
        with open(output_dir / "summary.json") as f:
            data = json.load(f)
        
        assert "n_splits" in data
        assert "splits" in data
        
        # Save to CSV
        df = results.to_dataframe()
        csv_path = temp_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        
        assert csv_path.exists()


class TestVisualizationIntegration:
    """Test visualization integration with evaluation results."""
    
    @requires_anndata
    def test_basic_visualization(self, sample_anndata, temp_dir):
        """Test generating basic visualizations."""
        from gge import evaluate
        from gge.visualization import EvaluationVisualizer
        from gge.metrics import PearsonCorrelation, Wasserstein1Distance
        
        real, generated = sample_anndata
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run evaluation
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
            metrics=[PearsonCorrelation(), Wasserstein1Distance()],
            verbose=False,
        )
        
        # Create visualizer
        visualizer = EvaluationVisualizer(results)
        
        # Generate plots
        output_dir = temp_dir / "plots"
        output_dir.mkdir()
        
        # Test boxplot (actual method name is boxplot_metrics)
        fig = visualizer.boxplot_metrics()
        assert fig is not None
        
        # Save figure
        fig.savefig(output_dir / "boxplot.png")
        assert (output_dir / "boxplot.png").exists()
    
    @requires_anndata
    def test_radar_plot(self, sample_anndata, temp_dir):
        """Test radar plot generation."""
        from gge import evaluate
        from gge.visualization import EvaluationVisualizer
        from gge.metrics import (
            PearsonCorrelation, 
            SpearmanCorrelation, 
            Wasserstein1Distance
        )
        
        real, generated = sample_anndata
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run evaluation with multiple metrics
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
            metrics=[
                PearsonCorrelation(),
                SpearmanCorrelation(),
                Wasserstein1Distance(),
            ],
            verbose=False,
        )
        
        visualizer = EvaluationVisualizer(results)
        
        # Generate radar plot (actual method name is radar_plot)
        fig = visualizer.radar_plot()
        assert fig is not None
    
    @pytest.mark.skip(reason="scipy KDE segfault - numpy/scipy version incompatibility")
    @requires_anndata
    def test_generate_all_plots(self, sample_anndata, temp_dir):
        """Test generating all plots at once."""
        from gge import evaluate
        from gge.visualization import EvaluationVisualizer
        from gge.metrics import PearsonCorrelation
        
        real, generated = sample_anndata
        
        # Save data
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Run evaluation
        results = evaluate(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
            metrics=[PearsonCorrelation()],
            verbose=False,
        )
        
        # Generate all plots (actual method is save_all)
        visualizer = EvaluationVisualizer(results)
        output_dir = temp_dir / "all_plots"
        
        visualizer.save_all(output_dir)
        
        assert output_dir.exists()
        # Check at least some plots were created
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) > 0


class TestDataLoaderIntegration:
    """Test data loader with various data formats."""
    
    @requires_anndata
    def test_load_unaligned_genes(self, mock_generator, temp_dir):
        """Test loading data with different genes."""
        import anndata as ad
        
        # Create data with overlapping but not identical genes
        real = mock_generator.generate_real_data()
        generated = mock_generator.generate_generated_data(real, noise_level=0.3)
        
        # Modify gene names in generated data
        new_var = generated.var.copy()
        new_var.index = [f"new_gene_{i}" for i in range(25)] + list(real.var_names[25:])
        generated = ad.AnnData(
            X=generated.X,
            obs=generated.obs,
            var=new_var,
        )
        
        # Save
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Load and align
        from gge.data.loader import GeneExpressionDataLoader
        
        loader = GeneExpressionDataLoader(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
        )
        loader.load()
        loader.align_genes()
        
        # Should have subset of genes
        assert loader.gene_names is not None
        assert len(loader.gene_names) == 25  # Only overlapping genes
    
    @requires_anndata
    def test_condition_iteration(self, sample_anndata, temp_dir):
        """Test iterating over conditions."""
        from gge.data.loader import GeneExpressionDataLoader
        
        real, generated = sample_anndata
        
        # Save
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Load
        loader = GeneExpressionDataLoader(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
        )
        loader.load()
        loader.align_genes()
        
        # Iterate conditions (yields 4 values: key, real, gen, info)
        conditions = list(loader.iterate_conditions())
        
        assert len(conditions) > 0
        
        for condition_key, real_data, gen_data, cond_info in conditions:
            assert isinstance(condition_key, str)
            assert real_data.shape[1] == gen_data.shape[1]  # Same genes


class TestMetricConsistency:
    """Test metric consistency across different data sizes."""
    
    def test_metric_determinism(self, sample_arrays, gene_names):
        """Test that metrics give consistent results."""
        from gge.metrics import PearsonCorrelation, Wasserstein1Distance
        
        real, generated = sample_arrays
        
        pearson = PearsonCorrelation()
        wasserstein = Wasserstein1Distance()
        
        # Run twice
        result1_p = pearson.compute(real, generated, gene_names)
        result2_p = pearson.compute(real, generated, gene_names)
        
        result1_w = wasserstein.compute(real, generated, gene_names)
        result2_w = wasserstein.compute(real, generated, gene_names)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1_p.per_gene_values,
            result2_p.per_gene_values,
        )
        np.testing.assert_almost_equal(
            result1_p.aggregate_value,
            result2_p.aggregate_value,
        )
        
        np.testing.assert_array_almost_equal(
            result1_w.per_gene_values,
            result2_w.per_gene_values,
        )
    
    def test_identical_data_gives_perfect_score(self, identical_arrays, gene_names):
        """Test that identical data gives perfect scores."""
        from gge.metrics import PearsonCorrelation, Wasserstein1Distance
        
        real, generated = identical_arrays
        
        pearson = PearsonCorrelation()
        wasserstein = Wasserstein1Distance()
        
        result_p = pearson.compute(real, generated, gene_names)
        result_w = wasserstein.compute(real, generated, gene_names)
        
        # Pearson should be 1.0
        np.testing.assert_almost_equal(result_p.aggregate_value, 1.0, decimal=5)
        
        # Wasserstein should be 0.0
        np.testing.assert_almost_equal(result_w.aggregate_value, 0.0, decimal=5)
    
    def test_metrics_scale_with_noise(self, mock_metric_data, gene_names):
        """Test that metrics degrade with increasing noise."""
        from gge.metrics import PearsonCorrelation
        
        pearson = PearsonCorrelation()
        
        # Get identical distributions (very similar)
        real, similar = mock_metric_data.identical_distributions(100, 50)
        
        # Get different distributions
        _, different = mock_metric_data.different_distributions(100, 50)
        
        result_similar = pearson.compute(real, similar, gene_names)
        result_different = pearson.compute(real, different, gene_names)
        
        # Similar should have higher correlation
        assert result_similar.aggregate_value > result_different.aggregate_value


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample(self, gene_names):
        """Test metrics with single sample."""
        from gge.metrics import Wasserstein1Distance
        
        np.random.seed(42)
        real = np.random.randn(1, 50)
        generated = np.random.randn(1, 50)
        
        wasserstein = Wasserstein1Distance()
        result = wasserstein.compute(real, generated, gene_names)
        
        # Should still compute (though not statistically meaningful)
        assert result is not None
    
    def test_sparse_data(self, mock_metric_data, gene_names):
        """Test metrics with sparse data."""
        from gge.metrics import PearsonCorrelation
        
        real, generated = mock_metric_data.sparse_data(100, 50, sparsity=0.8)
        
        pearson = PearsonCorrelation()
        result = pearson.compute(real, generated, gene_names)
        
        # Should handle sparse data
        assert result is not None
        assert not np.isnan(result.aggregate_value)
    
    def test_constant_genes(self):
        """Test handling of constant genes."""
        from gge.metrics import PearsonCorrelation
        
        real = np.ones((100, 10))
        generated = np.ones((100, 10))
        gene_names = [f"gene_{i}" for i in range(10)]
        
        pearson = PearsonCorrelation()
        result = pearson.compute(real, generated, gene_names)
        
        # Should handle without crashing (might have NaN values)
        assert result is not None
    
    @requires_anndata
    def test_empty_condition(self, temp_dir):
        """Test handling when a condition has no samples."""
        import anndata as ad
        from gge.data.loader import GeneExpressionDataLoader
        
        # Create minimal data with a condition that exists only in real
        np.random.seed(42)
        n_genes = 20
        
        real = ad.AnnData(
            X=np.random.randn(10, n_genes),
            obs={
                "perturbation": ["drug_a"] * 5 + ["drug_b"] * 5,
            },
        )
        real.var_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Generated has only drug_a
        generated = ad.AnnData(
            X=np.random.randn(10, n_genes),
            obs={
                "perturbation": ["drug_a"] * 10,
            },
        )
        generated.var_names = [f"gene_{i}" for i in range(n_genes)]
        
        # Save
        real_path = temp_dir / "real.h5ad"
        gen_path = temp_dir / "generated.h5ad"
        real.write(real_path)
        generated.write(gen_path)
        
        # Load
        loader = GeneExpressionDataLoader(
            real_data=real_path,
            generated_data=gen_path,
            condition_columns=["perturbation"],
        )
        loader.load()
        loader.align_genes()
        
        # Iterate - should only get matching conditions
        conditions = list(loader.iterate_conditions())
        
        # Should only have drug_a (the one present in both)
        assert len(conditions) == 1
        # condition key is a string (condition values joined by separator)
        assert conditions[0][0] == "drug_a"


class TestCustomMetrics:
    """Test custom metric registration and usage."""
    
    def test_register_custom_metric(self, sample_arrays, gene_names):
        """Test registering and using a custom metric."""
        from gge.metrics.base_metric import BaseMetric, MetricResult
        from gge.evaluator import MetricRegistry
        import numpy as np
        
        class MeanAbsoluteDifference(BaseMetric):
            """Custom metric: mean absolute difference per gene."""
            
            def __init__(self):
                super().__init__(
                    name="mean_absolute_difference",
                    description="Mean absolute difference per gene",
                    higher_is_better=False,
                )
            
            def compute_per_gene(self, real, generated, **kwargs):
                """Compute per-gene mean absolute difference."""
                return np.abs(real.mean(axis=0) - generated.mean(axis=0))
            
            def compute(self, real, generated, gene_names=None, **kwargs):
                per_gene = self.compute_per_gene(real, generated)
                if gene_names is None:
                    gene_names = [f"gene_{i}" for i in range(len(per_gene))]
                return MetricResult(
                    name=self.name,
                    per_gene_values=per_gene,
                    aggregate_value=float(per_gene.mean()),
                    gene_names=gene_names,
                )
        
        # Register (takes just the class)
        MetricRegistry.register(MeanAbsoluteDifference)
        
        # Use - get the class and instantiate
        MetricClass = MetricRegistry.get("mean_absolute_difference")
        assert MetricClass is not None
        
        metric = MetricClass()
        real, generated = sample_arrays
        result = metric.compute(real, generated, gene_names)
        
        assert result.name == "mean_absolute_difference"
        assert len(result.per_gene_values) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
