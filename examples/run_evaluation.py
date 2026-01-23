#!/usr/bin/env python
"""
Example: Running GenEval evaluation on gene expression data.

This script demonstrates how to use GenEval to evaluate generated
gene expression data against real data.
"""
import argparse
from pathlib import Path
import numpy as np

# Import GenEval
from geneval import (
    evaluate,
    load_data,
    GeneEvalEvaluator,
    EvaluationVisualizer,
)
from geneval.metrics import (
    PearsonCorrelation,
    SpearmanCorrelation,
    Wasserstein1Distance,
    MMDDistance,
)


def run_basic_evaluation(real_path: str, generated_path: str, output_dir: str):
    """
    Run basic evaluation with default settings.
    """
    print("=" * 60)
    print("GenEval: Basic Evaluation Example")
    print("=" * 60)
    
    # Run evaluation with one line
    results = evaluate(
        real_path=real_path,
        generated_path=generated_path,
        condition_columns=["perturbation"],  # Match by perturbation
        split_column="split",  # Use split column for train/test
        output_dir=output_dir,
        verbose=True,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(results.summary())
    
    return results


def run_custom_evaluation(real_path: str, generated_path: str, output_dir: str):
    """
    Run evaluation with custom metrics and settings.
    """
    print("=" * 60)
    print("GenEval: Custom Evaluation Example")
    print("=" * 60)
    
    # Step 1: Load and align data
    print("\n1. Loading data...")
    loader = load_data(
        real_path=real_path,
        generated_path=generated_path,
        condition_columns=["perturbation", "cell_type"],
        split_column="split",
        min_samples_per_condition=5,  # Require at least 5 samples
    )
    
    print(f"   Real data: {loader.real.n_obs} samples x {loader.n_genes} genes")
    print(f"   Generated: {loader.generated.n_obs} samples x {loader.n_genes} genes")
    print(f"   Splits: {loader.get_splits()}")
    
    # Step 2: Create evaluator with custom metrics
    print("\n2. Creating evaluator with custom metrics...")
    custom_metrics = [
        PearsonCorrelation(),
        SpearmanCorrelation(),
        Wasserstein1Distance(),
        MMDDistance(sigma=1.0),  # Custom bandwidth
    ]
    
    evaluator = GeneEvalEvaluator(
        data_loader=loader,
        metrics=custom_metrics,
        aggregate_method="median",  # Use median instead of mean
        include_multivariate=False,  # Skip multivariate metrics
        verbose=True,
    )
    
    # Step 3: Run evaluation on specific splits
    print("\n3. Running evaluation on test split only...")
    results = evaluator.evaluate(
        splits=["test"],
        save_dir=output_dir,
    )
    
    # Step 4: Access per-condition results
    print("\n4. Per-condition results:")
    test_split = results.get_split("test")
    if test_split:
        for cond_key, cond in list(test_split.conditions.items())[:5]:
            pearson = cond.get_metric_value("pearson")
            w1 = cond.get_metric_value("wasserstein_1")
            print(f"   {cond.perturbation}: Pearson={pearson:.3f}, W1={w1:.4f}")
    
    # Step 5: Get per-gene analysis
    print("\n5. Per-gene analysis (top 5 worst genes by Pearson):")
    for cond in list(test_split.conditions.values())[:1]:
        if "pearson" in cond.metrics:
            worst_genes = cond.metrics["pearson"].top_genes(n=5, ascending=True)
            for gene, value in worst_genes.items():
                print(f"   {gene}: {value:.3f}")
    
    return results


def run_visualization_example(results, loader, output_dir: str):
    """
    Demonstrate visualization capabilities.
    """
    print("\n" + "=" * 60)
    print("GenEval: Visualization Example")
    print("=" * 60)
    
    # Create visualizer
    viz = EvaluationVisualizer(results, dpi=150)
    
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual plots
    print("\nGenerating plots...")
    
    # 1. Boxplot of all metrics
    print("  - Boxplot of metrics...")
    fig_box = viz.boxplot_metrics()
    fig_box.savefig(plot_dir / "boxplot_metrics.png", bbox_inches='tight')
    
    # 2. Violin plots
    print("  - Violin plots...")
    fig_violin = viz.violin_metrics()
    fig_violin.savefig(plot_dir / "violin_metrics.png", bbox_inches='tight')
    
    # 3. Radar plot for split comparison
    print("  - Radar plot...")
    try:
        fig_radar = viz.radar_split_comparison()
        fig_radar.savefig(plot_dir / "radar_comparison.png", bbox_inches='tight')
    except Exception as e:
        print(f"    Skipped radar (need multiple splits): {e}")
    
    # 4. Scatter grid
    print("  - Scatter grid...")
    fig_scatter = viz.scatter_grid(max_conditions=9)
    fig_scatter.savefig(plot_dir / "scatter_grid.png", bbox_inches='tight')
    
    # 5. Heatmap
    print("  - Heatmap...")
    try:
        fig_heat = viz.heatmap_metrics_summary()
        fig_heat.savefig(plot_dir / "heatmap_summary.png", bbox_inches='tight')
    except Exception as e:
        print(f"    Skipped heatmap: {e}")
    
    # 6. Embedding plots
    print("  - PCA embedding...")
    try:
        fig_pca = viz.embedding_plot(loader, method="pca")
        fig_pca.savefig(plot_dir / "embedding_pca.png", bbox_inches='tight')
    except Exception as e:
        print(f"    Skipped PCA: {e}")
    
    print(f"\nPlots saved to: {plot_dir}")


def create_synthetic_data():
    """
    Create synthetic data for demonstration.
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError:
        print("anndata and pandas required. Install with: pip install anndata pandas")
        return None, None
    
    np.random.seed(42)
    
    n_samples = 500
    n_genes = 100
    n_perturbations = 5
    
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
    gen_X = real_X + np.random.randn(n_samples, n_genes) * 0.5
    gen_obs = real_obs.copy()
    gen = ad.AnnData(X=gen_X, obs=gen_obs)
    gen.var_names = [f"gene_{i}" for i in range(n_genes)]
    
    return real, gen


def main():
    parser = argparse.ArgumentParser(description="GenEval Example")
    parser.add_argument("--real", type=str, help="Path to real data (h5ad)")
    parser.add_argument("--generated", type=str, help="Path to generated data (h5ad)")
    parser.add_argument("--output", type=str, default="example_output", help="Output directory")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for demo")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.synthetic or (not args.real and not args.generated):
        print("Creating synthetic data for demonstration...")
        real, gen = create_synthetic_data()
        if real is None:
            return
        
        # Save synthetic data
        real_path = output_dir / "synthetic_real.h5ad"
        gen_path = output_dir / "synthetic_generated.h5ad"
        real.write(real_path)
        gen.write(gen_path)
        print(f"Saved synthetic data to {output_dir}")
    else:
        real_path = args.real
        gen_path = args.generated
    
    # Run evaluations
    results = run_basic_evaluation(str(real_path), str(gen_path), str(output_dir / "basic"))
    
    # Run custom evaluation
    results_custom = run_custom_evaluation(str(real_path), str(gen_path), str(output_dir / "custom"))
    
    # Generate visualizations
    loader = load_data(str(real_path), str(gen_path), ["perturbation"])
    run_visualization_example(results, loader, str(output_dir))
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()