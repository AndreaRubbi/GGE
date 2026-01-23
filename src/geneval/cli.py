"""
Command-line interface for GenEval gene expression evaluation.

Provides comprehensive CLI for evaluating generated vs real gene expression data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="geneval",
        description="""
GenEval: Comprehensive evaluation of generated gene expression data.

Computes metrics between real and generated datasets, matching samples
by condition columns (e.g., perturbation, cell type). Supports train/test
splits and generates publication-quality visualizations.

Metrics computed:
  - Pearson and Spearman correlation
  - Wasserstein-1 and Wasserstein-2 distance  
  - Maximum Mean Discrepancy (MMD)
  - Energy distance
  - Multivariate versions of distance metrics

All metrics are computed per-gene and aggregated.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--real", "-r",
        type=str,
        required=True,
        help="Path to real data file (h5ad format)",
    )
    required.add_argument(
        "--generated", "-g",
        type=str,
        required=True,
        help="Path to generated data file (h5ad format)",
    )
    required.add_argument(
        "--conditions", "-c",
        type=str,
        nargs="+",
        required=True,
        help="Condition columns to match (e.g., perturbation cell_type)",
    )
    required.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for results and plots",
    )
    
    # Optional arguments
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument(
        "--split-column", "-s",
        type=str,
        default=None,
        help="Column indicating train/test split. If not provided, all data treated as one split.",
    )
    optional.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Specific splits to evaluate (e.g., 'test' or 'train test'). Default: all splits.",
    )
    optional.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "pearson", "spearman", "mean_pearson", "mean_spearman",
            "wasserstein_1", "wasserstein_2", "mmd", "energy",
            "multivariate_wasserstein", "multivariate_mmd", "all"
        ],
        help="Metrics to compute. Default: all metrics.",
    )
    optional.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum samples per condition to include (default: 2)",
    )
    optional.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median", "std"],
        help="How to aggregate per-gene metrics (default: mean)",
    )
    
    # Plotting arguments
    plotting = parser.add_argument_group("Plotting options")
    plotting.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    plotting.add_argument(
        "--plot-formats",
        type=str,
        nargs="+",
        default=["png", "pdf"],
        help="Output formats for plots (default: png pdf)",
    )
    plotting.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved plots (default: 150)",
    )
    plotting.add_argument(
        "--embedding",
        type=str,
        nargs="+",
        default=["pca"],
        choices=["pca", "umap", "both", "none"],
        help="Embedding methods for visualization (default: pca)",
    )
    
    # Output options
    output = parser.add_argument_group("Output options")
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors",
    )
    output.add_argument(
        "--save-per-gene",
        action="store_true",
        help="Save per-gene metric values (can be large files)",
    )
    
    return parser


def get_metric_classes(metric_names: Optional[List[str]] = None):
    """Get metric classes from names."""
    from .metrics.correlation import (
        PearsonCorrelation,
        SpearmanCorrelation,
        MeanPearsonCorrelation,
        MeanSpearmanCorrelation,
    )
    from .metrics.distances import (
        Wasserstein1Distance,
        Wasserstein2Distance,
        MMDDistance,
        EnergyDistance,
        MultivariateWasserstein,
        MultivariateMMD,
    )
    
    all_metrics = {
        "pearson": PearsonCorrelation,
        "spearman": SpearmanCorrelation,
        "mean_pearson": MeanPearsonCorrelation,
        "mean_spearman": MeanSpearmanCorrelation,
        "wasserstein_1": Wasserstein1Distance,
        "wasserstein_2": Wasserstein2Distance,
        "mmd": MMDDistance,
        "energy": EnergyDistance,
        "multivariate_wasserstein": MultivariateWasserstein,
        "multivariate_mmd": MultivariateMMD,
    }
    
    if metric_names is None or "all" in metric_names:
        return list(all_metrics.values())
    
    return [all_metrics[name] for name in metric_names if name in all_metrics]


def main(args: Optional[List[str]] = None):
    """Main entry point for CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    # Set verbosity
    verbose = not parsed.quiet
    if parsed.verbose:
        verbose = True
    
    # Validate paths
    real_path = Path(parsed.real)
    gen_path = Path(parsed.generated)
    output_dir = Path(parsed.output)
    
    if not real_path.exists():
        print(f"Error: Real data file not found: {real_path}", file=sys.stderr)
        sys.exit(1)
    
    if not gen_path.exists():
        print(f"Error: Generated data file not found: {gen_path}", file=sys.stderr)
        sys.exit(1)
    
    # Import here to avoid slow startup
    from .data.loader import load_data
    from .evaluator import GeneEvalEvaluator
    from .visualization.visualizer import EvaluationVisualizer
    
    if verbose:
        print("=" * 60)
        print("GenEval: Gene Expression Evaluation")
        print("=" * 60)
        print(f"\nReal data:      {real_path}")
        print(f"Generated data: {gen_path}")
        print(f"Conditions:     {parsed.conditions}")
        print(f"Output:         {output_dir}")
        if parsed.split_column:
            print(f"Split column:   {parsed.split_column}")
        print()
    
    # Load data
    if verbose:
        print("Loading data...")
    
    try:
        loader = load_data(
            real_path=real_path,
            generated_path=gen_path,
            condition_columns=parsed.conditions,
            split_column=parsed.split_column,
            min_samples_per_condition=parsed.min_samples,
        )
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        summary = loader.summary()
        print(f"  Real:      {summary['real']['n_samples']} samples x {summary['real']['n_genes']} genes")
        print(f"  Generated: {summary['generated']['n_samples']} samples x {summary['generated']['n_genes']} genes")
        print(f"  Common genes: {summary.get('n_common_genes', 'N/A')}")
        print(f"  Splits: {summary.get('splits', ['all'])}")
        print()
    
    # Get metrics
    metric_classes = get_metric_classes(parsed.metrics)
    
    # Determine if multivariate metrics should be included
    include_multivariate = (
        parsed.metrics is None or
        "all" in parsed.metrics or
        any(m.startswith("multivariate") for m in (parsed.metrics or []))
    )
    
    # Create evaluator
    evaluator = GeneEvalEvaluator(
        data_loader=loader,
        metrics=metric_classes,
        aggregate_method=parsed.aggregate,
        include_multivariate=include_multivariate,
        verbose=verbose,
    )
    
    # Run evaluation
    if verbose:
        print("Running evaluation...")
    
    results = evaluator.evaluate(
        splits=parsed.splits,
        save_dir=output_dir,
    )
    
    # Generate plots
    if not parsed.no_plots:
        if verbose:
            print("\nGenerating visualizations...")
        
        plot_dir = output_dir / "plots"
        
        try:
            viz = EvaluationVisualizer(results, dpi=parsed.dpi)
            
            # Determine embedding methods
            embedding_methods = parsed.embedding
            if "none" in embedding_methods:
                embedding_methods = []
            elif "both" in embedding_methods:
                embedding_methods = ["pca", "umap"]
            
            viz.save_all(
                output_dir=plot_dir,
                formats=parsed.plot_formats,
                data_loader=loader if embedding_methods else None,
            )
        except Exception as e:
            print(f"Warning: Failed to generate some plots: {e}", file=sys.stderr)
    
    # Print final summary
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS SAVED")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")
        print("\nFiles generated:")
        print(f"  - summary.json: Aggregate metrics and metadata")
        print(f"  - results.csv: Per-condition metrics")
        if parsed.save_per_gene:
            print(f"  - per_gene_*.csv: Per-gene metric values")
        if not parsed.no_plots:
            print(f"  - plots/: Visualization figures")
        print()
    
    return results


def run():
    """Entry point for console script."""
    main()


if __name__ == "__main__":
    run()