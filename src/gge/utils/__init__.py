"""
Utilities module for GGE.

Provides:
- DEG analysis and space evaluation
- PC-space (PCA) analysis and evaluation
- I/O utilities
- Preprocessing functions
"""

from .deg import (
    identify_degs,
    get_deg_mask,
    filter_to_degs,
    compute_perturbation_effects,
    DEGSpaceEvaluator,
    evaluate_deg_space,
)

from .pca import (
    compute_pca,
    get_pc_coordinates,
    project_to_pc_space,
    PCSpaceEvaluator,
    evaluate_pc_space,
    compute_pc_variance_explained,
)

__all__ = [
    # DEG utilities
    "identify_degs",
    "get_deg_mask",
    "filter_to_degs",
    "compute_perturbation_effects",
    "DEGSpaceEvaluator",
    "evaluate_deg_space",
    # PC-space utilities
    "compute_pca",
    "get_pc_coordinates",
    "project_to_pc_space",
    "PCSpaceEvaluator",
    "evaluate_pc_space",
    "compute_pc_variance_explained",
]