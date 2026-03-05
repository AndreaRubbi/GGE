"""
Utilities module for GGE.

Provides:
- DEG analysis and space evaluation
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

__all__ = [
    # DEG utilities
    "identify_degs",
    "get_deg_mask",
    "filter_to_degs",
    "compute_perturbation_effects",
    "DEGSpaceEvaluator",
    "evaluate_deg_space",
]