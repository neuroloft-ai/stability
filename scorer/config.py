"""
Default configuration for the Category Scorer.

Business weights and alpha are user-configurable at runtime.
When adding a new category, register its default family weights here.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Per-category default business weights
#
# Keys must match the 'family' field in test_defs (registry).
# All weights within a category must sum to 1.0.
# ---------------------------------------------------------------------------

CATEGORY_DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Stability": {
        "paraphrase":  0.20,
        "noise":       0.10,
        "format":      0.10,
        "distraction": 0.20,
        "conflict":    0.25,
        "context":     0.15,
    },
    # Future categories registered here, e.g.:
    # "Robustness": {...},
    # "Fairness":   {...},
}

# ---------------------------------------------------------------------------
# Global scoring parameters
# ---------------------------------------------------------------------------

# Blend ratio α ∈ [0, 1]
#   0.0 → pure FWE weight   (data-driven)
#   1.0 → pure business weight (user-defined)
DEFAULT_ALPHA: float = 0.6

# Guard against division by zero in the harmonic mean formula
DEFAULT_EPSILON: float = 1e-6
