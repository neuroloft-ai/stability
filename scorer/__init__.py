from .config import CATEGORY_DEFAULT_WEIGHTS, DEFAULT_ALPHA, DEFAULT_EPSILON
from .weights import compute_fwe_weights, compute_final_weights, normalize_weights
from .scorer import CategoryScorer, FamilyInput, ScoringResult, FamilyBreakdown
from .meta_models import RunRecord, BinStats, MetaConfig, MetaMetricResult
from .meta_metrics import compute_meta_metrics
from .fwe import compute_fwe

__all__ = [
    # Part 1 — Stability Score
    "CategoryScorer",
    "FamilyInput",
    "ScoringResult",
    "FamilyBreakdown",
    # Config
    "CATEGORY_DEFAULT_WEIGHTS",
    "DEFAULT_ALPHA",
    "DEFAULT_EPSILON",
    # Weight utilities (available for testing / custom pipelines)
    "normalize_weights",
    "compute_fwe_weights",
    "compute_final_weights",
    # Part 2 — Meta-Metrics
    "RunRecord",
    "BinStats",
    "MetaConfig",
    "MetaMetricResult",
    "compute_meta_metrics",
    # FWE — canonical definition (scorer/fwe.py)
    "compute_fwe",
]
