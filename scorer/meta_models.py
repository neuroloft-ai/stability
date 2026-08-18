"""
Data models for the Part 2 Meta-Metrics module.

Meta-metrics characterise *behavioural dynamics* — how model accuracy changes
as stress intensity increases across the K = 6 bins (L0 – L5).

  L0       = baseline   (no perturbation, level_edge = 0.0)
  L1 – L5  = escalating stress, ordered by perturbation_score

These types are consumed exclusively by compute_meta_metrics(); they do NOT
feed back into the Part-1 CategoryScorer (no p(S), no P recalculation).
"""

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """
    One evaluated test run, ready for meta-metric computation.

    stress_type         Stress family name, e.g. "paraphrase".
    level               Bin index: 0 = baseline (L0), 1–5 = stress levels.
    perturbation_score  x-axis value produced by the generator ∈ [0, 1].
    match               True if the model response matched the expected output.
    phase               Optional: "pre_collapse" | "recovery" — required only
                        for RecGap computation.
    false_signature     Optional: True when a run is flagged as a false-masking
                        event — required only for FM_norm computation.
    """
    stress_type:        str
    level:              int
    perturbation_score: float
    match:              bool
    phase:              Optional[str]  = None   # for RecGap
    false_signature:    Optional[bool] = None   # for FM_norm


# ---------------------------------------------------------------------------
# Intermediate / aggregated types
# ---------------------------------------------------------------------------

@dataclass
class BinStats:
    """Aggregated run statistics for one level bin."""
    level:      int
    level_edge: float   # min perturbation_score in this bin (L0 forced = 0.0)
    n_runs:     int
    n_match:    int
    acc:        float   # n_match / n_runs;  float('nan') when n_runs == 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MetaConfig:
    """
    Hyper-parameters for meta-metric computation.

    K                 Number of bins (L0 = baseline + L1 – L(K-1) stress).
    baseline_fallback Used when L0 has < 2 runs (default 0.80).
    alpha_tau         τ = alpha_tau × acc_base  (default 0.80).
    SS_floor          Lower bound for SS_worst normalization (default −1.0).
    threshold_cs      Minimum drop magnitude that counts as catastrophic
                      (default 0.10).
    RecGap_g          Normalization half-range for RecGap (default 0.30).
                        RecGap = −g → norm 0.0  (no recovery)
                        RecGap =  0 → norm 0.5  (neutral)
                        RecGap = +g → norm 1.0  (full over-recovery)
    """
    K:                 int   = 6
    baseline_fallback: float = 0.80
    alpha_tau:         float = 0.80
    SS_floor:          float = -1.0
    threshold_cs:      float = 0.10
    RecGap_g:          float = 0.30

    @property
    def L_max(self) -> int:
        """Highest valid level index (= K − 1)."""
        return self.K - 1


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass
class MetaMetricResult:
    """
    Full meta-metric output for one stress type.

    All normalized metrics are ∈ [0, 1] with higher = better.
    Optional fields (RecGap*, FM_norm) are None when input data
    is absent.

    Curves
    ──────
    acc_curve          acc(k) for k = 0 … K-1;  float('nan') if bin empty.
    level_edges        Perturbation-score lower bound per bin.
    divergence_curve   1 − acc(k);  float('nan') when acc is nan.
    bin_stats          BinStats list (one per bin).

    Baseline
    ────────
    acc_base           Accuracy at L0 (or fallback value).
    tau                τ = alpha_tau × acc_base.

    Raw meta-metrics
    ────────────────
    SR          Survival Rate   — highest level k where acc(k) ≥ τ.
    CP          Collapse Point  — lowest level k where acc(k) < τ;
                                  None if model never collapses.
    SS_worst    Worst Stability Slope — most negative slope across bins.
    CS          Catastrophic-Slope fraction ∈ [0, 1].
    Osc_rate    Oscillation rate ∈ [0, 1].
    RecGap      Recovery gap (requires phase data); None otherwise.
    FM_norm     False-masking metric (requires false_signature data).

    Normalized (higher = better)
    ────────────────────────────
    SR_norm, CP_norm, SS_worst_norm, CS_norm, Osc_rate_norm, RecGap_norm
    """
    stress_type: str

    # Curves
    acc_curve:        List[float]
    level_edges:      List[float]
    divergence_curve: List[float]
    bin_stats:        List[BinStats]

    # Baseline
    acc_base: float
    tau:      float

    # Raw meta-metrics
    SR:       float
    CP:       Optional[float]
    SS_worst: float
    CS:       float
    Osc_rate: float
    RecGap:   Optional[float]
    FM_norm:  Optional[float]

    # Normalized (higher = better)
    SR_norm:       float
    CP_norm:       float
    SS_worst_norm: float
    CS_norm:       float
    Osc_rate_norm: float
    RecGap_norm:   Optional[float]
