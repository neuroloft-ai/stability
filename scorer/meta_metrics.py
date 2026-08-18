"""
Part 2 — Meta-Metrics for behavioural dynamics.

Characterises how model accuracy changes across K = 6 stress bins:

  L0      = baseline  (no perturbation, level_edge = 0.0)
  L1 – L5 = escalating stress, ordered by perturbation_score

Steps 5 / 6 / 7 (p(S), component totals, P recalculation) are handled
exclusively by Part 1 CategoryScorer and are NOT repeated here.

Algorithm
─────────
Step 1  Build acc_curve and divergence_curve from RunRecords.
Step 2  Resolve acc_base and τ.
Step 3  Compute raw meta-metrics: SR, CP, SS_worst, CS, Osc_rate,
        and (optionally) RecGap, FM_norm.
Step 4  Normalise all metrics to [0, 1]  (higher = better).

Quick usage
───────────
    from scorer import compute_meta_metrics, RunRecord

    records = [RunRecord("paraphrase", level=0, perturbation_score=0.0, match=True), ...]
    result  = compute_meta_metrics(records)
    print(result.SR_norm, result.SS_worst_norm)
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

from .meta_models import BinStats, MetaConfig, MetaMetricResult, RunRecord


_DEFAULT_CONFIG = MetaConfig()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_meta_metrics(
    records: List[RunRecord],
    config:  Optional[MetaConfig] = None,
) -> MetaMetricResult:
    """
    Compute meta-metrics for one stress type from a list of RunRecords.

    All records should belong to the same stress_type; level 0 records
    are treated as the baseline (L0).

    Args:
        records  RunRecord list for a single stress type.
        config   MetaConfig override (default MetaConfig() used if None).

    Returns:
        MetaMetricResult with raw + normalised meta-metrics.
    """
    cfg = config or _DEFAULT_CONFIG

    if not records:
        return _empty_result("", cfg)

    stress_type = records[0].stress_type

    # ── Step 1: Build bins ───────────────────────────────────────────────
    bins      = _build_bins(records, cfg)
    acc_curve = [b.acc for b in bins]
    level_edges = [b.level_edge for b in bins]
    div_curve = [
        (1.0 - a) if not math.isnan(a) else float("nan")
        for a in acc_curve
    ]

    # ── Step 2: Baseline and τ ───────────────────────────────────────────
    acc_base = _resolve_baseline(bins[0], cfg)
    tau      = cfg.alpha_tau * acc_base

    # ── Step 3: Raw meta-metrics ─────────────────────────────────────────
    SR        = _compute_sr(acc_curve, tau)
    CP        = _compute_cp(acc_curve, tau)
    SS_worst  = _compute_ss_worst(acc_curve, level_edges)
    CS        = _compute_cs(acc_curve, cfg.threshold_cs, cfg.K)
    Osc_rate  = _compute_osc_rate(acc_curve, cfg.K)
    recgap    = _compute_recgap(records)
    fm_norm   = _compute_fm_norm(records)

    # ── Step 4: Normalise ────────────────────────────────────────────────
    SR_norm       = SR / cfg.L_max
    CP_norm       = (CP / cfg.L_max) if CP is not None else 1.0
    SS_worst_norm = _normalize_ss(SS_worst, cfg.SS_floor)
    CS_norm       = 1.0 - CS
    Osc_rate_norm = 1.0 - Osc_rate
    recgap_norm   = _normalize_recgap(recgap, cfg.RecGap_g) if recgap is not None else None

    return MetaMetricResult(
        stress_type      = stress_type,
        acc_curve        = [round(a, 4) if not math.isnan(a) else a for a in acc_curve],
        level_edges      = [round(e, 4) if not math.isnan(e) else e for e in level_edges],
        divergence_curve = [round(d, 4) if not math.isnan(d) else d for d in div_curve],
        bin_stats        = bins,
        acc_base         = round(acc_base, 4),
        tau              = round(tau, 4),
        SR               = SR,
        CP               = CP,
        SS_worst         = round(SS_worst, 4),
        CS               = round(CS, 4),
        Osc_rate         = round(Osc_rate, 4),
        RecGap           = round(recgap, 4)    if recgap   is not None else None,
        FM_norm          = round(fm_norm, 4)   if fm_norm  is not None else None,
        SR_norm          = round(SR_norm,       4),
        CP_norm          = round(CP_norm,       4),
        SS_worst_norm    = round(SS_worst_norm, 4),
        CS_norm          = round(CS_norm,       4),
        Osc_rate_norm    = round(Osc_rate_norm, 4),
        RecGap_norm      = round(recgap_norm, 4) if recgap_norm is not None else None,
    )


# ---------------------------------------------------------------------------
# Step 1 — Build bins
# ---------------------------------------------------------------------------

def _build_bins(records: List[RunRecord], cfg: MetaConfig) -> List[BinStats]:
    """Aggregate RunRecords into K bins; L0 level_edge is forced to 0.0."""
    grouped: Dict[int, List[RunRecord]] = defaultdict(list)
    for r in records:
        if 0 <= r.level < cfg.K:
            grouped[r.level].append(r)

    bins: List[BinStats] = []
    for k in range(cfg.K):
        recs    = grouped.get(k, [])
        n       = len(recs)
        n_match = sum(1 for r in recs if r.match)
        acc     = n_match / n if n > 0 else float("nan")

        if k == 0:
            edge = 0.0
        else:
            scores = [r.perturbation_score for r in recs]
            edge   = min(scores) if scores else float("nan")

        bins.append(BinStats(
            level      = k,
            level_edge = edge,
            n_runs     = n,
            n_match    = n_match,
            acc        = acc,
        ))
    return bins


# ---------------------------------------------------------------------------
# Step 2 — Baseline resolution
# ---------------------------------------------------------------------------

def _resolve_baseline(l0_bin: BinStats, cfg: MetaConfig) -> float:
    """acc_base = acc(L0) when ≥ 2 runs available, else configured fallback."""
    if l0_bin.n_runs >= 2 and not math.isnan(l0_bin.acc):
        return l0_bin.acc
    return cfg.baseline_fallback


# ---------------------------------------------------------------------------
# Step 3 — Raw meta-metric computation
# ---------------------------------------------------------------------------

def _compute_sr(acc_curve: List[float], tau: float) -> float:
    """
    SR = highest level k where acc(k) ≥ τ.

    Initialised to 0.0 (L0 baseline).  Iterates all bins and keeps the
    maximum k that satisfies the threshold.
    """
    sr = 0.0
    for k, a in enumerate(acc_curve):
        if not math.isnan(a) and a >= tau:
            sr = float(k)
    return sr


def _compute_cp(acc_curve: List[float], tau: float) -> Optional[float]:
    """
    CP = lowest level k where acc(k) < τ.
    Returns None when the model never collapses (all acc ≥ τ).
    """
    for k, a in enumerate(acc_curve):
        if not math.isnan(a) and a < tau:
            return float(k)
    return None


def _compute_ss_worst(
    acc_curve:   List[float],
    level_edges: List[float],
) -> float:
    """
    Worst (most negative) slope over consecutive valid bin pairs.

        SS(k) = (acc(k+1) − acc(k)) / (level_edge(k+1) − level_edge(k))

    Pairs are skipped when Δlevel ≤ 0 or either acc is NaN.
    Returns 0.0 when no valid pair exists.
    """
    slopes: List[float] = []
    for k in range(len(acc_curve) - 1):
        a0, a1 = acc_curve[k], acc_curve[k + 1]
        if math.isnan(a0) or math.isnan(a1):
            continue
        e0, e1 = level_edges[k], level_edges[k + 1]
        if math.isnan(e0) or math.isnan(e1):
            continue
        delta = e1 - e0
        if delta <= 0:
            continue
        slopes.append((a1 - a0) / delta)
    return min(slopes) if slopes else 0.0


def _compute_cs(
    acc_curve: List[float],
    threshold: float,
    K:         int,
) -> float:
    """
    Catastrophic-slope fraction.

    A catastrophic drop occurs when acc(k+1) < acc(k)
    AND  |acc(k+1) − acc(k)| > threshold.

    CS = count(catastrophic drops) / (K − 1).
    Pairs where either acc is NaN are skipped; denominator is always K − 1.
    """
    cata_count = 0
    for k in range(len(acc_curve) - 1):
        a0, a1 = acc_curve[k], acc_curve[k + 1]
        if math.isnan(a0) or math.isnan(a1):
            continue
        if a1 < a0 and (a0 - a1) > threshold:
            cata_count += 1
    return cata_count / (K - 1)


def _compute_osc_rate(acc_curve: List[float], K: int) -> float:
    """
    Oscillation rate.

    diffs = [acc(k+1) − acc(k)  for valid consecutive pairs]
    Osc_rate = count(sign changes in diffs) / (K − 2).

    Returns 0.0 when fewer than 3 valid bins exist.
    Sign changes detected as  diffs[i] × diffs[i+1] < 0.
    """
    diffs: List[float] = []
    for k in range(len(acc_curve) - 1):
        a0, a1 = acc_curve[k], acc_curve[k + 1]
        if not (math.isnan(a0) or math.isnan(a1)):
            diffs.append(a1 - a0)

    if len(diffs) < 2:
        return 0.0

    sign_changes = sum(
        1 for i in range(len(diffs) - 1)
        if diffs[i] * diffs[i + 1] < 0
    )
    return sign_changes / (K - 2)


def _compute_recgap(records: List[RunRecord]) -> Optional[float]:
    """
    Recovery gap = acc(recovery phase) − acc(pre_collapse phase).

    Returns None when phase data is absent or either phase has no runs.
    Positive RecGap → model recovers beyond its pre-collapse accuracy.
    Negative RecGap → model fails to fully recover.
    """
    pre = [r for r in records if r.phase == "pre_collapse"]
    rec = [r for r in records if r.phase == "recovery"]
    if not pre or not rec:
        return None
    acc_pre = sum(1 for r in pre if r.match) / len(pre)
    acc_rec = sum(1 for r in rec if r.match) / len(rec)
    return acc_rec - acc_pre


def _compute_fm_norm(records: List[RunRecord]) -> Optional[float]:
    """
    False-masking metric.

    FM_raw = fraction of runs flagged as false_signature = True.
    FM_norm = 1 − FM_raw  (higher = better; fewer false masking events).

    Returns None when false_signature is absent from all records.
    """
    flagged = [r for r in records if r.false_signature is not None]
    if not flagged:
        return None
    fm_raw = sum(1 for r in flagged if r.false_signature) / len(flagged)
    return 1.0 - fm_raw


# ---------------------------------------------------------------------------
# Step 4 — Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_ss(ss_worst: float, ss_floor: float) -> float:
    """
    SS_worst_norm = clip((SS_worst − SS_floor) / (0 − SS_floor), 0, 1)

      SS_worst = SS_floor  →  0.0  (worst possible degradation)
      SS_worst = 0         →  1.0  (no degradation)
      SS_worst > 0 (improvement) →  capped at 1.0
    """
    denom = -ss_floor          # = 0 − SS_floor  (positive when SS_floor < 0)
    if denom <= 0:
        return 1.0
    raw = (ss_worst - ss_floor) / denom
    return max(0.0, min(1.0, raw))


def _normalize_recgap(recgap: float, g: float) -> float:
    """
    RecGap_norm = clip((RecGap + g) / (2g), 0, 1)

      RecGap = −g  →  0.0   (no recovery — dropped by g below pre-collapse)
      RecGap =  0  →  0.5   (neutral — returned to same level)
      RecGap = +g  →  1.0   (over-recovery — exceeded pre-collapse by g)
    """
    if g <= 0:
        return 1.0 if recgap >= 0 else 0.0
    raw = (recgap + g) / (2.0 * g)
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Empty result helper
# ---------------------------------------------------------------------------

def _empty_result(stress_type: str, cfg: MetaConfig) -> MetaMetricResult:
    nan  = float("nan")
    bins = [
        BinStats(level=k, level_edge=(0.0 if k == 0 else nan),
                 n_runs=0, n_match=0, acc=nan)
        for k in range(cfg.K)
    ]
    return MetaMetricResult(
        stress_type      = stress_type,
        acc_curve        = [nan] * cfg.K,
        level_edges      = [0.0] + [nan] * cfg.L_max,
        divergence_curve = [nan] * cfg.K,
        bin_stats        = bins,
        acc_base         = cfg.baseline_fallback,
        tau              = round(cfg.alpha_tau * cfg.baseline_fallback, 4),
        SR               = 0.0,
        CP               = 0.0,
        SS_worst         = 0.0,
        CS               = 0.0,
        Osc_rate         = 0.0,
        RecGap           = None,
        FM_norm          = None,
        SR_norm          = 0.0,
        CP_norm          = 0.0,
        SS_worst_norm    = 1.0,   # no data → assume no degradation
        CS_norm          = 1.0,
        Osc_rate_norm    = 1.0,
        RecGap_norm      = None,
    )
