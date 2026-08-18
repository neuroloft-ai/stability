"""
compute_metrics.py — Pipeline module 5: Compute behavioural metrics from eval results.

All metrics are computed for three grouping dimensions simultaneously:
  • level    — Classified Level (L0 … L5)
  • category — stress family (Paraphrase, Noise, …)
  • dimension — Stability, Coherence, …

Metrics computed
----------------
  1. MetaMetricResult  — behavioural dynamics (SR, CP, SS_worst, CS, Osc_rate)
                         overall + by_category + by_dimension
  2. P-score           — perturbation-weighted similarity
                         Σ(Score_i × Sim_i) / Σ(Score_i)  [stress rows, Score > 0]
                         mean(Sim: outputs)                [L0 rows, Score = 0]
  3. FWE (raw)         — Failure-Weighted Entropy per group (scorer/fwe.py)
                         not normalised; normalisation happens at use-time in
                         P_primary / P_harmonic
  4. P_primary         — FWE-weighted average p-score
                         Σ(FWE(g) × p(g)) / Σ(FWE(g))
  5. P_harmonic        — FWE-weighted harmonic p-score
                         Σ(FWE(g)) / Σ(FWE(g) / (p(g) + ε))
     Both P metrics are scalars computed separately for each grouping scheme.

Public API
----------
compute_metrics(eval_df, intake) → MetricsResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from modules.data_intake import DataIntakeResult
from scorer.fwe          import compute_fwe
from scorer.meta_metrics import compute_meta_metrics
from scorer.meta_models  import MetaMetricResult, RunRecord


_EPSILON = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    """Output of compute_metrics()."""

    threshold_used:     float
    threshold_strategy: str

    # ── Level summary table ──────────────────────────────────────────────────
    level_summary:      pd.DataFrame
    # Columns: Classified Level | n | mean_sim | pass_rate | mean_<key> …

    # ── Behavioural meta-metrics (SR, CP, SS_worst, CS, Osc_rate) ───────────
    overall_meta:      MetaMetricResult          # all rows (L0–L5)
    meta_by_category:  dict                      # {cat:  MetaMetricResult}
    meta_by_dimension: dict                      # {dim:  MetaMetricResult}

    # ── P-score (perturbation-weighted similarity) ───────────────────────────
    p_score_overall:       float                 # across all stress rows
    p_score_by_level:      dict                  # {level_str: float}
    p_score_by_category:   dict                  # {cat:       float}
    p_score_by_dimension:  dict                  # {dim:       float}

    # ── FWE (raw, un-normalised) ─────────────────────────────────────────────
    fwe_by_level:      dict                      # {L1…L5: float}  (L0 = floor 0.5)
    fwe_by_category:   dict                      # {cat:   float}
    fwe_by_dimension:  dict                      # {dim:   float}

    # ── Composite P (scalar per grouping scheme) ─────────────────────────────
    # by_level uses stress levels only (L1–L5); L0 excluded from composite
    P_primary_by_level:      Optional[float]
    P_primary_by_category:   Optional[float]
    P_primary_by_dimension:  Optional[float]
    P_harmonic_by_level:     Optional[float]
    P_harmonic_by_category:  Optional[float]
    P_harmonic_by_dimension: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _level_int(level_str) -> int:
    """'L3' → 3.  Returns -1 on parse failure."""
    try:
        return int(str(level_str).strip().upper().replace("L", ""))
    except (ValueError, TypeError):
        return -1


def _resolve_threshold(eval_df: pd.DataFrame, intake: DataIntakeResult) -> float:
    """
    Return the acceptance threshold.

    strategy='baseline' → mean(Sim: outputs) on all L0 rows.
                          Falls back to intake.acceptance_threshold when no
                          valid L0 rows exist.
    strategy='fixed'    → intake.acceptance_threshold as-is.
    """
    if intake.threshold_strategy == "baseline":
        col     = "Sim: outputs"
        lvl_col = "Classified Level"
        if col in eval_df.columns and lvl_col in eval_df.columns:
            l0_sims = eval_df.loc[eval_df[lvl_col] == "L0", col].dropna()
            if not l0_sims.empty:
                return float(l0_sims.mean())
    return intake.acceptance_threshold


# ─────────────────────────────────────────────────────────────────────────────
# P-score helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_p_score(df: pd.DataFrame) -> Optional[float]:
    """
    Perturbation-weighted similarity for a slice of eval_df.

    Stress rows (Score > 0) : Σ(Score_i × Sim_i) / Σ(Score_i)
    L0 rows (Score = 0)     : mean(Sim: outputs)

    When a slice contains both, only stress rows drive the weighted formula.
    Fallback to mean(Sim: outputs) if no stress rows or all scores are 0.
    Returns None if no valid Sim: outputs data.
    """
    sim_col   = "Sim: outputs"
    score_col = "Score"
    lvl_col   = "Classified Level"

    if sim_col not in df.columns:
        return None

    # Split L0 (baseline) from stress rows
    if lvl_col in df.columns:
        stress = df[df[lvl_col] != "L0"]
    else:
        stress = df

    # Score-weighted formula over stress rows
    if score_col in stress.columns and not stress.empty:
        valid = stress[[score_col, sim_col]].dropna()
        valid = valid[valid[score_col] > 0]
        if not valid.empty:
            total_score = valid[score_col].sum()
            return round(float((valid[score_col] * valid[sim_col]).sum() / total_score), 4)

    # Fallback: mean(Sim: outputs) — handles L0-only slices
    sims = df[sim_col].dropna()
    return round(float(sims.mean()), 4) if not sims.empty else None


def _compute_p_score_by_group(eval_df: pd.DataFrame, group_col: str) -> dict:
    """Return {group_value: p_score} for each unique value in group_col."""
    if group_col not in eval_df.columns:
        return {}
    result = {}
    for group_val, grp in eval_df.groupby(group_col):
        p = _compute_p_score(grp)
        if p is not None:
            result[str(group_val)] = p
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FWE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fwe_for_slice(df: pd.DataFrame, threshold: float) -> float:
    """
    FWE for a slice of eval_df.

    L0 rows are excluded (Score = 0; no stress signal).
    Returns compute_fwe(scores, matches) from scorer/fwe.py.
    Falls back to 0.5 (floor) when slice is empty after L0 exclusion.
    """
    sim_col   = "Sim: outputs"
    score_col = "Score"
    lvl_col   = "Classified Level"

    if lvl_col in df.columns:
        df = df[df[lvl_col] != "L0"]

    if df.empty or sim_col not in df.columns or score_col not in df.columns:
        return 0.5

    valid = df[[score_col, sim_col]].dropna()
    if valid.empty:
        return 0.5

    scores  = valid[score_col].tolist()
    matches = [float(s) >= threshold for s in valid[sim_col].tolist()]
    return compute_fwe(scores, matches)


def _compute_fwe_by_group(
    eval_df:   pd.DataFrame,
    group_col: str,
    threshold: float,
) -> dict:
    """Return {group_value: fwe_raw} for each unique value in group_col."""
    if group_col not in eval_df.columns:
        return {}
    return {
        str(g): _compute_fwe_for_slice(grp, threshold)
        for g, grp in eval_df.groupby(group_col)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite P helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_composite(
    p_dict:   dict,
    fwe_dict: dict,
    epsilon:  float = _EPSILON,
) -> tuple:
    """
    Compute FWE-weighted composite P scores.

    P_primary  = Σ(FWE(g) × p(g))        / Σ(FWE(g))
    P_harmonic = Σ(FWE(g))               / Σ(FWE(g) / (p(g) + ε))

    Only groups present in BOTH dicts are used.
    Returns (None, None) when no valid groups exist.
    """
    groups = [g for g in p_dict if g in fwe_dict]
    if not groups:
        return None, None

    pairs   = [(p_dict[g], fwe_dict[g]) for g in groups]
    sum_fwe = sum(f for _, f in pairs)

    if sum_fwe <= 0:
        return None, None

    P_primary = sum(f * p for p, f in pairs) / sum_fwe

    denom_h   = sum(f / (p + epsilon) for p, f in pairs)
    P_harmonic = sum_fwe / denom_h if denom_h > 0 else None

    return (
        round(P_primary, 4),
        round(P_harmonic, 4) if P_harmonic is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Meta-metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_records(
    df:          pd.DataFrame,
    threshold:   float,
    stress_type: str,
) -> list:
    """Convert eval_df rows to RunRecord list for compute_meta_metrics."""
    records = []
    for _, row in df.iterrows():
        level = _level_int(row.get("Classified Level", "L0"))
        if level < 0:
            continue

        run_ok  = str(row.get("Run Status", "ok")) == "ok"
        sim_out = row.get("Sim: outputs")

        if not run_ok or sim_out is None or pd.isna(sim_out):
            match = False
        else:
            match = float(sim_out) >= threshold

        perturb = row.get("Score", 0.0)
        if pd.isna(perturb):
            perturb = 0.0

        records.append(RunRecord(
            stress_type        = stress_type,
            level              = level,
            perturbation_score = float(perturb),
            match              = match,
        ))
    return records


def _compute_meta_by_group(
    eval_df:   pd.DataFrame,
    group_col: str,
    threshold: float,
) -> dict:
    """
    One MetaMetricResult per unique value of group_col.

    L0 rows are injected into every group so each group has a proper
    acc_base for normalisation.
    """
    if group_col not in eval_df.columns:
        return {}

    lvl_col   = "Classified Level"
    l0_df     = eval_df[eval_df[lvl_col] == "L0"] if lvl_col in eval_df.columns else pd.DataFrame()
    stress_df = eval_df[eval_df[lvl_col] != "L0"] if lvl_col in eval_df.columns else eval_df

    results = {}
    for group_name, grp_df in stress_df.groupby(group_col):
        combined = pd.concat([l0_df, grp_df], ignore_index=True)
        records  = _build_records(combined, threshold, str(group_name))
        results[str(group_name)] = compute_meta_metrics(records)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Level summary table
# ─────────────────────────────────────────────────────────────────────────────

def _compute_level_summary(
    eval_df:   pd.DataFrame,
    threshold: float,
    intake:    DataIntakeResult,
) -> pd.DataFrame:
    """
    Per Classified Level summary DataFrame.

    Columns: Classified Level | n | mean_sim | pass_rate | mean_<key> …
    Sorted L0 → L5.
    """
    sim_col    = "Sim: outputs"
    lvl_col    = "Classified Level"
    agent_keys = [od["agent_key"] for od in intake.output_defs]

    if lvl_col not in eval_df.columns:
        return pd.DataFrame()

    rows = []
    for level_str, grp in eval_df.groupby(lvl_col):
        sims = grp[sim_col].dropna() if sim_col in grp.columns else pd.Series(dtype=float)

        row = {
            "Classified Level": level_str,
            "n":                len(grp),
            "mean_sim":         round(float(sims.mean()), 4) if not sims.empty else None,
            "pass_rate":        round(float((sims >= threshold).mean()), 4) if not sims.empty else None,
        }

        for key in agent_keys:
            col  = f"Sim: {key}"
            vals = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            row[f"mean_{key}"] = round(float(vals.mean()), 4) if not vals.empty else None

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_sort"] = df["Classified Level"].apply(_level_int)
        df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    eval_df: pd.DataFrame,
    intake:  DataIntakeResult,
) -> MetricsResult:
    """
    Compute all behavioural metrics from eval_df.

    All metrics are computed for three grouping dimensions simultaneously:
    level, category, and dimension.

    Parameters
    ----------
    eval_df : DataFrame from eval_output()
    intake  : DataIntakeResult — threshold strategy + output_defs

    Returns
    -------
    MetricsResult
    """
    if eval_df.empty:
        raise ValueError("eval_df is empty — nothing to compute.")

    # ── Step 1: Resolve threshold ─────────────────────────────────────────────
    threshold = _resolve_threshold(eval_df, intake)

    # ── Step 2: Level summary table ───────────────────────────────────────────
    level_summary = _compute_level_summary(eval_df, threshold, intake)

    # ── Step 3: Overall meta-metrics ──────────────────────────────────────────
    overall_meta = compute_meta_metrics(
        _build_records(eval_df, threshold, "overall")
    )

    # ── Step 4: Meta-metrics by category and dimension ────────────────────────
    meta_by_category  = _compute_meta_by_group(eval_df, "Category",  threshold)
    meta_by_dimension = _compute_meta_by_group(eval_df, "Dimension", threshold)

    # ── Step 5: P-score by level, category, dimension ─────────────────────────
    p_score_overall      = _compute_p_score(eval_df)
    p_score_by_level     = _compute_p_score_by_group(eval_df, "Classified Level")
    p_score_by_category  = _compute_p_score_by_group(eval_df, "Category")
    p_score_by_dimension = _compute_p_score_by_group(eval_df, "Dimension")

    # ── Step 6: FWE (raw) by level, category, dimension ───────────────────────
    fwe_by_level     = _compute_fwe_by_group(eval_df, "Classified Level", threshold)
    fwe_by_category  = _compute_fwe_by_group(eval_df, "Category",         threshold)
    fwe_by_dimension = _compute_fwe_by_group(eval_df, "Dimension",        threshold)

    # ── Step 7: Composite P scores (self-normalising FWE weights) ─────────────
    # For by_level: exclude L0 — its FWE = 0.5 (floor) and has no stress signal
    stress_levels  = [k for k in p_score_by_level if k != "L0"]
    p_stress_lvl   = {k: p_score_by_level[k] for k in stress_levels}
    fwe_stress_lvl = {k: fwe_by_level[k]     for k in stress_levels}

    P_primary_by_level,     P_harmonic_by_level     = _compute_composite(p_stress_lvl,       fwe_stress_lvl)
    P_primary_by_category,  P_harmonic_by_category  = _compute_composite(p_score_by_category, fwe_by_category)
    P_primary_by_dimension, P_harmonic_by_dimension = _compute_composite(p_score_by_dimension, fwe_by_dimension)

    return MetricsResult(
        threshold_used          = round(threshold, 4),
        threshold_strategy      = intake.threshold_strategy,
        level_summary           = level_summary,
        overall_meta            = overall_meta,
        meta_by_category        = meta_by_category,
        meta_by_dimension       = meta_by_dimension,
        p_score_overall         = p_score_overall,
        p_score_by_level        = p_score_by_level,
        p_score_by_category     = p_score_by_category,
        p_score_by_dimension    = p_score_by_dimension,
        fwe_by_level            = fwe_by_level,
        fwe_by_category         = fwe_by_category,
        fwe_by_dimension        = fwe_by_dimension,
        P_primary_by_level      = P_primary_by_level,
        P_primary_by_category   = P_primary_by_category,
        P_primary_by_dimension  = P_primary_by_dimension,
        P_harmonic_by_level     = P_harmonic_by_level,
        P_harmonic_by_category  = P_harmonic_by_category,
        P_harmonic_by_dimension = P_harmonic_by_dimension,
    )
