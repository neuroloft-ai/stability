"""
display_metrics.py — Rich display of a MetricsResult in Jupyter notebooks.

Public API
----------
display_metrics(metrics, title=None)  — prints all sections to the notebook
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from modules.compute_metrics import MetricsResult


def _level_sort_key(lvl: str) -> int:
    try:
        return int(str(lvl).strip().upper().replace("L", ""))
    except (ValueError, TypeError):
        return -1


def display_metrics(metrics: MetricsResult, title: Optional[str] = None) -> None:
    """
    Print a structured summary of a MetricsResult.

    Sections
    --------
    1. Header (threshold)
    2. Per-level summary table + p_score/fwe columns
    3. Overall meta-metrics
    4. Per-category table  (p_score, fwe_raw, SR, CP, SS_worst, CS, Osc) + composites
    5. Per-dimension table (p_score, fwe_raw, SR, CP, SS_worst)           + composites
    """
    from IPython.display import display as ipy_display

    sep = "─" * 70

    if title:
        print(f"\n{'═' * 70}")
        print(f"  {title}")
        print(f"{'═' * 70}")

    print(f"Threshold : {metrics.threshold_used}  (strategy: {metrics.threshold_strategy})")
    print()

    # ── 1. Level summary ──────────────────────────────────────────────────────
    print(f"── Per Classified Level  {sep[:44]}")
    if not metrics.level_summary.empty:
        # Merge p_score and fwe_raw columns into the summary table
        extra = {
            "p_score": metrics.p_score_by_level,
            "fwe_raw": metrics.fwe_by_level,
        }
        df = metrics.level_summary.copy()
        for col_name, lookup in extra.items():
            df[col_name] = df["Classified Level"].map(lookup)
        ipy_display(df)
    else:
        print("  (no level data)")
    print()
    print(f"  P_primary  (by level) = {metrics.P_primary_by_level}")
    print(f"  P_harmonic (by level) = {metrics.P_harmonic_by_level}")
    print()

    # ── 2. Overall meta-metrics ───────────────────────────────────────────────
    o = metrics.overall_meta
    print(f"── Overall meta-metrics  {sep[:44]}")
    print(f"  acc_base={o.acc_base:.4f}  τ={o.tau:.4f}   p_score={metrics.p_score_overall}")
    print(f"  SR={o.SR_norm:.4f}  CP={o.CP_norm:.4f}  SS={o.SS_worst_norm:.4f}  "
          f"CS={o.CS_norm:.4f}  Osc={o.Osc_rate_norm:.4f}")
    curve = [round(a, 3) if a == a else None for a in o.acc_curve]
    print(f"  acc_curve : {curve}")
    print()

    # ── 3. Per category ───────────────────────────────────────────────────────
    print(f"── By Category  {sep[:53]}")
    _cat_rows = []
    for cat in sorted(metrics.p_score_by_category):
        mmr = metrics.meta_by_category.get(cat)
        _cat_rows.append({
            "Category": cat,
            "p_score":  metrics.p_score_by_category.get(cat),
            "fwe_raw":  metrics.fwe_by_category.get(cat),
            "SR":       mmr.SR_norm       if mmr else None,
            "CP":       mmr.CP_norm       if mmr else None,
            "SS_worst": mmr.SS_worst_norm if mmr else None,
            "CS":       mmr.CS_norm       if mmr else None,
            "Osc":      mmr.Osc_rate_norm if mmr else None,
        })
    if _cat_rows:
        ipy_display(pd.DataFrame(_cat_rows).set_index("Category").round(4))
    print()
    print(f"  P_primary  (by category)  = {metrics.P_primary_by_category}")
    print(f"  P_harmonic (by category)  = {metrics.P_harmonic_by_category}")
    print()

    # ── 4. Per dimension ──────────────────────────────────────────────────────
    print(f"── By Dimension  {sep[:52]}")
    _dim_rows = []
    for dim in sorted(metrics.p_score_by_dimension):
        mmr = metrics.meta_by_dimension.get(dim)
        _dim_rows.append({
            "Dimension": dim,
            "p_score":   metrics.p_score_by_dimension.get(dim),
            "fwe_raw":   metrics.fwe_by_dimension.get(dim),
            "SR":        mmr.SR_norm       if mmr else None,
            "CP":        mmr.CP_norm       if mmr else None,
            "SS_worst":  mmr.SS_worst_norm if mmr else None,
        })
    if _dim_rows:
        ipy_display(pd.DataFrame(_dim_rows).set_index("Dimension").round(4))
    print()
    print(f"  P_primary  (by dimension) = {metrics.P_primary_by_dimension}")
    print(f"  P_harmonic (by dimension) = {metrics.P_harmonic_by_dimension}")
