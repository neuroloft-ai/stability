"""
stress_metrics.py — Per-group aggregation of stress test metrics.

Metrics
-------
Per row (Gate == True only):
  Score           — perturbation magnitude  (difference: generated vs original input)
  Sim: outputs    — output stability        (similarity: generated vs original output)
  risk            — vulnerability           Score × (1 − Sim: outputs)

Aggregated per group:
  n               — number of test cases
  mean_score      — mean perturbation magnitude
  mean_stability  — mean output stability
  mean_risk       — mean vulnerability
  p_score         — perturbation-weighted robustness
                    Σ(Score_i × Sim_i) / Σ(Score_i)

Public API
----------
compute_stress_metrics(eval_df, group_by, threshold=None) → dict of DataFrames
    group_by: "level" | "category" | "dimension" | "all"

Adding a new metric
-------------------
1. Add computation in _aggregate_group(df) → update the returned dict
2. To remove: delete those lines — nothing else is affected.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Group-column mapping
# ─────────────────────────────────────────────────────────────────────────────

_GROUP_COL = {
    "level":     "Classified Level",
    "category":  "Category",
    "dimension": "Dimension",
}


# ─────────────────────────────────────────────────────────────────────────────
# Row-level risk
# ─────────────────────────────────────────────────────────────────────────────

def _add_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'risk' column = Score × (1 − Sim: outputs)."""
    df = df.copy()
    sim   = df.get("Sim: outputs", pd.Series(dtype=float))
    score = df.get("Score",        pd.Series(dtype=float))
    df["risk"] = score * (1 - sim)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Group aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_group(df: pd.DataFrame) -> dict:
    """
    Aggregate one group slice into summary metrics.

    Returns
    -------
    dict with keys: n, mean_score, mean_stability, mean_risk, p_score
    """
    sim_col   = "Sim: outputs"
    score_col = "Score"
    risk_col  = "risk"

    n = len(df)

    score = df[score_col].dropna() if score_col in df.columns else pd.Series(dtype=float)
    sim   = df[sim_col].dropna()   if sim_col   in df.columns else pd.Series(dtype=float)
    risk  = df[risk_col].dropna()  if risk_col  in df.columns else pd.Series(dtype=float)

    mean_score     = round(float(score.mean()), 4) if not score.empty else None
    mean_stability = round(float(sim.mean()),   4) if not sim.empty   else None
    mean_risk      = round(float(risk.mean()),  4) if not risk.empty  else None

    # p_score: perturbation-weighted robustness Σ(Score×Sim) / Σ(Score)
    if score_col in df.columns and sim_col in df.columns:
        valid = df[[score_col, sim_col]].dropna()
        valid = valid[valid[score_col] > 0]
        if not valid.empty:
            total  = valid[score_col].sum()
            p_score = round(float((valid[score_col] * valid[sim_col]).sum() / total), 4)
        else:
            p_score = mean_stability   # fallback: plain mean (e.g. L0 baseline)
    else:
        p_score = None

    return {
        "n":              n,
        "mean_score":     mean_score,
        "mean_stability": mean_stability,
        "mean_risk":      mean_risk,
        "p_score":        p_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sorting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sort_level(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def _key(v):
        try:
            return int(str(v).strip().upper().replace("L", ""))
        except (ValueError, TypeError):
            return -1
    df = df.copy()
    df["_sort"] = df[col].apply(_key)
    return df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _level_metrics(
    df:        pd.DataFrame,
    threshold: float,
    alpha_tau: float,
    l0_df:     pd.DataFrame,
) -> dict:
    """
    Compute SR, SR_pct, CP, CD, SS for one group slice.

    All metrics share the same acc-per-level foundation:
        acc(Lk) = mean(Sim: outputs >= threshold) for rows at level Lk

    SR  — count of levels where acc(Lk) >= τ
    CP  — first level where acc(Lk) < τ
    CD  — acc drop at collapse: acc(L_{CP-1}) − acc(L_CP)
    SS  — worst (steepest) slope between adjacent levels
    """
    sim_col = "Sim: outputs"
    lvl_col = "Classified Level"
    LEVELS  = [1, 2, 3, 4, 5]
    _none   = {"SR": None, "SR_pct": None, "CP": None, "CD": None, "SS": None}

    if lvl_col not in df.columns or sim_col not in df.columns:
        return _none

    # No stress data → all metrics N/A
    has_stress = any(
        not df.loc[df[lvl_col] == f"L{lk}", sim_col].dropna().empty
        for lk in LEVELS
    )
    if not has_stress:
        return _none

    # acc_base from L0
    l0_sims  = l0_df[sim_col].dropna() if sim_col in l0_df.columns else pd.Series(dtype=float)
    acc_base = float((l0_sims >= threshold).mean()) if not l0_sims.empty else alpha_tau
    tau      = round(alpha_tau * acc_base, 4)

    # acc per level (None when no data)
    acc = {}
    for lk in LEVELS:
        sims = df.loc[df[lvl_col] == f"L{lk}", sim_col].dropna()
        acc[lk] = round(float((sims >= threshold).mean()), 4) if not sims.empty else None

    # ── SR ────────────────────────────────────────────────────────────────────
    sr_count = sum(1 for lk in LEVELS if acc[lk] is not None and acc[lk] >= tau)

    # ── CP ────────────────────────────────────────────────────────────────────
    cp = None
    for lk in LEVELS:
        if acc[lk] is not None and acc[lk] < tau:
            cp = lk
            break

    # ── CD ────────────────────────────────────────────────────────────────────
    if cp is None:
        cd = 0.0
    else:
        acc_pre  = acc[cp - 1] if cp > 1 and acc.get(cp - 1) is not None else acc_base
        cd       = round(acc_pre - acc[cp], 4)

    # ── SS (worst slope across adjacent levels, L0 included as anchor) ────────
    level_acc = [(0, acc_base)] + [(lk, acc[lk]) for lk in LEVELS if acc[lk] is not None]
    slopes    = [b - a for (_, a), (_, b) in zip(level_acc, level_acc[1:])]
    ss        = round(min(slopes), 4) if slopes else None

    return {
        "SR":     sr_count,
        "SR_pct": round(sr_count / len(LEVELS) * 100, 1),
        "CP":     f"L{cp}" if cp is not None else None,
        "CD":     cd,
        "SS":     ss,
    }


def compute_stress_metrics(
    eval_df:   pd.DataFrame,
    group_by:  str,
    threshold: Optional[float] = None,
    alpha_tau: float           = 0.80,
) -> pd.DataFrame:
    """
    Aggregate stress metrics per group from eval_df.

    Parameters
    ----------
    eval_df   : DataFrame from eval_output() — must contain Gate, Score, Sim: outputs
    group_by  : "level" | "category" | "dimension"
    threshold : acceptance cutoff for Sim: outputs (used for SR computation)
    alpha_tau : survival threshold multiplier for SR (default 0.80)

    Returns
    -------
    DataFrame with columns:
        <group_by> | n | mean_score | mean_stability | mean_risk | p_score
                   | SR | SR_pct | CP | CD | SS
    Gate == True rows only. Sorted L0→L5 for level, alphabetical otherwise.
    """
    if group_by not in _GROUP_COL:
        raise ValueError(
            f"group_by={group_by!r} not recognised. "
            f"Use one of: {list(_GROUP_COL)}"
        )

    col = _GROUP_COL[group_by]
    if col not in eval_df.columns:
        raise ValueError(f"Column {col!r} not found in eval_df.")

    # ── Filter to Gate == True ────────────────────────────────────────────────
    if "Gate" in eval_df.columns:
        df = eval_df[eval_df["Gate"] == True].copy()
    else:
        df = eval_df.copy()

    # ── Exclude measurement-only rows (e.g. Decision Complexity dcdef) ─────
    if "Test Type" in df.columns:
        df = df[df["Test Type"] != "measurement"].copy()

    # ── Shared L0 rows for acc_base ───────────────────────────────────────────
    lvl_col = "Classified Level"
    l0_df   = df[df[lvl_col] == "L0"] if lvl_col in df.columns else pd.DataFrame()

    # ── Add risk column ───────────────────────────────────────────────────────
    df = _add_risk(df)

    # ── Aggregate per group ───────────────────────────────────────────────────
    rows = []
    for group_val, grp in df.groupby(col):
        row = {group_by: str(group_val)}
        row.update(_aggregate_group(grp))

        # SR, CP, CD, SS — inject L0 for category/dimension; level uses global l0_df
        if threshold is not None:
            combined = pd.concat([l0_df, grp], ignore_index=True) if group_by != "level" else grp
            row.update(_level_metrics(combined, threshold, alpha_tau, l0_df))

        rows.append(row)

    result = pd.DataFrame(rows)

    if group_by == "level":
        result = _sort_level(result, "level")
    else:
        result = result.sort_values(group_by).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scatter plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_stress_scatter(
    eval_df:     pd.DataFrame,
    group_by:    str,
    threshold:   float,
    x_threshold: Optional[float] = None,
    aggregated:  bool            = False,
    sigma:       float           = 1.0,
    figsize:     Tuple[int, int] = (10, 5),
    bg_color:    str             = "#ffffff",
) -> None:
    """
    Scatter plot: X = Perturbation Magnitude (Score)
                  Y = Output Stability       (Sim: outputs)
                  Color = group (level / category / dimension)
    """
    if group_by not in _GROUP_COL:
        raise ValueError(
            f"group_by={group_by!r} not recognised. "
            f"Use one of: {list(_GROUP_COL)}"
        )

    col     = _GROUP_COL[group_by]
    sim_col = "Sim: outputs"
    scr_col = "Score"

    # Short display labels to reduce clutter
    SHORT_LABELS = {
        "Conflict Instruction Stress": "Conflict Instr.",
        "Context Interference": "Context Interf.",
        "Context Load": "Context Load",
        "Input Quality": "Input Quality",
        "Knowledge Boundary": "Knowledge Bound.",
        "Semantic Variation": "Semantic Var.",
        "Structural/Format": "Format",
        "Temporal Consistency": "Temporal Consist.",
    }

    # ── Build plot data ───────────────────────────────────────────────────────
    raw = eval_df[eval_df["Gate"] == True].copy() if "Gate" in eval_df.columns else eval_df.copy()
    if "Test Type" in raw.columns:
        raw = raw[raw["Test Type"] != "measurement"].copy()
    raw = raw[[col, scr_col, sim_col]].dropna()

    if aggregated:
        agg = (
            raw.groupby(col)[[scr_col, sim_col]]
            .agg(mean_x=(scr_col, "mean"), mean_y=(sim_col, "mean"),
                 std_x=(scr_col,  "std"),  std_y=(sim_col,  "std"))
            .reset_index()
        )
        agg["std_x"] = agg["std_x"].fillna(0.0)
        agg["std_y"] = agg["std_y"].fillna(0.0)
        df          = agg
        point_size  = 100
    else:
        df          = raw
        point_size  = 36

    # ── Colour palette ────────────────────────────────────────────────────────
    def _sort_key(v):
        try:
            return int(str(v).replace("L", ""))
        except (ValueError, TypeError):
            return str(v)

    groups = sorted(df[col].unique(), key=_sort_key)
    palette = plt.colormaps.get_cmap("tab10")
    color_map = {g: palette(i) for i, g in enumerate(groups)}

    # ── X threshold default ───────────────────────────────────────────────────
    if x_threshold is None:
        x_col = "mean_x" if aggregated else scr_col
        x_threshold = float(df[x_col].mean())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    for group_val in groups:
        mask  = df[col] == group_val
        color = color_map[group_val]
        label_txt = SHORT_LABELS.get(str(group_val), str(group_val))

        if aggregated:
            row_data = df[mask].iloc[0]
            mx, my   = row_data["mean_x"], row_data["mean_y"]
            sx, sy   = row_data["std_x"],  row_data["std_y"]

            # lighter ellipse
            if sx > 0 or sy > 0:
                ellipse = mpatches.Ellipse(
                    (mx, my),
                    width=max(2 * sigma * sx, 0.01),
                    height=max(2 * sigma * sy, 0.01),
                    facecolor=color,
                    alpha=0.08,
                    edgecolor=color,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=2,
                )
                ax.add_patch(ellipse)

            # point
            ax.scatter(
                mx, my,
                color=color,
                label=label_txt,
                s=point_size,
                edgecolors="white",
                linewidths=0.7,
                alpha=0.9,
                zorder=4,
            )

            # offset label
            ax.annotate(
                label_txt,
                xy=(mx, my),
                xytext=(0, -12),
                ha="center",
                textcoords="offset points",
                fontsize=7,
                color=color,
                zorder=5,
            )
        else:
            ax.scatter(
                df.loc[mask, scr_col],
                df.loc[mask, sim_col],
                color=color,
                label=label_txt,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                s=point_size,
                zorder=3,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.05
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Threshold lines ───────────────────────────────────────────────────────
    ax.axhline(
        threshold,
        color="grey",
        linewidth=1.0,
        linestyle="--",
        alpha=0.45,
        label=f"y threshold = {threshold}",
        zorder=1,
    )
    ax.axvline(
        x_threshold,
        color="grey",
        linewidth=1.0,
        linestyle="--",
        alpha=0.45,
        label=f"x threshold = {x_threshold:.3f}",
        zorder=1,
    )

    # ── Quadrant labels ───────────────────────────────────────────────────────
    ax.tick_params(axis="both", labelsize=10)

    quad_kw = dict(
        fontsize=9,
        color="grey",
        alpha=0.35,
        ha="center",
        va="center",
    )

    ax.text(
        (x_min + x_threshold) / 2,
        (threshold + y_max) / 2,
        "Stable Behavior",
        **quad_kw,
    )
    ax.text(
        (x_threshold + x_max) / 2,
        (threshold + y_max) / 2,
        "Robust Under Stress",
        **quad_kw,
    )
    ax.text(
        (x_min + x_threshold) / 2,
        (y_min + threshold) / 2,
        "Unstable Response",
        **quad_kw,
    )
    ax.text(
        (x_threshold + x_max) / 2,
        (y_min + threshold) / 2,
        "Failure Under Stress",
        **quad_kw,
    )

    # ── Labels & legend ───────────────────────────────────────────────────────
    ax.set_xlabel("Perturbation Magnitude", fontsize=11)
    ax.set_ylabel("Output Stability", fontsize=11)
    ax.legend(
        loc="lower right",
        title=group_by.capitalize(),
        title_fontsize=9,
        fontsize=7,
        markerscale=0.6,
        handlelength=0.8,
        handletextpad=0.3,
        borderpad=0.5,
        labelspacing=0.3,
        facecolor=bg_color,
        edgecolor="#d1d5db",
        framealpha=0.88,
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Survival Rate (SR)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sr(
    eval_df:   pd.DataFrame,
    threshold: float,
    group_by:  str           = "overall",
    alpha_tau: float         = 0.80,
) -> pd.DataFrame:
    """
    Compute Survival Rate (SR) per group.

    A stress level Lk is considered "survived" when:
        acc(Lk) >= τ
    where:
        acc(Lk) = fraction of rows at level Lk where Sim: outputs >= threshold
        τ       = alpha_tau × acc_base
        acc_base= mean(match | L0)    [fallback: alpha_tau if no L0 rows]

    Parameters
    ----------
    eval_df   : DataFrame from eval_output()
    threshold : acceptance cutoff for Sim: outputs
    group_by  : "overall" | "category" | "dimension"
    alpha_tau : survival threshold multiplier (default 0.80)

    Returns
    -------
    DataFrame with columns:
        group | acc_base | tau | L1 | L2 | L3 | L4 | L5 | SR | SR_pct
        L1–L5 : True = survived (acc >= τ), False = failed, None = no data
        SR     : count of survived levels (0–5)
        SR_pct : SR / 5  as percentage
    """
    if group_by not in ("overall", "category", "dimension"):
        raise ValueError(
            f"group_by={group_by!r} not recognised. "
            f"Use one of: 'overall', 'category', 'dimension'"
        )

    sim_col = "Sim: outputs"
    lvl_col = "Classified Level"
    LEVELS  = [1, 2, 3, 4, 5]

    def _sr_for_slice(df: pd.DataFrame, label: str) -> dict:
        # ── acc_base from L0 ──────────────────────────────────────────────────
        l0 = df[df[lvl_col] == "L0"][sim_col].dropna() if lvl_col in df.columns else pd.Series(dtype=float)
        if not l0.empty:
            acc_base = float((l0 >= threshold).mean())
        else:
            acc_base = alpha_tau          # fallback when no L0 rows in slice

        tau = round(alpha_tau * acc_base, 4)

        # ── Per-level survival ────────────────────────────────────────────────
        row = {"group": label, "acc_base": round(acc_base, 4), "tau": tau}
        sr_count = 0

        for lk in LEVELS:
            mask = df[lvl_col] == f"L{lk}" if lvl_col in df.columns else pd.Series(False, index=df.index)
            sims = df.loc[mask, sim_col].dropna() if sim_col in df.columns else pd.Series(dtype=float)

            if sims.empty:
                row[f"L{lk}"] = None
            else:
                acc = float((sims >= threshold).mean())
                survived = acc >= tau
                row[f"L{lk}"] = survived
                if survived:
                    sr_count += 1

        row["SR"]     = sr_count
        row["SR_pct"] = round(sr_count / len(LEVELS) * 100, 1)  # always out of 5

        return row

    # ── Filter Gate == True ───────────────────────────────────────────────────
    df = eval_df[eval_df["Gate"] == True].copy() if "Gate" in eval_df.columns else eval_df.copy()

    # ── Exclude measurement-only rows (e.g. Decision Complexity dcdef) ─────
    if "Test Type" in df.columns:
        df = df[df["Test Type"] != "measurement"].copy()

    # ── Build rows ────────────────────────────────────────────────────────────
    if group_by == "overall":
        rows = [_sr_for_slice(df, "overall")]

    else:
        col  = _GROUP_COL[group_by]
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in eval_df.")
        rows = []
        # Inject L0 into every group so acc_base is always available
        l0_df     = df[df[lvl_col] == "L0"] if lvl_col in df.columns else pd.DataFrame()
        stress_df = df[df[lvl_col] != "L0"] if lvl_col in df.columns else df
        for group_val, grp in stress_df.groupby(col):
            combined = pd.concat([l0_df, grp], ignore_index=True)
            rows.append(_sr_for_slice(combined, str(group_val)))

    result = pd.DataFrame(rows)
    if group_by != "overall":
        result = result.sort_values("group").reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Bar + Line chart: ranked by Perturbation Magnitude
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_stability_bars(
    eval_df:  pd.DataFrame,
    bin_size: int            = 1,
    figsize:  Tuple[int,int] = (10, 5),
    bg_color: str            = "#ffffff",
) -> None:
    """
    Bar chart of Perturbation Magnitude (Score) ranked low → high,
    with an overlaid line for Output Stability (Sim: outputs).

    Parameters
    ----------
    eval_df  : DataFrame from eval_output() — Gate == True rows used
    bin_size : number of tests per bin (default 1 = no binning).
               When > 1, adjacent tests (after sorting by Score) are grouped
               and their mean Score / mean Sim: outputs is plotted per bin.
    figsize  : figure size tuple

    Style
    -----
    Black background, gray bars, light-gray line.
    """
    sim_col = "Sim: outputs"
    scr_col = "Score"

    # ── Filter & sort ─────────────────────────────────────────────────────────
    df = eval_df[eval_df["Gate"] == True].copy() if "Gate" in eval_df.columns else eval_df.copy()
    df = df[[scr_col, sim_col]].dropna().sort_values(scr_col).reset_index(drop=True)

    # ── Bin ───────────────────────────────────────────────────────────────────
    bin_size = max(1, int(bin_size))
    n        = len(df)
    indices  = range(0, n, bin_size)

    scores = []
    stabs  = []
    for i in indices:
        chunk = df.iloc[i : i + bin_size]
        scores.append(float(chunk[scr_col].mean()))
        stabs.append(float(chunk[sim_col].mean()))

    x     = list(range(len(scores)))
    x_lbl = (
        [str(i + 1) for i in x]            if bin_size == 1
        else [f"{i*bin_size+1}–{min((i+1)*bin_size, n)}" for i in x]
    )

    # ── Colours ───────────────────────────────────────────────────────────────
    BG        = bg_color
    BAR_CLR   = "#aaaaaa"
    BAR_EDGE  = "#888888"
    LINE_CLR  = "#000000"
    TICK_CLR  = "#000000"
    LABEL_CLR = "#000000"
    GRID_CLR  = "#e0e0e0"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)

    # Bars — Perturbation Magnitude
    ax.bar(
        x, scores,
        color=BAR_CLR, edgecolor=BAR_EDGE, linewidth=0.5,
        width=0.65, zorder=2,
        label="Perturbation Magnitude",
    )

    # Line — Output Stability
    ax.plot(
        x, stabs,
        color=LINE_CLR, linewidth=1.2,
        marker="None", zorder=3,
        label="Output Stability",
    )

    # ── Grid & spines ─────────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.7, linestyle="--")
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(BAR_EDGE)

    # ── Ticks & labels ────────────────────────────────────────────────────────
    show_every = max(1, len(x) // 20)       # avoid crowding on large datasets
    ax.set_xticks(x[::show_every])
    ax.set_xticklabels(x_lbl[::show_every], rotation=45, ha="right",
                      fontsize=7, color=TICK_CLR)
    ax.tick_params(axis="y", colors=TICK_CLR, labelsize=7)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, 1.05)

    bin_lbl = f"  (bin size = {bin_size})" if bin_size > 1 else ""
    ax.set_xlabel(f"Tests ranked by Perturbation Magnitude{bin_lbl}",
                  fontsize=8, color=LABEL_CLR, labelpad=6)
    ax.set_ylabel("Value  [0 – 1]", fontsize=8, color=LABEL_CLR)

    # ── Legend ────────────────────────────────────────────────────────────────
    leg = ax.legend(
        fontsize=8, facecolor=BG, edgecolor="#cccccc",
        labelcolor=LABEL_CLR, loc="upper left",
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-category stability chart
# ─────────────────────────────────────────────────────────────────────────────

# Short labels reused from scatter
_CAT_SHORT = {
    "Conflict Instruction Stress": "Conflict Instr.",
    "Context Interference": "Context Interf.",
    "Context Load": "Context Load",
    "Input Quality": "Input Quality",
    "Knowledge Boundary": "Knowledge Bound.",
    "Semantic Variation": "Semantic Var.",
    "Structural/Format": "Format",
    "Temporal Consistency": "Temporal Consist.",
}


def plot_category_stability(
    eval_df:      pd.DataFrame,
    categories:   Union[List[str], str, None] = None,
    show_bars:    bool                        = False,
    bin_size:     int                         = 1,
    figsize:      Tuple[int, int]             = (12, 5),
    bg_color:     str                         = "#ffffff",
) -> None:
    """
    Output Stability line per category, plotted against Perturbation Magnitude.

    Parameters
    ----------
    eval_df    : DataFrame with Gate, Score, Sim: outputs, Category columns.
    categories : Which categories to plot.
                 - None or "all"  → every category present
                 - list of names  → only those (e.g. ["Semantic Variation", "Knowledge Boundary"])
    show_bars  : If True, show gray bars for Perturbation Magnitude behind the lines.
    bin_size   : Group N adjacent tests per bin (after sorting by Score). Default 1.
    figsize    : Figure size.
    bg_color   : Background colour.
    """
    sim_col = "Sim: outputs"
    scr_col = "Score"
    cat_col = "Category"

    # ── Filter gated rows ────────────────────────────────────────────────────
    df = eval_df[eval_df["Gate"] == True].copy() if "Gate" in eval_df.columns else eval_df.copy()
    if "Test Type" in df.columns:
        df = df[df["Test Type"] != "measurement"].copy()
    df = df[[cat_col, scr_col, sim_col]].dropna()

    # ── Select categories ────────────────────────────────────────────────────
    all_cats = sorted(df[cat_col].unique())
    if categories is None or categories == "all":
        selected = all_cats
    elif isinstance(categories, str):
        selected = [categories]
    else:
        selected = [c for c in categories if c in all_cats]
        missing = [c for c in categories if c not in all_cats]
        if missing:
            print(f"[WARN] Categories not found: {missing}")

    if not selected:
        print("[WARN] No matching categories to plot.")
        return None

    # ── Colours ──────────────────────────────────────────────────────────────
    palette = plt.colormaps.get_cmap("tab10")
    cat_colors = {c: palette(i) for i, c in enumerate(selected)}

    BG        = bg_color
    BAR_CLR   = "#d0d0d0"
    BAR_EDGE  = "#bbbbbb"
    TICK_CLR  = "#000000"
    LABEL_CLR = "#000000"
    GRID_CLR  = "#e0e0e0"

    # ── Build per-category series (sorted by Score, then binned) ─────────────
    bin_size = max(1, int(bin_size))

    # Global sort to get a shared x-axis
    all_rows = df[df[cat_col].isin(selected)].sort_values(scr_col).reset_index(drop=True)
    n = len(all_rows)

    # Bin indices
    indices = range(0, n, bin_size)

    # Global binned scores (for x-axis labels and optional bars)
    g_scores = []
    for i in indices:
        chunk = all_rows.iloc[i : i + bin_size]
        g_scores.append(float(chunk[scr_col].mean()))

    x = list(range(len(g_scores)))
    x_lbl = (
        [f"{s:.2f}" for s in g_scores] if bin_size == 1
        else [f"{i*bin_size+1}-{min((i+1)*bin_size, n)}" for i in x]
    )

    # Per-category: bin stability values aligned to global sort
    cat_stabs = {}
    for cat in selected:
        cat_mask = all_rows[cat_col] == cat
        stabs = []
        for i in indices:
            chunk = all_rows.iloc[i : i + bin_size]
            cat_chunk = chunk[chunk[cat_col] == cat]
            if not cat_chunk.empty:
                stabs.append(float(cat_chunk[sim_col].mean()))
            else:
                stabs.append(None)
        cat_stabs[cat] = stabs

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)

    # Optional bars for Perturbation Magnitude
    if show_bars:
        ax.bar(
            x, g_scores,
            color=BAR_CLR, edgecolor=BAR_EDGE, linewidth=0.4,
            width=0.7, zorder=1, alpha=0.5,
            label="Perturbation Magnitude",
        )

    # Stability lines per category
    for cat in selected:
        label_txt = _CAT_SHORT.get(cat, cat)
        vals = cat_stabs[cat]
        # Filter out None for plotting
        xs_plot = [xi for xi, v in zip(x, vals) if v is not None]
        ys_plot = [v  for v in vals if v is not None]
        if xs_plot:
            ax.plot(
                xs_plot, ys_plot,
                color=cat_colors[cat],
                linewidth=1.8,
                marker="o",
                markersize=4,
                alpha=0.85,
                zorder=3,
                label=label_txt,
            )

    # ── Grid & spines ────────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.7, linestyle="--")
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    # ── Ticks & labels ───────────────────────────────────────────────────────
    show_every = max(1, len(x) // 25)
    ax.set_xticks(x[::show_every])
    ax.set_xticklabels(x_lbl[::show_every], rotation=45, ha="right",
                       fontsize=8, color=TICK_CLR)
    ax.tick_params(axis="y", colors=TICK_CLR, labelsize=9)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, 1.05)

    bin_lbl = f"  (bin={bin_size})" if bin_size > 1 else ""
    ax.set_xlabel(f"Tests ranked by Perturbation Magnitude{bin_lbl}",
                  fontsize=11, color=LABEL_CLR, labelpad=6)
    ax.set_ylabel("Output Stability", fontsize=11, color=LABEL_CLR)

    # ── Legend ───────────────────────────────────────────────────────────────
    ax.legend(
        fontsize=9, facecolor=BG, edgecolor="#cccccc",
        labelcolor=LABEL_CLR, loc="lower left",
    )

    plt.tight_layout()
    return fig
