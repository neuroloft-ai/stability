"""
qualify_metrics.py — Enrich compute_stress_metrics output with qualitative bands,
                     profile matching, and actionable recommendations.

Driven by CSV files in AIQQ/data/metrics/:
    Layer 1 — Per-Metric Recommendations.csv
    Layer 2 — Combined Profile Recommendations.csv
    Layer 3 — Action Categories.csv

Public API
----------
qualify_metrics(stress_df, eval_df, data_dir=None) → pd.DataFrame
    Adds *_band columns, profile, fix_priority, root_problem, and actions
    to the output of compute_stress_metrics().
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default data directory
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path(r"./data/metrics")


# ─────────────────────────────────────────────────────────────────────────────
# CSV loaders — lazy, module-level cache
# ─────────────────────────────────────────────────────────────────────────────

_CACHE: dict = {}


def _load_csvs(data_dir: Path) -> dict:
    key = str(data_dir)
    if key in _CACHE:
        return _CACHE[key]

    layer1 = pd.read_csv(data_dir / "Layer 1 — Per-Metric Recommendations.csv")
    layer2 = pd.read_csv(data_dir / "Layer 2 — Combined Profile Recommendations.csv")
    layer3 = pd.read_csv(data_dir / "Layer 3 — Action Categories.csv")
    levels = pd.read_csv(data_dir / "levels.csv")

    # Strip whitespace from string columns, drop empty rows
    for df in (layer1, layer2, layer3):
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].str.strip()

    layer1 = layer1.dropna(subset=["Metric", "Band"]).reset_index(drop=True)
    layer2 = layer2.dropna(subset=["Profile"]).reset_index(drop=True)
    layer3 = layer3.dropna(subset=["Category"]).reset_index(drop=True)

    result = {"layer1": layer1, "layer2": layer2, "layer3": layer3, "levels": levels}
    _CACHE[key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Hard-threshold binning functions
# Each returns "Low" | "Medium" | "High" | None
# ─────────────────────────────────────────────────────────────────────────────

def _is_null(val) -> bool:
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _bin_sr(val) -> Optional[str]:
    """SR 0–1 = Low, 2–3 = Medium, 4–5 = High. None if no stress rows."""
    if _is_null(val):
        return None
    v = int(val)
    if v <= 1:  return "Low"
    if v <= 3:  return "Medium"
    return "High"


def _bin_cp(val) -> Optional[str]:
    """CP: L1 = Low resilience, L2-L3 = Medium, L4-L5/None = High (resilient)."""
    if _is_null(val):
        return "High"   # None = never collapsed = most resilient
    s = str(val).strip().upper()
    if s == "L1":              return "Low"
    if s in ("L2", "L3"):      return "Medium"
    return "High"              # L4, L5, or None


def _bin_cd(val) -> Optional[str]:
    """CD: 0–0.3 = Low severity, 0.3–0.6 = Medium, 0.6–1.0 = High (cliff)."""
    if _is_null(val):
        return None
    v = float(val)
    if v < 0.3:  return "Low"
    if v < 0.6:  return "Medium"
    return "High"


def _bin_ss(val) -> Optional[str]:
    """SS: 0 to -0.3 = Low sensitivity, -0.3 to -0.6 = Medium, -0.6 to -1.0 = High (Steep)."""
    if _is_null(val):
        return None
    v = float(val)
    if v >= -0.3:  return "Low"
    if v >= -0.6:  return "Medium"
    return "High"


def _bin_p_score(val) -> Optional[str]:
    """p_score: 0–0.4 = Low, 0.4–0.7 = Medium, 0.7–1.0 = High robustness."""
    if _is_null(val):
        return None
    v = float(val)
    if v < 0.4:  return "Low"
    if v < 0.7:  return "Medium"
    return "High"


def _bin_stability(val) -> Optional[str]:
    """Stability (Sim: outputs): 0–0.6 = Low, 0.6–0.85 = Medium, 0.85–1.0 = High."""
    if _is_null(val):
        return None
    v = float(val)
    if v < 0.6:   return "Low"
    if v < 0.85:  return "Medium"
    return "High"


def _bin_dynamic(val, p33: float, p66: float) -> Optional[str]:
    """Percentile-based binning for Score and Risk (distribution-dependent)."""
    if _is_null(val):
        return None
    v = float(val)
    if v <= p33:  return "Low"
    if v <= p66:  return "Medium"
    return "High"


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Layer 1 lookup: (metric, internal_band) → CSV band label → signal + action
# ─────────────────────────────────────────────────────────────────────────────

# Maps (metric, internal band) → CSV Band string in Layer 1
_L1_BAND_MAP: dict[str, dict[str, str]] = {
    "SR": {
        "Low":    "Low 0–1",
        "Medium": "Medium 2–3",
        "High":   "High 4–5",
    },
    "CP": {
        "Low":    "L1",
        "Medium": "L2–L3",
        "High":   "L4–L5 / None",
    },
    "CD": {
        "Low":    "Low 0.0–0.3",
        "Medium": "Medium 0.3–0.6",
        "High":   "High 0.6–1.0",
    },
    "SS": {
        "Low":    "Low 0 to −0.3",
        "Medium": "Medium −0.3 to −0.6",
        "High":   "Steep −0.6 to −1.0",
    },
    "p_score": {
        "Low":    "Low 0.0–0.4",
        "Medium": "Medium 0.4–0.7",
        "High":   "High 0.7–1.0",
    },
    "Stability": {
        "Low":    "Low 0.0–0.6",
        "Medium": "Medium 0.6–0.85",
        "High":   "High 0.85–1.0",
    },
    "Risk": {
        "Low":    "Low",
        "Medium": "Medium",
        "High":   "High (top tercile)",
    },
}


def _lookup_layer1(layer1_df: pd.DataFrame, metric: str, band: Optional[str]) -> dict:
    """Return {signal, action} from Layer 1 CSV for (metric, band). Returns Nones on miss."""
    if band is None or metric not in _L1_BAND_MAP or band not in _L1_BAND_MAP[metric]:
        return {"signal": None, "action": None}

    csv_band = _L1_BAND_MAP[metric][band]
    row = layer1_df[
        (layer1_df["Metric"] == metric) &
        (layer1_df["Band"]   == csv_band)
    ]
    if row.empty:
        return {"signal": None, "action": None}

    r = row.iloc[0]
    return {
        "signal": r.get("What it signals"),
        "action": r.get("Recommended Action"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Layer 2: Profile matching
# Priority order: most severe first (first full match wins on tie)
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE_RULES = [
    {
        "name":         "Cliff Brittle",
        "sr":           "Low",
        "cp":           "Low",      # L1
        "cd":           "High",
        "ss":           "High",     # Steep
        "fix_priority": "Critical — retrain or major overhaul",
        "root_problem": "Fundamental robustness gap",
    },
    {
        "name":         "Gradual Fader",
        "sr":           "Low",
        "cp":           "Medium",   # L3–L4
        "cd":           "Low",
        "ss":           "Medium",
        "fix_priority": "High — progressive augmentation L1→L5",
        "root_problem": "Steady degradation, no cliff",
    },
    {
        "name":         "Late Collapse",
        "sr":           "High",
        "cp":           "High",     # L4–L5
        "cd":           "Medium",
        "ss":           "Medium",
        "fix_priority": "Medium — targeted hardening at extreme stress",
        "root_problem": "Strong but not fully robust",
    },
    {
        "name":         "Output Drifter",
        "sr":           "High",
        "cp":           "High",     # None
        "cd":           "Low",
        "ss":           "Low",
        "p_score_not":  "High",     # distinguishes from Fully Robust
        "fix_priority": "Medium — inference tuning, not retraining",
        "root_problem": "Correct answers, inconsistent phrasing",
    },
    {
        "name":         "Fully Robust",
        "sr":           "High",
        "cp":           "High",
        "cd":           "Low",
        "ss":           "Low",
        "fix_priority": "None",
        "root_problem": "No issue",
    },
]


def _match_profile(bands: dict) -> dict:
    """
    Score each profile by how many conditions match the given bands.
    Returns the best-matching profile (highest score; priority order breaks ties).
    """
    sr = bands.get("SR_band")
    cp = bands.get("CP_band")
    cd = bands.get("CD_band")
    ss = bands.get("SS_band")
    ps = bands.get("p_score_band")

    best = {"name": "Unknown", "fix_priority": "N/A", "root_problem": "N/A"}
    best_score = -1

    for rule in _PROFILE_RULES:
        score = 0.0
        if sr and rule.get("sr") and sr == rule["sr"]:   score += 1
        if cp and rule.get("cp") and cp == rule["cp"]:   score += 1
        if cd and rule.get("cd") and cd == rule["cd"]:   score += 1
        if ss and rule.get("ss") and ss == rule["ss"]:   score += 1
        # p_score differentiator: Output Drifter vs Fully Robust
        if "p_score_not" in rule:
            if ps and ps != rule["p_score_not"]:
                score += 0.5

        if score > best_score:
            best_score = score
            best = {
                "name":         rule["name"],
                "fix_priority": rule["fix_priority"],
                "root_problem": rule["root_problem"],
            }

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Layer 3: Action category matching
# ─────────────────────────────────────────────────────────────────────────────

def _match_actions(bands: dict) -> list:
    """
    Return triggered action categories based on band combinations.
    Ordered by severity: Critical → High → Medium → Monitor.
    """
    sr  = bands.get("SR_band")
    cp  = bands.get("CP_band")
    cd  = bands.get("CD_band")
    ss  = bands.get("SS_band")
    ps  = bands.get("p_score_band")
    stb = bands.get("Stability_band")
    rsk = bands.get("Risk_band")

    triggered = []

    # Training data: SR Low + CP Low + CD High
    if sr == "Low" and cp == "Low" and cd == "High":
        triggered.append("Training data")

    # Model architecture: SR Low AND SS High (steep)
    if sr == "Low" and ss == "High":
        triggered.append("Model architecture")

    # Prompt / instructions: early collapse with poor output consistency
    if cp in ("Low", "Medium") and ps in ("Low", "Medium"):
        triggered.append("Prompt / instructions (LLMs)")

    # Inference settings: low stability or low p_score
    if stb == "Low" or ps == "Low":
        triggered.append("Inference settings")

    # Deployment guardrails: high risk exposure
    if rsk == "High":
        triggered.append("Deployment guardrails")

    # Monitoring only: if nothing else triggered and model is healthy
    if not triggered:
        triggered.append("Monitoring only")

    return triggered


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Main public function
# ─────────────────────────────────────────────────────────────────────────────

def qualify_metrics(
    stress_df: pd.DataFrame,
    eval_df:   pd.DataFrame,
    data_dir:  Optional[str] = None,
) -> pd.DataFrame:
    """
    Enrich compute_stress_metrics() output with qualitative bands,
    profile matching, and actionable recommendations.

    Parameters
    ----------
    stress_df : DataFrame from compute_stress_metrics()
    eval_df   : original eval DataFrame (for Score/Risk percentile computation)
    data_dir  : path to the metrics CSV folder (defaults to AIQQ/data/metrics)

    Returns
    -------
    DataFrame with all original columns plus:
        <metric>_band    — qualitative band (Low / Medium / High)
        <metric>_signal  — what the band means (from Layer 1 CSV)
        <metric>_action  — recommended action  (from Layer 1 CSV)
        profile          — matched failure profile (from Layer 2)
        fix_priority     — priority label for the profile
        root_problem     — root cause description
        actions          — list of triggered action categories (from Layer 3)

    Band strategies
    ---------------
    Hard threshold : SR, CP, CD, SS, p_score, Stability
    Percentile     : Score (p33/p66), Risk (p33/p66) — distribution-dependent
    """
    dir_path = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    csvs     = _load_csvs(dir_path)
    layer1   = csvs["layer1"]

    result = stress_df.copy()

    # ── Percentile thresholds for Score and Risk (from eval_df) ───────────────
    gated = eval_df[eval_df["Gate"] == True].copy() if "Gate" in eval_df.columns else eval_df.copy()

    score_vals = gated["Score"].dropna() if "Score" in gated.columns else pd.Series(dtype=float)
    if not score_vals.empty:
        score_p33, score_p66 = float(score_vals.quantile(0.33)), float(score_vals.quantile(0.66))
    else:
        score_p33, score_p66 = 0.33, 0.66

    if "Score" in gated.columns and "Sim: outputs" in gated.columns:
        risk_vals = (gated["Score"] * (1 - gated["Sim: outputs"])).dropna()
    else:
        risk_vals = pd.Series(dtype=float)
    if not risk_vals.empty:
        risk_p33, risk_p66 = float(risk_vals.quantile(0.33)), float(risk_vals.quantile(0.66))
    else:
        risk_p33, risk_p66 = 0.10, 0.25

    # ── Bin each metric → _band column ────────────────────────────────────────
    if "mean_score" in result.columns:
        result["Score_band"] = result["mean_score"].apply(
            lambda v: _bin_dynamic(v, score_p33, score_p66)
        )
    if "mean_stability" in result.columns:
        result["Stability_band"] = result["mean_stability"].apply(_bin_stability)
    if "mean_risk" in result.columns:
        result["Risk_band"] = result["mean_risk"].apply(
            lambda v: _bin_dynamic(v, risk_p33, risk_p66)
        )
    if "p_score" in result.columns:
        result["p_score_band"] = result["p_score"].apply(_bin_p_score)
    if "SR" in result.columns:
        result["SR_band"] = result["SR"].apply(_bin_sr)
    if "CP" in result.columns:
        result["CP_band"] = result["CP"].apply(_bin_cp)
    if "CD" in result.columns:
        result["CD_band"] = result["CD"].apply(_bin_cd)
    if "SS" in result.columns:
        result["SS_band"] = result["SS"].apply(_bin_ss)

    # ── Layer 1: per-metric signal + action ───────────────────────────────────
    _METRIC_BAND_PAIRS = [
        ("Stability", "Stability_band"),
        ("Risk",      "Risk_band"),
        ("p_score",   "p_score_band"),
        ("SR",        "SR_band"),
        ("CP",        "CP_band"),
        ("CD",        "CD_band"),
        ("SS",        "SS_band"),
    ]
    for metric, band_col in _METRIC_BAND_PAIRS:
        if band_col not in result.columns:
            continue
        signals, actions_col = [], []
        for band in result[band_col]:
            info = _lookup_layer1(layer1, metric, band)
            signals.append(info["signal"])
            actions_col.append(info["action"])
        result[f"{metric}_signal"] = signals
        result[f"{metric}_action"] = actions_col

    # ── Layer 2 + 3: profile + actions per row ────────────────────────────────
    profiles, fix_pris, root_pbs, actions_list = [], [], [], []

    for _, row in result.iterrows():
        bands = {
            "SR_band":        row.get("SR_band"),
            "CP_band":        row.get("CP_band"),
            "CD_band":        row.get("CD_band"),
            "SS_band":        row.get("SS_band"),
            "p_score_band":   row.get("p_score_band"),
            "Stability_band": row.get("Stability_band"),
            "Risk_band":      row.get("Risk_band"),
        }
        prof = _match_profile(bands)
        acts = _match_actions(bands)
        profiles.append(prof["name"])
        fix_pris.append(prof["fix_priority"])
        root_pbs.append(prof["root_problem"])
        actions_list.append(", ".join(acts))

    result["profile"]      = profiles
    result["fix_priority"] = fix_pris
    result["root_problem"] = root_pbs
    result["actions"]      = actions_list

    return result
