"""
eval_output.py — Pipeline module 4: Compare predicted outputs to ground truth.

Receives the results DataFrame from test_runner.run_tests() and computes a
similarity score for each output field using the method appropriate to its type.

Similarity by output type
--------------------------
  categorical       — exact match (case-insensitive)        → 0 or 1
  multi_categorical — Jaccard:  |pred ∩ gt| / |pred ∪ gt|  → [0, 1]
  text              — cosine similarity via embed_fn         → [0, 1]
                      falls back to SequenceMatcher ratio if embed_fn is None
  numeric           — 1 − normalised absolute difference     → [0, 1]

New columns appended
---------------------
  GT: <agent_key>   — ground-truth value used for comparison (for inspection)
  Sim: <agent_key>  — similarity score in [0, 1]; None if run failed
  Sim: outputs      — weighted average of all Sim: <key> scores using output_def weights

Public functions
----------------
eval_output(results_df, intake, embed_fn=None)  → DataFrame
"""

from __future__ import annotations

import ast
import math
import re
from difflib import SequenceMatcher
from typing import Callable, Optional

import pandas as pd

from modules.data_intake import DataIntakeResult


# ─────────────────────────────────────────────────────────────────────────────
# Similarity functions
# ─────────────────────────────────────────────────────────────────────────────

def _exact_match(pred, gt: str) -> float:
    """Categorical: 1.0 if pred matches gt (case-insensitive), else 0.0."""
    if pred is None:
        return 0.0
    return 1.0 if str(pred).strip().lower() == str(gt).strip().lower() else 0.0


def _parse_multi(pred) -> set:
    """
    Normalise any pred format to a lowercase string set for Jaccard comparison.

    Handles all forms that arrive depending on data source:
      - Python list/set/tuple  (in-memory from run_tests)
      - "['a', 'b', 'c']"     (string-serialised list from CSV)
      - "{'a', 'b'}"           (string-serialised set from CSV)
      - "a, b, c"              (comma-joined string from some LLMs)
      - "a"                    (single-value string — categorical-like)
      - None / NaN             (failed or missing prediction)
    """
    if pred is None:
        return set()
    # Already a proper collection
    if isinstance(pred, (list, set, tuple, frozenset)):
        return {str(x).strip().lower() for x in pred if str(x).strip()}
    s = str(pred).strip()
    if not s or s.lower() in ("nan", "none", "n/a", "null"):
        return set()
    # String representation of a Python literal: "['a', 'b']" or "{'a', 'b'}"
    if s[0] in ("[", "{"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, set, tuple, frozenset)):
                return {str(x).strip().lower() for x in parsed if str(x).strip()}
        except (ValueError, SyntaxError):
            pass
    # Comma-separated fallback: "Feature, Documentation, Feedback"
    if "," in s:
        return {part.strip().lower() for part in s.split(",") if part.strip()}
    # Single value
    return {s.lower()}


def _jaccard(pred, gt_set: set) -> float:
    """
    Multi-categorical: Jaccard similarity between predicted and ground-truth sets.
    Jaccard = |pred ∩ gt| / |pred ∪ gt|  — both sides lowercased.
    pred is normalised via _parse_multi, so lists, CSV strings, and
    comma-separated strings are all handled correctly.
    """
    pred_set     = _parse_multi(pred)
    union        = pred_set | gt_set
    intersection = pred_set & gt_set
    return round(len(intersection) / len(union), 4) if union else 0.0


def _cosine_text(pred: str, gt: str, embed_fn: Optional[Callable]) -> float:
    """
    Text: cosine similarity between embeddings if embed_fn is provided,
    otherwise SequenceMatcher character-level ratio as a lightweight fallback.
    """
    pred = (pred or "").strip()
    gt   = (gt   or "").strip()
    if not pred or not gt:
        return 0.0
    if embed_fn is not None:
        v1  = embed_fn(pred)
        v2  = embed_fn(gt)
        dot = sum(x * y for x, y in zip(v1, v2))
        n1  = math.sqrt(sum(x * x for x in v1))
        n2  = math.sqrt(sum(x * x for x in v2))
        return round(dot / (n1 * n2), 4) if n1 and n2 else 0.0
    return round(SequenceMatcher(None, pred.lower(), gt.lower()).ratio(), 4)


def _to_number(x):
    """Extract the numeric value from a string, ignoring $, commas, units, words, etc.
    '$1,200' -> 1200.0 ; '694 dollars' -> 694.0 ; '8.0' -> 8.0 ; '8.' -> 8.0 ; returns None if none."""
    m = re.search(r'-?\s*\$?\s*\d[\d,]*(?:\.\d+)?', str(x))
    if not m:
        return None
    tok = m.group(0).replace('$', '').replace(',', '').replace(' ', '')
    try:
        return float(tok)
    except ValueError:
        return None


def _numeric_sim(pred, gt: str) -> float:
    """
    Numeric: compare ONLY the numeric value (strip $, commas, units, whitespace).
    Exact match with 0.1 % relative tolerance. Returns 1.0 if match, 0.0 otherwise.
    """
    p = _to_number(pred)
    g = _to_number(gt)
    if p is None or g is None:
        return 0.0
    if abs(g) > 1e-9:
        return 1.0 if abs(p - g) / abs(g) < 1e-3 else 0.0
    return 1.0 if abs(p) < 1e-9 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gt(intake_row: pd.Series, output_def: dict):
    """
    Extract the ground-truth value for one output_def from an intake row.

    multi_categorical → frozenset of lowercase non-null values across all gt_cols
    categorical / text / numeric → string value from the single gt_col
    """
    out_type = output_def["type"]
    gt_cols  = output_def["gt_cols"]

    if out_type == "multi_categorical":
        vals = set()
        for col in gt_cols:
            v = str(intake_row.get(col, "")).strip()
            if v.lower() not in ("", "nan", "none", "n/a", "null"):
                vals.add(v.lower())
        return frozenset(vals)

    # categorical / text / numeric — single column
    col = gt_cols[0]
    return str(intake_row.get(col, "")).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def eval_output(
    results_df: pd.DataFrame,
    intake: DataIntakeResult,
    embed_fn: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Compute output similarity scores for every row in results_df.

    Parameters
    ----------
    results_df : DataFrame from test_runner.run_tests()
    intake     : DataIntakeResult — provides output_defs and ground-truth rows
    embed_fn   : optional callable  str -> list[float]
                 Used for 'text' output type.  If None, falls back to
                 SequenceMatcher character-level ratio.

    Returns
    -------
    DataFrame — all original columns preserved, with these added per output_def:
        GT: <agent_key>   — ground-truth value (for inspection)
        Sim: <agent_key>  — similarity score in [0, 1]
                            None when Run Status != "ok"
        Sim: outputs      — weighted average: Σ(weight_i × Sim_i)
                            weights from intake.output_defs; None fields are skipped
                            and remaining weights are re-normalized

    Row matching
    ------------
    Uses the 'Source Row' column (added by test_generator and test_runner) to
    look up the corresponding intake row.  Falls back to intake.df.iloc[0] if
    the column is absent or the index is not found (safe for single-row intake).
    """
    if results_df.empty:
        return results_df.copy()

    has_source_row = "Source Row" in results_df.columns
    output_defs    = intake.output_defs
    records        = []

    for _, row in results_df.iterrows():
        record = row.to_dict()

        # ── Resolve the intake row for ground truth ───────────────────────────
        if has_source_row and pd.notna(row.get("Source Row")):
            try:
                intake_row = intake.df.loc[int(row["Source Row"])]
            except KeyError:
                intake_row = intake.df.iloc[0]
        else:
            intake_row = intake.df.iloc[0]

        run_ok = str(row.get("Run Status", "ok")) == "ok"

        # ── Score each output ─────────────────────────────────────────────────
        for od in output_defs:
            agent_key = od["agent_key"]
            out_type  = od["type"]
            pred      = row.get(f"Pred: {agent_key}")
            gt        = _extract_gt(intake_row, od)

            if not run_ok:
                sim = None
            elif out_type == "categorical":
                sim = _exact_match(pred, gt)
            elif out_type == "multi_categorical":
                sim = _jaccard(pred, gt)
            elif out_type == "text":
                sim = _cosine_text(str(pred or ""), str(gt), embed_fn)
            elif out_type == "numeric":
                sim = _numeric_sim(pred, gt)
            else:
                sim = None

            record[f"GT: {agent_key}"]  = gt
            record[f"Sim: {agent_key}"] = sim

        # ── Weighted aggregate: Sim: outputs ──────────────────────────────────
        valid = [
            (od["weight"], record[f"Sim: {od['agent_key']}"])
            for od in output_defs
            if record.get(f"Sim: {od['agent_key']}") is not None
        ]
        if valid:
            total_w = sum(w for w, _ in valid) or 1.0
            record["Sim: outputs"] = round(
                sum(w * s for w, s in valid) / total_w, 4
            )
        else:
            record["Sim: outputs"] = None

        records.append(record)

    return pd.DataFrame(records)
