"""
eval_all.py — Unified output evaluator.

Routes each row to the appropriate evaluation strategy based on its Category,
producing a consistent per-output-field scoring structure for all rows.

Standard evaluation (default for all categories):
    Compares Pred: <key> against ground truth from intake.
    Supported output types: categorical, multi_categorical, text, numeric.

Override evaluation (configured via CATEGORY_EVAL_OVERRIDES):
    method="llm_judge"   — KB behavioral compliance scoring.
                           Each output field judged against expected epistemic behavior.
                           Score in {0.0, 0.25, 0.75, 1.0}.
    method="dc_judge"    — DC decision correctness scoring.
                           Judges whether the model reaches the correct decision given
                           the task rules and case facts. Score in {0.0, 0.25, 0.75, 1.0}.

Output columns — identical structure for all rows regardless of strategy:
    GT: <agent_key>   — ground truth value, or "(behavioral)"/"(decision)" for judge rows
    Sim: <agent_key>  — score in [0, 1]; None when run failed
    Sim: outputs      — weighted average of all Sim: <key> scores
    Expected Output   — GT value(s) or source text (all paths)
    Judge Method      — "llm_judge", "cosine", "exact_match", "jaccard", etc.
    Judge Reason      — LLM explanation when method="llm_judge"; else empty

Additional columns for llm_judge rows:
    Level Boundary, Expected Output, Judge Method, Judge Reason

Additional columns for dc_judge rows:
    Expected Output (task text excerpt), Judge Method, Judge Reason

Public functions
----------------
eval_all(results_df, intake, overrides=None, embed_fn=None, client=None)
    → DataFrame
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

import pandas as pd

from modules.data_intake import DataIntakeResult
from modules.eval_output import (
    _extract_gt,
    _exact_match,
    _jaccard,
    _cosine_text,
    _numeric_sim,
)
from modules.eval_kb import (
    _resolve_level,
    _resolve_kb_question,
    _resolve_prediction,
    _call_judge,
    _pattern_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper — handles OpenAI 429 rate-limit responses
# ─────────────────────────────────────────────────────────────────────────────

def _chat_create(client: Any, max_retries: int = 6, **kwargs) -> Any:
    """
    Call client.chat.completions.create with exponential backoff on 429.
    Waits 2 → 4 → 8 → 16 → 32 → 60 s between attempts.
    Raises the last exception if all retries are exhausted.
    """
    delay = 2.0
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate_limit" in msg.lower():
                last_exc = exc
                time.sleep(min(delay, 60))
                delay *= 2
            else:
                raise
    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _judge_field(
    pred: Any,
    kb_question: str,
    expected_behavior: str,
    score_criteria: dict,
    client: Any,
    judge_model: str,
) -> tuple[float, str]:
    """
    Score one output field for an llm_judge row.

    Returns (score, method) where method is "llm_judge" or "pattern_match".
    """
    try:
        _na = bool(pd.isna(pred))
    except (TypeError, ValueError):
        _na = False
    pred_str = str(pred).strip() if (pred is not None and not _na) else ""
    if not pred_str:
        return 0.0, "pattern_match"

    score = None
    method = "pattern_match"

    if client is not None:
        score = _call_judge(
            client, judge_model,
            kb_question, pred_str,
            expected_behavior, score_criteria,
        )
        if score is not None:
            method = "llm_judge"

    if score is None:
        score = _pattern_score(pred_str, expected_behavior)

    return score, method


# ─────────────────────────────────────────────────────────────────────────────
# Stability LLM judge — for text-type output fields in standard eval
# ─────────────────────────────────────────────────────────────────────────────

_VALID_SCORES = {1.0, 0.75, 0.25, 0.0}

_STABILITY_JUDGE_SYSTEM = (
    "You are an evaluator assessing whether an AI model's response matches "
    "the expected output.\n\n"
    "Score the response on this 4-point scale:\n"
    "  1.00 — Semantically equivalent to the expected output\n"
    "  0.75 — Minor differences; core meaning preserved\n"
    "  0.25 — Noticeable differences; answer partially correct\n"
    "  0.00 — Fundamentally different from expected output or incorrect\n\n"
    'Output ONLY a JSON object with keys "score" and "reason".\n'
    '"score" must be one of: 1.0, 0.75, 0.25, 0.0\n'
    '"reason" must be a concise one-sentence explanation.\n'
    'Example: {"score": 0.75, "reason": "Core answer preserved but a minor detail was omitted."}'
)


# ─────────────────────────────────────────────────────────────────────────────
# DC judge — decision correctness for Decision Complexity category
# ─────────────────────────────────────────────────────────────────────────────

_DC_JUDGE_SYSTEM = (
    "You are a strict evaluator assessing whether an AI model's response correctly "
    "applies the given decision rules to reach the right decision.\n\n"
    "Score the response on this 4-point scale:\n"
    "  1.00 — Decision is clearly correct and fully justified by the rules\n"
    "  0.75 — Decision is correct but justification is incomplete or partially sound\n"
    "  0.25 — Partially correct — right direction but missing key rule application\n"
    "  0.00 — Wrong decision or no justification provided\n\n"
    'Output ONLY a JSON object with keys "score" and "reason".\n'
    '"score" must be one of: 1.0, 0.75, 0.25, 0.0\n'
    '"reason" must be a concise one-sentence explanation of the score.\n'
    'Example: {"score": 0.75, "reason": "Correct outcome but Rule 3 exception was not applied."}'
)


def _dc_call_judge(
    client: Any,
    judge_model: str,
    task_text: str,
    response: str,
) -> Optional[tuple[float, str]]:
    """
    Score a DC response against the task rules.
    Returns (score, reason) or None on failure.
    score is in {0.0, 0.25, 0.75, 1.0}.
    """
    try:
        completion = _chat_create(
            client,
            model=judge_model,
            messages=[
                {"role": "system", "content": _DC_JUDGE_SYSTEM},
                {"role": "user",   "content": (
                    f"DECISION TASK (rules + case):\n{task_text}\n\n"
                    f"MODEL RESPONSE:\n{response}\n\n"
                    "Score this response (output JSON only):"
                )},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        raw    = completion.choices[0].message.content.strip()
        parsed = json.loads(raw)
        score  = float(parsed.get("score", -1))
        if score not in _VALID_SCORES:
            score = min(_VALID_SCORES, key=lambda s: abs(s - score))
        reason = str(parsed.get("reason", "")).strip()
        return score, reason
    except Exception:
        return None


def _stability_judge(
    pred_str: str,
    gt_str: str,
    client: Any,
    judge_model: str,
) -> Optional[tuple[float, str]]:
    """Score a text output against the expected GT. Returns (score, reason) or None."""
    try:
        completion = _chat_create(
            client,
            model=judge_model,
            messages=[
                {"role": "system", "content": _STABILITY_JUDGE_SYSTEM},
                {"role": "user",   "content": (
                    f"Expected output:\n{gt_str}\n\n"
                    f"Model response:\n{pred_str}\n\n"
                    "Score this response (output JSON only):"
                )},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        raw    = completion.choices[0].message.content.strip()
        parsed = json.loads(raw)
        score  = float(parsed.get("score", -1))
        if score not in _VALID_SCORES:
            score = min(_VALID_SCORES, key=lambda s: abs(s - score))
        reason = str(parsed.get("reason", "")).strip()
        return score, reason
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def eval_all(
    results_df: pd.DataFrame,
    intake: DataIntakeResult,
    overrides: Optional[dict] = None,
    embed_fn: Optional[Callable] = None,
    client: Any = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Unified output evaluator — routes each row to the correct eval strategy.

    Parameters
    ----------
    results_df : DataFrame from test_runner.run_tests()
    intake     : DataIntakeResult — provides output_defs and ground-truth rows
    overrides  : dict mapping category name → eval config dict.
                 Supported config keys:
                   method            : "llm_judge"
                   expected_behavior : dict mapping level key → behavior metadata
                                       (e.g. KB_EXPECTED_BEHAVIOR from registry.seed_kb)
                   judge_model       : model name (default: "gpt-4o-mini")
                 Example:
                   {
                     "Knowledge Boundary": {
                         "method":            "llm_judge",
                         "expected_behavior": KB_EXPECTED_BEHAVIOR,
                         "judge_model":       "gpt-4o-mini",
                     }
                   }
    embed_fn   : optional str → list[float] for text similarity (standard eval)
    client     : OpenAI-compatible client used for "llm_judge" strategy

    Returns
    -------
    DataFrame — all original columns preserved, plus for every output_def:
        GT: <agent_key>   — ground truth, or "(behavioral)" for judge rows
        Sim: <agent_key>  — score in [0, 1]; None when run failed
        Sim: outputs      — weighted average of all Sim: <key> scores
    For llm_judge rows, also adds:
        KB Level, KB Boundary, KB Expected, Judge Method

    Row matching
    ------------
    Uses the 'Source Row' column to look up the corresponding intake row for
    ground truth. Falls back to intake.df.iloc[0] if absent or index not found.
    """
    if results_df.empty:
        return results_df.copy()

    overrides = overrides or {}
    has_source_row = "Source Row" in results_df.columns
    output_defs = intake.output_defs
    records = []
    _total = len(results_df)

    for _i, (_, row) in enumerate(results_df.iterrows()):
        record = row.to_dict()
        category = str(row.get("Category", "")).strip()
        override = overrides.get(category)
        run_ok = str(row.get("Run Status", "ok")) == "ok"

        # ── Resolve intake row for GT look-up (standard eval) ─────────────────
        if has_source_row and pd.notna(row.get("Source Row")):
            try:
                intake_row = intake.df.loc[int(row["Source Row"])]
            except KeyError:
                intake_row = intake.df.iloc[0]
        else:
            intake_row = intake.df.iloc[0]

        # ── LLM-judge path ─────────────────────────────────────────────────────
        if override and override.get("method") == "llm_judge":
            kb_expected_behavior = override.get("expected_behavior", {})
            judge_model = override.get("judge_model", "gpt-4o-mini")

            level = _resolve_level(row)
            level_meta = kb_expected_behavior.get(level, {}) if level else {}
            expected_behavior = level_meta.get("expected_behavior", "")
            score_criteria = level_meta.get("score_criteria", {})
            boundary_type = level_meta.get("boundary_type", "")

            record["Level Boundary"] = boundary_type
            record["Expected Output"] = expected_behavior

            if not run_ok or not expected_behavior:
                for od in output_defs:
                    record[f"GT: {od['agent_key']}"]  = "(behavioral)"
                    record[f"Sim: {od['agent_key']}"] = None
                record["Judge Method"]  = None
                record["Judge Reason"]  = None
                record["Sim: outputs"]  = None
                record["Test Failures"] = None
                records.append(record)
                continue

            kb_question = _resolve_kb_question(row)

            # ── Join all output fields into one combined prediction ────────────
            prediction = _resolve_prediction(row, intake)

            if not prediction:
                for od in output_defs:
                    record[f"GT: {od['agent_key']}"]  = "(behavioral)"
                    record[f"Sim: {od['agent_key']}"] = None
                record["Judge Method"]  = None
                record["Judge Reason"]  = None
                record["Sim: outputs"]  = None
                record["Test Failures"] = None
                records.append(record)
                continue

            # ── One judge call for the combined response ───────────────────────
            score  = None
            reason = ""
            method = "pattern_match"

            if client is not None:
                result = _call_judge(
                    client, judge_model,
                    kb_question, prediction,
                    expected_behavior, score_criteria,
                )
                if result is not None:
                    score, reason = result
                    method = "llm_judge"

            if score is None:
                score = _pattern_score(prediction, expected_behavior)

            # ── Apply single score to all output fields ────────────────────────
            for od in output_defs:
                record[f"GT: {od['agent_key']}"]  = "(behavioral)"
                record[f"Sim: {od['agent_key']}"] = score

            record["Sim: outputs"]  = score
            record["Judge Method"]  = method
            record["Judge Reason"]  = reason
            record["Test Failures"] = None

        # ── DC-judge path ──────────────────────────────────────────────────────
        elif override and override.get("method") == "dc_judge":
            judge_model = override.get("judge_model", "gpt-4o-mini")

            # The "Stress Input" column holds the generated task text (rules + case + question)
            task_text = ""
            for col in ("Stress Input", "stress_input", "Generated Input", "Input", "input"):
                val = row.get(col)
                if val and pd.notna(val) and str(val).strip():
                    task_text = str(val).strip()
                    break

            record["Expected Output"] = task_text[:300] if task_text else ""

            if not run_ok or not task_text:
                for od in output_defs:
                    record[f"GT: {od['agent_key']}"]  = "(decision)"
                    record[f"Sim: {od['agent_key']}"] = None
                record["Judge Method"]  = None
                record["Judge Reason"]  = None
                record["Sim: outputs"]  = None
                record["Test Failures"] = None
                records.append(record)
                continue

            # Combine all prediction fields into one response string
            prediction = _resolve_prediction(row, intake)

            if not prediction:
                for od in output_defs:
                    record[f"GT: {od['agent_key']}"]  = "(decision)"
                    record[f"Sim: {od['agent_key']}"] = None
                record["Judge Method"]  = None
                record["Judge Reason"]  = None
                record["Sim: outputs"]  = None
                record["Test Failures"] = None
                records.append(record)
                continue

            # One judge call for the combined response
            score  = None
            reason = ""
            method = "pattern_match"

            if client is not None:
                result = _dc_call_judge(client, judge_model, task_text, prediction)
                if result is not None:
                    score, reason = result
                    method = "llm_judge"

            # Pattern-match fallback: look for any decision word in response
            if score is None:
                _decision_words = {
                    'approve', 'reject', 'select', 'assign', 'choose', 'recommend',
                    'eligible', 'ineligible', 'qualify', 'accept', 'deny', 'grant',
                    'rank', 'tier', 'route', 'escalate',
                }
                pred_lower = prediction.lower()
                score = 0.25 if any(w in pred_lower for w in _decision_words) else 0.0

            for od in output_defs:
                record[f"GT: {od['agent_key']}"]  = "(decision)"
                record[f"Sim: {od['agent_key']}"] = score

            record["Sim: outputs"]  = score
            record["Judge Method"]  = method
            record["Judge Reason"]  = reason
            record["Test Failures"] = None

        # ── Standard eval path ─────────────────────────────────────────────────
        else:
            judge_model  = "gpt-4o-mini"
            methods_used = []
            reasons      = []
            gt_parts     = []

            for od in output_defs:
                agent_key = od["agent_key"]
                out_type  = od["type"]
                pred      = row.get(f"Pred: {agent_key}")
                gt        = _extract_gt(intake_row, od)
                gt_str    = str(gt).strip() if gt is not None else ""
                if gt_str:
                    gt_parts.append(gt_str)

                if not run_ok:
                    sim    = None
                    method = None
                    reason = None
                elif out_type == "text" and client is not None:
                    try:
                        _na = bool(pd.isna(pred))
                    except (TypeError, ValueError):
                        _na = False
                    pred_str = str(pred).strip() if (pred is not None and not _na) else ""
                    result = _stability_judge(pred_str, gt_str, client, judge_model) if pred_str and gt_str else None
                    if result is not None:
                        sim, reason = result
                        method = "llm_judge"
                    else:
                        sim    = _cosine_text(pred_str, gt_str, embed_fn)
                        method = "cosine"
                        reason = None
                elif out_type == "categorical":
                    sim    = _exact_match(pred, gt)
                    method = "exact_match"
                    reason = None
                elif out_type == "multi_categorical":
                    sim    = _jaccard(pred, gt)
                    method = "jaccard"
                    reason = None
                elif out_type == "text":
                    sim    = _cosine_text(str(pred or ""), str(gt), embed_fn)
                    method = "cosine"
                    reason = None
                elif out_type == "numeric":
                    sim    = _numeric_sim(pred, gt)
                    method = "numeric"
                    reason = None
                else:
                    sim    = None
                    method = None
                    reason = None

                record[f"GT: {agent_key}"]  = gt
                record[f"Sim: {agent_key}"] = sim
                if method:
                    methods_used.append(method)
                if reason:
                    reasons.append(reason)

            # Weighted aggregate
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

            record["Expected Output"] = " | ".join(gt_parts) if gt_parts else ""
            record["Judge Method"]    = "llm_judge" if "llm_judge" in methods_used else (methods_used[0] if methods_used else None)
            record["Judge Reason"]    = next((r for r in reasons if r), "")
            record["Test Failures"]   = None

        records.append(record)
        if progress_cb:
            progress_cb(_i + 1, _total)

    return pd.DataFrame(records)
