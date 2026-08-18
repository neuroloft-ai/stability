"""
test_generator.py — Pipeline module 2: Stress test generation.

Public functions
----------------
get_available_tests()          → list of test names from registry
get_test_specs()               → DataFrame of all levels, parameters, metrics
generate_tests(...)            → DataFrame of generated stress variants

Gate layers (modular, reusable across families)
-----------------------------------------------
gate_structural()   Layer 1 — deterministic validator_rules checks  (no API)
gate_level()        Layer 2 — TCR must fall within target level band (no API)
gate_nli()          Layer 3 — LLaMA textual entailment check         (1 Groq call)
compute_temperature()          — parametric temp with retry escalation
"""

from __future__ import annotations

import re
import json as _json
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

import random as _random
from generator.level_specs import build_validity_block
from generator.gates import _run_nli_check as _meaning_check
from generator.core_extractor import extract_core as _extract_core

# Families that should preserve meaning (prompt enhancement + meaning gate)
_MEANING_FAMILIES = {"paraphrase", "noise", "distractor", "conflict", "context", "kb"}

_MEANING_SYSTEM_SUFFIX = (
    "\n- Despite the requested changes, the overall meaning of the text must "
    "remain unchanged. A reader must understand the exact same content "
    "and reach the exact same conclusion."
)


def _build_anchors_block(core: dict) -> str:
    """Build a semantic anchors block from extract_core() output.

    Injected into paraphrase user templates to prevent logic drift.
    """
    if not core:
        return ""
    parts = ["SEMANTIC ANCHORS (must be preserved exactly):"]
    nums = core.get("numbers")
    if nums:
        parts.append(f"- Numbers: {', '.join(sorted(nums))}")
    units = core.get("units")
    if units:
        parts.append(f"- Units: {', '.join(sorted(units))}")
    goal = core.get("goal_type")
    if goal and goal != "find_unknown":
        parts.append(f"- Goal: {goal}")
    claims = core.get("core_claims")
    if claims:
        parts.append("- Key facts:")
        for c in claims:
            parts.append(f"  * {c}")
    return "\n".join(parts) + "\n"


# ── Registry map ─────────────────────────────────────────────────────────────
_REGISTRY = {
    "Semantic Variation":          ("registry.seed_paraphrase",  "PARAPHRASE_STRESS_PACKAGE",             "paraphrase"),
    "Input Quality":              ("registry.seed_noise",       "INPUT_QUALITY_PACKAGE",                 "noise"),
    "Structural/Format":       ("registry.seed_format",      "FORMAT_STRESS_PACKAGE",                 "format"),
    "Context Interference":       ("registry.seed_distraction", "CONTEXT_INTERFERENCE_PACKAGE",          "distractor"),
    "Context Load":               ("registry.seed_context",     "CONTEXT_LOAD_PACKAGE",                  "context"),
    "Conflict Instruction Stress":("registry.seed_conflict",    "CONFLICT_INSTRUCTION_STRESS_PACKAGE",   "conflict"),
    "Knowledge Boundary":         ("registry.seed_kb",          "KNOWLEDGE_BOUNDARY_PACKAGE",            "kb"),
}

_PHASE = {"L1": "Minimal", "L2": "Light", "L3": "Moderate", "L4": "Heavy", "L5": "Extreme"}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_packages() -> dict:
    result = {}
    for name, (mod_path, attr, family) in _REGISTRY.items():
        mod = __import__(mod_path, fromlist=[attr])
        result[name] = {"pkg": getattr(mod, attr), "family": family}
    return result


def _tcr(orig: str, gen: str) -> float:
    """Text Change Rate: 1 - SequenceMatcher ratio (character-level). Equals 1 - SIM (same
    SequenceMatcher as _sim), so the magnitude score [(1-SIM)+TCR]/2 reduces to 1 - SIM.
    This is character-level, NOT token Jaccard. Range [0, 1]."""
    if not orig and not gen:
        return 0.0
    return round(1.0 - SequenceMatcher(None, orig.lower(), gen.lower()).ratio(), 4)


def _sim(orig: str, gen: str, embed_fn=None) -> float:
    """
    Cosine similarity between original and generated text.
    Uses embedding cosine if embed_fn provided, else lexical ratio fallback.
    """
    if embed_fn and orig and gen:
        import numpy as np
        v1 = embed_fn(orig)
        v2 = embed_fn(gen)
        return float(round(float(np.clip(float(v1 @ v2), 0.0, 1.0)), 4))
    return round(SequenceMatcher(None, orig.lower(), gen.lower()).ratio(), 4)


def _classify_level(tcr_val: float, level_criteria: dict) -> str:
    """Map a TCR value to a level key using tcr_band from registry."""
    for lk in ["L1", "L2", "L3", "L4", "L5"]:
        band = level_criteria.get(lk, {}).get("tcr_band")
        if band and band[0] <= tcr_val < band[1]:
            return lk
    # Generic fallback: 5 equal bands over [0, 1]
    bands = [(0.00, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    for i, (lo, hi) in enumerate(bands):
        if lo <= tcr_val < hi:
            return f"L{i + 1}"
    return "L5"


def _extract_numbers(text: str) -> list:
    """Extract all numeric tokens (integers, decimals, currency) from text."""
    return re.findall(r'\$?\d+(?:[.,]\d+)*', text)


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on terminal punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


# ── Gate Layer 1: Structural (deterministic, no API) ─────────────────────────

def gate_structural(orig: str, gen: str, validator_rules: dict, extra_ctx: dict = None) -> tuple:
    """
    Layer 1 — Deterministic structural checks driven by validator_rules.

    Each enabled rule is checked independently; all failures are collected.
    Returns (passed: bool, failures: list[str]).

    Supported rules
    ---------------
    preserve_numbers              — every numeric token in orig must appear in gen
    preserve_word_count           — word count must be identical
    no_word_additions_or_deletions— alias for preserve_word_count
    same_sentence_count           — sentence count must be identical
    preserve_words_exact          — every word, identical and in the same order
    preserve_word_order           — original words must appear in the same relative order
    no_word_reordering            — alias for preserve_word_order
    preserve_original_question             — full original text must appear as substring of gen (case-insensitive)
    preserve_original_question_verbatim   — full original text must appear verbatim (case-sensitive)
    no_new_numbers                — distractor (extra_ctx["distractor_part"]) must contain no numbers
                                    (applies when extra_ctx["num_new_numbers"] == 0, i.e. D1)
    no_number_reuse               — distractor numbers must not overlap with original numbers
                                    (applies when extra_ctx["num_new_numbers"] > 0, i.e. D2-D5)
    no_extra_words                — word tokens (\\w+) must be identical to original
                                    (rejects added headers/labels; ignores markdown symbols)
    max_length_ratio              — len(gen)/len(orig) must not exceed max_ratio
                                    (catches empty-line padding, content duplication, junk)

    Permissive / semantic rules (not enforced here, noted for completeness):
    allow_whitespace_changes, allow_harmless_symbols,
    allow_extra_sentences, no_new_required_computation, same_sentence_order
    """
    if extra_ctx is None:
        extra_ctx = {}
    failures = []

    # ── preserve_numbers ─────────────────────────────────────────────────────
    if validator_rules.get("preserve_numbers", {}).get("enabled"):
        orig_nums = _extract_numbers(orig)
        gen_nums  = _extract_numbers(gen)
        # Multiset check: every number in orig must appear at least as often in gen
        orig_counts: dict = {}
        for n in orig_nums:
            orig_counts[n] = orig_counts.get(n, 0) + 1
        gen_counts: dict = {}
        for n in gen_nums:
            gen_counts[n] = gen_counts.get(n, 0) + 1
        missing = [n for n, c in orig_counts.items() if gen_counts.get(n, 0) < c]
        if missing:
            failures.append(f"preserve_numbers: missing {missing}")

    # ── preserve_word_count / no_word_additions_or_deletions ─────────────────
    if (validator_rules.get("preserve_word_count", {}).get("enabled") or
            validator_rules.get("no_word_additions_or_deletions", {}).get("enabled")):
        ow = len(orig.split())
        gw = len(gen.split())
        if ow != gw:
            failures.append(f"preserve_word_count: orig={ow} gen={gw}")

    # ── same_sentence_count ───────────────────────────────────────────────────
    if validator_rules.get("same_sentence_count", {}).get("enabled"):
        os_ = _count_sentences(orig)
        gs_ = _count_sentences(gen)
        if os_ != gs_:
            failures.append(f"same_sentence_count: orig={os_} gen={gs_}")

    # ── preserve_words_exact ──────────────────────────────────────────────────
    if validator_rules.get("preserve_words_exact", {}).get("enabled"):
        if orig.lower().split() != gen.lower().split():
            failures.append("preserve_words_exact: words changed or reordered")

    # ── preserve_word_order / no_word_reordering ─────────────────────────────
    if (validator_rules.get("preserve_word_order", {}).get("enabled") or
            validator_rules.get("no_word_reordering", {}).get("enabled")):
        # Extract pure word tokens (strip punctuation) so that formatting symbols
        # like separating "solution?" into "solution" + "?" don't cause false failures
        orig_words = re.findall(r'\b\w+\b', orig.lower())
        gen_words  = re.findall(r'\b\w+\b', gen.lower())
        gen_idx = 0
        violated = False
        for w in orig_words:
            while gen_idx < len(gen_words) and gen_words[gen_idx] != w:
                gen_idx += 1
            if gen_idx >= len(gen_words):
                violated = True
                break
            gen_idx += 1
        if violated:
            failures.append("preserve_word_order: word order violated")

    # ── no_extra_words ────────────────────────────────────────────────────────
    # Strict word-token equality: same words, same count, same order.
    # Uses \w+ extraction so markdown symbols (#, -, >, |, *) are ignored.
    if validator_rules.get("no_extra_words", {}).get("enabled"):
        _orig_w = re.findall(r'\b\w+\b', orig.lower())
        _gen_w  = re.findall(r'\b\w+\b', gen.lower())
        if _gen_w != _orig_w:
            failures.append(
                f"no_extra_words: word tokens differ "
                f"(orig={len(_orig_w)}, gen={len(_gen_w)})"
            )

    # ── max_length_ratio ──────────────────────────────────────────────────────
    # Reject outputs where char length is too many times the original.
    # Catches: empty-line padding (#/#/>/> junk), content duplication, char-level splitting.
    _mlr_cfg = validator_rules.get("max_length_ratio", {})
    if _mlr_cfg.get("enabled"):
        _max_r = _mlr_cfg.get("max_ratio", 10)
        _ratio = len(gen) / max(len(orig), 1)
        if _ratio > _max_r:
            failures.append(
                f"max_length_ratio: {_ratio:.1f}x exceeds limit {_max_r}x"
            )

    # ── preserve_original_question ────────────────────────────────────────────
    if validator_rules.get("preserve_original_question", {}).get("enabled"):
        if orig.lower().strip() not in gen.lower():
            failures.append("preserve_original_question: original text not found in generated")

    # ── preserve_original_question_verbatim (context load / wrap strategy) ──────
    # For wrap strategy: check original primary field appears verbatim in wrapped doc.
    # Falls back to case-sensitive substring check on concatenated strings.
    if validator_rules.get("preserve_original_question_verbatim", {}).get("enabled"):
        orig_primary = extra_ctx.get("orig_primary", "")
        gen_primary  = extra_ctx.get("gen_primary", "")
        if orig_primary and gen_primary:
            if orig_primary.strip() not in gen_primary:
                failures.append("preserve_original_question_verbatim: original text not found verbatim in generated")
        elif orig.strip() not in gen:
            failures.append("preserve_original_question_verbatim: original text not found verbatim in generated")

    # ── no_new_numbers (D1: zero numbers allowed in distractor) ──────────────
    # Applies only when num_new_numbers == 0 (distractor must be number-free).
    if validator_rules.get("no_new_numbers", {}).get("enabled") and extra_ctx:
        if extra_ctx.get("num_new_numbers", -1) == 0:
            distractor = extra_ctx.get("distractor_part", "")
            dist_nums  = _extract_numbers(distractor)
            if dist_nums:
                failures.append(f"no_new_numbers: distractor contains numbers {dist_nums}")

    # ── no_number_reuse (D2-D5: distractor numbers must not overlap original) ─
    # Applies only when num_new_numbers > 0 (new numbers are expected but must be fresh).
    if validator_rules.get("no_number_reuse", {}).get("enabled") and extra_ctx:
        if extra_ctx.get("num_new_numbers", 0) > 0:
            orig_nums_set = set(_extract_numbers(orig))
            distractor    = extra_ctx.get("distractor_part", "")
            dist_nums     = _extract_numbers(distractor)
            reused = [n for n in dist_nums if n in orig_nums_set]
            if reused:
                failures.append(f"no_number_reuse: distractor reuses numbers {reused}")

    return len(failures) == 0, failures


# ── Delimiter-based response parser (used by Format and other plain-text families) ──

def _parse_delimited(raw: str, key_map: dict, orig_fields: dict) -> dict:
    """
    Parse plain-text response where each field is labeled on its own line.

    Expected model output format:
      [FIELD_1]
      <formatted content for field 1>
      [FIELD_2]
      <formatted content for field 2>

    Falls back to original field value if a label is not found.
    """
    keys   = list(key_map.keys())   # ["field_1", "field_2", ...]
    result = {}
    for i, generic_key in enumerate(keys):
        real_name = key_map[generic_key]
        label     = f"[{generic_key.upper()}]"
        start     = raw.find(label)
        if start == -1:
            result[real_name] = orig_fields.get(real_name, "")
            continue
        content_start = start + len(label)
        # End at next label or end of string
        next_label = f"[{keys[i + 1].upper()}]" if i + 1 < len(keys) else None
        if next_label:
            end = raw.find(next_label, content_start)
            content = raw[content_start:end].strip() if end != -1 else raw[content_start:].strip()
        else:
            content = raw[content_start:].strip()
        result[real_name] = content if content else orig_fields.get(real_name, "")
    return result


# ── Gate Layer 2: Level (TCR band) ───────────────────────────────────────────

def gate_level(tcr_val: float, target_lk: str, level_criteria: dict) -> bool:
    """
    Layer 2 — TCR must fall within the target level's tcr_band.
    Falls back to 'something changed' (TCR > 0.005) if no band is defined.
    """
    band = level_criteria.get(target_lk, {}).get("tcr_band")
    if band:
        return band[0] <= tcr_val < band[1]
    return tcr_val > 0.005


# ── Gate Layer 3: NLI — Textual Entailment via LLaMA ─────────────────────────

_NLI_SYSTEM = (
    "You are a textual entailment classifier.\n"
    "Given a Premise and a Hypothesis, classify their relationship.\n"
    "Respond with ONLY one word from: entailment, neutral, contradiction.\n\n"
    "Definitions:\n"
    "  entailment    — the hypothesis is a paraphrase / restatement of the premise\n"
    "  neutral       — the hypothesis is related but not equivalent\n"
    "  contradiction — the hypothesis contradicts or significantly changes the premise"
)


def gate_nli(
    orig: str,
    gen: str,
    llm_client,
    model: str,
    pass_labels: list,
) -> tuple:
    """
    Layer 3 — NLI via LLaMA: checks whether the generated text entails the original.

    Direction: Premise=generated, Hypothesis=original.
    We ask "does the generated text entail the original meaning?"
    A truncated or meaning-changed output will fail because it cannot entail
    the full original.

    Returns (passed: bool, label: str).
    label is one of: entailment | neutral | contradiction | error:<msg>

    pass_labels controls what counts as a pass per family:
      Paraphrase  → ["entailment"]               (meaning must be fully preserved)
      Distraction → ["neutral", "contradiction"]  (meaning must change)
    """
    user_msg = f"Premise: {gen}\nHypothesis: {orig}"
    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _NLI_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw   = (resp.choices[0].message.content or "").strip().lower()
        label = next(
            (l for l in ["entailment", "neutral", "contradiction"] if l in raw),
            raw,
        )
    except Exception as e:
        return False, f"error:{e}"

    return label in [p.lower() for p in pass_labels], label


# ── Temperature schedule ──────────────────────────────────────────────────────

def compute_temperature(base_temp: float, attempt: int, cap: float = 0.95) -> float:
    """
    Parametric temperature with retry escalation.
      attempt=1 → base_temp
      attempt=2 → base_temp + 0.10
      attempt=3 → base_temp + 0.20
      ...capped at cap (default 0.95)
    """
    return round(min(base_temp + 0.10 * (attempt - 1), cap), 2)


# ── Public functions ──────────────────────────────────────────────────────────

def get_available_tests() -> list:
    """Return list of available stress test names from the registry."""
    return list(_REGISTRY.keys())


def get_test_specs() -> pd.DataFrame:
    """
    Return a complete DataFrame with everything defined in the registry for each
    (test × level) combination. Each row = one level of one test.

    Column groups
    -------------
    Identity        : Dimension, Category, Test, Family, Level, Phase
    Generation      : Temperature, Intensity, Generation Type
    Prompts         : System Generic, User Template Generic, System Math, User Template Math
    Style           : Style Hints
    Family params   : (noise) CER Min/Max, WCR Min/Max
                      (distraction) Num Distractors, Placement, Distractor Strength, Max Extra Chars
                      (format) Pattern Targets
    Criteria        : w_sim, w_tcr, sim_min, TCR Band Min, TCR Band Max, Invariance Min
    Validator Rules : Validator Rules  (enabled rules as comma-separated string)
    Gate Config     : Gate Struct, Gate NLI Pass Labels, Gate NLI Sim Direction,
                      Gate Score Formula, Gate Temp Formula
    Metric          : Metric, Metric Type, Metric Weight, Severity,
                      UI Label, UI Description, Aggregation Rule
    """
    pkgs = _load_packages()
    rows = []

    for test_name, info in pkgs.items():
        pkg    = info["pkg"]
        family = info["family"]
        dim    = pkg["dimension"]["name"]
        cat    = pkg["categories"][0]["name"]
        td     = pkg["test_defs"][0]
        md     = pkg["metric_defs"][0]
        lc     = {x["level"]: x["criteria"] for x in md.get("level_criteria", [])}

        # ── Test-level constants (same for all 5 levels) ──────────────────────
        gen_type   = td.get("generation_type", "-")
        vr         = td.get("validator_rules", {})
        vr_str     = ", ".join(k for k, v in vr.items() if v.get("enabled"))
        gate_cfg   = td.get("gate_config", {})
        g_struct   = gate_cfg.get("structural", {})
        g_nli      = gate_cfg.get("nli_gate", {})
        g_score    = gate_cfg.get("score", {})
        g_temp_cfg = gate_cfg.get("temperature", {})

        gate_struct_str  = ", ".join(k for k, v in g_struct.items() if v)
        nli_pass_labels  = str(g_nli.get("pass_labels", "-"))
        nli_sim_dir      = g_nli.get("sim_direction", "-")
        score_formula    = g_score.get("formula", "-")
        temp_formula     = g_temp_cfg.get("formula", "-")

        # ── Metric constants ──────────────────────────────────────────────────
        m_name  = md["name"]
        m_type  = md["metric_type"]
        m_desc  = md.get("description", "-")
        m_weight= md.get("weight", "-")
        m_sev   = md.get("severity", "-")
        m_label = md.get("ui_label", "-")
        m_uidesc= md.get("ui_description", "-")
        m_agg   = md.get("aggregation_rule", "-")

        for lk, spec in td["level_specs"].items():
            crit = lc.get(lk, {})

            # TCR band split into min/max for readability
            band = crit.get("tcr_band", ["-", "-"])
            tcr_min = band[0] if isinstance(band, list) and len(band) > 0 else "-"
            tcr_max = band[1] if isinstance(band, list) and len(band) > 1 else "-"

            # Family-specific level params
            cer    = spec.get("cer_target", {})
            wcr    = spec.get("wcr_target", {})
            pt     = spec.get("pattern_targets", {})

            rows.append({
                # ── Identity ──────────────────────────────────────────────────
                "Dimension":              dim,
                "Category":               cat,
                "Test":                   test_name,
                "Family":                 family,
                "Level":                  lk,
                "Phase":                  _PHASE.get(lk, lk),

                # ── Generation ────────────────────────────────────────────────
                "Temperature":            spec.get("temperature", "-"),
                "Intensity":              spec.get("intensity", "-"),
                "Generation Type":        gen_type,

                # ── Prompts ───────────────────────────────────────────────────
                "System Generic":         spec.get("system_generic", "-"),
                "User Template Generic":  spec.get("user_template_generic", "-"),
                "System Math":            spec.get("system", "-"),
                "User Template Math":     spec.get("user_template", "-"),

                # ── Style hints ───────────────────────────────────────────────
                "Style Hints":            " | ".join(spec.get("style_hints", [])),

                # ── Family-specific params (sparse) ───────────────────────────
                "CER Min":                cer.get("min", "-"),
                "CER Max":                cer.get("max", "-"),
                "WCR Min":                wcr.get("min", "-"),
                "WCR Max":                wcr.get("max", "-"),
                "Num Distractors":        spec.get("num_distractors", "-"),
                "Placement":              spec.get("placement", "-"),
                "Distractor Strength":    spec.get("distractor_strength", "-"),
                "Max Extra Chars":        spec.get("max_extra_chars", "-"),
                "Pattern Targets":        str(pt) if pt else "-",

                # ── Scoring criteria ──────────────────────────────────────────
                "w_sim":                  crit.get("w_sim", "-"),
                "w_tcr":                  crit.get("w_tcr", "-"),
                "sim_min":                crit.get("sim_min", "-"),
                "TCR Band Min":           tcr_min,
                "TCR Band Max":           tcr_max,
                "Invariance Min":         crit.get("invariance_min", "-"),

                # ── Validator rules ───────────────────────────────────────────
                "Validator Rules":        vr_str,

                # ── Gate config ───────────────────────────────────────────────
                "Gate Struct":            gate_struct_str,
                "Gate NLI Pass Labels":   nli_pass_labels,
                "Gate NLI Sim Direction": nli_sim_dir,
                "Gate Score Formula":     score_formula,
                "Gate Temp Formula":      temp_formula,

                # ── Metric definition ─────────────────────────────────────────
                "Metric":                 m_name,
                "Metric Type":            m_type,
                "Metric Description":     m_desc,
                "Metric Weight":          m_weight,
                "Severity":               m_sev,
                "UI Label":               m_label,
                "UI Description":         m_uidesc,
                "Aggregation Rule":       m_agg,
            })

    return pd.DataFrame(rows)


def generate_tests(
    test_names,
    intake,
    max_attempts: int = 3,
    llm_client=None,
    gen_model: str = "",
    embed_fn=None,
    use_nli: bool = True,
    # ── Meaning preservation (Steps 1-3) ──────────────────────────────────
    system_description: str = "",
    local_nli_gate=None,
    min_entailment: float = 0.5,
    max_contradiction: float = 0.3,
    llm_judge=None,
) -> pd.DataFrame:
    """
    Generate stress test variants for all selected rows in intake.

    Parameters
    ----------
    test_names         : list of test names, or "all"
    intake             : DataIntakeResult (from load_data)
    max_attempts       : attempts per (row × level × test); temperature escalates each retry
    llm_client         : OpenAI-compatible client (openai.OpenAI, Groq, etc.)
    gen_model          : model name (e.g. "llama-3.3-70b-versatile")
    embed_fn           : optional — embedding function (str → np.ndarray) for cosine SIM
    use_nli            : run Layer 3 NLI gate (1 extra Groq call per attempt); default True
    system_description : what the SUT does (e.g. "classifies support tickets"); empty = skip
    local_nli_gate     : NLIGate instance for local meaning preservation check; None = skip
    min_entailment     : reject if local NLI entailment < this (default 0.5)
    max_contradiction  : reject if local NLI contradiction > this (default 0.3)
    llm_judge          : LLMJudge instance for LLM meaning preservation check; None = skip

    Returns
    -------
    DataFrame — all attempts kept regardless of gate result.
    Columns:
      Dimension, Category, Phase, Target Level, Classified Level,
      Attempt, Temperature,
      Struct Gate, Struct Failures,
      Level Gate,
      NLI Label, NLI Gate,
      Meaning Gate, Meaning Reason,   ← local NLI + LLM judge (when provided)
      Gate,          ← True only if ALL gates pass
      SIM, TCR, Score,
      Generated Input
    """
    pkgs = _load_packages()

    if test_names == "all":
        selected = list(pkgs.keys())
    else:
        if isinstance(test_names, str):
            test_names = [test_names]
        selected = [n for n in test_names if n in pkgs]
        unknown  = [n for n in test_names if n not in pkgs]
        if unknown:
            print(f"  [!] Unknown tests ignored: {unknown}")
            print(f"  [i] Available: {list(pkgs.keys())}")

    records      = []
    total_combos = len(intake.df) * len(selected) * 5 * max_attempts
    done         = 0

    for row_idx, row in intake.df.iterrows():
        fields = intake.input_fields

        # Collect original value for every input field (preserves each separately)
        orig_fields: dict = {
            f: str(row.get(f, "")).strip()
            for f in fields
            if str(row.get(f, "")).strip() not in ("", "nan", "NaN")
        }
        orig_text = " ".join(orig_fields.values())

        # Build validity block for meaning-preservation prompts
        expected_str = None
        if hasattr(intake, 'output_defs') and intake.output_defs:
            _exp_parts = []
            for od in intake.output_defs:
                col = getattr(od, 'name', None)
                if col is None and isinstance(od, dict):
                    col = od.get('name')
                if col:
                    val = str(row.get(col, '')).strip()
                    if val and val.lower() not in ('', 'nan', 'none'):
                        _exp_parts.append(f"{col}: {val}")
            if _exp_parts:
                expected_str = "\n".join(_exp_parts)
        validity_block_str = build_validity_block(
            expected=expected_str,
            system_description=system_description or None,
        )

        # Extract semantic anchors once per source row (for paraphrase family)
        _row_core = None
        _row_anchors_block = ""
        if llm_client:
            try:
                _row_core = _extract_core(orig_text, llm_client, gen_model)
                _row_anchors_block = _build_anchors_block(_row_core)
            except Exception:
                _row_core = None
                _row_anchors_block = ""

        # Generic key mapping: field_1, field_2, ... → actual field names
        # This decouples the LLM from knowing real column names
        key_map   = {f"field_{i+1}": f for i, f in enumerate(orig_fields)}   # field_1 → "subject"
        key_map_r = {f: f"field_{i+1}" for i, f in enumerate(orig_fields)}   # "subject" → field_1

        # Prompt block: generic labels (FIELD_1, FIELD_2, ...) with original values
        text_for_prompt = "\n".join(
            f"{key_map_r[f].upper()}:\n{orig_fields[f]}" for f in orig_fields
        )

        # JSON output spec using generic keys only — no real column names exposed
        json_spec = "{" + ", ".join(f'"{k}": "..."' for k in key_map) + "}"

        for test_name in selected:
            # ── Decision Complexity: measurement-only (no LLM) ────────────
            if test_name == "Decision Complexity":
                from generator.metrics import dcdef_all_features as _dcdef_af
                _result = _dcdef_af(orig_text)
                _sc     = _result["score"]
                _raw    = _result["raw"]
                _ranked = _result["ranked"]
                _clf    = _dcdef_classify(_sc)
                _top3   = ", ".join(f"{k}:{v:.3f}" for k, v in _ranked[:3])
                _feat   = " | ".join(f"{k}={_raw[k]}" for k in [
                    "num_conditions", "num_variables", "num_actions",
                    "num_constraints", "dep_tree_depth", "sentence_length",
                    "clause_count", "logical_operators", "decision_branches"])
                done += 1
                print(
                    f"  [{done:>3}/{total_combos}]  row={row_idx}  "
                    f"Decision Complexity  dcdef_score={_sc:.4f}  "
                    f"level={_clf}  top3=[{_top3}]"
                )
                records.append({
                    "Dimension": "Decision Quality", "Category": "Decision Complexity",
                    "Phase": _PHASE.get(_clf, _clf),
                    "Target Level": _clf, "Classified Level": _clf,
                    "Attempt": 1, "Temperature": 0.0,
                    "Structural Gate": True, "Struct Failures": "",
                    "NLI Gate": True, "NLI Label": "skipped",
                    "Level Gate": True, "Gate": True,
                    "SIM": 1.0, "TCR": 0.0, "Score": _sc,
                    "Source Row": row_idx,
                    "Generated Input": "\n".join(
                        f"[{f.upper()}] {v}" for f, v in orig_fields.items()),
                    "Meaning Gate": True, "Meaning Reason": "",
                    "Judge Same": "", "Judge Conf": "",
                    "Judge Expl": f"dcdef_score={_sc:.4f} | {_feat} | top3: {_top3}",
                    "Sim: outputs": _sc, "Judge Method": "dcdef_measure",
                    "Test Type": "measurement",
                    "DCDef Score": _sc,
                    "DCDef Conditions": _raw["num_conditions"],
                    "DCDef Variables": _raw["num_variables"],
                    "DCDef Actions": _raw["num_actions"],
                    "DCDef Constraints": _raw["num_constraints"],
                    "DCDef Depth": _raw["dep_tree_depth"],
                    "DCDef Sent Len": _raw["sentence_length"],
                    "DCDef Clauses": _raw["clause_count"],
                    "DCDef Logical Ops": _raw["logical_operators"],
                    "DCDef Branches": _raw["decision_branches"],
                    "DCDef Top Features": _top3,
                })
                continue  # skip level/attempt loops for this test

            info   = pkgs[test_name]
            pkg    = info["pkg"]
            dim    = pkg["dimension"]["name"]
            cat    = pkg["categories"][0]["name"]
            td     = pkg["test_defs"][0]
            md     = pkg["metric_defs"][0]
            lc     = {x["level"]: x["criteria"] for x in md.get("level_criteria", [])}

            validator_rules = td.get("validator_rules", {})
            gate_cfg        = td.get("gate_config", {})
            nli_cfg         = gate_cfg.get("nli_gate", {"pass_labels": ["entailment"]})
            # Per-family NLI toggle: seed can set nli_gate.enabled=False to skip NLI
            use_nli_here    = use_nli and nli_cfg.get("enabled", True)
            # Generation strategy: "json" (default) or "delimited" (plain-text with [FIELD_N] markers)
            gen_strategy    = td.get("generation_strategy", "json")

            for lk, spec in td["level_specs"].items():
                crit      = lc.get(lk, {})
                w_sim     = crit.get("w_sim", 0.5)
                w_tcr     = crit.get("w_tcr", 0.5)
                base_temp = spec.get("temperature", 0.5)

                # Base system prompt from spec — output format depends on strategy
                base_system = spec.get("system_generic") or spec.get("system", "")
                if gen_strategy in ("per_field", "append", "wrap", "ri_task", "ri_source", "dc", "measure"):
                    # Plain-text output — no JSON wrapping added to system prompt.
                    # ri_task / ri_source produce plain text naturally; JSON mode would
                    # cause the LLM to wrap the output in {"field_1": "..."} which
                    # breaks verb detection and source parsing.
                    system_prompt = base_system
                elif gen_strategy == "delimited":
                    delim_output_spec = "\n".join(f"[{gk.upper()}]" for gk in key_map)
                    system_prompt = (
                        f"{base_system}\n"
                        f"Output each field with its label on its own line, "
                        f"then the formatted content immediately after:\n"
                        f"{delim_output_spec}\n"
                        f"No other text or commentary."
                    )
                else:  # json
                    system_prompt = (
                        f"{base_system}\n"
                        f"Output ONLY valid JSON with exactly these keys: {json_spec}"
                    )

                user_tmpl = (
                    spec.get("user_template_generic")
                    or spec.get("user_template", "TEXT:\n{text}\n")
                )
                hints = spec.get("style_hints", [])

                # Per-level sentence delta (paraphrase: L1-L2=0, L3=1, L4-L5=None)
                _max_sent_delta = spec.get("max_sentence_delta", 0)

                # Style modes (L3+ paraphrase: textbook/conversational/concise/narrative)
                _style_modes = spec.get("style_modes")
                _style_suffix = ""
                if _style_modes:
                    _chosen_style = _random.choice(_style_modes)
                    _style_suffix = f"\nWrite in a {_chosen_style} tone."

                # Anchors block — inject for paraphrase family only
                _anchors_for_tmpl = (
                    _row_anchors_block
                    if info["family"] == "paraphrase" and _row_anchors_block
                    else ""
                )

                # Meaning preservation: enhance prompts for relevant families
                meaning_user_suffix = ""
                if info["family"] in _MEANING_FAMILIES:
                    system_prompt += _MEANING_SYSTEM_SUFFIX
                    # Skip suffix if template already has {validity_block} inline
                    if validity_block_str and "{validity_block}" not in user_tmpl:
                        meaning_user_suffix = "\n" + validity_block_str

                for attempt in range(1, max_attempts + 1):
                    done += 1
                    temp = compute_temperature(base_temp, attempt)
                    print(
                        f"  [{done:>3}/{total_combos}]  row={row_idx}  "
                        f"{test_name}  {lk}  attempt={attempt}  temp={temp} ...",
                        end=" ", flush=True,
                    )

                    # ── Meaning preservation defaults ──────────────────────
                    meaning_ok     = True
                    meaning_reason = ""
                    meaning_result = None

                    # ── Generation ────────────────────────────────────────────
                    # Rotate style hint per attempt for output variety
                    hint      = hints[(attempt - 1) % len(hints)] if hints else ""
                    gen_ok    = parse_ok = False
                    gen_fields = {}
                    extra_ctx  = {}   # populated by "append" strategy for gate checks

                    if gen_strategy == "per_field":
                        # ── Per-field: one plain-text API call per input field ──
                        # No JSON mode, no delimiters — model outputs formatted text directly.
                        #
                        # Hint injection: added to system prompt (not user message) so
                        # the model treats it as a generation instruction, not as text to echo.
                        # Controlled per-family via hint_placement in gate_config:
                        #   "system" → appended to system prompt (noise)
                        #   "user"   → appended to user message (format, default)
                        hint_placement   = gate_cfg.get("hint_placement", "user")
                        per_field_system = (
                            system_prompt + f"\nApproach for this attempt: {hint}"
                            if hint and hint_placement == "system" else system_prompt
                        )
                        try:
                            for generic_key, real_name in key_map.items():
                                field_val = orig_fields.get(real_name, "")
                                if not field_val:
                                    gen_fields[real_name] = field_val
                                    continue
                                field_user_msg = user_tmpl.format(
                                    text=field_val,
                                    question=field_val,
                                    anchors_block=_anchors_for_tmpl,
                                    validity_block=validity_block_str,
                                )
                                field_user_msg += meaning_user_suffix
                                field_user_msg += _style_suffix
                                if hint and hint_placement == "user":
                                    field_user_msg += f"\nStyle guidance: {hint}"
                                resp      = llm_client.chat.completions.create(
                                    model=gen_model,
                                    messages=[
                                        {"role": "system", "content": per_field_system},
                                        {"role": "user",   "content": field_user_msg},
                                    ],
                                    temperature=temp,
                                    max_tokens=spec.get("max_tokens", 1000),
                                )
                                field_out = (resp.choices[0].message.content or "").strip()
                                # Guard against model returning empty or "None"
                                gen_fields[real_name] = (
                                    field_out
                                    if field_out and field_out.lower() != "none"
                                    else field_val
                                )
                            gen_text       = " ".join(gen_fields.values())
                            gen_ok = parse_ok = True
                        except Exception as e:
                            print(f"FAIL ({e})")

                    elif gen_strategy == "append":
                        # ── Append: LLM generates ONLY the distractor sentence ─
                        # Original text is always preserved; distractor is appended.
                        # Provides {text}/{question} and {protected_numbers} to template.
                        try:
                            orig_nums     = _extract_numbers(orig_text)
                            protected_str = ", ".join(orig_nums) if orig_nums else "none"
                            field_user_msg = user_tmpl.format(
                                text=orig_text,
                                question=orig_text,          # math-specific alias
                                protected_numbers=protected_str,
                            )
                            field_user_msg += meaning_user_suffix
                            if hint:
                                field_user_msg += f"\nStyle guidance: {hint}"
                            resp = llm_client.chat.completions.create(
                                model=gen_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": field_user_msg},
                                ],
                                temperature=temp,
                                max_tokens=200,   # distractor is 1-2 sentences
                            )
                            distractor = (resp.choices[0].message.content or "").strip()
                            if not distractor or distractor.lower() == "none":
                                print("FAIL (empty distractor)")
                            else:
                                # Append distractor to last input field
                                last_field = list(orig_fields.keys())[-1]
                                gen_fields  = dict(orig_fields)
                                gen_fields[last_field] = orig_fields[last_field] + "\n" + distractor
                                gen_text    = " ".join(gen_fields.values())
                                gen_ok      = parse_ok = True
                                extra_ctx   = {
                                    "distractor_part":  distractor,
                                    "num_new_numbers":  spec.get("num_new_numbers", 0),
                                }
                        except Exception as e:
                            print(f"FAIL ({e})")

                    elif gen_strategy == "wrap":
                        # ── Wrap: LLM embeds the primary (last) field verbatim ─
                        # One API call. Target word count derived from ctx_cer_target band midpoint.
                        # Provides {text}/{question}, {input_wc}, {target_wc} to template.
                        try:
                            last_field   = list(orig_fields.keys())[-1]
                            primary_orig = orig_fields[last_field]
                            input_wc     = max(1, len(primary_orig.split()))
                            cer_target   = spec.get("ctx_cer_target", {})
                            cer_min      = cer_target.get("min", 1.1)
                            cer_max      = cer_target.get("max", None)
                            # Use band midpoint; if open-ended use 1.25× the minimum
                            mult         = (cer_min + cer_max) / 2 if cer_max is not None else cer_min * 1.25
                            target_wc    = int(input_wc * mult)
                            # Token budget: proportional to target, capped at 8192
                            dyn_tokens   = min(8192, max(
                                spec.get("max_tokens", 1000),
                                int(target_wc * 1.5 * 1.3),
                            ))
                            field_user_msg = user_tmpl.format(
                                text=primary_orig,
                                question=primary_orig,
                                input_wc=input_wc,
                                target_wc=target_wc,
                            )
                            field_user_msg += meaning_user_suffix
                            if hint:
                                field_user_msg += f"\nStyle guidance: {hint}"
                            resp = llm_client.chat.completions.create(
                                model=gen_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": field_user_msg},
                                ],
                                temperature=temp,
                                max_tokens=dyn_tokens,
                            )
                            wrapped_doc = (resp.choices[0].message.content or "").strip()
                            if not wrapped_doc:
                                print("FAIL (empty document)")
                            else:
                                gen_fields = dict(orig_fields)
                                gen_fields[last_field] = wrapped_doc
                                gen_text   = " ".join(gen_fields.values())
                                gen_ok     = parse_ok = True
                                ctx_cer    = round(len(wrapped_doc.split()) / input_wc, 4)
                                extra_ctx  = {
                                    "orig_primary":   primary_orig,
                                    "gen_primary":    wrapped_doc,
                                    "ctx_cer":        ctx_cer,
                                    "ctx_cer_target": spec.get("ctx_cer_target", {}),
                                }
                        except Exception as e:
                            print(f"FAIL ({e})")

                    elif gen_strategy == "ri_task":
                        # ── RI Mode A: generate task instruction from source ───
                        # Source = original intake input fields concatenated.
                        # Generates a task instruction per level (RI1–RI5).
                        # Gate 1: instruction length >= 30 AND action verb present.
                        # Gate 2: NLI(source, task_instruction) != contradiction.
                        # Stress input = [TASK] {instruction} [SOURCE] {source}.
                        _ACTION_VERBS = {
                            "extract", "list", "identify", "find", "locate",
                            "restate", "rewrite", "paraphrase", "rephrase",
                            "summarize", "summarise", "condense", "compress",
                            "combine", "compare", "synthesize", "synthesise",
                            "relate", "describe", "explain", "abstract",
                            "interpret", "provide", "give", "state",
                        }
                        try:
                            field_user_msg = user_tmpl.format(text=orig_text, question=orig_text)
                            field_user_msg += meaning_user_suffix
                            if hint:
                                field_user_msg += f"\nStyle guidance: {hint}"
                            resp = llm_client.chat.completions.create(
                                model=gen_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": field_user_msg},
                                ],
                                temperature=temp,
                                max_tokens=200,
                            )
                            task_instruction = (resp.choices[0].message.content or "").strip()
                            if not task_instruction or task_instruction.lower() == "none":
                                print("FAIL (empty instruction)")
                            else:
                                # Gate 1a: length
                                task_len = len(task_instruction)
                                task_len_ok = task_len >= gate_cfg.get("task_min_len", 30)
                                # Gate 1b: action verb
                                instr_words = set(task_instruction.lower().split())
                                has_verb = bool(instr_words & _ACTION_VERBS)
                                # Build stress input (structured format)
                                source_block = "\n".join(
                                    f"{f.upper()}: {v}" for f, v in orig_fields.items()
                                )
                                # Build gen_fields using the actual intake field names
                                # so the runner's _parse_generated_input can parse them.
                                # Task instruction → first field
                                # Source text     → remaining fields (original values)
                                field_items = list(orig_fields.items())
                                ri_gen_fields = {}
                                for _fi, (_fname, _fval) in enumerate(field_items):
                                    if _fi == 0:
                                        ri_gen_fields[_fname] = (
                                            f"{task_instruction}\n\nSource:\n{_fval}"
                                        )
                                    else:
                                        ri_gen_fields[_fname] = _fval
                                gen_fields = ri_gen_fields
                                gen_text   = "\n".join(
                                    f"[{f.upper()}] {v}" for f, v in gen_fields.items()
                                )
                                gen_ok     = parse_ok = True
                                extra_ctx  = {
                                    "task_instruction": task_instruction,
                                    "task_len":         task_len,
                                    "task_len_ok":      task_len_ok,
                                    "has_verb":         has_verb,
                                }
                        except Exception as e:
                            print(f"FAIL ({e})")

                    elif gen_strategy == "ri_source":
                        # ── RI Mode B: enrich source with additional facts ──────
                        # Task/question stays constant from intake.
                        # Source = last input field (body) of the original row.
                        # Gate 1: len(enriched) > len(original)  (len_ratio > 1.0)
                        # Gate 2: NLI(original, enriched) != contradiction
                        try:
                            field_user_msg = user_tmpl.format(
                                text=orig_text, question=orig_text
                            )
                            field_user_msg += meaning_user_suffix
                            if hint:
                                field_user_msg += f"\nStyle guidance: {hint}"
                            resp = llm_client.chat.completions.create(
                                model=gen_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": field_user_msg},
                                ],
                                temperature=temp,
                                max_tokens=spec.get("max_tokens", 800),
                            )
                            enriched = (resp.choices[0].message.content or "").strip()
                            if not enriched or enriched.lower() == "none":
                                print("FAIL (empty enriched source)")
                            else:
                                # Replace the last input field (body) with the enriched
                                # source. Subject stays constant; body gets enriched.
                                field_items   = list(orig_fields.items())
                                last_fname    = field_items[-1][0]          # "body"
                                orig_source   = orig_fields.get(last_fname, "")
                                ri_gen_fields = dict(orig_fields)
                                ri_gen_fields[last_fname] = enriched
                                gen_fields    = ri_gen_fields
                                gen_text      = " ".join(gen_fields.values())
                                len_ratio_val = round(
                                    len(enriched) / max(len(orig_source), 1), 4
                                )
                                gen_ok = parse_ok = True
                                extra_ctx = {
                                    "original_source": orig_source,
                                    "enriched_source": enriched,
                                    "len_ratio":       len_ratio_val,
                                    "len_ratio_ok":    len_ratio_val > gate_cfg.get(
                                        "len_ratio_min", 1.0
                                    ),
                                }
                        except Exception as e:
                            print(f"FAIL ({e})")

                    elif gen_strategy == "measure":
                        # ── Measure: no LLM — compute metrics on original text ─
                        gen_text   = orig_text
                        gen_fields = dict(orig_fields)
                        gen_ok = parse_ok = True
                        extra_ctx  = {}

                    elif gen_strategy == "dc":
                        # ── DC: generate a brand new decision task from scratch ─
                        # Original text provides thematic context (optional inspiration).
                        # The generated task is fully self-contained: rules + case + question.
                        # No SIM/TCR comparison is meaningful — gates use dc_struct_score.
                        context = orig_text or "general business scenario"
                        try:
                            field_user_msg = user_tmpl.format(context=context)
                            field_user_msg += meaning_user_suffix
                            if hint:
                                field_user_msg += f"\nStyle guidance: {hint}"
                            resp = llm_client.chat.completions.create(
                                model=gen_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": field_user_msg},
                                ],
                                temperature=temp,
                                max_tokens=spec.get("max_tokens", 800),
                            )
                            dc_task = (resp.choices[0].message.content or "").strip()
                            if dc_task and dc_task.lower() not in ("none", ""):
                                gen_text = dc_task
                                # Map DC task into actual intake field names so the runner's
                                # _parse_generated_input can find [FIELD] markers correctly.
                                # Put the full task text in the last (body-level) field;
                                # leave earlier fields (e.g. subject) empty.
                                _dc_items = list(orig_fields.items())
                                gen_fields = {fname: "" for fname, _ in _dc_items}
                                if _dc_items:
                                    gen_fields[_dc_items[-1][0]] = dc_task
                                gen_ok = parse_ok = True
                                extra_ctx = {"dc_task": dc_task}
                            else:
                                print("FAIL (empty DC task)")
                        except Exception as e:
                            print(f"FAIL ({e})")

                    else:
                        # ── Single-call strategies: delimited or json ──────────
                        raw_content = ""
                        try:
                            user_msg = user_tmpl.format(text=text_for_prompt)
                            if _anchors_for_tmpl:
                                user_msg += "\n" + _anchors_for_tmpl
                            user_msg += meaning_user_suffix
                            user_msg += _style_suffix
                            if hint:
                                user_msg += f"\nStyle guidance: {hint}"
                            api_kwargs = dict(
                                model=gen_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user",   "content": user_msg},
                                ],
                                temperature=temp,
                                max_tokens=700,
                            )
                            if gen_strategy == "json":
                                api_kwargs["response_format"] = {"type": "json_object"}
                            resp        = llm_client.chat.completions.create(**api_kwargs)
                            raw_content = resp.choices[0].message.content or ""
                            gen_ok      = True
                        except Exception as e:
                            print(f"FAIL ({e})")

                        if gen_ok:
                            try:
                                if gen_strategy == "delimited":
                                    gen_fields = _parse_delimited(raw_content, key_map, orig_fields)
                                else:
                                    data = _json.loads(raw_content)
                                    gen_fields = {
                                        real_name: str(data.get(generic_key, orig_fields.get(real_name, ""))).strip()
                                        for generic_key, real_name in key_map.items()
                                    }
                                gen_text = " ".join(gen_fields.values())
                                parse_ok = True
                            except Exception:
                                gen_text   = raw_content.strip()
                                gen_fields = {f: raw_content.strip() for f in orig_fields}

                    ok = gen_ok

                    if not gen_ok:
                        gen_fields = dict(orig_fields)
                        gen_text   = orig_text

                    if ok:
                        # ── Pre-gate: no-change detection ────────────────────
                        if gen_strategy != "measure" and gen_text.strip() == orig_text.strip():
                            structural_gate  = False
                            struct_failures  = ["no_change_applied"]
                            nli_gate, nli_label = True, "skipped"
                            tcr_val    = 0.0
                            level_gate = False
                            clf_lk     = _classify_level(0.0, lc)
                            sim_val    = 1.0
                            gate       = False
                            denom      = w_sim + w_tcr
                            score      = 0.0
                            print("no_change  gate=FAIL")
                            records.append({
                                "Dimension":        dim,
                                "Category":         cat,
                                "Phase":            _PHASE.get(lk, lk),
                                "Target Level":     lk,
                                "Classified Level": clf_lk,
                                "Attempt":          attempt,
                                "Temperature":      temp,
                                "Structural Gate":  structural_gate,
                                "Struct Failures":  "; ".join(struct_failures),
                                "NLI Gate":         nli_gate,
                                "NLI Label":        nli_label,
                                "Level Gate":       level_gate,
                                "Gate":             gate,
                                "SIM":              sim_val,
                                "TCR":              tcr_val,
                                "Score":            score,
                                "Source Row":       row_idx,
                                "Original Input":  orig_text,
                                "Generated Input":  "\n".join(
                                    f"[{f.upper()}] {v}" for f, v in gen_fields.items()
                                ),
                                "Meaning Gate":   True,
                                "Meaning Reason": "",
                            })
                            continue  # retry — model failed to apply any change

                        # ── Gate 1: Structural (deterministic) ────────────────
                        structural_gate, struct_failures = gate_structural(
                            orig_text, gen_text, validator_rules, extra_ctx
                        )
                        if not parse_ok:
                            struct_failures = ["json_parse_failed"] + struct_failures

                        # ── Gate 1b: Per-level sentence delta + length ratio ──
                        if info["family"] == "paraphrase" and structural_gate:
                            from generator.gates import split_sentences as _split_s
                            # Sentence delta check
                            if _max_sent_delta is not None:
                                _o_s = len(_split_s(orig_text))
                                _g_s = len(_split_s(gen_text))
                                if abs(_g_s - _o_s) > _max_sent_delta:
                                    structural_gate = False
                                    struct_failures.append(
                                        f"sentence_delta={abs(_g_s - _o_s)}"
                                        f" exceeds max={_max_sent_delta}"
                                    )
                            # Length ratio check (max 1.3×)
                            _o_w = len(orig_text.split())
                            _g_w = len(gen_text.split())
                            if _o_w > 0 and _g_w / _o_w > 1.3:
                                structural_gate = False
                                struct_failures.append(
                                    f"length_ratio={_g_w/_o_w:.2f} exceeds 1.3"
                                )

                        # ── Gate 2: NLI (semantic validity) ──────────────────
                        if use_nli_here and llm_client:
                            nli_gate, nli_label = gate_nli(
                                orig_text, gen_text,
                                llm_client, gen_model,
                                nli_cfg.get("pass_labels", ["entailment"]),
                            )
                        else:
                            nli_gate, nli_label = True, "skipped"

                        # ── Gate 2b: Meaning preservation (Local NLI + LLM Judge) ──
                        if (info["family"] in _MEANING_FAMILIES
                                and (local_nli_gate is not None or llm_judge is not None)):
                            meaning_ok, meaning_reason, meaning_result = _meaning_check(
                                orig_text, gen_text,
                                local_nli_gate, min_entailment, max_contradiction,
                                llm_judge,
                                expected=expected_str,
                                system_description=system_description,
                                family=info["family"],
                            )

                        # ── Gate 3: Level gate ───────────────────────────────
                        tcr_val = _tcr(orig_text, gen_text)
                        if info["family"] == "kb":
                            # KB: condition-based — Classified Level = Target Level
                            clf_lk     = lk
                            level_gate = tcr_val > 0.005   # something must have changed
                        elif info["family"] == "dc":
                            # DC: condition-based — gates use passes_dc_global_gates + level banding
                            from generator.gates import passes_dc_global_gates, passes_dc_level_gates
                            from generator.metrics import (
                                dc_struct_score as _dc_struct_score,
                                dc_num_rules as _dc_num_rules,
                                dc_num_factors as _dc_num_factors,
                                dc_num_exceptions as _dc_num_exceptions,
                                dc_branch_depth as _dc_branch_depth,
                                dc_has_tradeoff as _dc_has_tradeoff,
                                dc_decision_type as _dc_decision_type,
                            )
                            dc_task  = extra_ctx.get("dc_task", gen_text)
                            lvl_int  = int(lk[1:])  # "L3" → 3

                            # Override structural gate with DC global validity
                            struct_ok, struct_reason = passes_dc_global_gates(dc_task)
                            structural_gate = struct_ok
                            struct_failures = [] if struct_ok else [struct_reason]

                            # Level gate: dc_struct_score banding
                            level_ok, level_reason = passes_dc_level_gates(dc_task, lvl_int)
                            level_gate = level_ok
                            clf_lk     = lk  # condition-based: target level = classified level

                            # Store DC metrics in extra_ctx for record columns
                            extra_ctx["dc_struct_score_val"] = _dc_struct_score(dc_task)
                            extra_ctx["dc_num_rules_val"]    = _dc_num_rules(dc_task)
                            extra_ctx["dc_num_factors_val"]  = _dc_num_factors(dc_task)
                            extra_ctx["dc_num_exc_val"]      = _dc_num_exceptions(dc_task)
                            extra_ctx["dc_branch_depth_val"] = _dc_branch_depth(dc_task)
                            extra_ctx["dc_has_tradeoff_val"] = _dc_has_tradeoff(dc_task)
                            extra_ctx["dc_decision_type_val"]= _dc_decision_type(dc_task)
                        elif info["family"] == "dcdef":
                            # DCDef: measurement-only — compute 9 features, classify level
                            from generator.metrics import dcdef_all_features as _dcdef_all_features
                            dcdef_result = _dcdef_all_features(gen_text)
                            extra_ctx["dcdef_features"]  = dcdef_result
                            extra_ctx["dcdef_score_val"] = dcdef_result["score"]
                            # Classify level from score bands
                            _s = dcdef_result["score"]
                            if   _s < 0.16: clf_lk = "L1"
                            elif _s < 0.32: clf_lk = "L2"
                            elif _s < 0.52: clf_lk = "L3"
                            elif _s < 0.76: clf_lk = "L4"
                            else:           clf_lk = "L5"
                            structural_gate = True
                            struct_failures = []
                            level_gate      = True

                        elif info["family"] == "conflict":
                            # Conflict uses ces_norm for level classification
                            from generator.metrics import ces as _ces
                            _ces_val  = _ces(orig_text, gen_text)
                            _ces_norm = 0.0 if _ces_val == 0 else (_ces_val - 1) / 4
                            extra_ctx["ces_val"]  = _ces_val
                            extra_ctx["ces_norm"] = round(_ces_norm, 4)
                            # Classify using ces_norm bands
                            _ci_bands = [
                                ("L1", 0.00, 0.13),
                                ("L2", 0.13, 0.38),
                                ("L3", 0.38, 0.63),
                                ("L4", 0.63, 0.88),
                                ("L5", 0.88, 1.01),
                            ]
                            clf_lk = "L5"
                            for _lbl, _lo, _hi in _ci_bands:
                                if _lo <= _ces_norm < _hi:
                                    clf_lk = _lbl
                                    break
                            # Level gate: ces_norm falls in target level's band
                            _tgt_spec = td["level_specs"].get(lk, {})
                            _tgt_band = _tgt_spec.get("ces_norm_target", {})
                            if _tgt_band:
                                level_gate = (
                                    _tgt_band.get("min", 0.0) <= _ces_norm
                                    < _tgt_band.get("max", 1.01)
                                )
                            else:
                                level_gate = clf_lk == lk

                        elif gen_strategy == "wrap" and extra_ctx:
                            # Context load uses ctx_cer (expansion ratio) instead of TCR
                            clf_lk     = _classify_level(tcr_val, lc)
                            cer        = extra_ctx.get("ctx_cer", 0.0)
                            cer_target = extra_ctx.get("ctx_cer_target", {})
                            cer_min    = cer_target.get("min", 0.0)
                            cer_max    = cer_target.get("max", float("inf"))
                            level_gate = (cer >= cer_min) and (
                                cer < cer_max if "max" in cer_target else True
                            )
                        else:
                            clf_lk     = _classify_level(tcr_val, lc)
                            level_gate = gate_level(tcr_val, lk, lc)

                        # ── Post-classification: forbidden chars ─────────────
                        # Check against CLASSIFIED level (not target level).
                        # FORMAT family: reclassify L1→L2 (markdown is valid
                        #   formatting, just not L1-level whitespace-only).
                        # Other families: reject (original behaviour).
                        _clf_spec  = td["level_specs"].get(clf_lk, {})
                        _forbidden = _clf_spec.get("forbidden_chars", "")
                        if _forbidden and structural_gate:
                            _found = [c for c in set(_forbidden)
                                     if c in gen_text and c not in orig_text]
                            if _found:
                                if info["family"] == "format" and clf_lk == "L1":
                                    clf_lk = "L2"
                                    struct_failures.append(
                                        f"reclassified_L1->L2: format_tokens {_found}"
                                    )
                                else:
                                    structural_gate = False
                                    struct_failures.append(
                                        f"forbidden_chars: {_found}"
                                    )

                        # ── SIM + Score ───────────────────────────────────────
                        sim_val = _sim(orig_text, gen_text, embed_fn)
                        if info["family"] == "dc":
                            # DC gate = global validity AND level banding AND relevance
                            # SIM = cosine relevance of generated task to original domain
                            # (measures whether the task stayed in the same topic area)
                            dc_sim_ok = True
                            if embed_fn:
                                # Relevance to original domain — expect ≥ min_sim_relevance
                                min_sim_rel = gate_cfg.get("min_sim_relevance", 0.30)
                                dc_sim_ok   = sim_val >= min_sim_rel
                                extra_ctx["dc_sim_ok"]      = dc_sim_ok
                                extra_ctx["dc_sim_min"]     = min_sim_rel
                            gate    = structural_gate and level_gate and dc_sim_ok
                            tcr_val = 0.0   # not applicable (no transformation)
                            score   = round(extra_ctx.get("dc_struct_score_val", 0.0), 4)
                        elif info["family"] == "dcdef":
                            # DCDef: measurement-only — always pass
                            gate    = True
                            tcr_val = 0.0
                            score   = round(extra_ctx.get("dcdef_score_val", 0.0), 4)
                        else:
                            gate = structural_gate and nli_gate and meaning_ok
                        if info["family"] not in ("dc", "dcdef"):
                            # Magnitude: weighted avg of (1-SIM) and TCR. Since TCR = 1-SIM
                            # (see _tcr), this reduces to score = 1 - SIM regardless of the weights.
                            denom = w_sim + w_tcr
                            score = (
                                round(((1.0 - sim_val) * w_sim + tcr_val * w_tcr) / denom, 4)
                                if denom else 0.0
                            )

                        if gen_strategy == "wrap" and extra_ctx:
                            cer = extra_ctx.get("ctx_cer", 0.0)
                            print(
                                f"structural={'PASS' if structural_gate else 'FAIL'}  "
                                f"nli={nli_label}  "
                                f"level={'PASS' if level_gate else 'FAIL'}  "
                                f"gate={'PASS' if gate else 'FAIL'}  "
                                f"ctx_cer={cer:.2f}  tcr={tcr_val:.0%}  sim={sim_val:.0%}"
                            )
                        elif info["family"] == "dc":
                            sim_ok_str = f"  sim={sim_val:.2f}({'OK' if extra_ctx.get('dc_sim_ok', True) else 'LOW'})" if embed_fn else ""
                            print(
                                f"global={'PASS' if structural_gate else 'FAIL'}  "
                                f"level={'PASS' if level_gate else 'FAIL'}  "
                                f"gate={'PASS' if gate else 'FAIL'}  "
                                f"dc_score={extra_ctx.get('dc_struct_score_val', 0.0):.3f}  "
                                f"rules={extra_ctx.get('dc_num_rules_val', 0)}  "
                                f"factors={extra_ctx.get('dc_num_factors_val', 0)}"
                                f"{sim_ok_str}"
                            )
                        elif info["family"] == "dcdef":
                            dcdef_feat = extra_ctx.get("dcdef_features", {})
                            top3 = dcdef_feat.get("ranked", [])[:3]
                            top3_str = ", ".join(f"{k}={v:.3f}" for k, v in top3) if top3 else ""
                            print(
                                f"dcdef_score={extra_ctx.get('dcdef_score_val', 0.0):.4f}  "
                                f"level={clf_lk}  "
                                f"top3=[{top3_str}]"
                            )
                        elif info["family"] == "conflict":
                            _m_str = (
                                f"  meaning={'PASS' if meaning_ok else 'FAIL'}"
                                if (local_nli_gate or llm_judge) else ""
                            )
                            print(
                                f"structural={'PASS' if structural_gate else 'FAIL'}  "
                                f"nli={nli_label}{_m_str}  "
                                f"level={'PASS' if level_gate else 'FAIL'}  "
                                f"gate={'PASS' if gate else 'FAIL'}  "
                                f"ces={extra_ctx.get('ces_val', 0)}  "
                                f"ces_norm={extra_ctx.get('ces_norm', 0.0):.2f}  "
                                f"tcr={tcr_val:.0%}  sim={sim_val:.0%}"
                            )
                        else:
                            _m_str = (
                                f"  meaning={'PASS' if meaning_ok else 'FAIL'}"
                                if info["family"] in _MEANING_FAMILIES
                                   and (local_nli_gate or llm_judge)
                                else ""
                            )
                            print(
                                f"structural={'PASS' if structural_gate else 'FAIL'}  "
                                f"nli={nli_label}{_m_str}  "
                                f"level={'PASS' if level_gate else 'FAIL'}  "
                                f"gate={'PASS' if gate else 'FAIL'}  "
                                f"tcr={tcr_val:.0%}  sim={sim_val:.0%}"
                            )
                    else:
                        structural_gate = False
                        struct_failures = ["generation_failed"]
                        nli_gate        = False
                        nli_label       = "-"
                        level_gate      = False
                        clf_lk          = "-"
                        tcr_val = sim_val = score = 0.0
                        gate    = False
                        gen_fields      = dict(orig_fields)

                    rec = {
                        "Dimension":        dim,
                        "Category":         cat,
                        "Phase":            _PHASE.get(lk, lk),
                        "Target Level":     lk,
                        "Classified Level": clf_lk,
                        "Attempt":          attempt,
                        "Temperature":      temp,
                        "Structural Gate":  structural_gate,
                        "Struct Failures":  "; ".join(struct_failures) if struct_failures else "",
                        "NLI Gate":         nli_gate,
                        "NLI Label":        nli_label,
                        "Level Gate":       level_gate,
                        "Gate":             gate,
                        "SIM":              sim_val,
                        "TCR":              tcr_val,
                        "Score":            score,
                        "Source Row":       row_idx,
                        "Original Input":  orig_text,
                        "Generated Input":  "\n".join(
                            f"[{f.upper()}] {v}" for f, v in gen_fields.items()
                        ),
                        "Test Type":        "stress",
                    }
                    # Meaning preservation columns
                    if info["family"] in _MEANING_FAMILIES:
                        rec["Meaning Gate"]   = meaning_ok
                        rec["Meaning Reason"] = meaning_reason
                        if meaning_result:
                            _nli_r   = meaning_result.get('nli', {})
                            _judge_r = meaning_result.get('judge', {})
                            if _nli_r:
                                rec["Local NLI Ent"] = _nli_r.get('entailment_score', '')
                                rec["Local NLI Con"] = _nli_r.get('contradiction_score', '')
                            if _judge_r:
                                rec["Judge Same"]  = _judge_r.get('same_meaning', '')
                                rec["Judge Conf"]  = _judge_r.get('confidence', '')
                                rec["Judge Expl"]  = _judge_r.get('explanation', '')

                    # RI-specific extra columns
                    if info["family"] == "dc" and extra_ctx:
                        rec["DC Struct Score"]  = extra_ctx.get("dc_struct_score_val", 0.0)
                        rec["DC Num Rules"]     = extra_ctx.get("dc_num_rules_val", 0)
                        rec["DC Num Factors"]   = extra_ctx.get("dc_num_factors_val", 0)
                        rec["DC Num Exc"]       = extra_ctx.get("dc_num_exc_val", 0)
                        rec["DC Branch Depth"]  = extra_ctx.get("dc_branch_depth_val", 0)
                        rec["DC Has Tradeoff"]  = extra_ctx.get("dc_has_tradeoff_val", False)
                        rec["DC Decision Type"] = extra_ctx.get("dc_decision_type_val", "unknown")
                        # SIM = domain relevance to original input (replaces meaningless 0.0)
                        rec["DC SIM OK"]        = extra_ctx.get("dc_sim_ok", True)
                        # For DC, "Stress Input" is the generated task itself
                        rec["Stress Input"]     = extra_ctx.get("dc_task", "")
                    elif info["family"] == "dcdef" and extra_ctx:
                        dcdef_feat = extra_ctx.get("dcdef_features", {})
                        raw = dcdef_feat.get("raw", {})
                        _dcdef_sc = extra_ctx.get("dcdef_score_val", 0.0)
                        rec["DCDef Score"]        = _dcdef_sc
                        rec["DCDef Conditions"]   = raw.get("num_conditions", 0)
                        rec["DCDef Variables"]    = raw.get("num_variables", 0)
                        rec["DCDef Actions"]      = raw.get("num_actions", 0)
                        rec["DCDef Constraints"]  = raw.get("num_constraints", 0)
                        rec["DCDef Depth"]        = raw.get("dep_tree_depth", 1)
                        rec["DCDef Sent Len"]     = raw.get("sentence_length", 5.0)
                        rec["DCDef Clauses"]      = raw.get("clause_count", 1)
                        rec["DCDef Logical Ops"]  = raw.get("logical_operators", 0)
                        rec["DCDef Branches"]     = raw.get("decision_branches", 0)
                        ranked = dcdef_feat.get("ranked", [])
                        rec["DCDef Top Features"] = ", ".join(f"{k}:{v:.3f}" for k, v in ranked[:3])
                        # Standard columns (consistency with other categories)
                        rec["Meaning Gate"]   = True
                        rec["Meaning Reason"] = ""
                        rec["Judge Same"]     = ""
                        rec["Judge Conf"]     = ""
                        # Populate eval columns directly (no SUT/eval needed)
                        rec["Sim: outputs"]  = _dcdef_sc
                        rec["Judge Method"]  = "dcdef_measure"
                        rec["Judge Expl"]    = (
                            f"dcdef_score={_dcdef_sc:.4f} | "
                            + " | ".join(
                                f"{k}={raw.get(k, 0)}" for k in
                                ["num_conditions", "num_variables", "num_actions",
                                 "num_constraints", "dep_tree_depth", "sentence_length",
                                 "clause_count", "logical_operators", "decision_branches"]
                            )
                            + " | top3: "
                            + ", ".join(f"{k}={v:.3f}" for k, v in ranked[:3])
                        )
                    # Conflict-specific columns
                    if info["family"] == "conflict" and extra_ctx:
                        rec["CES"]      = extra_ctx.get("ces_val", 0)
                        rec["CES Norm"] = extra_ctx.get("ces_norm", 0.0)
                    records.append(rec)

                    # ── Early stop logic ──────────────────────────────────────
                    # Gate = structural + NLI valid; Level Gate = correct intensity
                    if not gate:
                        continue   # meaning invalid or constraint violated → retry

                    if not level_gate:
                        continue   # valid generation, wrong intensity band → retry

                    break          # Gate=True AND Level Gate=True → perfect sample

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Decision Complexity — Definition (measurement-only, no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

_DCDEF_BANDS = [(0.00, 0.16, "L1"), (0.16, 0.32, "L2"), (0.32, 0.52, "L3"),
                (0.52, 0.76, "L4"), (0.76, 1.01, "L5")]


def _dcdef_classify(score: float) -> str:
    for lo, hi, lbl in _DCDEF_BANDS:
        if lo <= score < hi:
            return lbl
    return "L5"


