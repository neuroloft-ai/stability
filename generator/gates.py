"""
Content-integrity gates for generated test cases.

All 6 meaning-preservation families return a consistent 3-tuple:
    (passed: bool, reason: str, nli_result: dict | None)

All 6 accept optional NLI parameters for meaning-preservation checking:
    nli_gate          : NLIGate instance (None = skip NLI)
    min_entailment    : reject if entailment < this  (0.0 = disabled)
    max_contradiction : reject if contradiction > this (1.0 = disabled)

NLI uses direct pair comparison (original → candidate) via check_meaning_preserved().

FORMAT gates:      Gate 1 TPS=1.0 | Gate 2 numbers | Gate 3 whitespace-only | Gate 4 NLI
PARAPHRASE gates:  Gate 1 numbers | Gate 2 length ratio | Gate 3 sentence delta | Gate 4 goal type | Gate 5 NLI
DISTRACTOR gates:  Gate 1 numbers subset | Gate 2 goal type | Gate 3 sent increased | Gate 4 NLI
NOISE gates:       Gate 1 numbers PSV=0 | Gate 2 word count | Gate 3 NLI
CONFLICT gates:    Gate 1 numbers subset | Gate 2 content subset | Gate 3 cc>=1 | Gate 4 NLI
CONTEXT gates:     Gate 1 numbers subset | Gate 2 verbatim present | Gate 3 ctx_cer>1 | Gate 4 NLI

RI Mode A/B gates return 2-tuple and use LLM-based NLI (separate from the local model).

Usage:
    from generator.nli_gate import NLIGate

    # Without NLI (backward compatible)
    ok, reason, nli = passes_paraphrase_gates(original, candidate)

    # With NLI as hard gate
    gate = NLIGate()
    ok, reason, nli = passes_paraphrase_gates(
        original, candidate,
        nli_gate=gate, min_entailment=0.5, max_contradiction=0.3,
    )

    # With NLI + LLM Judge (both layers)
    from generator.llm_judge import LLMJudge
    judge = LLMJudge(client=groq_client, model='llama-3.3-70b-versatile')
    ok, reason, result = passes_paraphrase_gates(
        original, candidate,
        nli_gate=gate, min_entailment=0.5, max_contradiction=0.3,
        llm_judge=judge, expected='42',
        system_description='solves math word problems',
    )

    # All 6 families have the same interface:
    ok, reason, result = passes_format_gates(original, candidate, nli_gate=gate, ...)
    ok, reason, result = passes_noise_gates(original, candidate, nli_gate=gate, ...)
    ok, reason, result = passes_distractor_gates(original, candidate, nli_gate=gate, ...)
    ok, reason, result = passes_conflict_gates(original, candidate, nli_gate=gate, ...)
    ok, reason, result = passes_context_gates(original, candidate, nli_gate=gate, ...)

    # RI modes (unchanged — 2-tuple, LLM NLI)
    ok, reason = passes_ri_a_gates(source, task_instruction, nli_client=..., nli_model=...)
    ok, reason = passes_ri_b_gates(original, rewritten, nli_client=..., nli_model=...)
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"   # comma-formatted  e.g. 1,000.50
    r"|-?\d+(?:\.\d+)?"                  # plain integer or decimal
)

_WS_RE = re.compile(r"\s+")

# Lexical tokens: words (with apostrophes) OR numbers — ignores punctuation,
# markdown symbols, bullets, headers, code fences, pipes, dashes, etc.
_LEX_TOK_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"          # word (with optional apostrophe)
    r"|"
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"   # comma-formatted number
    r"|-?\d+(?:\.\d+)?"                  # plain number
)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", text.strip())


def extract_numbers(text: str) -> List[str]:
    """Return list of numbers found in text, normalised (commas removed, trailing dot stripped)."""
    raw = _NUM_RE.findall(text or "")
    return [n.replace(",", "").rstrip(".") for n in raw if n]


def numbers_signature(text: str) -> Counter:
    """Multiset of normalised numbers in text."""
    return Counter(extract_numbers(text))


def lexical_tokens(text: str) -> List[str]:
    """
    Ordered list of words and numbers, ignoring all punctuation and markdown.
    Numbers are normalised (commas removed, trailing dot stripped).
    """
    raw = _LEX_TOK_RE.findall(text or "")
    cleaned: List[str] = []
    for t in raw:
        if _NUM_RE.fullmatch(t):
            t = t.replace(",", "").rstrip(".")
        cleaned.append(t)
    return cleaned


# ---------------------------------------------------------------------------
# Shared NLI hard gate check  (used by all 6 meaning-preservation families)
# ---------------------------------------------------------------------------

def _run_nli_check(
    original: str,
    candidate: str,
    nli_gate,
    min_entailment: float,
    max_contradiction: float,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
    family: Optional[str] = None,
) -> Tuple[bool, str, Optional[dict]]:
    """
    Run meaning-preservation checks (NLI and/or LLM judge) if provided.

    Layer 1 — Local NLI (fast, free, ~180 MB model):
        Runs when nli_gate is not None.
        Rejects if contradiction > max_contradiction or entailment < min_entailment.

    Layer 2 — LLM Judge (slower, higher accuracy, uses API):
        Runs when llm_judge is not None AND Layer 1 passed (or was skipped).
        For distractor family: uses judge_distractor_valid (checks original
        preserved + distractor doesn't change the answer).
        For other families: uses judge_meaning_preserved (same meaning check).
        Rejects if judge says same_meaning=False.

    Returns:
        (True,  "", result_dict)  — all checks passed or skipped
        (False, "<reason>", result_dict) — check failed (hard rejection)

    The result_dict contains both nli and judge results when available.
    """
    result: dict = {}

    # Layer 1: Local NLI
    # For distractor/context: skip local NLI — appended/wrapped content
    # always lowers entailment score, but that's expected (content was
    # added around the original, not changed).
    _SKIP_LOCAL_NLI = {"distractor", "context", "conflict", "kb"}
    if nli_gate is not None and family not in _SKIP_LOCAL_NLI:
        nli = nli_gate.check_meaning_preserved(original, candidate)
        result['nli'] = nli

        if nli['contradiction_score'] > max_contradiction:
            return False, (
                f"NLI: contradiction {nli['contradiction_score']:.3f} "
                f"> max {max_contradiction:.3f}"
            ), result

        if nli['entailment_score'] < min_entailment:
            return False, (
                f"NLI: entailment {nli['entailment_score']:.3f} "
                f"< min {min_entailment:.3f}"
            ), result

    # Layer 2: LLM Judge
    if llm_judge is not None:
        if family == "context":
            # Context Load judge: checks original preserved verbatim +
            # surrounding context (before AND after) doesn't change answer.
            verdict = llm_judge.judge_context_load_valid(
                original, candidate,
                expected=expected,
                system_description=system_description,
            )
        elif family == "distractor":
            # Distractor judge: checks original preserved +
            # appended content does not change the correct answer.
            verdict = llm_judge.judge_distractor_valid(
                original, candidate,
                expected=expected,
                system_description=system_description,
            )
        elif family == "conflict":
            # Conflict judge: checks original preserved +
            # conflict instruction doesn't embed the answer.
            verdict = llm_judge.judge_conflict_valid(
                original, candidate,
                expected=expected,
                system_description=system_description,
            )
        elif family == "kb":
            # KB judge: checks the transformation creates a valid
            # knowledge-boundary challenge (unanswerable / unknowable).
            verdict = llm_judge.judge_kb_valid(
                original, candidate,
                expected=expected,
                system_description=system_description,
            )
        else:
            verdict = llm_judge.judge_meaning_preserved(
                original, candidate,
                expected=expected,
                system_description=system_description,
            )
        result['judge'] = verdict

        if verdict.get('error') is None and verdict.get('same_meaning') is False:
            return False, (
                f"LLM Judge: meaning NOT preserved "
                f"(confidence={verdict.get('confidence', 0):.2f}, "
                f"reason={verdict.get('explanation', '')})"
            ), result

    return True, "", result if result else None


# ---------------------------------------------------------------------------
# FORMAT gates
# ---------------------------------------------------------------------------

def passes_format_gates(
    original: str,
    candidate: str,
    nli_gate=None,
    min_entailment: float = 0.0,
    max_contradiction: float = 1.0,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
) -> Tuple:
    """
    Verify a FORMAT candidate preserves all content exactly.

    Gates (in order):
      1. Token sequence identical  (TPS = 1.0)
         — no word/number added, removed, changed, or reordered
      2. Numbers multiset unchanged
         — redundant safety check (covered by Gate 1, but explicit)
      3. Only whitespace / formatting symbols changed
         — no alphabetic or digit characters were altered in the diff
      4. NLI meaning preservation (optional, when nli_gate provided)

    Returns:
      (True,  "", nli_result)  — all gates pass
      (False, "<reason>", None) — first failing gate and why
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return False, "original or candidate is not a string", None
    if not original.strip():
        return False, "original is empty", None
    if not candidate.strip():
        return False, "candidate is empty", None

    # Gate 1: token sequence must be identical (TPS = 1.0)
    orig_toks = lexical_tokens(original)
    cand_toks = lexical_tokens(candidate)
    if cand_toks != orig_toks:
        for i, (o, c) in enumerate(zip(orig_toks, cand_toks)):
            if o != c:
                return False, (
                    f"token changed at position {i}: "
                    f"expected '{o}', got '{c}'"
                ), None
        if len(cand_toks) != len(orig_toks):
            return False, (
                f"token count changed: "
                f"original={len(orig_toks)}, candidate={len(cand_toks)}"
            ), None

    # Gate 2: numbers multiset unchanged
    orig_nums = numbers_signature(original)
    cand_nums = numbers_signature(candidate)
    if cand_nums != orig_nums:
        added   = list((cand_nums - orig_nums).elements())
        removed = list((orig_nums - cand_nums).elements())
        return False, (
            f"numbers changed — "
            f"added={added if added else 'none'}, "
            f"removed={removed if removed else 'none'}"
        ), None

    # Gate 3: only whitespace / formatting symbols changed
    import difflib
    matcher = difflib.SequenceMatcher(None, original, candidate, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        changed = original[i1:i2] + candidate[j1:j2]
        for ch in changed:
            if ch.isalpha() or ch.isdigit():
                return False, (
                    f"alphanumeric character changed: {repr(ch)} — "
                    f"only whitespace and formatting symbols are allowed"
                ), None

    # Gate 4: NLI + LLM Judge meaning preservation (optional)
    return _run_nli_check(original, candidate, nli_gate, min_entailment, max_contradiction,
                          llm_judge, expected, system_description)


# ---------------------------------------------------------------------------
# Temperature schedule — Format (level-specific)
# ---------------------------------------------------------------------------

_FORMAT_BASE_TEMPS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6}


def format_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for FORMAT generation.

    Base temperatures: F1→0.10, F2→0.20, F3→0.30, F4→0.40, F5→0.60
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Lower levels stay conservative (subtle whitespace only); higher levels
    are more creative (aggressive structure). Retries nudge temperature up
    so the model explores different layouts on each attempt.
    """
    base = _FORMAT_BASE_TEMPS.get(lvl, 0.3)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# PARAPHRASE gates
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter for GSM8K-style math word problems.
    Used as a gate: sentence count check only.
    """
    text = normalize_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def passes_paraphrase_gates(
    original:  str,
    candidate: str,
    core:      Optional[dict] = None,
    nli_gate   = None,
    min_entailment: float = 0.0,
    max_contradiction: float = 1.0,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
    max_sentence_delta: Optional[int] = 0,
    max_length_ratio: float = 1.3,
) -> Tuple:
    """
    Verify a PARAPHRASE candidate preserves the required invariants.

    Gates (in order):
      1. Numbers multiset unchanged          — always checked
      2. Length ratio within bound            — always checked (default ≤1.3×)
      3. Sentence count within delta          — controlled by max_sentence_delta:
            0    = exact match (L1-L2)
            1    = ±1 allowed (L3)
            None = no check (L4-L5)
      4. Goal type preserved                 — checked if core provided
      5. NLI meaning preservation            — when nli_gate or llm_judge provided

    Returns:
        (True,  "", nli_result)    — all gates pass; nli_result is dict or None
        (False, "<reason>", None)  — first failing gate and why
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return False, "original or candidate is not a string", None
    if not original.strip():
        return False, "original is empty", None
    if not candidate.strip():
        return False, "candidate is empty", None

    # Gate 1: numbers multiset
    orig_nums = numbers_signature(original)
    cand_nums = numbers_signature(candidate)
    if cand_nums != orig_nums:
        added   = list((cand_nums - orig_nums).elements())
        removed = list((orig_nums - cand_nums).elements())
        return False, (
            f"numbers changed — "
            f"added={added if added else 'none'}, "
            f"removed={removed if removed else 'none'}"
        ), None

    # Gate 2: length ratio (prevent runaway text growth)
    orig_words = len(original.split())
    cand_words = len(candidate.split())
    if orig_words > 0:
        ratio = cand_words / orig_words
        if ratio > max_length_ratio:
            return False, (
                f"length ratio too high: {ratio:.2f} "
                f"(max {max_length_ratio}, original={orig_words}w, "
                f"candidate={cand_words}w)"
            ), None

    # Gate 3: sentence count (respects max_sentence_delta)
    if max_sentence_delta is not None:
        orig_sents = len(split_sentences(original))
        cand_sents = len(split_sentences(candidate))
        delta = abs(cand_sents - orig_sents)
        if delta > max_sentence_delta:
            return False, (
                f"sentence count delta={delta} exceeds max={max_sentence_delta}: "
                f"original={orig_sents}, candidate={cand_sents}"
            ), None

    # Gate 4: goal type  (requires core)
    if core is not None:
        from generator.core_extractor import classify_goal_type
        orig_goal = core.get('goal_type', 'find_unknown')
        if orig_goal != 'find_unknown':
            cand_goal = classify_goal_type(candidate)
            if cand_goal != orig_goal:
                return False, (
                    f"goal type changed: "
                    f"original={orig_goal}, candidate={cand_goal}"
                ), None

    # Gate 5: NLI + LLM Judge meaning preservation (optional)
    return _run_nli_check(original, candidate, nli_gate, min_entailment, max_contradiction,
                          llm_judge, expected, system_description)


# ---------------------------------------------------------------------------
# Temperature schedule — Paraphrase (level-specific)
# ---------------------------------------------------------------------------

_PARAPHRASE_BASE_TEMPS = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8}


def paraphrase_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for PARAPHRASE generation.

    Base temperatures: L1→0.10, L2→0.20, L3→0.40, L4→0.60, L5→0.80
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Higher temps at L4-L5 give the model more creative freedom for
    sentence restructuring and narrative rewrites.
    """
    base = _PARAPHRASE_BASE_TEMPS.get(lvl, 0.4)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# DISTRACTOR gates
# ---------------------------------------------------------------------------

def passes_distractor_gates(
    original:  str,
    candidate: str,
    core:      Optional[dict] = None,
    nli_gate   = None,
    min_entailment: float = 0.0,
    max_contradiction: float = 1.0,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
) -> Tuple:
    """
    Verify a DISTRACTOR candidate preserves required invariants.

    Gates (in order):
      1. Original numbers all preserved  — all original numbers still present
         (candidate may ADD new numbers; it must NOT remove any original ones)
      2. Goal type preserved             — checked if core provided
      3. Sentence count INCREASED        — candidate must have more sentences
         than original (otherwise no distractor was added)
      4. NLI meaning preservation        — when nli_gate provided

    Returns:
      (True,  "",        nli_result)  — all gates pass
      (False, "<reason>", None)       — first failing gate and why
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return False, "original or candidate is not a string", None
    if not original.strip():
        return False, "original is empty", None
    if not candidate.strip():
        return False, "candidate is empty", None

    # Gate 1: all original numbers still present in candidate
    orig_nums = numbers_signature(original)
    cand_nums = numbers_signature(candidate)
    removed   = list((orig_nums - cand_nums).elements())
    if removed:
        return False, (
            f"original numbers removed from candidate — removed={removed}"
        ), None

    # Gate 2: goal type preserved  (requires core)
    if core is not None:
        from generator.core_extractor import classify_goal_type
        orig_goal = core.get('goal_type', 'find_unknown')
        if orig_goal != 'find_unknown':
            cand_goal = classify_goal_type(candidate)
            if cand_goal != orig_goal:
                return False, (
                    f"goal type changed: "
                    f"original={orig_goal}, candidate={cand_goal}"
                ), None

    # Gate 3: sentence count must be STRICTLY INCREASED
    orig_sents = len(split_sentences(original))
    cand_sents = len(split_sentences(candidate))
    if cand_sents <= orig_sents:
        return False, (
            f"sentence count not increased: "
            f"original={orig_sents}, candidate={cand_sents}"
        ), None

    # Gate 4: NLI + LLM Judge meaning preservation (optional)
    return _run_nli_check(original, candidate, nli_gate, min_entailment, max_contradiction,
                          llm_judge, expected, system_description)


# ---------------------------------------------------------------------------
# Temperature schedule — Distractor (level-specific)
# ---------------------------------------------------------------------------

_DISTRACTOR_BASE_TEMPS = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.7}


def distractor_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for DISTRACTOR generation.

    Base temperatures: D1→0.30, D2→0.40, D3→0.50, D4→0.60, D5→0.70
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Higher base temperatures reflect the need for creative distractor content —
    the model must add plausible but irrelevant sentences without altering
    the original problem structure. Higher levels demand more confounding numbers
    and denser false leads, requiring more generation diversity.
    """
    base = _DISTRACTOR_BASE_TEMPS.get(lvl, 0.5)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# NOISE gates
# ---------------------------------------------------------------------------

# Word token pattern for Gate 2 (alphabetic only — same logic as metrics.word_tokens)
_NOISE_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def passes_noise_gates(
    original: str,
    candidate: str,
    nli_gate=None,
    min_entailment: float = 0.0,
    max_contradiction: float = 1.0,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
) -> Tuple:
    """
    Verify a NOISE candidate preserves required invariants.

    Gates (in order):
      1. Numeric token multiset unchanged  (PSV = 0)
         — all numeric values preserved exactly; no digit corruption allowed
      2. Word token count unchanged
         — no word insertions or deletions; noise only corrupts characters
      3. NLI meaning preservation (optional, when nli_gate provided)

    Returns:
      (True,  "", nli_result) — all gates pass
      (False, "<reason>", None) — first failing gate and why
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return False, "original or candidate is not a string", None
    if not original.strip():
        return False, "original is empty", None
    if not candidate.strip():
        return False, "candidate is empty", None

    # Gate 1: numeric multiset unchanged  (Protected Span Violations = 0)
    orig_nums = numbers_signature(original)
    cand_nums = numbers_signature(candidate)
    if orig_nums != cand_nums:
        added   = list((cand_nums - orig_nums).elements())
        removed = list((orig_nums - cand_nums).elements())
        return False, (
            f"numeric tokens changed (PSV > 0) — "
            f"added={added if added else 'none'}, "
            f"removed={removed if removed else 'none'}"
        ), None

    # Gate 2: word token count unchanged  (no insertions or deletions)
    orig_count = len(_NOISE_WORD_RE.findall(original))
    cand_count = len(_NOISE_WORD_RE.findall(candidate))
    if orig_count != cand_count:
        return False, (
            f"word token count changed: "
            f"original={orig_count}, candidate={cand_count}"
        ), None

    # Gate 3: NLI + LLM Judge meaning preservation (optional)
    return _run_nli_check(original, candidate, nli_gate, min_entailment, max_contradiction,
                          llm_judge, expected, system_description)


# ---------------------------------------------------------------------------
# Temperature schedule — Noise (level-specific)
# ---------------------------------------------------------------------------

_NOISE_BASE_TEMPS = {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.6, 5: 0.7}


def noise_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for NOISE generation.

    Base temperatures: N1→0.20, N2→0.30, N3→0.40, N4→0.60, N5→0.70
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Conservative bases reflect the precision required — the model must corrupt
    characters without accidentally changing words or numbers.  Higher levels
    need more aggressive corruption (higher base) but still benefit from a
    lower ceiling than distractor/paraphrase since word order must be exact.
    """
    base = _NOISE_BASE_TEMPS.get(lvl, 0.4)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# CONFLICT INSTRUCTION gates
# ---------------------------------------------------------------------------

def passes_conflict_gates(
    original: str,
    candidate: str,
    nli_gate=None,
    min_entailment: float = 0.0,
    max_contradiction: float = 1.0,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
) -> Tuple:
    """
    Verify a CONFLICT INSTRUCTION candidate preserves the original problem.

    Gates (in order):
      1. Original numbers preserved  — all original numbers still in candidate.
      2. Original problem content preserved  — all original lowercase lexical
         tokens present as a multiset subset of candidate tokens.
      3. At least one conflict instruction detected  — cc(original, candidate) >= 1.
      4. NLI meaning preservation (optional, when nli_gate provided)

    Returns:
      (True,  "", nli_result) — all gates pass
      (False, "<reason>", None) — first failing gate and why
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return False, "original or candidate is not a string", None
    if not original.strip():
        return False, "original is empty", None
    if not candidate.strip():
        return False, "candidate is empty", None

    # Gate 1: original numbers preserved in candidate (multiset subset)
    orig_nums = numbers_signature(original)
    cand_nums = numbers_signature(candidate)
    removed   = list((orig_nums - cand_nums).elements())
    if removed:
        return False, f"original numbers removed — removed={removed}", None

    # Gate 2: original problem content preserved (core logic unchanged)
    # Lowercase lexical token multiset from original must be a subset of candidate.
    from collections import Counter as _Counter
    orig_toks = _Counter(t.lower() for t in lexical_tokens(original))
    cand_toks = _Counter(t.lower() for t in lexical_tokens(candidate))
    missing   = list((orig_toks - cand_toks).elements())
    if missing:
        return False, (
            f"original problem content altered — "
            f"{len(missing)} token(s) removed/changed: {missing[:8]}"
        ), None

    # Gate 3: at least one conflict instruction present (cc >= 1)
    from generator.metrics import cc as _cc
    if _cc(original, candidate) < 1:
        return False, "no conflict instructions detected (cc = 0)", None

    # Gate 4: NLI + LLM Judge meaning preservation (optional)
    return _run_nli_check(original, candidate, nli_gate, min_entailment, max_contradiction,
                          llm_judge, expected, system_description)


# ---------------------------------------------------------------------------
# Temperature schedule — Conflict Instruction (level-specific)
# ---------------------------------------------------------------------------

_CONFLICT_BASE_TEMPS = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.7}


def conflict_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for CONFLICT INSTRUCTION generation.

    Base temperatures: CI1→0.30, CI2→0.40, CI3→0.50, CI4→0.60, CI5→0.70
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Higher base temperatures for higher levels reflect the need for more
    elaborate conflict constructions — authority markers, multi-layer hierarchies,
    and prompt-injection style instructions at CI5.
    """
    base = _CONFLICT_BASE_TEMPS.get(lvl, 0.5)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# CONTEXT-LENGTH gates
# ---------------------------------------------------------------------------

def passes_context_gates(
    original: str,
    candidate: str,
    nli_gate=None,
    min_entailment: float = 0.0,
    max_contradiction: float = 1.0,
    llm_judge=None,
    expected: Optional[str] = None,
    system_description: Optional[str] = None,
) -> Tuple:
    """
    Verify a CONTEXT-LENGTH candidate preserves the original problem intact.

    Gates (in order):
      1. Original numbers preserved  — all original numbers still present in
         candidate as a multiset subset.
      2. Original text present verbatim  — original appears as a contiguous
         substring within candidate (whitespace-normalised comparison).
      3. Candidate is longer than original  — ctx_cer > 1.0.
      4. NLI meaning preservation (optional, when nli_gate provided)

    Returns:
      (True,  "", nli_result) — all gates pass
      (False, "<reason>", None) — first failing gate and why
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return False, "original or candidate is not a string", None
    if not original.strip():
        return False, "original is empty", None
    if not candidate.strip():
        return False, "candidate is empty", None

    # Gate 1: original numbers preserved in candidate (multiset subset)
    orig_nums = numbers_signature(original)
    cand_nums = numbers_signature(candidate)
    removed   = list((orig_nums - cand_nums).elements())
    if removed:
        return False, f"original numbers removed — removed={removed}", None

    # Gate 2: original text present verbatim (substring match after normalising whitespace)
    orig_norm = re.sub(r'\s+', ' ', original.strip())
    cand_norm = re.sub(r'\s+', ' ', candidate.strip())
    if orig_norm not in cand_norm:
        return False, (
            "original problem text not found verbatim in candidate — "
            "the core question must appear unchanged inside the extended context"
        ), None

    # Gate 3: candidate is longer than original (irrelevant context was added)
    from generator.metrics import ctx_cer as _ctx_cer
    ratio = _ctx_cer(original, candidate)
    if ratio <= 1.0:
        return False, f"candidate is not longer than original (ctx_cer={ratio:.2f} ≤ 1.0)", None

    # Gate 4: NLI + LLM Judge meaning preservation (optional)
    return _run_nli_check(original, candidate, nli_gate, min_entailment, max_contradiction,
                          llm_judge, expected, system_description)


# ---------------------------------------------------------------------------
# Temperature schedule — Context-Length (level-specific)
# ---------------------------------------------------------------------------

_CONTEXT_BASE_TEMPS = {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.7, 5: 0.8}


def context_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for CONTEXT-LENGTH generation.

    Base temperatures: CL1→0.40, CL2→0.50, CL3→0.60, CL4→0.70, CL5→0.80
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Higher temperatures for higher levels reflect the need for more
    diverse, creative narrative content — short filler at CL1 can be
    repetitive, but CL4/CL5 requires sustained varied text across many paragraphs.
    """
    base = _CONTEXT_BASE_TEMPS.get(lvl, 0.6)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# Representation Integrity — shared LLM NLI helper
# ---------------------------------------------------------------------------

_RI_NLI_SYSTEM = (
    "You are a textual entailment classifier.\n"
    "Given a Premise and a Hypothesis, classify their relationship.\n"
    "Respond with ONLY one word from: entailment, neutral, contradiction.\n\n"
    "Definitions:\n"
    "  entailment    — the hypothesis is a paraphrase or restatement of the premise\n"
    "  neutral       — the hypothesis adds or omits information but does not contradict\n"
    "  contradiction — the hypothesis contradicts or significantly changes the premise"
)


def _llm_nli(client, model: str, premise: str, hypothesis: str) -> str:
    """
    Call an LLM to classify NLI relationship between premise and hypothesis.
    Returns 'entailment', 'neutral', 'contradiction', or 'unknown' on error.
    Used by both RI Mode A and Mode B gate functions.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RI_NLI_SYSTEM},
                {"role": "user",   "content": f"Premise: {premise}\nHypothesis: {hypothesis}"},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        return next(
            (label for label in ("entailment", "neutral", "contradiction") if label in raw),
            "unknown",
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Representation Integrity — Mode A gates  (task instruction generation)
# ---------------------------------------------------------------------------

_RI_ACTION_VERBS = {
    "extract", "list", "identify", "find", "locate",
    "restate", "rewrite", "paraphrase", "rephrase",
    "summarize", "summarise", "condense", "compress",
    "combine", "compare", "synthesize", "synthesise",
    "relate", "describe", "explain", "abstract",
    "interpret", "provide", "give", "state",
}


def passes_ri_a_gates(
    source: str,
    task_instruction: str,
    nli_client=None,
    nli_model: str = "gpt-4o-mini",
    task_min_len: int = 30,
) -> Tuple[bool, str]:
    """
    Validate an RI Mode A task instruction candidate.

    Gate 1a: len(task_instruction) >= task_min_len (default 30 chars)
    Gate 1b: at least one recognised action verb present
    Gate 2:  LLM NLI — task instruction must NOT contradict the source
             (skipped when nli_client is None)

    Returns (True, '') on pass; (False, reason) on fail.
    """
    if not task_instruction.strip():
        return False, "empty task instruction"

    # Gate 1a: minimum length
    if len(task_instruction) < task_min_len:
        return False, f"task too short: {len(task_instruction)} < {task_min_len} chars"

    # Gate 1b: action verb
    words = set(task_instruction.lower().split())
    if not (words & _RI_ACTION_VERBS):
        return False, "no recognised action verb"

    # Gate 2: LLM NLI (only when client provided)
    if nli_client is not None:
        label = _llm_nli(nli_client, nli_model, source, task_instruction)
        if label == "contradiction":
            return False, f"NLI: task contradicts source (label={label})"

    return True, ""


# ---------------------------------------------------------------------------
# Temperature schedule — RI Mode A  (level-specific)
# ---------------------------------------------------------------------------

_RI_A_BASE_TEMPS = {1: 0.30, 2: 0.40, 3: 0.50, 4: 0.60, 5: 0.70}


def ri_a_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for RI Mode A generation.

    Base temperatures: RI1→0.30, RI2→0.40, RI3→0.50, RI4→0.60, RI5→0.70
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Higher temperatures for higher levels reflect the need for more varied
    and creative task instructions at harder transformation levels.
    """
    base = _RI_A_BASE_TEMPS.get(lvl, 0.50)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# Representation Integrity — Mode B gates  (source enrichment generation)
# ---------------------------------------------------------------------------

def passes_ri_b_gates(
    original: str,
    rewritten: str,
    nli_client=None,
    nli_model: str = "gpt-4o-mini",
) -> Tuple[bool, str]:
    """
    Validate an RI Mode B rewritten source candidate.

    Gate 1: len(rewritten) > len(original)   — content must be added, not removed
    Gate 2: LLM NLI — rewritten source must NOT contradict the original
            (skipped when nli_client is None)

    Returns (True, '') on pass; (False, reason) on fail.
    """
    if not rewritten.strip():
        return False, "empty rewritten source"

    # Gate 1: length > original (content was added)
    if len(rewritten) <= len(original):
        return False, (
            f"rewritten not longer than original: "
            f"{len(rewritten)} <= {len(original)} chars"
        )

    # Gate 2: LLM NLI (only when client provided)
    if nli_client is not None:
        label = _llm_nli(nli_client, nli_model, original, rewritten)
        if label == "contradiction":
            return False, f"NLI: rewritten contradicts original (label={label})"

    return True, ""


# ---------------------------------------------------------------------------
# Temperature schedule — RI Mode B  (level-specific)
# ---------------------------------------------------------------------------

_RI_B_BASE_TEMPS = {1: 0.30, 2: 0.40, 3: 0.50, 4: 0.60, 5: 0.70}


def ri_b_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for RI Mode B generation.

    Base temperatures: RI1→0.30, RI2→0.40, RI3→0.50, RI4→0.60, RI5→0.70
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Higher temperatures for higher levels reflect the need for more nuanced
    and varied source enrichments (subtle facts, attributions at RI5).
    """
    base = _RI_B_BASE_TEMPS.get(lvl, 0.50)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ---------------------------------------------------------------------------
# DECISION COMPLEXITY gates
# ---------------------------------------------------------------------------
#
# DC generates self-contained new tasks (Rules + Case + Question).
# There is no original/candidate pair — all gates validate the generated text.
#
# Global validity gates (all levels):
#   Gate G1 — question present      : text contains a question mark
#   Gate G2 — rules section present : at least one rule/condition pattern detected
#   Gate G3 — decision verb present : at least one decision action verb found
#
# Level-specific structural gates (dc_struct_score banding):
#   DC1: dc_struct_score in [0.00, 0.16)  — simple 1-2 rules, 1-2 factors
#   DC2: dc_struct_score in [0.16, 0.32)  — 2-3 rules, 2-3 factors, shallow branch
#   DC3: dc_struct_score in [0.32, 0.52)  — 3-4 rules, exceptions, nested branch
#   DC4: dc_struct_score in [0.52, 0.76)  — 5-6 rules, multi-factor, deep branch
#   DC5: dc_struct_score in [0.76, 1.01)  — 7-8 rules, tradeoffs, complex tree
#
# ---------------------------------------------------------------------------

_DC_LEVEL_BANDS = {
    1: (0.00, 0.16),
    2: (0.16, 0.32),
    3: (0.32, 0.52),
    4: (0.52, 0.76),
    5: (0.76, 1.01),
}

_DC_QUESTION_RE = re.compile(r'\?')

_DC_RULE_PRESENCE_RE = re.compile(
    r'rule\s*\d|if\b.{2,50}then\b|when\b.{2,50},|criteria|condition\s*\d|requirement|must\b',
    re.IGNORECASE | re.DOTALL,
)

_DC_ACTION_VERBS = {
    'approve', 'reject', 'select', 'assign', 'choose', 'recommend', 'determine',
    'decide', 'rank', 'classify', 'evaluate', 'route', 'escalate', 'allocate',
    'eligible', 'qualify', 'accept', 'deny', 'grant', 'prioritize', 'compare',
}


def passes_dc_global_gates(text: str) -> Tuple[bool, str]:
    """
    Validate global (level-independent) DC generation constraints.

    Gate G1: question mark present — task must contain a question
    Gate G2: at least one rule/condition pattern detected
    Gate G3: at least one decision action verb present

    Returns (True, '') on pass; (False, reason) on fail.
    """
    if not text or not text.strip():
        return False, "generated text is empty"

    # Gate G1: question mark present
    if not _DC_QUESTION_RE.search(text):
        return False, "no question found in generated text (missing '?')"

    # Gate G2: rules section present
    if not _DC_RULE_PRESENCE_RE.search(text):
        return False, (
            "no rule or condition pattern detected — "
            "text must contain at least one rule/if-then/criteria/condition clause"
        )

    # Gate G3: decision action verb present
    words = set(re.findall(r'[a-z]+', text.lower()))
    if not (words & _DC_ACTION_VERBS):
        return False, (
            "no decision action verb found — "
            "text must ask to approve/reject/select/rank/route/etc."
        )

    return True, ""


def passes_dc_level_gates(text: str, level: int) -> Tuple[bool, str]:
    """
    Validate that dc_struct_score falls within the expected band for the given level.

    Uses dc_struct_score from generator.metrics.
    Level must be 1–5; anything else fails immediately.

    Returns (True, '') on pass; (False, reason) on fail.
    """
    from generator.metrics import dc_struct_score as _dc_struct_score
    band = _DC_LEVEL_BANDS.get(level)
    if band is None:
        return False, f"unknown DC level: {level}"

    score = _dc_struct_score(text)
    lo, hi = band
    if not (lo <= score < hi):
        return False, (
            f"dc_struct_score={score:.4f} outside DC{level} band [{lo:.2f}, {hi:.2f}) — "
            f"task complexity does not match target level"
        )

    return True, ""


def passes_dc_gates(text: str, level: int) -> Tuple[bool, str]:
    """
    Combined DC gate check — global validity + level-specific banding.

    Runs passes_dc_global_gates first, then passes_dc_level_gates.
    Returns (True, '') if both pass; (False, reason) at first failure.
    """
    ok, reason = passes_dc_global_gates(text)
    if not ok:
        return False, reason
    return passes_dc_level_gates(text, level)


# ---------------------------------------------------------------------------
# Temperature schedule — Decision Complexity (level-specific)
# ---------------------------------------------------------------------------

_DC_BASE_TEMPS = {1: 0.30, 2: 0.40, 3: 0.55, 4: 0.65, 5: 0.75}


def dc_temperature(lvl: int, attempt: int) -> float:
    """
    Level-specific temperature for DECISION COMPLEXITY generation.

    Base temperatures: DC1→0.30, DC2→0.40, DC3→0.55, DC4→0.65, DC5→0.75
    Retry increment  : +0.05 per attempt beyond the first, capped at 0.95

    Simple decisions (DC1-DC2) use lower temperatures for crisp, rule-based tasks.
    Complex decisions (DC4-DC5) use higher temperatures for diverse multi-criteria
    scenarios with tradeoffs and deep branching.
    """
    base = _DC_BASE_TEMPS.get(lvl, 0.50)
    return min(0.95, round(base + 0.05 * max(0, attempt - 1), 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Decision Complexity: Definition (DCDef) — level banding via dcdef_score
# ═══════════════════════════════════════════════════════════════════════════════

_DCDEF_LEVEL_BANDS = {
    1: (0.00, 0.16),
    2: (0.16, 0.32),
    3: (0.32, 0.52),
    4: (0.52, 0.76),
    5: (0.76, 1.01),
}


def passes_dcdef_level_gates(text: str, level: int) -> Tuple[bool, str]:
    """
    Validate that dcdef_score falls within the expected band for the given level.

    Uses dcdef_score from generator.metrics (9-feature composite).
    Level must be 1–5; anything else fails immediately.

    Returns (True, '') on pass; (False, reason) on fail.
    """
    from generator.metrics import dcdef_score as _dcdef_score
    band = _DCDEF_LEVEL_BANDS.get(level)
    if band is None:
        return False, f"unknown DCDef level: {level}"

    score = _dcdef_score(text)
    lo, hi = band
    if not (lo <= score < hi):
        return False, (
            f"dcdef_score={score:.4f} outside DCDef{level} band [{lo:.2f}, {hi:.2f}) — "
            f"task complexity does not match target level"
        )

    return True, ""


def passes_dcdef_gates(text: str, level: int) -> Tuple[bool, str]:
    """
    Combined DCDef gate check — DC global validity + dcdef level banding.

    Runs passes_dc_global_gates first (question + rules + action verb),
    then passes_dcdef_level_gates (9-feature score banding).
    Returns (True, '') if both pass; (False, reason) at first failure.
    """
    ok, reason = passes_dc_global_gates(text)
    if not ok:
        return False, reason
    return passes_dcdef_level_gates(text, level)
