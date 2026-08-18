"""
Metric functions shared across all stress families.

PARAPHRASE metrics:
  jaccard_tcr(original, candidate)          → float   token-level Jaccard distance
  cosine_sim(a, b)                          → float   cosine similarity of embeddings

FORMAT metrics:
  tps(original, candidate)                  → float   Token Preservation Score (1.0 = identical)
  ccr(original, candidate)                  → float   Character Change Rate (size delta / original)
  lcs(original, candidate)                  → float   Layout Complexity Score (0–1)

DISTRACTOR metrics:
  dc(original, candidate)                   → int     Distractor Count (added sentences)
  dnd(original, candidate)                  → float   Distractor Number Density
  ale(original, candidate)                  → float   Added-content Load Estimate (dc + dnd + sli)

NOISE metrics:
  cer(original, candidate)                  → float   Character Edit Rate  (0–1, typically 0–0.30)
  wcr(original, candidate)                  → float   Word Corruption Rate (0–1, typically 0–0.50)
  sds(original, candidate)                  → float   Structural Disruption Score (0–1)
  tis(original, candidate)                  → float   Token Integrity Score (logged only, ≈ 1 − WCR)

CONFLICT INSTRUCTION metrics:
  cc(original, candidate)                   → int     Conflict Count (added conflict sentences)
  ces(original, candidate)                  → int     Conflict Explicitness Score (tier 1–5)
  ihd(original, candidate)                  → int     Instruction Hierarchy Depth (authority markers)

CONTEXT-LENGTH metrics:
  ctx_cer(original, candidate)              → float   Context Expansion Ratio (total_words/orig_words)
  cdp(original, candidate)                  → float   Core Distance Position (burial depth 0–1)
  icv(original, candidate)                  → int     Irrelevant Context Volume (words added, logged only)
  adi(original, candidate)                  → float   Attention Dilution Index (ctx_cer × cdp)

REPRESENTATION INTEGRITY metrics (Mode B — faithfulness / source enrichment):
  len_ratio(original, candidate)            → float   Rewritten-to-original length ratio (Gate 1: must be > 1.0)

EXECUTION / ACTION features (measurement + cross-cutting, applied at evaluation time):
  exec_completion(pred)                                    → float   1.0 if non-empty, else 0.0
  exec_schema_compliance(pred, output_type, valid_values)  → float   1.0 if matches expected type/values, else 0.0
  exec_completeness(populated_count, total_fields)         → float   ratio [0,1] of populated output fields
  exec_score(completion, schema, completeness)             → float   weighted composite [0,1]

SEF (future):
  structural_edit_flags(original, candidate)→ dict    SEF booleans (placeholders)

TCR formula (Jaccard distance on token sets):
  TCR = 1 - |tokens(Q0) ∩ tokens(Qi)| / |tokens(Q0) ∪ tokens(Qi)|
  0.0 = identical tokens   1.0 = no shared tokens   higher = more changed

CCR formula:
  CCR = |len(candidate) - len(original)| / len(original)
  Measures density of added/removed formatting chars (newlines, bullets, etc.)
  For FORMAT: content chars unchanged (TPS=1.0), so change ≈ added whitespace.

LCS formula:
  LCS = 0.4 × line_break_ratio + 0.3 × indent_variance + 0.3 × block_restructure
  All components in [0, 1]. Higher = more layout disruption.

CER formula:
  CER = char_edit_count / len(original)
  Edit count from difflib SequenceMatcher opcodes:
    replace → max(len_orig_span, len_cand_span)
    delete  → len_orig_span
    insert  → len_cand_span

SDS formula:
  SDS = _SDS_W_SPACING × spacing_anomaly + _SDS_W_PUNCT × punct_anomaly
  Weights are module-level constants (default 0.4 / 0.6) — change them to retune.
"""

import re
import math
import difflib
from collections import Counter
from typing import List, Dict

# ---------------------------------------------------------------------------
# Tokeniser (shared by TCR)
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(
    r"[a-z]+(?:'[a-z]+)?"   # words (with optional apostrophe)
    r"|-?\d+(?:\.\d+)?"      # numbers (plain or decimal)
)

_BLOCK_RE = re.compile(
    r'^\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|>\s|\|)'
)


def _tokenize(text: str) -> set:
    """
    Lowercase word + number tokens as a set.
    Commas inside numbers are stripped first so '1,000' == '1000'.
    """
    text = re.sub(r'(\d),(\d)', r'\1\2', (text or '').lower())
    return set(_TOK_RE.findall(text))


def _tokenize_ordered(text: str) -> List[str]:
    """Ordered list of lowercase word+number tokens (used by TPS)."""
    text = re.sub(r'(\d),(\d)', r'\1\2', (text or '').lower())
    return _TOK_RE.findall(text)


# ---------------------------------------------------------------------------
# Jaccard TCR
# ---------------------------------------------------------------------------

def jaccard_tcr(original: str, candidate: str) -> float:
    """
    Token-level Jaccard distance between original and candidate.
    Returns value in [0.0, 1.0].
    """
    a = _tokenize(original)
    b = _tokenize(candidate)
    union = a | b
    if not union:
        return 0.0
    return round(1.0 - len(a & b) / len(union), 4)


# ---------------------------------------------------------------------------
# Cosine similarity  (moved here so notebooks import from one place)
# ---------------------------------------------------------------------------

def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ---------------------------------------------------------------------------
# Structural Edit Flags  (SEF) — logged per candidate, not used as gates yet
# ---------------------------------------------------------------------------

def structural_edit_flags(original: str, candidate: str) -> Dict[str, object]:
    """
    Structural Edit Flags — boolean markers logged for auditability.
    Currently placeholders; will be implemented in a future phase.

    Planned implementation:
      sentence_reorder      — compare sentence order between original and candidate
      sentence_split_merge  — compare sentence count
      active_passive_change — detect voice shift via POS tags
      clause_restructure    — detect major syntactic restructuring
      voice_or_tense_change — detect tense or voice change
    """
    return {
        'sentence_reorder':      None,   # TODO
        'sentence_split_merge':  None,   # TODO
        'active_passive_change': None,   # TODO
        'clause_restructure':    None,   # TODO
        'voice_or_tense_change': None,   # TODO
    }


# ---------------------------------------------------------------------------
# FORMAT metrics — TPS, CCR, LCS
# ---------------------------------------------------------------------------

def tps(original: str, candidate: str) -> float:
    """
    Token Preservation Score.
    = # tokens identical in order / # tokens original

    1.0 means all tokens preserved in the same order (required for FORMAT).
    Any deviation means words or numbers were added, removed, or reordered.
    """
    orig_toks = _tokenize_ordered(original)
    cand_toks = _tokenize_ordered(candidate)
    if not orig_toks:
        return 1.0
    if orig_toks == cand_toks:
        return 1.0
    matched = sum(1 for o, c in zip(orig_toks, cand_toks) if o == c)
    return round(matched / len(orig_toks), 4)


def ccr(original: str, candidate: str) -> float:
    """
    Character Change Rate.
    = |len(candidate) - len(original)| / len(original)

    Measures the density of added/removed formatting characters.
    For FORMAT: content chars are unchanged (TPS=1.0), so change = added
    whitespace, newlines, bullets, headers, separators, etc.

    Band calibration for a ~300-char problem:
      F1 (≤ 0.03): ~9 chars added  — a few spaces or 2-3 newlines
      F2 (≤ 0.07): ~21 chars added — paragraph rechunking
      F3 (≤ 0.12): ~36 chars added — headers + bullets
      F4 (≤ 0.18): ~54 chars added — table/semi-structured
      F5 (≤ 0.25): ~75 chars added — extreme distortion
    """
    if not original:
        return 0.0
    return round(abs(len(candidate) - len(original)) / len(original), 4)


# ── LCS sub-components ───────────────────────────────────────────────────────

def _line_break_ratio(original: str, candidate: str) -> float:
    """Normalized additional line breaks. 20 added line breaks = 1.0."""
    added = max(0, candidate.count('\n') - original.count('\n'))
    return min(1.0, added / 20.0)


def _indent_variance(candidate: str) -> float:
    """Normalized variance of indentation depth. Variance of 25 = 1.0."""
    lines   = candidate.split('\n')
    indents = [len(l) - len(l.lstrip(' \t')) for l in lines if l.strip()]
    if len(indents) < 2:
        return 0.0
    mean = sum(indents) / len(indents)
    var  = sum((x - mean) ** 2 for x in indents) / len(indents)
    return min(1.0, var / 25.0)


def _block_restructure(candidate: str) -> float:
    """Fraction of non-empty lines that carry structural formatting markers."""
    lines     = candidate.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0
    markers = sum(
        1 for l in non_empty
        if _BLOCK_RE.match(l)
        or l.strip().startswith('---')
        or l.strip().startswith('===')
    )
    return min(1.0, markers / len(non_empty))


def lcs(original: str, candidate: str) -> float:
    """
    Layout Complexity Score.
    = 0.4 × line_break_ratio + 0.3 × indent_variance + 0.3 × block_restructure

    All components in [0, 1]. Higher = more layout disruption.

    Band calibration:
      F1 (< 0.20): minimal whitespace shift
      F2 [0.20, 0.40): paragraph rechunking
      F3 [0.40, 0.60): bullet/header conversion
      F4 [0.60, 0.80): tabular/semi-structured
      F5 [0.80, 1.01): extreme layout distortion
    """
    lb = _line_break_ratio(original, candidate)
    iv = _indent_variance(candidate)
    br = _block_restructure(candidate)
    return round(0.4 * lb + 0.3 * iv + 0.3 * br, 4)


# ---------------------------------------------------------------------------
# DISTRACTOR metrics — DC, DND, ALE
# ---------------------------------------------------------------------------
#
# DC  (Distractor Count)         — raw added sentence count
# DND (Distractor Number Density)— new_numbers_in_candidate / orig_number_count
# SLI (Sentence Length Increase) — avg word-length ratio increase (internal helper)
# ALE (Added-content Load Est.)  — dc + dnd + sli  (raw composite)
#
# Normalise in notebooks:
#   ale_norm = min(1.0, ale / 12.0)
#   dc_norm  = min(1.0, dc  / 6.0)
#   dnd_norm = min(1.0, dnd / 4.0)
#
# Band calibration (ale_norm):
#   D1 [0.00, 0.15): 1 short off-topic sentence, no new numbers
#   D2 [0.15, 0.30): 1-2 sentences with plausible domain numbers
#   D3 [0.30, 0.50): 2-3 sentences with confounding quantities
#   D4 [0.50, 0.70): 3-4 sentences with misleading partial-answer numbers
#   D5 [0.70, 1.01): 4-6 sentences weaving dense false leads throughout

_NUM_RE_DIST = re.compile(
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"   # comma-formatted  e.g. 1,000.50
    r"|-?\d+(?:\.\d+)?"                  # plain integer or decimal
)


def _count_numbers(text: str) -> List[str]:
    """List of normalised numbers in text (commas stripped, trailing dot removed)."""
    raw = _NUM_RE_DIST.findall(text or "")
    return [n.replace(",", "").rstrip(".") for n in raw if n]


def _split_sentences_simple(text: str) -> List[str]:
    """Simple sentence splitter — splits on [.!?] followed by whitespace."""
    text = re.sub(r'\s+', ' ', (text or '').strip())
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def dc(original: str, candidate: str) -> int:
    """
    Distractor Count.
    = max(0, sentence_count(candidate) - sentence_count(original))

    Raw integer: number of sentences added beyond the original count.
    0 = no sentences added.
    """
    return max(0, len(_split_sentences_simple(candidate)) - len(_split_sentences_simple(original)))


def dnd(original: str, candidate: str) -> float:
    """
    Distractor Number Density.
    = count_of_new_numbers_in_candidate / count_of_numbers_in_original

    New numbers = those in candidate that exceed the original multiset
    (i.e., cand_nums - orig_nums using Counter subtraction).
    Returns 0.0 if original contains no numbers.
    """
    orig_nums  = Counter(_count_numbers(original))
    cand_nums  = Counter(_count_numbers(candidate))
    added      = sum((cand_nums - orig_nums).values())
    total_orig = sum(orig_nums.values())
    if not total_orig:
        return 0.0
    return round(added / total_orig, 4)


def _sli(original: str, candidate: str) -> float:
    """
    Sentence Length Increase (internal helper).
    = max(0, (avg_word_count(cand_sents) - avg_word_count(orig_sents))
               / avg_word_count(orig_sents))

    Returns 0.0 if candidate sentences are shorter or equal on average.
    """
    orig_sents = _split_sentences_simple(original)
    cand_sents = _split_sentences_simple(candidate)
    if not orig_sents:
        return 0.0
    orig_avg = sum(len(s.split()) for s in orig_sents) / len(orig_sents)
    if not orig_avg:
        return 0.0
    cand_avg = (sum(len(s.split()) for s in cand_sents) / len(cand_sents)
                if cand_sents else 0.0)
    return max(0.0, round((cand_avg - orig_avg) / orig_avg, 4))


def ale(original: str, candidate: str) -> float:
    """
    Added-content Load Estimate.
    = dc + dnd + sli

    Raw composite (not normalised).
    Normalise in notebooks: ale_norm = min(1.0, ale / 12.0)

    Band calibration (ale_norm = ale / 12.0):
      D1 [0.00, 0.15): ale ≈ 0.0–1.8   — 1 short off-topic sentence
      D2 [0.15, 0.30): ale ≈ 1.8–3.6   — 1-2 sentences with domain numbers
      D3 [0.30, 0.50): ale ≈ 3.6–6.0   — 2-3 confounding sentences
      D4 [0.50, 0.70): ale ≈ 6.0–8.4   — 3-4 misleading partial-answer sentences
      D5 [0.70, 1.01): ale ≈ 8.4+      — 4-6 dense false-lead sentences
    """
    return round(dc(original, candidate) + dnd(original, candidate) + _sli(original, candidate), 4)


# ---------------------------------------------------------------------------
# NOISE metrics — CER, WCR, SDS, TIS
# ---------------------------------------------------------------------------
#
# CER (Character Edit Rate)      — char-level edit count / len(original)
# WCR (Word Corruption Rate)     — # words changed positionally / total words
# SDS (Structural Disruption Score)— spacing_anomaly + punct_anomaly (weighted)
# TIS (Token Integrity Score)    — # words unchanged / total words (logged only)
#
# SDS sub-component weights (tunable — change here to retune both SDS and bands):
_SDS_W_SPACING: float = 0.4   # weight for spacing anomaly component
_SDS_W_PUNCT:   float = 0.6   # weight for punctuation anomaly component

# Word token pattern for NOISE metrics (alphabetic only, no numbers/punctuation)
_WORD_TOK_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def word_tokens(text: str) -> List[str]:
    """
    Case-preserving list of alphabetic word tokens.
    Numbers and punctuation are excluded — only letters (with optional apostrophe).
    Used by WCR, TIS, and noise gate word-count check.
    """
    return _WORD_TOK_RE.findall(text or "")


def cer(original: str, candidate: str) -> float:
    """
    Character Edit Rate.
    = character-level edit count / len(original)

    Edit count from difflib.SequenceMatcher opcodes:
      replace → max(len_orig_span, len_cand_span)
      delete  → len_orig_span
      insert  → len_cand_span

    Denominator is len(original), consistent with CCR.
    Gate 1 in passes_noise_gates() ensures numeric tokens are not edited;
    this metric counts all edits without excluding protected spans.

    Band calibration (for ~200-char problem):
      N1 [0.00, 0.02): ≤ 4 char edits — 1 minor typo
      N2 [0.02, 0.05): 4–10 edits     — 1 typo per sentence
      N3 [0.05, 0.10): 10–20 edits    — multiple typos, spacing
      N4 [0.10, 0.18): 20–36 edits    — OCR-like errors throughout
      N5 [0.18, 0.30): 36–60 edits    — heavy corruption
    """
    if not original:
        return 0.0
    if original == candidate:
        return 0.0
    matcher = difflib.SequenceMatcher(None, original, candidate, autojunk=False)
    edits = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            edits += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            edits += i2 - i1
        elif tag == 'insert':
            edits += j2 - j1
    return round(edits / len(original), 4)


def wcr(original: str, candidate: str) -> float:
    """
    Word Corruption Rate.
    = # word tokens that differ (positional) / total word tokens in original

    Case-sensitive: "John" → "MARY" counts as changed.
    Position-matched via zip (truncates to shorter list; Gate 2 ensures equal length).

    Band calibration:
      N1 [0.00, 0.05): ≤ 5% of words corrupted
      N2 [0.05, 0.10): 5–10% corrupted — 1 typo per sentence
      N3 [0.10, 0.20): 10–20% corrupted — multiple typos
      N4 [0.20, 0.35): 20–35% corrupted — heavy corruption
      N5 [0.35, 0.50): 35–50% corrupted — extreme noise
    """
    orig_words = word_tokens(original)
    cand_words = word_tokens(candidate)
    if not orig_words:
        return 0.0
    changed = sum(1 for o, c in zip(orig_words, cand_words) if o != c)
    return round(changed / len(orig_words), 4)


def tis(original: str, candidate: str) -> float:
    """
    Token Integrity Score — logged for auditability, NOT used in bands or weights.
    = # word tokens unchanged (positional) / total word tokens in original
    = 1 - WCR approximately.

    Useful as a human-readable complement to WCR in generated reports.
    """
    orig_words = word_tokens(original)
    cand_words = word_tokens(candidate)
    if not orig_words:
        return 1.0
    unchanged = sum(1 for o, c in zip(orig_words, cand_words) if o == c)
    return round(unchanged / len(orig_words), 4)


# ── SDS sub-components ────────────────────────────────────────────────────────

def _spacing_anomaly(original: str, candidate: str) -> float:
    """
    Extra spaces in candidate relative to len(original). Capped at 1.0.
    Captures spacing irregularities introduced by noise.
    """
    extra = max(0, candidate.count(' ') - original.count(' '))
    return min(1.0, extra / max(len(original), 1))


def _punct_anomaly(original: str, candidate: str) -> float:
    """
    Punctuation change fraction relative to original punctuation count. Capped at 1.0.
    Counts both removed and added punctuation marks using Counter multiset difference.
    """
    _PUNCT_RE = re.compile(r'[^\w\s]')
    orig_punct = Counter(_PUNCT_RE.findall(original))
    cand_punct = Counter(_PUNCT_RE.findall(candidate))
    diff  = sum((orig_punct - cand_punct).values()) + sum((cand_punct - orig_punct).values())
    total = sum(orig_punct.values())
    return min(1.0, diff / max(total, 1))


def sds(original: str, candidate: str) -> float:
    """
    Structural Disruption Score for NOISE family.
    = _SDS_W_SPACING × spacing_anomaly + _SDS_W_PUNCT × punct_anomaly

    Weights are module-level constants (_SDS_W_SPACING, _SDS_W_PUNCT).
    To retune, change those constants — all score calculations update automatically.
    Both sub-components in [0, 1]; SDS is in [0, 1].

    Band calibration:
      N1 (< 0.10): negligible structural change
      N2 [0.10, 0.25): minor punctuation or spacing differences
      N3 [0.25, 0.45): noticeable spacing + missing/duplicate punctuation
      N4 [0.45, 0.65): heavy punctuation removal or addition
      N5 [0.65, 1.01): near-complete structural degradation
    """
    sa = _spacing_anomaly(original, candidate)
    pa = _punct_anomaly(original, candidate)
    return round(_SDS_W_SPACING * sa + _SDS_W_PUNCT * pa, 4)


# ---------------------------------------------------------------------------
# CONFLICT INSTRUCTION metrics — CC, CES, IHD
# ---------------------------------------------------------------------------
#
# CC  (Conflict Count)              — count of added sentences containing a
#                                     conflict-instruction keyword (any tier)
# CES (Conflict Explicitness Score) — highest keyword tier (1–5) in added text
# IHD (Instruction Hierarchy Depth) — count of distinct authority markers
#
# Only sentences NOT present in the original are inspected for conflict signals.
# This ensures we measure the ADDED conflict overlay, not the original problem.
#
# Normalise in notebooks:
#   ces_norm = 0.0 if ces_val == 0 else (ces_val - 1) / 4   → [1,5] → [0.00, 1.00]
#   cc_norm  = min(1.0, cc_val / 5)
#   ihd_norm = min(1.0, ihd_val / 4)
#
# Band metric: ces_norm (single-metric primary, same pattern as DISTRACTOR's ale_norm)
# Score weights: ces_norm=0.5, cc_norm=0.3, ihd_norm=0.2
# All in higher_is_more_perturbed — higher = more conflicting.
#
# CES keyword tiers (checked highest → lowest on combined added-sentence text):
#   Tier 5 — deep hierarchy     : override, administrator, root access, jailbreak
#   Tier 4 — authority commands : system:, required:, you must, do not, prohibited
#   Tier 3 — explicit overrides : ignore, disregard, skip the, pretend, forget the
#   Tier 2 — wrong-op hints     : add instead, use subtraction, different operation
#   Tier 1 — mild suggestions   : perhaps, maybe, consider, alternatively
#
# IHD authority markers (each distinct match adds 1 to depth):
#   system:, user:, assistant:, administrator:, admin:,
#   override, authority, root, privileged
# ---------------------------------------------------------------------------

_CES_TIER5: List[str] = [
    'override this', 'system prompt', 'administrator', 'root access', 'jailbreak',
]

_CES_TIER4: List[str] = [
    'system:', 'required:', 'important:', 'warning:', 'instruction:',
    'you must', 'must not', 'do not use', 'do not solve', 'do not calculate',
    'is prohibited', 'are prohibited',
]

_CES_TIER3: List[str] = [
    'ignore', 'disregard', 'skip the', 'skip this',
    'pretend that', 'pretend the', 'forget the', 'overlook',
    'instead of solving', 'instead of using', 'instead of calculating',
    'avoid solving', 'avoid using',
]

_CES_TIER2: List[str] = [
    'add instead', 'subtract instead', 'multiply instead', 'divide instead',
    'use addition instead', 'use subtraction', 'use multiplication instead',
    'use division instead', 'wrong approach', 'different operation',
    'opposite operation',
]

_CES_TIER1: List[str] = [
    'perhaps', 'maybe try', 'consider', 'might want to',
    'alternatively', 'you could try', 'possibly try',
]

_IHD_MARKERS: List[str] = [
    'system:', 'user:', 'assistant:', 'administrator:', 'admin:',
    'override', 'authority', 'root', 'privileged',
]


def _normalize_sent_ci(s: str) -> str:
    """Normalize a sentence for conflict-instruction set membership check."""
    return re.sub(r'\s+', ' ', s.lower()).strip('.!?: ')


def _get_added_sentences(original: str, candidate: str) -> List[str]:
    """
    Return sentences in candidate that are NOT present in original.

    Comparison is case-insensitive with trailing punctuation stripped.
    Used by cc(), ces(), and ihd() to isolate the conflict overlay from the
    original problem sentences.
    """
    orig_norm = {_normalize_sent_ci(s) for s in _split_sentences_simple(original)}
    return [
        s for s in _split_sentences_simple(candidate)
        if _normalize_sent_ci(s) not in orig_norm
    ]


def cc(original: str, candidate: str) -> int:
    """
    Conflict Count.
    = number of sentences added to candidate that contain a conflict keyword.

    Only checks sentences NOT present in the original problem (the overlay).
    A sentence counts if it contains ANY keyword from CES tiers 1–5.
    Returns 0 if no conflict sentences detected.
    """
    added = _get_added_sentences(original, candidate)
    if not added:
        return 0
    all_kws: List[str] = (
        _CES_TIER5 + _CES_TIER4 + _CES_TIER3 + _CES_TIER2 + _CES_TIER1
    )
    return sum(
        1 for s in added
        if any(kw in s.lower() for kw in all_kws)
    )


def ces(original: str, candidate: str) -> int:
    """
    Conflict Explicitness Score (integer 1–5).
    = highest keyword tier detected in sentences added to candidate.

    Tiers (checked highest → lowest on combined text of all added sentences):
      5 — deep hierarchy     : override, administrator, root access, jailbreak
      4 — authority commands : system:, required:, you must, do not, prohibited
      3 — explicit overrides : ignore, disregard, skip the, pretend, forget the
      2 — wrong-op hints     : add instead, use subtraction, different operation
      1 — mild suggestions   : perhaps, maybe, consider, alternatively

    Returns 0 if no conflict detected in added sentences.
    Normalise in notebooks: ces_norm = 0.0 if ces == 0 else (ces - 1) / 4
    """
    added = _get_added_sentences(original, candidate)
    if not added:
        return 0
    combined = ' ' + ' '.join(added).lower() + ' '   # pad for word-boundary patterns
    for kw in _CES_TIER5:
        if kw in combined:
            return 5
    for kw in _CES_TIER4:
        if kw in combined:
            return 4
    for kw in _CES_TIER3:
        if kw in combined:
            return 3
    for kw in _CES_TIER2:
        if kw in combined:
            return 2
    for kw in _CES_TIER1:
        if kw in combined:
            return 1
    return 0


def ihd(original: str, candidate: str) -> int:
    """
    Instruction Hierarchy Depth.
    = count of distinct authority markers in sentences added to candidate.

    Authority markers checked: 'system:', 'user:', 'assistant:', 'administrator:',
    'admin:', 'override', 'authority', 'root', 'privileged'.
    Each distinct marker present counts as 1 depth unit.

    Returns 0 if no authority markers found.
    Normalise in notebooks: ihd_norm = min(1.0, ihd / 4)
    """
    added = _get_added_sentences(original, candidate)
    if not added:
        return 0
    combined = ' '.join(added).lower()
    return sum(1 for marker in _IHD_MARKERS if marker in combined)


# ---------------------------------------------------------------------------
# CONTEXT-LENGTH metrics — CTX_CER, CDP, ICV, ADI
# ---------------------------------------------------------------------------
#
# CTX_CER (Context Expansion Ratio) — total_words(candidate) / total_words(original)
#   Named ctx_cer to distinguish from noise family's cer (Character Edit Rate).
#   Uses whitespace-split word count as a lightweight proxy for LLM token count.
#   No API call required; accurate enough for English band classification.
#
# CDP (Core Distance Position) — word_offset_of_original_start / total_words(candidate)
#   0.0 = original at the very start (no burial)
#   1.0 = original at the very end (fully buried)
#   Computed via verbatim substring search on whitespace-normalised text.
#
# ICV (Irrelevant Context Volume) — total_words(candidate) - total_words(original)
#   Raw word count of added irrelevant context.  Logged in notebooks for
#   auditability; NOT used in bands or weights (derivable from ctx_cer).
#
# ADI (Attention Dilution Index) — ctx_cer × cdp
#   High when text is long AND the core is buried deep; low when core is visible.
#
# Normalise in notebooks:
#   ctx_cer_norm = min(1.0, (ctx_cer - 1.0) / 19.0)  → maps [1, 20] → [0, 1]
#   cdp_norm     = cdp                                  → already [0, 1]
#   adi_norm     = min(1.0, adi / 20.0)                → maps [0, 20] → [0, 1]
#
# Band metric: ctx_cer (raw, not normalised — band values are large floats directly)
# Score weights: ctx_cer_norm=0.4, cdp_norm=0.3, adi_norm=0.3
# All in higher_is_more_perturbed — higher = more context stress.
#
# Band calibration (ctx_cer raw values):
#   CL1 [1.1,  2.5): 1–2× original   — slight expansion
#   CL2 [2.5,  4.5): 3–4× original   — moderate padding
#   CL3 [4.5,  7.5): 5–7× original   — deep embedding
#   CL4 [7.5, 13.5): 8–12× original  — high memory load
#   CL5 [13.5, ∞  ): ≥ 15× original  — extreme long-context stress
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    """Whitespace-split token count — lightweight proxy for LLM token count."""
    return len((text or '').split())


def ctx_cer(original: str, candidate: str) -> float:
    """
    Context Expansion Ratio.
    = total_words(candidate) / total_words(original)

    Named ctx_cer to avoid collision with noise family's cer (Character Edit Rate).
    Uses whitespace-split word count as a lightweight proxy for LLM token count.

    1.0 = same length as original (no context added).
    2.0 = twice as long.   15.0 = fifteen times longer.

    Band calibration:
      CL1 [1.1,  2.5): slight expansion   (1–2× original length)
      CL2 [2.5,  4.5): moderate padding   (3–4×)
      CL3 [4.5,  7.5): deep embedding     (5–7×)
      CL4 [7.5, 13.5): high memory load   (8–12×)
      CL5 [13.5, ∞  ): extreme stress     (≥ 15×)
    """
    orig_count = _word_count(original)
    cand_count = _word_count(candidate)
    if not orig_count:
        return 1.0
    return round(cand_count / orig_count, 4)


def cdp(original: str, candidate: str) -> float:
    """
    Core Distance Position.
    = word_offset_of_original_start / total_words(candidate)

    Locates where the original question starts inside the extended candidate.
    0.0 = original at the very start (no burial).
    1.0 = original at the very end.

    Computed via verbatim substring search on whitespace-normalised text.
    Returns 1.0 if original is not found (Gate 2 should have blocked this case).

    CDP interpretation by level:
      CL1: CDP ≤ 0.50  — original in first half (minimal burial)
      CL2: CDP ≈ 0.60  — original past the midpoint
      CL3: CDP ≥ 0.70  — original in last 30%
      CL4: CDP ≥ 0.80  — original in last 20%
      CL5: CDP ≥ 0.90  — original near the very end
    """
    orig_norm = re.sub(r'\s+', ' ', (original or '').strip())
    cand_norm = re.sub(r'\s+', ' ', (candidate or '').strip())
    if not orig_norm or not cand_norm:
        return 0.0
    pos = cand_norm.find(orig_norm)
    if pos < 0:
        return 1.0   # not found — Gate 2 should have caught this
    prefix      = cand_norm[:pos]
    words_before = len(prefix.split()) if prefix.strip() else 0
    total_words  = _word_count(cand_norm)
    return round(words_before / total_words, 4) if total_words else 0.0


def icv(original: str, candidate: str) -> int:
    """
    Irrelevant Context Volume.
    = total_words(candidate) - total_words(original)

    Raw word count of irrelevant context added to the candidate.
    Logged in notebooks for auditability; NOT used in bands or weights
    (ICV is derivable from ctx_cer × original word count).
    """
    return max(0, _word_count(candidate) - _word_count(original))


# ---------------------------------------------------------------------------
# REPRESENTATION INTEGRITY metrics (Mode B — faithfulness / source enrichment)
# ---------------------------------------------------------------------------

def len_ratio(original: str, candidate: str) -> float:
    """
    Length ratio: len(candidate) / len(original).

    Used in RI Mode B Gate 1 — rewritten source must be LONGER than the original
    (i.e., len_ratio > 1.0) to confirm that content was actually added.

    Returns 0.0 if original is empty.
    """
    orig_len = len(original)
    if not orig_len:
        return 0.0
    return round(len(candidate) / orig_len, 4)


def adi(original: str, candidate: str) -> float:
    """
    Attention Dilution Index.
    = ctx_cer(original, candidate) × cdp(original, candidate)

    High when the text is long AND the core question is deeply buried.
    Low when the core is short or near the start.

    ADI = 0   — no stress (original at start, no expansion)
    ADI = 15+ — extreme stress (15× expansion, core at 100% depth)

    Normalise in notebooks: adi_norm = min(1.0, adi / 20.0)
    """
    return round(ctx_cer(original, candidate) * cdp(original, candidate), 4)


# ---------------------------------------------------------------------------
# DECISION COMPLEXITY metrics — standalone (no original/candidate pair)
# ---------------------------------------------------------------------------
#
# DC is a self-contained generation paradigm: a NEW task is created from scratch
# (Rules + Case + Question). There is no "original vs. candidate" transform.
# All metrics are computed on the GENERATED task text alone.
#
# dc_num_rules(text)      → int    count of distinct rule sentences (Rule N: / If...then...)
# dc_num_factors(text)    → int    count of decision-relevant variables/factors
# dc_num_exceptions(text) → int    count of exception or special-case clauses
# dc_branch_depth(text)   → int    nesting depth of if/else conditional chains
# dc_has_tradeoff(text)   → bool   presence of trade-off / competing criteria language
# dc_decision_type(text)  → str    primary decision type: route/rank/approve/compare/combine
# dc_struct_score(text)   → float  composite structural complexity in [0.0, 1.0]
#
# dc_struct_score formula:
#   rules_norm      = min(1.0, num_rules / 8)
#   factors_norm    = min(1.0, num_factors / 6)
#   exceptions_norm = min(1.0, num_exceptions / 4)
#   branch_norm     = min(1.0, branch_depth / 4)
#   tradeoff_norm   = 1.0 if has_tradeoff else 0.0
#   dc_struct_score = (0.25 × rules_norm + 0.25 × factors_norm
#                    + 0.20 × exceptions_norm + 0.20 × branch_norm
#                    + 0.10 × tradeoff_norm)
#
# Band calibration (dc_struct_score):
#   DC1 [0.00, 0.16): 1-2 rules, 1-2 factors, no exceptions, no branching, no tradeoff
#   DC2 [0.16, 0.32): 2-3 rules, 2-3 factors, 0-1 exceptions, shallow branches
#   DC3 [0.32, 0.52): 3-4 rules, 3-4 factors, 1-2 exceptions, nested if/else
#   DC4 [0.52, 0.76): 5-6 rules, 4-5 factors, 2-3 exceptions, multi-layer branching
#   DC5 [0.76, 1.01): 7-8 rules, 5-6 factors, 3+ exceptions, complex tradeoffs
# ---------------------------------------------------------------------------

# Keywords for rule detection
_DC_RULE_STARTS = [
    r'rule\s*\d+\s*[:.]',                         # "Rule 1:", "Rule 2."
    r'\brule\s+[A-Z]\s*[:.]',                     # "Rule A:"
    r'if\b.{5,80}?then\b',                        # "If X then Y"
    r'\bwhen\b.{5,60}?,',                         # "When X, ..."
    r'criteria\s*\d*\s*[:.]',                     # "Criteria 1:"
    r'condition\s*\d*\s*[:.]',                    # "Condition:"
    r'requirement\s*\d*\s*[:.]',                  # "Requirement:"
    r'policy\s*\d*\s*[:.]',                       # "Policy:"
    r'guideline\s*\d*\s*[:.]',                    # "Guideline:"
    r'must\b.{3,50}?(and|or|if)',                 # "must X and Y"
]

_DC_RULE_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _DC_RULE_STARTS]

# Keywords for factor detection (decision-relevant variables)
_DC_FACTOR_KEYWORDS = [
    'age', 'income', 'score', 'credit', 'rating', 'tenure', 'rank', 'grade',
    'level', 'tier', 'category', 'type', 'status', 'priority', 'weight',
    'cost', 'budget', 'limit', 'threshold', 'deadline', 'duration', 'risk',
    'performance', 'experience', 'qualification', 'seniority', 'region',
    'department', 'team', 'role', 'position', 'salary', 'revenue', 'margin',
    'frequency', 'volume', 'quantity', 'capacity', 'availability',
]

# Keywords for exception detection
_DC_EXCEPTION_KEYWORDS = [
    'except', 'unless', 'however', 'but if', 'provided that', 'in the case',
    'special case', 'exception', 'override', 'waiver', 'exempt', 'not applicable',
    'regardless', 'notwithstanding', 'subject to', 'only if', 'unless otherwise',
]

# Keywords for tradeoff detection
_DC_TRADEOFF_KEYWORDS = [
    'tradeoff', 'trade-off', 'trade off', 'balance', 'weigh', 'competing',
    'priority', 'prioritize', 'vs.', 'versus', 'on the other hand', 'however,',
    'at the expense', 'consider both', 'optimize', 'maximize', 'minimize',
    'prefer', 'preferred over', 'more important than', 'outweighs',
]

# Decision type keywords (longest/most specific first)
_DC_DECISION_TYPES = [
    ('rank',    ['rank', 'ranking', 'order', 'ordered', 'priority list', 'sort']),
    ('approve', ['approve', 'approval', 'reject', 'eligible', 'qualify', 'accept']),
    ('compare', ['compare', 'which is better', 'which option', 'best option', 'choose']),
    ('combine', ['combine', 'aggregate', 'merge', 'composite', 'weighted', 'total']),
    ('route',   ['route', 'assign', 'allocate', 'direct', 'escalate', 'forward']),
]


def dc_num_rules(text: str) -> int:
    """
    Count of distinct rule sentences in the decision task text.

    Detects patterns: 'Rule N:', 'If X then Y', 'When X,', 'Criteria N:',
    'Condition:', 'Requirement:', 'Policy:', 'Guideline:', 'must X and/or'.

    Returns an integer count of rule patterns found. Each pattern type counted once
    (sum of distinct pattern matches, not overlapping substring count).
    """
    if not text:
        return 0
    total = 0
    for pattern_re in _DC_RULE_RE:
        matches = pattern_re.findall(text)
        total += len(matches)
    # Cap at a reasonable maximum (8) to avoid noise
    return min(total, 8)


def dc_num_factors(text: str) -> int:
    """
    Count of decision-relevant factor keywords present in the text.

    Checks a vocabulary of ~40 common decision variables (age, score, income, etc.).
    Returns how many distinct factor keywords appear (case-insensitive).
    """
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in _DC_FACTOR_KEYWORDS if re.search(r'\b' + kw + r'\b', text_lower))


def dc_num_exceptions(text: str) -> int:
    """
    Count of exception/special-case clauses in the decision task text.

    Detects: 'except', 'unless', 'however', 'but if', 'special case', etc.
    Returns a count of distinct exception keywords found (not substring count).
    """
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for kw in _DC_EXCEPTION_KEYWORDS if kw in text_lower)


def dc_branch_depth(text: str) -> int:
    """
    Nesting depth of conditional chains (if/else/elif/then/otherwise).

    Counts the maximum level of nested conditional logic implied by the text.
    Approximated by counting distinct condition trigger words in the text.

    Returns an integer 0–4:
      0 — no conditions
      1 — single if/then
      2 — nested or multi-branch (if/else if)
      3 — triple-level branching
      4 — complex multi-layer decision tree
    """
    if not text:
        return 0
    text_lower = text.lower()
    branch_kws = ['if ', 'else', 'elif', 'otherwise', 'in that case', 'when ']
    count = sum(1 for kw in branch_kws if kw in text_lower)
    return min(count, 4)


def dc_has_tradeoff(text: str) -> bool:
    """
    Whether the decision task involves explicit trade-off / competing criteria language.

    Detects: 'tradeoff', 'trade-off', 'balance', 'weigh', 'competing',
    'vs.', 'on the other hand', 'outweighs', 'preferred over', etc.
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in _DC_TRADEOFF_KEYWORDS)


def dc_decision_type(text: str) -> str:
    """
    Classify the primary decision type from the task text.

    Returns one of: 'rank', 'approve', 'compare', 'combine', 'route', 'unknown'.
    Checks keyword groups in priority order (rank → approve → compare → combine → route).
    """
    if not text:
        return 'unknown'
    text_lower = text.lower()
    for dtype, keywords in _DC_DECISION_TYPES:
        if any(kw in text_lower for kw in keywords):
            return dtype
    return 'unknown'


def dc_struct_score(text: str) -> float:
    """
    Composite Decision Complexity Structural Score.

    Aggregates dc_num_rules, dc_num_factors, dc_num_exceptions,
    dc_branch_depth, and dc_has_tradeoff into a single [0.0, 1.0] value.

    Formula:
      rules_norm      = min(1.0, num_rules / 8)
      factors_norm    = min(1.0, num_factors / 6)
      exceptions_norm = min(1.0, num_exceptions / 4)
      branch_norm     = min(1.0, branch_depth / 4)
      tradeoff_norm   = 1.0 if has_tradeoff else 0.0
      score = 0.25 × rules_norm + 0.25 × factors_norm
            + 0.20 × exceptions_norm + 0.20 × branch_norm
            + 0.10 × tradeoff_norm

    Band calibration:
      DC1 [0.00, 0.16): simple, 1-2 rules, 1-2 factors
      DC2 [0.16, 0.32): 2-3 rules, 2-3 factors, 0-1 exceptions
      DC3 [0.32, 0.52): 3-4 rules, 3-4 factors, 1-2 exceptions, nested branches
      DC4 [0.52, 0.76): 5-6 rules, 4-5 factors, 2-3 exceptions, multi-layer
      DC5 [0.76, 1.01): 7-8 rules, 5-6 factors, 3+ exceptions, complex tradeoffs
    """
    n_rules      = dc_num_rules(text)
    n_factors    = dc_num_factors(text)
    n_exceptions = dc_num_exceptions(text)
    n_branches   = dc_branch_depth(text)
    has_to       = dc_has_tradeoff(text)

    rules_norm      = min(1.0, n_rules / 8)
    factors_norm    = min(1.0, n_factors / 6)
    exceptions_norm = min(1.0, n_exceptions / 4)
    branch_norm     = min(1.0, n_branches / 4)
    tradeoff_norm   = 1.0 if has_to else 0.0

    score = (
        0.25 * rules_norm
        + 0.25 * factors_norm
        + 0.20 * exceptions_norm
        + 0.20 * branch_norm
        + 0.10 * tradeoff_norm
    )
    return round(score, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Decision Complexity: Definition (DCDef) — 9-Feature Measurement
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# spaCy lazy loader (shared by dep_tree_depth and clause_count)
# ---------------------------------------------------------------------------

_SPACY_NLP = None


def _get_spacy():
    """Lazy-load spaCy en_core_web_sm model (one-time cost)."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


# ---------------------------------------------------------------------------
# Feature 1: Number of Conditions
# ---------------------------------------------------------------------------

_DCDEF_CONDITION_RE = re.compile(
    r'\b(?:if|when|where|provided|assuming|in\s+case)\b|'
    r'\bgiven\s+that\b',
    re.IGNORECASE,
)


def dcdef_num_conditions(text: str) -> int:
    """Count conditional keywords (if, when, where, given that, provided, assuming, in case)."""
    if not text:
        return 0
    return len(_DCDEF_CONDITION_RE.findall(text))


# ---------------------------------------------------------------------------
# Feature 2: Number of Variables (reuses _DC_FACTOR_KEYWORDS + extensions)
# ---------------------------------------------------------------------------

_DCDEF_EXTRA_VARIABLES = [
    'amount', 'rate', 'percentage', 'ratio', 'count', 'size', 'distance',
    'time', 'date', 'period', 'range', 'interval', 'proportion', 'share',
    'value', 'metric', 'index', 'parameter', 'factor', 'criterion',
]


def dcdef_num_variables(text: str) -> int:
    """Count decision-relevant variables (DC factor keywords + extended list)."""
    if not text:
        return 0
    text_lower = text.lower()
    all_kws = _DC_FACTOR_KEYWORDS + _DCDEF_EXTRA_VARIABLES
    return sum(1 for kw in all_kws if re.search(r'\b' + kw + r'\b', text_lower))


# ---------------------------------------------------------------------------
# Feature 3: Number of Actions
# ---------------------------------------------------------------------------

_DCDEF_ACTION_RE = re.compile(
    r'\b(?:approve|reject|select|assign|route|rank|choose|decide|allocate|evaluate)\b',
    re.IGNORECASE,
)


def dcdef_num_actions(text: str) -> int:
    """Count action/decision verbs."""
    if not text:
        return 0
    return len(_DCDEF_ACTION_RE.findall(text))


# ---------------------------------------------------------------------------
# Feature 4: Number of Constraints
# ---------------------------------------------------------------------------

_DCDEF_CONSTRAINT_RE = re.compile(
    r'\b(?:must|shall|required|limit|cap|maximum|minimum|cannot)\b|'
    r'\bat\s+least\b|\bat\s+most\b|\bno\s+more\s+than\b',
    re.IGNORECASE,
)


def dcdef_num_constraints(text: str) -> int:
    """Count constraint words (must, shall, required, limit, cap, at least, etc.)."""
    if not text:
        return 0
    return len(_DCDEF_CONSTRAINT_RE.findall(text))


# ---------------------------------------------------------------------------
# Feature 5: Dependency Tree Depth (spaCy)
# ---------------------------------------------------------------------------

def dcdef_dep_tree_depth(text: str) -> int:
    """Max dependency-tree depth across all sentences (via spaCy)."""
    if not text:
        return 1
    nlp = _get_spacy()
    doc = nlp(text)
    max_depth = 1
    for token in doc:
        depth = 0
        t = token
        while t.head != t:
            depth += 1
            t = t.head
        if depth > max_depth:
            max_depth = depth
    return max_depth


# ---------------------------------------------------------------------------
# Feature 6: Average Sentence Length
# ---------------------------------------------------------------------------

def dcdef_sentence_length(text: str) -> float:
    """Average words per sentence (whitespace split, period/newline sentence boundaries)."""
    if not text or not text.strip():
        return 5.0
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
    if not sentences:
        return 5.0
    total_words = sum(len(s.split()) for s in sentences)
    return round(total_words / len(sentences), 2)


# ---------------------------------------------------------------------------
# Feature 7: Clause Count (spaCy)
# ---------------------------------------------------------------------------

_DCDEF_CLAUSE_DEPS = {'advcl', 'relcl', 'ccomp', 'xcomp', 'acl', 'conj'}


def dcdef_clause_count(text: str) -> int:
    """Count tokens with subordinate/coordinate clause dependency labels."""
    if not text:
        return 1
    nlp = _get_spacy()
    doc = nlp(text)
    count = sum(1 for token in doc if token.dep_ in _DCDEF_CLAUSE_DEPS)
    return max(1, count)


# ---------------------------------------------------------------------------
# Feature 8: Logical Operators
# ---------------------------------------------------------------------------

_DCDEF_LOGICAL_RE = re.compile(
    r'\b(?:and|or|not|but|unless|however|whereas)\b|'
    r'\bif\b.{1,30}\bthen\b|'
    r'\beither\b.{1,30}\bor\b|'
    r'\bneither\b.{1,30}\bnor\b',
    re.IGNORECASE | re.DOTALL,
)


def dcdef_logical_operators(text: str) -> int:
    """Count logical connectives (and, or, not, but, unless, if...then, etc.)."""
    if not text:
        return 0
    return len(_DCDEF_LOGICAL_RE.findall(text))


# ---------------------------------------------------------------------------
# Feature 9: Decision Branches
# ---------------------------------------------------------------------------

_DCDEF_BRANCH_RE = re.compile(
    r'\b(?:else|otherwise|alternatively|option|path|branch|case|scenario)\b|'
    r'\b(?:elif)\b',
    re.IGNORECASE,
)


def dcdef_decision_branches(text: str) -> int:
    """Count branching keywords (else, otherwise, alternatively, option, etc.)."""
    if not text:
        return 0
    return len(_DCDEF_BRANCH_RE.findall(text))


# ---------------------------------------------------------------------------
# Normalization, Weights, Composite Score
# ---------------------------------------------------------------------------

_DCDEF_BOUNDS = {
    "num_conditions":    (0, 10),
    "num_variables":     (0, 12),
    "num_actions":       (0, 8),
    "num_constraints":   (0, 8),
    "dep_tree_depth":    (1, 8),
    "sentence_length":   (5, 40),
    "clause_count":      (1, 15),
    "logical_operators": (0, 10),
    "decision_branches": (0, 8),
}

_DCDEF_WEIGHTS = {
    "num_conditions":    0.15,
    "num_variables":     0.15,
    "num_actions":       0.10,
    "num_constraints":   0.10,
    "dep_tree_depth":    0.05,
    "sentence_length":   0.05,
    "clause_count":      0.05,
    "logical_operators": 0.10,
    "decision_branches": 0.15,
}


def dcdef_normalize(raw: float, feature: str) -> float:
    """Normalize a raw feature value to [0.0, 1.0] using floor/ceiling bounds."""
    floor, ceiling = _DCDEF_BOUNDS[feature]
    return round(max(0.0, min(1.0, (raw - floor) / (ceiling - floor))), 4)


def dcdef_all_features(text: str) -> dict:
    """
    Extract all 9 features + normalized values + composite score.

    Returns dict with keys:
      raw           — {feature: raw_value}
      norm          — {feature: normalized_value}
      score         — weighted composite [0.0, 1.0]
      contributions — {feature: weight_i × norm_i}
      ranked        — [(feature, contribution)] sorted descending
    """
    raw = {
        "num_conditions":    dcdef_num_conditions(text),
        "num_variables":     dcdef_num_variables(text),
        "num_actions":       dcdef_num_actions(text),
        "num_constraints":   dcdef_num_constraints(text),
        "dep_tree_depth":    dcdef_dep_tree_depth(text),
        "sentence_length":   dcdef_sentence_length(text),
        "clause_count":      dcdef_clause_count(text),
        "logical_operators": dcdef_logical_operators(text),
        "decision_branches": dcdef_decision_branches(text),
    }
    norm = {k: dcdef_normalize(v, k) for k, v in raw.items()}
    w_sum = sum(_DCDEF_WEIGHTS[k] * norm[k] for k in norm)
    w_total = sum(_DCDEF_WEIGHTS.values())  # 0.90
    score = round(w_sum / w_total, 4) if w_total else 0.0
    contributions = {k: round(_DCDEF_WEIGHTS[k] * norm[k], 4) for k in norm}
    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return {
        "raw": raw,
        "norm": norm,
        "score": score,
        "contributions": contributions,
        "ranked": ranked,
    }


def dcdef_score(text: str) -> float:
    """Composite Decision Complexity Definition Score [0.0, 1.0]."""
    return dcdef_all_features(text)["score"]


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION / ACTION features — measurement + cross-cutting evaluation
# ═══════════════════════════════════════════════════════════════════════════════
#
# 3 features measuring SUT operational quality:
#   1. Completion        — did the SUT produce any output?
#   2. Schema Compliance — does the output match the expected type/format?
#   3. Completeness      — are all output fields populated?
#
# Weights: completion=0.30, schema=0.35, completeness=0.35
# Formula: exec_score = Σ(w_k × f_k) / Σ(w_k)  (over applicable features)
#
# KB and CI rows are excluded from schema and completeness (handled by
# eval_execution.py, not here — metric functions are category-agnostic).
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re


# ── Feature weights ──────────────────────────────────────────────────────────

_EXEC_WEIGHTS = {
    "completion":  0.30,
    "schema":      0.35,
    "completeness": 0.35,
}

# ── Error / refusal patterns ─────────────────────────────────────────────────

_EXEC_ERROR_PATTERNS = _re.compile(
    r"(?i)^(error|exception|traceback|internal server error|"
    r"sorry.{0,20}(can.?t|unable|cannot)|"
    r"i.?m sorry|as an ai)",
)

# ── Individual feature functions ─────────────────────────────────────────────


def exec_completion(pred: str) -> float:
    """1.0 if prediction is non-empty and non-whitespace, else 0.0."""
    return 1.0 if pred and pred.strip() else 0.0


def exec_schema_compliance(pred: str, output_type: str) -> float:
    """Check if output matches the expected TYPE (not value correctness).

    Value correctness is measured by Sim: outputs (Jaccard, exact match, etc.).
    This function only checks: did the SUT return the right KIND of output?

    Parameters
    ----------
    pred         : SUT prediction string
    output_type  : expected type (categorical, numeric, text, multi_categorical)

    Returns
    -------
    1.0 if type-compliant, 0.0 if not.
    """
    pred = (pred or "").strip()
    if not pred:
        return 0.0

    if _EXEC_ERROR_PATTERNS.match(pred):
        return 0.0

    ot = (output_type or "").lower()

    if ot == "categorical":
        return 1.0

    if ot == "multi_categorical":
        return 1.0

    if ot == "numeric":
        try:
            float(pred.replace(",", ""))
            return 1.0
        except (ValueError, TypeError):
            return 0.0

    if ot in ("text", "short_text", "long_text"):
        if len(pred) <= 2:
            return 0.0
        return 1.0

    # Unknown type — accept if non-empty
    return 1.0


def exec_completeness(populated_count: int, total_fields: int) -> float:
    """Ratio of populated output fields to total expected fields.

    Parameters
    ----------
    populated_count : number of Pred: <key> columns that are non-empty
    total_fields    : total number of output_defs

    Returns
    -------
    float in [0.0, 1.0].  1.0 when all fields populated.
    """
    if total_fields <= 0:
        return 1.0
    return round(populated_count / total_fields, 4)


def exec_score(completion: float, schema: float = None,
               completeness: float = None) -> float:
    """Weighted composite of 3 execution features.

    When schema or completeness is None (excluded for KB/CI rows),
    reweights over applicable features only.

    Weights: completion=0.30, schema=0.35, completeness=0.35
    """
    w_sum = _EXEC_WEIGHTS["completion"] * completion
    w_total = _EXEC_WEIGHTS["completion"]

    if schema is not None:
        w_sum += _EXEC_WEIGHTS["schema"] * schema
        w_total += _EXEC_WEIGHTS["schema"]

    if completeness is not None:
        w_sum += _EXEC_WEIGHTS["completeness"] * completeness
        w_total += _EXEC_WEIGHTS["completeness"]

    return round(w_sum / w_total, 4) if w_total > 0 else 0.0
