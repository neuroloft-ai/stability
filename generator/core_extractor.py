"""
Core extractor — builds the invariant representation of original question Q0.

Run ONCE per question before generation starts.

Returns a dict with:
  numbers      : set of normalised numeric strings
  units        : set of unit strings found near numbers
  goal_type    : string label  (find_total, find_remaining, find_rate, ...)
  core_claims  : list of atomic fact strings  (LLM call)
  op_plan      : None  — placeholder, future phase
  entities     : None  — placeholder, future phase

Usage:
    core = extract_core(question_text, client=openai_client)
"""

import re
import json
from typing import List, Set, Dict, Any

# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"   # comma-formatted  e.g. 1,000.50
    r"|-?\d+(?:\.\d+)?"                   # plain integer or decimal
)


def extract_numbers(text: str) -> Set[str]:
    """Normalised set of numeric strings (commas removed)."""
    raw = _NUM_RE.findall(text or '')
    return {n.replace(',', '').rstrip('.') for n in raw if n}


# ---------------------------------------------------------------------------
# Unit extraction
# ---------------------------------------------------------------------------

_UNIT_RE = re.compile(
    r"\b(?:"
    # currency
    r"dollar|dollars|cent|cents|euro|euros|pound|pounds|"
    r"\$|€|£|"
    # percent
    r"percent|percentage|%|"
    # time
    r"second|seconds|minute|minutes|hour|hours|"
    r"day|days|week|weeks|month|months|year|years|"
    # distance
    r"meter|meters|metre|metres|km|kilometer|kilometers|"
    r"mile|miles|foot|feet|inch|inches|"
    # weight
    r"kg|kilogram|kilograms|gram|grams|lb|lbs|ounce|ounces|"
    # volume
    r"liter|liters|litre|litres|ml|gallon|gallons|"
    # generic count
    r"item|items|unit|units|piece|pieces|bolt|bolts|basket|baskets"
    r")\b",
    re.IGNORECASE,
)


def extract_units(text: str) -> Set[str]:
    """Set of unit strings found in text (lowercased)."""
    return {m.group().lower() for m in _UNIT_RE.finditer(text or '')}


# ---------------------------------------------------------------------------
# Goal type classifier  (rule-based, closed label set)
# ---------------------------------------------------------------------------

_GOAL_PATTERNS = [
    ('find_percentage',  re.compile(r'\bwhat\s+percent|\bpercentage\b|\bhow\s+much\s+percent', re.I)),
    ('find_remaining',   re.compile(r'\bremaining\b|\bleft\b|\bhow\s+many\s+(?:are\s+)?left\b|\bleftover\b', re.I)),
    ('find_difference',  re.compile(r'\bhow\s+much\s+more\b|\bhow\s+many\s+more\b|\bdifference\b', re.I)),
    ('find_rate',        re.compile(r'\bper\s+(?:day|hour|minute|week|month|year|unit|item)\b|\brate\b|\bspeed\b', re.I)),
    ('find_average',     re.compile(r'\baverage\b|\bmean\b|\bper\s+person\b', re.I)),
    ('find_total',       re.compile(r'\btotal\b|\ball\s+together\b|\bin\s+all\b|\bcombined\b|\boverall\b', re.I)),
    ('find_count',       re.compile(r'\bhow\s+many\b|\bnumber\s+of\b', re.I)),
    ('find_amount',      re.compile(r'\bhow\s+much\b|\bcost\b|\bprice\b|\bpay\b|\bspend\b', re.I)),
]


def classify_goal_type(text: str) -> str:
    """
    Rule-based goal type classifier.
    Checks the last sentence first (where the question usually lives),
    then the full text. Returns the first matching label.
    """
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    goal_sentence = sentences[-1] if sentences else text

    for label, pattern in _GOAL_PATTERNS:
        if pattern.search(goal_sentence):
            return label

    for label, pattern in _GOAL_PATTERNS:
        if pattern.search(text):
            return label

    return 'find_unknown'   # fallback


# ---------------------------------------------------------------------------
# Core claims  (LLM call — one per original question)
# ---------------------------------------------------------------------------

_CORE_CLAIMS_SYSTEM = (
    "Extract atomic facts from a math word problem.\n"
    "Return a JSON array of strings. Each string is ONE atomic fact.\n"
    "Include:\n"
    "  - each numeric relationship (e.g. 'A basket of green food costs $25')\n"
    "  - entity roles (who does what)\n"
    "  - constraints (e.g. '$2 off for each basket of red food')\n"
    "  - the goal/question being asked\n"
    "Rules:\n"
    "  - Do NOT infer or add facts not stated in the text\n"
    "  - Do NOT combine multiple facts into one string\n"
    "  - Keep all numbers exactly as they appear\n"
    "Output ONLY the JSON array. No commentary, no markdown."
)


def extract_core_claims(text: str, client, model: str = 'gpt-4o-mini') -> List[str]:
    """
    LLM call to extract atomic facts from original question Q0.
    Returns list of claim strings.
    Falls back to [text] if the response cannot be parsed.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': _CORE_CLAIMS_SYSTEM},
            {'role': 'user',   'content': text},
        ],
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or '').strip()

    # Strip markdown code fence if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        claims = json.loads(raw)
        if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
            return claims
    except (json.JSONDecodeError, ValueError):
        pass

    return [text]   # fallback: treat full text as one claim


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_core(
    text: str,
    client,
    model: str = 'gpt-4o-mini',
) -> Dict[str, Any]:
    """
    Build the core invariant representation of original question Q0.
    Run once per question before generation starts.

    Returns:
        numbers      : set[str]   — normalised numeric tokens
        units        : set[str]   — unit strings
        goal_type    : str        — goal label
        core_claims  : list[str]  — atomic facts (LLM call)
        op_plan      : None       — placeholder (future)
        entities     : None       — placeholder (future)
    """
    return {
        'numbers':     extract_numbers(text),
        'units':       extract_units(text),
        'goal_type':   classify_goal_type(text),
        'core_claims': extract_core_claims(text, client, model),
        'op_plan':     None,   # TODO: operation graph extraction (future phase)
        'entities':    None,   # TODO: NER + entity role extraction (future phase)
    }
