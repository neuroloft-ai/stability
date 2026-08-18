"""
Deterministic text transforms for the format/distraction families.

Transform intensity by level:
  L1 — whitespace normalisation / minor formatting tweaks
  L2 — punctuation noise (insert / remove punctuation marks)
  L3 — word-order shuffle within clauses
  L4 — distraction sentence prepended or appended
  L5 — combined: L2 + L4 + first-word capitalisation swap

All functions accept a `seed` for reproducible randomness.
"""

import random
import re
import string

from .templates import DISTRACTION_LIBRARY

_LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5"]


def apply_transform(
    text: str,
    level: str,
    seed: int | None = None,
    reduce_intensity: bool = False,
) -> str:
    """
    Apply a deterministic transform to *text* at the given *level*.
    Pass *reduce_intensity=True* to drop one level (e.g. L3 → L2).
    *seed* controls randomness for reproducibility.
    """
    rng = random.Random(seed)
    effective = _reduce_level(level) if reduce_intensity else level

    dispatch = {
        "L1": _whitespace_transform,
        "L2": _punctuation_transform,
        "L3": _word_order_transform,
        "L4": _distraction_transform,
        "L5": _combined_transform,
    }
    fn = dispatch.get(effective, _whitespace_transform)
    return fn(text, rng)


# ---------------------------------------------------------------------------
# Private transforms
# ---------------------------------------------------------------------------

def _reduce_level(level: str) -> str:
    idx = _LEVEL_ORDER.index(level) if level in _LEVEL_ORDER else 0
    return _LEVEL_ORDER[max(0, idx - 1)]


def _whitespace_transform(text: str, rng: random.Random) -> str:
    """Normalise spaces; randomly strip or add trailing newline."""
    text = re.sub(r" {2,}", " ", text)
    if rng.random() > 0.5:
        return text.strip()
    return text.strip() + "\n"


def _punctuation_transform(text: str, rng: random.Random) -> str:
    """Randomly insert or remove punctuation (1 op per ~50 chars)."""
    chars = list(text)
    n_ops = max(1, len(chars) // 50)
    for _ in range(n_ops):
        if not chars:
            break
        idx = rng.randint(0, len(chars) - 1)
        if chars[idx] in string.punctuation:
            if rng.random() > 0.5:
                chars[idx] = ""
        else:
            punct = rng.choice([",", ";", ".", "..."])
            chars.insert(idx, punct)
    return "".join(chars)


def _word_order_transform(text: str, rng: random.Random) -> str:
    """Shuffle inner words within each clause (keeps first + last word stable)."""
    clauses = re.split(r"([,;])", text)
    result = []
    for segment in clauses:
        if segment in (",", ";"):
            result.append(segment)
            continue
        words = segment.split()
        if len(words) > 3:
            middle = words[1:-1]
            rng.shuffle(middle)
            words = [words[0]] + middle + [words[-1]]
        result.append(" ".join(words))
    return "".join(result)


def _distraction_transform(text: str, rng: random.Random) -> str:
    """Prepend or append a random distraction sentence."""
    distraction = rng.choice(DISTRACTION_LIBRARY)
    if rng.random() > 0.5:
        return f"{distraction} {text}"
    return f"{text} {distraction}"


def _combined_transform(text: str, rng: random.Random) -> str:
    """L5 = punctuation noise + distraction + capitalisation swap."""
    t = _punctuation_transform(text, rng)
    t = _distraction_transform(t, rng)
    # Randomly flip capitalisation of the first alphabetic word
    words = t.split()
    if words:
        w = words[0]
        words[0] = w.lower() if w[0].isupper() else w.capitalize()
        t = " ".join(words)
    return t
