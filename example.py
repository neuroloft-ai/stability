"""
Minimal, offline example — no API key required.

Shows the per-test severity metrics from §3.2 of the paper on a worked pair, then
cross-checks them against the generated test suite. For full (LLM-based) test
GENERATION and model runs, see the scripts in experiment/.

Run from the repository root:
    python example.py
"""
import difflib
from pathlib import Path

# SIM and TCR are both computed from difflib.SequenceMatcher below; TCR = 1 - SIM (character-level).


def sim(a: str, b: str) -> float:
    """SIM = difflib SequenceMatcher ratio over lowercased strings (paper §3.2)."""
    return round(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(), 4)


def tcr(a: str, b: str) -> float:
    """TCR = 1 - SequenceMatcher ratio = 1 - SIM (character-level; matches the pipeline, paper §3.2)."""
    return round(1 - difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(), 4)


def magnitude(original: str, perturbed: str):
    """Per-test quantities: SIM, TCR (= 1 - SIM), and the magnitude score. The pipeline computes
    score = [(1-SIM)*w_sim + TCR*w_tcr] / (w_sim + w_tcr); since TCR = 1 - SIM, this reduces to 1 - SIM."""
    SIM = sim(original, perturbed)
    TCR = tcr(original, perturbed)
    score = round(0.5 * (1 - SIM) + 0.5 * TCR, 4)   # = 1 - SIM (since TCR = 1 - SIM)
    return SIM, TCR, score


if __name__ == "__main__":
    original  = "Toula bought 3 dozen donuts that cost $68 per dozen. How much did she pay?"
    perturbed = "Toula purchased three dozen doughnuts priced at $68 each dozen. What was her total cost?"

    SIM, TCR, score = magnitude(original, perturbed)
    print("original :", original)
    print("perturbed:", perturbed)
    print(f"SIM={SIM}  TCR={TCR}  magnitude(score)={score}\n")

    suite = Path("experiment/output/benchmark/gsm8k_symbolic100/generated.csv")
    if suite.exists():
        import pandas as pd
        d = pd.read_csv(suite)
        admitted = d[d["Gate"] == True]
        print(f"Generated suite: {len(d)} candidates -> {len(admitted)} admitted "
              f"across {admitted['Category'].nunique()} families")
        print("\nMean magnitude (Score) per family among admitted tests:")
        print(admitted.groupby("Category")["Score"].mean().round(3).to_string())
    else:
        print("(generated suite not found — run from the repo root to see the cross-check)")
