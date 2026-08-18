"""
Family-Wide Error (FWE) — canonical definition.

Formula
───────
Given the stress test cases for one family:

    Score_i = (tcr_i × w_tcr + (1 − sim_i) × w_sim) / (w_sim + w_tcr)
              — perturbation intensity of test case i ∈ [0, 1]

Step 1  Filter to failed cases  F = { i : match_i = False }
Step 2  If |F| = 0  →  FWE = 0.0   (no failures, perfect robustness)
Step 3  Normalize failed scores to a probability distribution
            p_i = score_i / Σ_{j∈F} score_j
Step 4  Shannon entropy of the failure distribution
            H = −Σ_{i∈F} p_i · ln(p_i)
Step 5  Normalise to [0, 1]
            H_norm = H / ln(|F|)     (= 1.0 when |F| = 1, set to 0 by convention)
Step 6  FWE = 1 − 0.5 × H_norm  ∈ [0.5, 1]

Interpretation
──────────────
• FWE = 1.0  →  all failures concentrated at ONE stress level  (clear breakpoint)
• FWE = 0.5  →  failures spread uniformly across ALL stress levels (diffuse weakness)
• FWE = 0.5  →  also the floor when there are NO failures

Where FWE is used
─────────────────
1. Cell 8 (pipeline_single_qa.ipynb)  — Table 1 display column
2. CategoryScorer (scorer/scorer.py)   — FWE-derived family weight
       w_fwe(S) = FWE(S) / Σ_S FWE(S)
       w_final  = α·w_business + (1−α)·w_fwe
3. DiagnosisInput (diagnosis_eng/)     — evidence signal for root-cause analysis
"""

import math
from typing import List


def compute_fwe(scores: List[float], matches: List[bool]) -> float:
    """
    Compute FWE = 1 − H_norm for one stress family.

    Parameters
    ──────────
    scores  : perturbation score per stress test case  (Score_i ∈ [0, 1])
              L0 baseline cases must be EXCLUDED before calling.
    matches : True if the model answered correctly, False otherwise
              Must be the same length as scores.

    Returns
    ───────
    fwe : float ∈ [0.5, 1]
    """
    if len(scores) != len(matches):
        raise ValueError(
            f"scores and matches must be the same length "
            f"(got {len(scores)} vs {len(matches)})"
        )

    # Step 1 — failed cases only
    failed_scores = [s for s, m in zip(scores, matches) if not m]

    # Step 2 — no failures → FWE = 0.5 (floor of new range)
    if not failed_scores:
        return 0.5

    # Step 3 — normalize to probability distribution
    total = sum(failed_scores)
    if total <= 0:
        return 0.5  # no perturbation signal → floor
    p = [s / total for s in failed_scores]

    # Step 4 — Shannon entropy  H = −Σ p_i · ln(p_i)
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)

    # Step 5 — normalise by maximum possible entropy  ln(|F|)
    n_failed = len(failed_scores)
    if n_failed == 1:
        # Single failure → entropy = 0 → most concentrated → FWE = 1
        h_norm = 0.0
    else:
        h_max  = math.log(n_failed)
        h_norm = h / h_max if h_max > 0 else 0.0

    # Step 6 — rescale to [0.5, 1]:  FWE = 1 − 0.5 × H_norm
    return round(1.0 - 0.5 * h_norm, 6)
