"""
Weight computation for the adaptive category scorer.

Two weight systems
──────────────────
w_business(S)  User-configurable business sensitivity (read from config/registry).

w_fwe(S)       FWE-normalized information weight.
               FWE(S) is pre-computed (= 1 − H(S), already the divergence).
               If not explicitly supplied, derived as FWE(S) = 1 − p_score(S).

               w_fwe(S) = FWE(S) / Σ_S FWE(S)

               Higher FWE → more divergence → higher weight.

Combined weight
───────────────
   w_final(S) = α · w_business(S) + (1−α) · w_fwe(S)

Renormalized to Σ = 1.0 (floating-point safety).
"""

from typing import Dict, Optional


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Renormalize a weight dict so values sum to 1.0.
    Falls back to uniform distribution if total ≤ 0.
    """
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: v / total for k, v in weights.items()}


def compute_fwe_weights(
    fwe_scores: Dict[str, float],
    fallback_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Normalize FWE scores into weights.

    FWE(S) is already the divergence (= 1 − H(S)).
    No further transformation needed — normalize directly:

        w_fwe(S) = FWE(S) / Σ_S FWE(S)

    Fallback to fallback_weights (or uniform) when:
      • fwe_scores is empty
      • Σ FWE = 0  (all families perfectly pass — no discriminating signal)

    Args:
        fwe_scores:       {family: fwe_value}  — values ∈ [0, 1]
        fallback_weights: {family: weight}      — used when FWE unavailable

    Returns:
        {family: w_fwe}  normalized to sum 1.0
    """
    if not fwe_scores:
        return normalize_weights(fallback_weights) if fallback_weights else {}

    total = sum(fwe_scores.values())
    if total <= 0:
        # All FWE = 0 → no discriminating signal → use fallback
        if fallback_weights:
            return normalize_weights(fallback_weights)
        return normalize_weights({k: 1.0 for k in fwe_scores})

    return {k: v / total for k, v in fwe_scores.items()}


def compute_final_weights(
    w_business: Dict[str, float],
    w_fwe: Dict[str, float],
    alpha: float = 0.6,
) -> Dict[str, float]:
    """
    Blend business and FWE weights into final weights.

        w_final(S) = α · w_business(S) + (1−α) · w_fwe(S)

    Renormalized to Σ = 1.0 (floating-point safety).
    Families absent from either dict receive 0 for that component.

    Args:
        w_business: {family: weight}  — business sensitivity weights
        w_fwe:      {family: weight}  — FWE-normalized weights
        alpha:      blend ratio  (0 = pure FWE,  1 = pure business)

    Returns:
        {family: w_final}  normalized to sum 1.0
    """
    all_families = set(w_business) | set(w_fwe)
    combined = {
        s: alpha * w_business.get(s, 0.0) + (1 - alpha) * w_fwe.get(s, 0.0)
        for s in all_families
    }
    return normalize_weights(combined)
