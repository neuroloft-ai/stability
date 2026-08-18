"""
CategoryScorer — adaptive multi-family score aggregation.

Works for any scoring category (Stability, Robustness, Fairness, …).
One instance per category. FWE values must be pre-computed via
scorer.compute_fwe(scores, matches) and passed in FamilyInput.fwe.
If fwe is not supplied, it falls back to max(0, 1 − p_score) as an
approximation (not recommended — always pass the computed value).

Algorithm
─────────
Step 1  w_business(S) read from config (user-configurable)
Step 2  w_fwe(S) = FWE(S) / Σ_S FWE(S)
Step 3  w_final(S) = α·w_business(S) + (1−α)·w_fwe(S),  renormalized
Step 4  P          = Σ_S w_final(S) · p(S)            ← weighted mean
        P_harmonic = 1 / Σ_S (w_final(S) / (p(S)+ε))  ← weighted harmonic

Quick usage
───────────
    from scorer import CategoryScorer, FamilyInput
    from scorer.config import CATEGORY_DEFAULT_WEIGHTS

    scorer = CategoryScorer(
        category="Stability",
        business_weights=CATEGORY_DEFAULT_WEIGHTS["Stability"],
    )
    result = scorer.score({
        "paraphrase":  FamilyInput(p_score=0.85),
        "noise":       FamilyInput(p_score=0.92),
        "format":      FamilyInput(p_score=0.78),
        "distraction": FamilyInput(p_score=0.70),
        "conflict":    FamilyInput(p_score=0.60),
        "context":     FamilyInput(p_score=0.55),
    })
    print(result.p_primary)    # weighted mean
    print(result.p_harmonic)   # weighted harmonic mean

    # Override one business weight at runtime
    scorer.update_business_weight("conflict", 0.35)
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .config import DEFAULT_ALPHA, DEFAULT_EPSILON
from .weights import compute_fwe_weights, compute_final_weights, normalize_weights


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------

@dataclass
class FamilyInput:
    """
    Per-family inputs required for scoring.

    p_score : P-Score ∈ [0, 1]  — perturbation-weighted accuracy for this family.
              Compute via:  Σ(Score_i × Match_i) / Σ(Score_i)
    fwe     : FWE    ∈ [0, 1]  — Family-Wide Error = 1 − H_norm, where
              H_norm is the normalised Shannon entropy of the failure-score
              distribution.  Compute via:  scorer.compute_fwe(scores, matches)
              If None, falls back to max(0, 1 − p_score) as an approximation.
    """
    p_score: float
    fwe:     Optional[float] = None


@dataclass
class FamilyBreakdown:
    """Per-family scoring detail returned inside ScoringResult."""
    p_score:      float
    fwe:          float        # resolved value — never None in output
    w_business:   float
    w_fwe:        float
    w_final:      float
    contribution: float        # w_final × p_score  (additive share of P)


@dataclass
class ScoringResult:
    """
    Full scoring result for one category.

    p_primary   — weighted mean:     Σ_S w_final(S) · p(S)
    p_harmonic  — weighted harmonic: 1 / Σ_S (w_final(S) / (p(S) + ε))

    fallback    — True when FWE was unavailable (or degenerate) and
                  w_business was used as the sole weight system.
    alpha       — α used in this computation (logged for auditability).
    """
    category:   str
    p_primary:  float
    p_harmonic: float
    w_business: Dict[str, float]
    w_fwe:      Dict[str, float]
    w_final:    Dict[str, float]
    breakdown:  Dict[str, FamilyBreakdown]
    fallback:   bool
    alpha:      float


# ---------------------------------------------------------------------------
# CategoryScorer
# ---------------------------------------------------------------------------

class CategoryScorer:
    """
    Adaptive scorer for one evaluation category.

    Combines user-configurable business weights with FWE-derived weights
    to produce a single category-level score.

    Parameters
    ──────────
    category         Category name, e.g. "Stability"
    business_weights {family: weight}  Must sum to 1.0.
                     Update at runtime via update_business_weight().
    alpha            Blend ratio ∈ [0, 1]
                       0 → pure FWE weight    1 → pure business weight
    epsilon          Guard against div/0 in harmonic mean  (default 1e-6)
    """

    def __init__(
        self,
        category: str,
        business_weights: Dict[str, float],
        alpha:   float = DEFAULT_ALPHA,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.category = category
        self.alpha    = alpha
        self.epsilon  = epsilon
        self._w_business = normalize_weights(dict(business_weights))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        family_results: Dict[str, FamilyInput],
    ) -> ScoringResult:
        """
        Compute category scores from per-family p_score / fwe inputs.

        FWE should be pre-computed via compute_fwe(scores, matches) and
        passed in FamilyInput.fwe.  Falls back to max(0, 1 − p_score)
        when not supplied (approximate — use only when raw data unavailable).

        Fallback to pure business weights when:
          • Σ FWE = 0  (all families perform perfectly — no signal)
          • A family in family_results is not in the registered w_business

        Returns ScoringResult with p_primary, p_harmonic, and full breakdown.
        """
        if not family_results:
            return self._empty_result()

        # ── Resolve FWE ─────────────────────────────────────────────
        resolved_fwe: Dict[str, float] = {
            s: (r.fwe if r.fwe is not None else max(0.0, 1.0 - r.p_score))
            for s, r in family_results.items()
        }

        # ── Step 2: FWE weights ──────────────────────────────────────
        # Fallback when Σ FWE = 0 (all perfect) or unknown family present
        fwe_total   = sum(resolved_fwe.values())
        all_known   = all(s in self._w_business for s in family_results)
        use_fallback = (not all_known) or (fwe_total <= 0)

        w_fwe = compute_fwe_weights(
            fwe_scores       = resolved_fwe if not use_fallback else {},
            fallback_weights = self._w_business,
        )

        # ── Step 3: Final weights ────────────────────────────────────
        w_business_active = {
            s: self._w_business.get(s, 0.0) for s in family_results
        }
        w_fwe_active = {s: w_fwe.get(s, 0.0) for s in family_results}

        # When in fallback, force α = 1 so w_final = w_business
        effective_alpha = 1.0 if use_fallback else self.alpha

        w_final = compute_final_weights(
            w_business = w_business_active,
            w_fwe      = w_fwe_active,
            alpha      = effective_alpha,
        )

        # ── Step 4: Scores ───────────────────────────────────────────
        p_primary  = self._weighted_mean(family_results, w_final)
        p_harmonic = self._weighted_harmonic(family_results, w_final)

        # ── Breakdown ────────────────────────────────────────────────
        breakdown = {
            s: FamilyBreakdown(
                p_score      = r.p_score,
                fwe          = resolved_fwe[s],
                w_business   = round(w_business_active.get(s, 0.0), 6),
                w_fwe        = round(w_fwe_active.get(s, 0.0), 6),
                w_final      = round(w_final.get(s, 0.0), 6),
                contribution = round(w_final.get(s, 0.0) * r.p_score, 6),
            )
            for s, r in family_results.items()
        }

        return ScoringResult(
            category   = self.category,
            p_primary  = round(p_primary,  4),
            p_harmonic = round(p_harmonic, 4),
            w_business = {s: round(v, 4) for s, v in w_business_active.items()},
            w_fwe      = {s: round(v, 4) for s, v in w_fwe_active.items()},
            w_final    = {s: round(v, 4) for s, v in w_final.items()},
            breakdown  = breakdown,
            fallback   = use_fallback,
            alpha      = self.alpha,
        )

    def update_business_weight(self, family: str, weight: float) -> None:
        """
        Update one business weight and renormalize all weights.

        Other weights scale proportionally so the sum stays at 1.0.

        Args:
            family : stress family name (must be registered)
            weight : new weight value  (must be ≥ 0)
        """
        if weight < 0:
            raise ValueError(f"Weight must be ≥ 0, got {weight!r}")
        if family not in self._w_business:
            raise KeyError(
                f"Unknown family {family!r} for category {self.category!r}. "
                f"Registered: {sorted(self._w_business)}"
            )
        self._w_business[family] = weight
        self._w_business = normalize_weights(self._w_business)

    def set_alpha(self, alpha: float) -> None:
        """Update the blending coefficient α ∈ [0, 1]."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        self.alpha = alpha

    @property
    def business_weights(self) -> Dict[str, float]:
        """Current business weights (read-only copy)."""
        return dict(self._w_business)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _weighted_mean(
        self,
        results: Dict[str, FamilyInput],
        w_final: Dict[str, float],
    ) -> float:
        """P = Σ_S w_final(S) · p(S)"""
        return sum(
            w_final.get(s, 0.0) * r.p_score
            for s, r in results.items()
        )

    def _weighted_harmonic(
        self,
        results: Dict[str, FamilyInput],
        w_final: Dict[str, float],
    ) -> float:
        """P_harmonic = 1 / Σ_S (w_final(S) / (p(S) + ε))"""
        denom = sum(
            w_final.get(s, 0.0) / (r.p_score + self.epsilon)
            for s, r in results.items()
        )
        return 1.0 / denom if denom > 0 else 0.0

    def _empty_result(self) -> ScoringResult:
        return ScoringResult(
            category=self.category, p_primary=0.0, p_harmonic=0.0,
            w_business={}, w_fwe={}, w_final={}, breakdown={},
            fallback=True, alpha=self.alpha,
        )
