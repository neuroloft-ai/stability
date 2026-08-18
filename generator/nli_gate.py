"""
NLI-based meaning preservation gate.

Uses a local cross-encoder NLI model to verify that a generated candidate
preserves the meaning of the original text.

Model : cross-encoder/nli-deberta-v3-small  (~180 MB, auto-downloaded on first use)
Needs : pip install transformers torch

Two modes of operation:

1. check_meaning_preserved(original, candidate)       [RECOMMENDED]
   Direct pair NLI — no claim extraction needed.
   Returns entailment / contradiction / neutral scores for (original → candidate).

2. check_invariance(candidate, core_claims)            [LEGACY]
   Per-claim NLI — requires pre-extracted atomic claims.
   Less reliable for conditional/hedged text.

Usage (hard gate with configurable thresholds):
    gate = NLIGate()
    result = gate.check_meaning_preserved(original, candidate)
    ok = (result['entailment_score']  >= min_entailment and
          result['contradiction_score'] <= max_contradiction)
"""

from typing import List, Dict

_DEFAULT_MODEL = 'cross-encoder/nli-deberta-v3-small'


class NLIGate:

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        """
        Load the NLI pipeline.
        First call downloads the model (~180 MB) — subsequent calls use cache.
        """
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            'text-classification',
            model=model_name,
            top_k=None,          # return all three label scores
        )
        self.model_name = model_name
        print(f'NLIGate ready: {model_name}')

    # ------------------------------------------------------------------
    # Internal: score one (premise, hypothesis) pair
    # ------------------------------------------------------------------

    def _score_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Run NLI on one pair.
        Returns {'entailment': float, 'neutral': float, 'contradiction': float}.
        """
        results = self._pipe({'text': premise, 'text_pair': hypothesis})
        # results shape for single input: [{'label': ..., 'score': ...}, ...]
        # normalise: unwrap nested list if present
        if results and isinstance(results[0], list):
            results = results[0]
        out = {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 0.0}
        for item in results:
            label = item['label'].lower()
            if label in out:
                out[label] = round(float(item['score']), 4)
        return out

    # ------------------------------------------------------------------
    # Public: direct pair NLI  (recommended — no claims needed)
    # ------------------------------------------------------------------

    def check_meaning_preserved(
        self,
        original: str,
        candidate: str,
    ) -> Dict[str, float]:
        """
        Check whether candidate preserves the meaning of original.

        Uses direct pair NLI: premise = original, hypothesis = candidate.
        More reliable than per-claim checking for full-text comparison.

        Args:
            original  : original input text
            candidate : generated / perturbed text

        Returns dict:
            entailment_score    : float  — P(candidate entails original meaning)
            contradiction_score : float  — P(candidate contradicts original)
            neutral_score       : float  — P(relationship is neutral)
        """
        if not original or not candidate:
            return {
                'entailment_score':    1.0,
                'contradiction_score': 0.0,
                'neutral_score':       0.0,
            }

        scores = self._score_pair(original, candidate)
        return {
            'entailment_score':    scores['entailment'],
            'contradiction_score': scores['contradiction'],
            'neutral_score':       scores['neutral'],
        }

    # ------------------------------------------------------------------
    # Public: check invariance against all core claims  (legacy)
    # ------------------------------------------------------------------

    def check_invariance(
        self,
        candidate: str,
        core_claims: List[str],
    ) -> Dict:
        """
        Check that candidate does not contradict any core claim.

        Args:
            candidate    : generated paraphrase text
            core_claims  : list of atomic fact strings from extract_core()

        Returns dict:
            entailment_score   : float  — avg p_entailment across claims
            contradiction_max  : float  — max p_contradiction across claims
            per_claim          : list   — per-claim scores for auditing
        """
        if not core_claims:
            return {
                'entailment_score':  1.0,
                'contradiction_max': 0.0,
                'per_claim':         [],
            }

        per_claim = []
        for claim in core_claims:
            scores = self._score_pair(candidate, claim)
            per_claim.append({
                'claim':         claim,
                'entailment':    scores['entailment'],
                'neutral':       scores['neutral'],
                'contradiction': scores['contradiction'],
            })

        entailment_score  = round(
            sum(r['entailment']    for r in per_claim) / len(per_claim), 4
        )
        contradiction_max = round(
            max(r['contradiction'] for r in per_claim), 4
        )

        return {
            'entailment_score':  entailment_score,
            'contradiction_max': contradiction_max,
            'per_claim':         per_claim,
        }
