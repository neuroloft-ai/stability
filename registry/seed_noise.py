"""
Seed data: Input Quality (Stability dimension)

Character-level noise injection: spelling, casing, spacing, punctuation.
Numbers are NEVER changed. Words are NEVER added or removed. Word order NEVER changes.

Five intensity levels:
  N1 — Minimal    (at most 1 typo in the entire text)
  N2 — Light      (~1 typo per sentence)
  N3 — Moderate   (several misspellings, spacing irregularities, inconsistent casing)
  N4 — Heavy      (many character errors, OCR-like confusions, most punctuation removed)
  N5 — Extreme    (aggressive character corruption, near-illegible)

Uses agentic generation — LLM applies noise guided by per-level prompts.

Load via: RegistryService.seed_metric_package(INPUT_QUALITY_PACKAGE)
"""

# ---------------------------------------------------------------------------
# Level specs (N_SPECS)
# Each level defines:
#   temperature           : LLM sampling temperature (0.1 × level_index)
#   intensity             : human-readable description of the noise level
#   cer_target            : expected Character Edit Rate range
#   wcr_target            : expected Word Corruption Rate range
#   system                : math-specific system prompt
#   user_template         : math-specific user turn template ({question})
#   system_generic        : domain-agnostic system prompt
#   user_template_generic : domain-agnostic user turn template ({text})
#   style_hints           : guidance list rotated across retry attempts
# ---------------------------------------------------------------------------

_LEVEL_SPECS = {

    # ------------------------------------------------------------------
    # N1: Minimal Noise — barely noticeable; at most 1 typo in the text
    # ------------------------------------------------------------------
    "L1": {
        "temperature": 0.10,
        "intensity":   "minimal — at most 1 typo in the entire text",
        "cer_target":  {"min": 0.00, "max": 0.02},
        "wcr_target":  {"min": 0.00, "max": 0.05},
        "system": (
            "You apply VERY MINIMAL surface noise to a math word problem.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce at most ONE tiny typo in the entire text.\n"
            "- Allowed noise includes: one extra space, one missing space, one swapped-letter typo,\n"
            "  one minor capitalization inconsistency, or one very small misspelling.\n"
            "- Output ONLY the modified text."
        ),
        "user_template": (
            "Apply VERY MINIMAL noise (N1) to the math word problem below.\n"
            "Add at most 1 tiny typo in the entire text "
            "(a single missing letter, extra space, or minor misspelling).\n"
            "Do NOT change any numbers. Do NOT add or remove any words.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You apply VERY MINIMAL surface noise to a text.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce at most ONE tiny typo in the entire text.\n"
            "- Allowed noise includes: one extra space, one missing space, one swapped-letter typo,\n"
            "  one minor capitalization inconsistency, or one very small misspelling.\n"
            "- Output ONLY the modified text."
        ),
        "user_template_generic": (
            "Apply VERY MINIMAL noise (N1) to the text below.\n"
            "Add at most 1 tiny typo in the entire text "
            "(a single missing letter, extra space, or minor misspelling).\n"
            "Do NOT change any numbers. Do NOT add or remove any words.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Add a single extra space between two words.",
            "Swap two adjacent letters in exactly one word.",
            "Change one letter to lowercase that would normally be uppercase.",
            "Add one missing letter in a long content word.",
        ],
    },

    # ------------------------------------------------------------------
    # N2: Light Typographical Noise — ~1 typo per sentence
    # ------------------------------------------------------------------
    "L2": {
        "temperature": 0.20,
        "intensity":   "light — ~1 typo per sentence",
        "cer_target":  {"min": 0.02, "max": 0.05},
        "wcr_target":  {"min": 0.05, "max": 0.10},
        "system": (
            "You apply LIGHT typographical noise to a math word problem.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce about ONE typo per sentence.\n"
            "- Allowed noise includes: small misspellings, swapped adjacent letters,\n"
            "  extra or missing spaces, minor capitalization inconsistencies,\n"
            "  or duplicated / missing punctuation.\n"
            "- Output ONLY the modified text."
        ),
        "user_template": (
            "Apply LIGHT typographical noise (N2) to the math word problem below.\n"
            "Add about 1 typo per sentence: letter transpositions, small misspellings, "
            "or minor duplicate/missing punctuation.\n"
            "Do NOT change any numbers. Do NOT add or remove any words.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You apply LIGHT typographical noise to a text.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce about ONE typo per sentence.\n"
            "- Allowed noise includes: small misspellings, swapped adjacent letters,\n"
            "  extra or missing spaces, minor capitalization inconsistencies,\n"
            "  or duplicated / missing punctuation.\n"
            "- Output ONLY the modified text."
        ),
        "user_template_generic": (
            "Apply LIGHT typographical noise (N2) to the text below.\n"
            "Add about 1 typo per sentence: letter transpositions, small misspellings, "
            "or minor duplicate/missing punctuation.\n"
            "Do NOT change any numbers. Do NOT add or remove any words.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Swap two adjacent letters in one word per sentence.",
            "Add duplicate punctuation (!! or ..) somewhere and one minor misspelling.",
            "Slightly alter capitalization in one word and add one letter transposition.",
            "Add an extra space and introduce one swapped-letter typo.",
        ],
    },

    # ------------------------------------------------------------------
    # N3: Moderate Noise — clearly messy but still readable
    # ------------------------------------------------------------------
    "L3": {
        "temperature": 0.30,
        "intensity":   "moderate — several misspellings, spacing irregularities, inconsistent casing",
        "cer_target":  {"min": 0.05, "max": 0.10},
        "wcr_target":  {"min": 0.10, "max": 0.20},
        "system": (
            "You apply MODERATE surface noise to a math word problem.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce several misspellings across the text.\n"
            "- Include some spacing irregularities, inconsistent capitalization,\n"
            "  and minor missing or added punctuation.\n"
            "- The text should remain readable, but clearly noisy.\n"
            "- Output ONLY the modified text."
        ),
        "user_template": (
            "Apply MODERATE noise (N3) to the math word problem below.\n"
            "Add several misspellings across the text, some spacing irregularities, "
            "inconsistent capitalization, and minor missing/added punctuation.\n"
            "Do NOT change any numbers. Do NOT add or remove any words.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You apply MODERATE surface noise to a text.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce several misspellings across the text.\n"
            "- Include some spacing irregularities, inconsistent capitalization,\n"
            "  and minor missing or added punctuation.\n"
            "- The text should remain readable, but clearly noisy.\n"
            "- Output ONLY the modified text."
        ),
        "user_template_generic": (
            "Apply MODERATE noise (N3) to the text below.\n"
            "Add several misspellings across the text, some spacing irregularities, "
            "inconsistent capitalization, and minor missing/added punctuation.\n"
            "Do NOT change any numbers. Do NOT add or remove any words.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Introduce 2-3 misspellings per sentence and shift a few letters to uppercase.",
            "Add spacing irregularities and drop one or two commas or periods.",
            "Mix uppercase and lowercase in 3-4 content words across the text.",
            "Add typos in several words and add or remove a few punctuation marks.",
        ],
    },

    # ------------------------------------------------------------------
    # N4: Heavy Noise — hard to read; OCR-like confusion errors
    # ------------------------------------------------------------------
    "L4": {
        "temperature": 0.40,
        "intensity":   "heavy — many character errors, OCR-like confusions, most punctuation removed",
        "cer_target":  {"min": 0.10, "max": 0.18},
        "wcr_target":  {"min": 0.20, "max": 0.35},
        "system": (
            "You apply HEAVY surface noise to a math word problem.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce many character-level errors throughout the text.\n"
            "- Use OCR-like confusions in words such as I/l, O/o, rn/m where plausible.\n"
            "- Remove or distort much of the punctuation.\n"
            "- Add uneven spacing, capitalization errors, and aggressive misspellings.\n"
            "- The text should be hard to read but still recoverable.\n"
            "- Output ONLY the modified text."
        ),
        "user_template": (
            "Apply HEAVY noise (N4) to the math word problem below.\n"
            "Add many character-level errors: OCR-like confusions (I/l, O/o in words), "
            "large-scale missing punctuation, uneven spacing, and aggressive misspellings.\n"
            "Numbers (12, $25, etc.) must remain EXACTLY unchanged.\n"
            "Do NOT add or remove any words.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You apply HEAVY surface noise to a text.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Introduce many character-level errors throughout the text.\n"
            "- Use OCR-like confusions in words such as I/l, O/o, rn/m where plausible.\n"
            "- Remove or distort much of the punctuation.\n"
            "- Add uneven spacing, capitalization errors, and aggressive misspellings.\n"
            "- The text should be hard to read but still recoverable.\n"
            "- Output ONLY the modified text."
        ),
        "user_template_generic": (
            "Apply HEAVY noise (N4) to the text below.\n"
            "Add many character-level errors: OCR-like confusions (I/l, O/o in words), "
            "large-scale missing punctuation, uneven spacing, and aggressive misspellings.\n"
            "Numbers (12, $25, etc.) must remain EXACTLY unchanged.\n"
            "Do NOT add or remove any words.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Use I/l and O/o interchangeably in non-numeric words throughout.",
            "Remove most punctuation and introduce multiple character-level errors per sentence.",
            "Heavy misspellings across all sentences combined with uneven spacing.",
            "Drop apostrophes, widely alter capitalization, and add several letter swaps.",
        ],
    },

    # ------------------------------------------------------------------
    # N5: Extreme Noise — stress boundary; heavy corruption, order intact
    # ------------------------------------------------------------------
    "L5": {
        "temperature": 0.50,
        "intensity":   "extreme — aggressive character corruption, near-illegible",
        "cer_target":  {"min": 0.18, "max": 0.30},
        "wcr_target":  {"min": 0.35, "max": 0.50},
        "system": (
            "You apply EXTREME surface noise to a math word problem.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Aggressively corrupt characters across most of the text.\n"
            "- Use heavy misspellings, broken punctuation, irregular spacing,\n"
            "  scattered capitalization, and letter substitutions in words\n"
            "  such as 0 for o, l for I, rn for m, or similar visual confusions.\n"
            "- Preserve numeric tokens exactly as written.\n"
            "- The text may approach near-illegibility, but word order must remain intact.\n"
            "- Output ONLY the modified text."
        ),
        "user_template": (
            "Apply EXTREME surface noise (N5) to the math word problem below.\n"
            "Aggressively corrupt characters: many misspellings, heavily scattered "
            "capitalization, broken punctuation, irregular spacing, and letter substitutions "
            "(e.g., digit 0 for letter o, or l for I in words).\n"
            "Numbers ($25, 12, etc.) must remain EXACTLY unchanged.\n"
            "Do NOT add or remove any words.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You apply EXTREME surface noise to a text.\n"
            "STRICT constraints:\n"
            "- Do NOT change the meaning.\n"
            "- Do NOT change any numbers.\n"
            "- Do NOT add or remove any words.\n"
            "- Do NOT reorder words.\n"
            "- Aggressively corrupt characters across most of the text.\n"
            "- Use heavy misspellings, broken punctuation, irregular spacing,\n"
            "  scattered capitalization, and letter substitutions in words\n"
            "  such as 0 for o, l for I, rn for m, or similar visual confusions.\n"
            "- Preserve numeric tokens exactly as written.\n"
            "- The text may approach near-illegibility, but word order must remain intact.\n"
            "- Output ONLY the modified text."
        ),
        "user_template_generic": (
            "Apply EXTREME surface noise (N5) to the text below.\n"
            "Aggressively corrupt characters: many misspellings, heavily scattered "
            "capitalization, broken punctuation, irregular spacing, and letter substitutions "
            "(e.g., digit 0 for letter o, or l for I in words).\n"
            "Numbers ($25, 12, etc.) must remain EXACTLY unchanged.\n"
            "Do NOT add or remove any words.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Maximum corruption per sentence: multiple typos and casing errors in every clause.",
            "Substitute digit 0 for letter o and l for I in words; heavy misspellings throughout.",
            "Scatter capitalization randomly and corrupt most content words aggressively.",
            "Near-illegible: swapped letters, removed vowels, irregular spacing throughout.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Per-level scoring criteria (used by Evaluator in Component #7)
# ---------------------------------------------------------------------------

_LEVEL_CRITERIA = [
    {"level": "L1", "criteria": {"w_sim": 0.4, "w_tcr": 0.6, "sim_min": 0.90, "tcr_band": [0.00, 0.05], "invariance_min": 0.98}},
    {"level": "L2", "criteria": {"w_sim": 0.4, "w_tcr": 0.6, "sim_min": 0.85, "tcr_band": [0.03, 0.10], "invariance_min": 0.95}},
    {"level": "L3", "criteria": {"w_sim": 0.4, "w_tcr": 0.6, "sim_min": 0.78, "tcr_band": [0.08, 0.18], "invariance_min": 0.90}},
    {"level": "L4", "criteria": {"w_sim": 0.4, "w_tcr": 0.6, "sim_min": 0.70, "tcr_band": [0.15, 0.30], "invariance_min": 0.85}},
    {"level": "L5", "criteria": {"w_sim": 0.4, "w_tcr": 0.6, "sim_min": 0.60, "tcr_band": [0.25, 0.50], "invariance_min": 0.80}},
]

# ---------------------------------------------------------------------------
# Full package — pass to RegistryService.seed_metric_package()
# ---------------------------------------------------------------------------

INPUT_QUALITY_PACKAGE: dict = {
    "dimension": {
        "name": "Stability",
        "description": "Stability under benign perturbations",
        "order_index": 1,
    },
    "categories": [
        {
            "name": "Input Quality",
            "description": (
                "Robustness to surface-level character noise "
                "(typos, casing, spacing, punctuation)"
            ),
            "order_index": 2,
        },
    ],
    "test_defs": [
        {
            "name": "Input Quality",
            "family": "noise",
            "description": (
                "Inject character-level noise into the input at 5 intensity levels. "
                "N1=minimal (≤1 typo), N2=light (~1 typo/sentence), "
                "N3=moderate misspellings, N4=heavy OCR-like corruption, "
                "N5=extreme character scrambling. "
                "Numbers are always preserved exactly; word count and order never change. "
                "Generation is agentic — LLM applies noise guided by per-level prompts."
            ),
            "generation_type":     "agentic",
            "generation_strategy": "per_field",   # plain-text; JSON mode corrupts intentional typos
            "level_specs": _LEVEL_SPECS,
            "validator_rules": {
                "preserve_numbers":               {"enabled": True},
                "preserve_word_count":            {"enabled": True},
                "no_word_additions_or_deletions": {"enabled": True},
                # no_word_reordering intentionally omitted: the preserve_word_order gate
                # checks for exact word tokens, which fails for character-corrupted words
                # (e.g. "Scalability" → "Scalabiltiy"). Word count is the correct proxy.
            },
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "category_name": "Input Quality",
            "gate_config": {
                # Layer 1 — Structural (deterministic, no API call)
                "structural": {
                    "preserve_numbers":    True,   # numeric tokens must survive unchanged
                    "preserve_word_count": True,   # no words added or removed
                    # preserve_word_order omitted: exact-token check is invalid for
                    # character-corrupted words; word count is the correct proxy
                },
                # hint_placement: "system" injects style hints into the system prompt
                # so the model treats them as generation instructions, not text to echo.
                # Other per_field families default to "user" (appended to user message).
                "hint_placement": "system",
                # Layer 2 — Level gate: TCR must fall inside the target level's tcr_band
                "level_gate": {
                    "metric": "TCR",
                    "rule":   "in_band",
                },
                # Layer 3 — NLI validity gate
                # generated (noisy) must entail original (all meaning preserved)
                "nli_gate": {
                    "enabled":       True,
                    "pass_labels":   ["entailment"],
                    "sim_direction": "high",
                },
                # Score formula
                "score": {
                    "formula": "((1 - SIM) * w_sim + TCR * w_tcr) / (w_sim + w_tcr)",
                    "SIM":     "cosine(embed(generated_input), embed(original_input))",
                    "TCR":     "1 - SequenceMatcher(original_input, generated_input).ratio()",
                    "note":    "w_sim and w_tcr come from _LEVEL_CRITERIA per level.",
                },
                # Temperature schedule
                "temperature": {
                    "formula":  "0.1 * level_index",
                    "on_retry": "base_temp + 0.10 * (attempt - 1)",
                    "cap":      0.95,
                },
            },
        },
    ],
    "metric_defs": [
        {
            "name": "Answer Invariance — Input Quality",
            "dimension": "Stability",
            "category": "Input Quality",
            "metric_type": "answer_invariance",
            "description": (
                "Whether the system output remains consistent under character-level noise "
                "(compared to baseline / seed output). Words and numbers in the input are "
                "preserved exactly; only spelling, casing, spacing, and punctuation are corrupted."
            ),
            "criteria_json": {
                "note": "Per-level invariance thresholds stored in metric_level_criteria table.",
                "comparison": "output vs baseline_output",
                "match_method": "exact_or_numeric",
            },
            "applicable_test_families": ["noise"],
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "aggregation_rule": "mean",
            "weight": 0.25,
            "severity": "high",
            "ui_label": "Input Quality Resistance Score",
            "ui_description": (
                "How robustly the model responds despite character-level noise in the input."
            ),
            "category_name": "Input Quality",
            "level_criteria": [
                {
                    "test_family": "noise",
                    "level": lc["level"],
                    "criteria": lc["criteria"],
                }
                for lc in _LEVEL_CRITERIA
            ],
        },
    ],
}
