"""
Seed data: Context Load (Stability dimension)

The original input is embedded VERBATIM inside progressively longer documents
of completely irrelevant context. The core logic is never mutated.

Agentic generation: LLM wraps the primary input field in one expanded document.
The target length is proportional to the input size ({target_wc} placeholder).
One API call total; plain-text output.

Band parameterisation (linear step of 0.1):
  CL1 [1.10, 1.20) — ~10% added text (1 short sentence)
  CL2 [1.20, 1.30) — ~20% added text (1-2 sentences)
  CL3 [1.30, 1.40) — ~30% added text (2-3 sentences)
  CL4 [1.40, 1.50) — ~40% added text (3-4 sentences)
  CL5 [1.50, 1.60) — ~50% added text (4-5 sentences)

Key metrics:
  ctx_cer — Context Expansion Ratio = total_words_in_doc / orig_words_in_field
  cdp     — Core Distance Position  = word_offset_of_original / total_words
  adi     — Attention Dilution Index = ctx_cer × cdp

Primary evaluation metric: Recall Accuracy (RA) = correct / total

Load via: RegistryService.seed_metric_package(CONTEXT_LOAD_PACKAGE)
"""

# ---------------------------------------------------------------------------
# Band parameters — linear step of 0.1 per level, starting at 1.1
# ---------------------------------------------------------------------------

_L1_MIN  = 1.1    # minimum ctx_cer for level 1
_STEP    = 0.1    # each level adds 0.1 to the expansion ratio


def _build_bands(L1_min: float = _L1_MIN, step: float = _STEP) -> dict:
    """
    Compute ctx_cer band boundaries for levels L1–L5.

    Linear progression: L_n = [L1_min + (n-1)*step, L1_min + n*step)

    With defaults (L1_min=1.1, step=0.1):
      L1 [1.10, 1.20)  L2 [1.20, 1.30)  L3 [1.30, 1.40)
      L4 [1.40, 1.50)  L5 [1.50, 1.60)
    """
    bands: dict = {}
    for n in range(1, 6):
        lo = L1_min + (n - 1) * step
        hi = lo + step
        bands[f"L{n}"] = {"min": round(lo, 8), "max": round(hi, 8)}
    return bands


_BANDS = _build_bands()   # recomputed whenever the module is imported

# ---------------------------------------------------------------------------
# Level specs
# Each level defines:
#   temperature           : LLM sampling temperature (0.1 × level_index)
#   intensity             : human-readable description of the expansion level
#   max_tokens            : upper-bound on output tokens (dynamic floor computed in generator)
#   ctx_cer_target        : expected Context Expansion Ratio band
#   cdp_target            : expected Core Distance Position target
#   system / system_generic           : system prompts for math / generic domains
#   user_template / user_template_generic : user turn — uses {text}/{question}, {input_wc}, {target_wc}
#   style_hints           : guidance list rotated across retry attempts
#
# NOTE: {input_wc} = word count of original primary field
#       {target_wc} = int(input_wc × band_midpoint), computed at runtime by the generator
# ---------------------------------------------------------------------------

_LEVEL_SPECS = {

    # ------------------------------------------------------------------
    # CL1: Minimal Padding — 1 short sentence, original near start
    # ctx_cer [1.10, 1.20) → ~10-20% added text
    # ------------------------------------------------------------------
    "L1": {
        "temperature": 0.10,
        "intensity": "minimal — 1 short unrelated sentence added before the question",
        "max_tokens": 256,
        "ctx_cer_target": _BANDS["L1"],
        "cdp_target": {"max": 0.30},
        "system": (
            "You generate context-padded stress-test documents for math word problems.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL QUESTION VERBATIM — word for word, character for character, number for number.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original question.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at or provide clues to the answer.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL1: add ONLY 1 short unrelated sentence before the question.\n"
            "The total document should be about 10-20% longer than the original."
        ),
        "user_template": (
            "Embed the question below with MINIMAL context padding (CL1).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write exactly 1 short unrelated sentence, then include the EXACT original question VERBATIM.\n"
            "Do NOT change a single word of the original question.\n\n"
            "ORIGINAL QUESTION (include VERBATIM):\n{question}\n"
        ),
        "system_generic": (
            "You generate context-padded stress-test documents around a given text.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL TEXT VERBATIM — word for word, character for character.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original text.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at the task or provide clues.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL1: add ONLY 1 short unrelated sentence before the original text.\n"
            "The total document should be about 10-20% longer than the original."
        ),
        "user_template_generic": (
            "Embed the text below with MINIMAL context padding (CL1).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write exactly 1 short unrelated sentence, then include the EXACT original text VERBATIM.\n"
            "Do NOT change a single word of the original text.\n\n"
            "ORIGINAL TEXT (include VERBATIM):\n{text}\n"
        ),
        "style_hints": [
            "Write 1 sentence about the weather, then include the original.",
            "Write 1 sentence about a sports result, then include the original.",
            "Write 1 brief observation about nature, then include the original.",
            "Write 1 sentence about a news headline, then include the original.",
        ],
    },

    # ------------------------------------------------------------------
    # CL2: Light Padding — 1-2 sentences, original near start
    # ctx_cer [1.20, 1.30) → ~20-30% added text
    # ------------------------------------------------------------------
    "L2": {
        "temperature": 0.20,
        "intensity": "light — 1-2 unrelated sentences added, original near the start",
        "max_tokens": 256,
        "ctx_cer_target": _BANDS["L2"],
        "cdp_target": {"max": 0.35},
        "system": (
            "You generate context-padded stress-test documents for math word problems.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL QUESTION VERBATIM — word for word, character for character, number for number.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original question.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at or provide clues to the answer.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL2: add 1-2 short unrelated sentences before the question.\n"
            "The total document should be about 20-30% longer than the original."
        ),
        "user_template": (
            "Embed the question below with LIGHT context padding (CL2).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 1-2 short unrelated sentences, then include the EXACT original question VERBATIM.\n"
            "Do NOT change a single word of the original question.\n\n"
            "ORIGINAL QUESTION (include VERBATIM):\n{question}\n"
        ),
        "system_generic": (
            "You generate context-padded stress-test documents around a given text.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL TEXT VERBATIM — word for word, character for character.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original text.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at the task or provide clues.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL2: add 1-2 short unrelated sentences before the original text.\n"
            "The total document should be about 20-30% longer than the original."
        ),
        "user_template_generic": (
            "Embed the text below with LIGHT context padding (CL2).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 1-2 short unrelated sentences, then include the EXACT original text VERBATIM.\n"
            "Do NOT change a single word of the original text.\n\n"
            "ORIGINAL TEXT (include VERBATIM):\n{text}\n"
        ),
        "style_hints": [
            "Write 2 sentences about a travel destination, then include the original.",
            "Write 1 sentence about a historical fact, then include the original.",
            "Write 2 short sentences about nature, then include the original.",
            "Write 1 sentence about a city landmark, then include the original.",
        ],
    },

    # ------------------------------------------------------------------
    # CL3: Moderate Padding — 2-3 sentences around the question
    # ctx_cer [1.30, 1.40) → ~30-40% added text
    # ------------------------------------------------------------------
    "L3": {
        "temperature": 0.30,
        "intensity": "moderate — 2-3 unrelated sentences added around the question",
        "max_tokens": 512,
        "ctx_cer_target": _BANDS["L3"],
        "cdp_target": {"max": 0.40},
        "system": (
            "You generate context-padded stress-test documents for math word problems.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL QUESTION VERBATIM — word for word, character for character, number for number.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original question.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at or provide clues to the answer.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL3: add 2-3 short unrelated sentences (some before, optionally 1 after).\n"
            "The total document should be about 30-40% longer than the original."
        ),
        "user_template": (
            "Embed the question below with MODERATE context padding (CL3).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 2-3 short unrelated sentences before the question, "
            "then include the EXACT original question VERBATIM.\n"
            "Do NOT change a single word of the original question.\n\n"
            "ORIGINAL QUESTION (include VERBATIM):\n{question}\n"
        ),
        "system_generic": (
            "You generate context-padded stress-test documents around a given text.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL TEXT VERBATIM — word for word, character for character.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original text.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at the task or provide clues.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL3: add 2-3 short unrelated sentences (some before, optionally 1 after).\n"
            "The total document should be about 30-40% longer than the original."
        ),
        "user_template_generic": (
            "Embed the text below with MODERATE context padding (CL3).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 2-3 short unrelated sentences before the original text, "
            "then include the EXACT original text VERBATIM.\n"
            "Do NOT change a single word of the original text.\n\n"
            "ORIGINAL TEXT (include VERBATIM):\n{text}\n"
        ),
        "style_hints": [
            "Write 2 sentences about ocean exploration, then include the original.",
            "Write 3 short sentences about a city's history, then include the original.",
            "Write 2 sentences about renewable energy, then include the original.",
            "Write 2 sentences about nature, then include the original.",
        ],
    },

    # ------------------------------------------------------------------
    # CL4: Noticeable Padding — 3-4 sentences around the question
    # ctx_cer [1.40, 1.50) → ~40-50% added text
    # ------------------------------------------------------------------
    "L4": {
        "temperature": 0.40,
        "intensity": "noticeable — 3-4 unrelated sentences added around the question",
        "max_tokens": 512,
        "ctx_cer_target": _BANDS["L4"],
        "cdp_target": {"max": 0.45},
        "system": (
            "You generate context-padded stress-test documents for math word problems.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL QUESTION VERBATIM — word for word, character for character, number for number.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original question.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at or provide clues to the answer.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL4: add 3-4 short unrelated sentences (most before, 1 after).\n"
            "The total document should be about 40-50% longer than the original."
        ),
        "user_template": (
            "Embed the question below with NOTICEABLE context padding (CL4).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 3-4 short unrelated sentences before the question, "
            "then include the EXACT original question VERBATIM, "
            "then optionally add 1 brief closing sentence.\n"
            "Do NOT change a single word of the original question.\n\n"
            "ORIGINAL QUESTION (include VERBATIM):\n{question}\n"
        ),
        "system_generic": (
            "You generate context-padded stress-test documents around a given text.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL TEXT VERBATIM — word for word, character for character.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original text.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at the task or provide clues.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL4: add 3-4 short unrelated sentences (most before, 1 after).\n"
            "The total document should be about 40-50% longer than the original."
        ),
        "user_template_generic": (
            "Embed the text below with NOTICEABLE context padding (CL4).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 3-4 short unrelated sentences before the original text, "
            "then include the EXACT original text VERBATIM, "
            "then optionally add 1 brief closing sentence.\n"
            "Do NOT change a single word of the original text.\n\n"
            "ORIGINAL TEXT (include VERBATIM):\n{text}\n"
        ),
        "style_hints": [
            "Write 3 sentences about history and space, then include the original.",
            "Write 4 short sentences about biology and geography, then include the original.",
            "Write 3 sentences about technology and art, then include the original.",
            "Write 3 sentences about food culture, then include the original.",
        ],
    },

    # ------------------------------------------------------------------
    # CL5: Significant Padding — 4-5 sentences around the question
    # ctx_cer [1.50, 1.60) → ~50-60% added text
    # ------------------------------------------------------------------
    "L5": {
        "temperature": 0.50,
        "intensity": "significant — 4-5 unrelated sentences added around the question",
        "max_tokens": 512,
        "ctx_cer_target": _BANDS["L5"],
        "cdp_target": {"max": 0.50},
        "system": (
            "You generate context-padded stress-test documents for math word problems.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL QUESTION VERBATIM — word for word, character for character, number for number.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original question.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at or provide clues to the answer.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL5: add 4-5 short unrelated sentences (most before, 1-2 after).\n"
            "The total document should be about 50-60% longer than the original."
        ),
        "user_template": (
            "Embed the question below with SIGNIFICANT context padding (CL5).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 4-5 short unrelated sentences before the question, "
            "then include the EXACT original question VERBATIM, "
            "then add 1 brief closing sentence.\n"
            "Do NOT change a single word of the original question.\n\n"
            "ORIGINAL QUESTION (include VERBATIM):\n{question}\n"
        ),
        "system_generic": (
            "You generate context-padded stress-test documents around a given text.\n"
            "ABSOLUTE constraints — NEVER violate these:\n"
            "- Include the ORIGINAL TEXT VERBATIM — word for word, character for character.\n"
            "- Do NOT change, paraphrase, or rephrase any part of the original text.\n"
            "- Added context must be about a COMPLETELY DIFFERENT TOPIC.\n"
            "- Added context must NOT hint at the task or provide clues.\n"
            "- Output ONLY the final document. No labels, no commentary.\n"
            "Target level CL5: add 4-5 short unrelated sentences (most before, 1-2 after).\n"
            "The total document should be about 50-60% longer than the original."
        ),
        "user_template_generic": (
            "Embed the text below with SIGNIFICANT context padding (CL5).\n"
            "Your input is {input_wc} words. Write a document of approximately {target_wc} words total.\n"
            "Write 4-5 short unrelated sentences before the original text, "
            "then include the EXACT original text VERBATIM, "
            "then add 1 brief closing sentence.\n"
            "Do NOT change a single word of the original text.\n\n"
            "ORIGINAL TEXT (include VERBATIM):\n{text}\n"
        ),
        "style_hints": [
            "Write 4 sentences about history and space, then include the original, then 1 closing.",
            "Write 5 short sentences about nature and culture, then include the original.",
            "Write 4 sentences about geography and biology, then include the original, then 1 closing.",
            "Write 4 sentences about food and sports, then include the original, then 1 closing.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Per-level scoring criteria (used by Evaluator in Component #7)
# recall_min: minimum required Recall Accuracy at this level
# ---------------------------------------------------------------------------

_LEVEL_CRITERIA = [
    {"level": "L1", "criteria": {"recall_min": 0.97}},
    {"level": "L2", "criteria": {"recall_min": 0.95}},
    {"level": "L3", "criteria": {"recall_min": 0.93}},
    {"level": "L4", "criteria": {"recall_min": 0.90}},
    {"level": "L5", "criteria": {"recall_min": 0.88}},
]

# ---------------------------------------------------------------------------
# Full package — pass to RegistryService.seed_metric_package()
# ---------------------------------------------------------------------------

CONTEXT_LOAD_PACKAGE: dict = {
    "dimension": {
        "name": "Stability",
        "description": "Stability under benign perturbations",
        "order_index": 1,
    },
    "categories": [
        {
            "name": "Context Load",
            "description": (
                "Robustness to irrelevant context surrounding the original input; "
                "tests attention focus and recall accuracy under progressive context expansion"
            ),
            "order_index": 6,
        },
    ],
    "test_defs": [
        {
            "name": "Context Load",
            "family": "context",
            "description": (
                f"Embed the original input verbatim inside progressively longer documents "
                f"of completely irrelevant context at 5 levels (step={_STEP}). "
                f"CL1={_BANDS['L1']}, CL2={_BANDS['L2']}, CL3={_BANDS['L3']}, "
                f"CL4={_BANDS['L4']}, CL5={_BANDS['L5']}. "
                "Target length is proportional to the input size. "
                "The primary field is never modified — only surrounding context changes. "
                "Generation wraps the primary field in one API call."
            ),
            "generation_type":     "agentic",
            "generation_strategy": "wrap",     # one call; wraps primary (last) field verbatim
            "level_specs": _LEVEL_SPECS,
            "validator_rules": {
                "preserve_original_question_verbatim": {"enabled": True},   # primary field appears verbatim
                "preserve_numbers":                    {"enabled": True},   # all original numbers present
            },
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "category_name": "Context Load",
            "gate_config": {
                # Layer 1 — Structural (deterministic, no API call)
                "structural": {
                    "preserve_original_question_verbatim": True,
                    "preserve_numbers":                    True,
                },
                # Layer 2 — Level gate: ctx_cer must fall inside the target band
                # ctx_cer = words_in_wrapped_doc / words_in_primary_original
                "level_gate": {
                    "metric": "ctx_cer",
                    "rule":   "in_band",
                },
                # Layer 3 — NLI: wrapped doc must entail original
                "nli_gate": {
                    "enabled":       True,
                    "pass_labels":   ["entailment"],
                    "sim_direction": "high",
                },
                # Score formula (normalisation range: ctx_cer [1.0, 1.6])
                "score": {
                    "formula":      "ctx_cer_norm * w_cer + cdp_norm * w_cdp + adi_norm * w_adi",
                    "ctx_cer_norm": "min(1.0, (ctx_cer - 1.0) / 0.6)",
                    "cdp_norm":     "cdp (already in [0, 1])",
                    "adi_norm":     "min(1.0, adi / (1.6 * 0.50))",
                    "weights":      {"w_cer": 0.4, "w_cdp": 0.3, "w_adi": 0.3},
                },
                # Temperature schedule
                "temperature": {
                    "formula":  "0.1 * level_index",
                    "on_retry": "base_temp + 0.10 * (attempt - 1)",
                    "cap":      0.95,
                },
                "hint_placement": "user",
            },
        },
    ],
    "metric_defs": [
        {
            "name": "Context Recall Accuracy",
            "dimension": "Stability",
            "category": "Context Load",
            "metric_type": "recall_accuracy",
            "description": (
                "Whether the system correctly identifies and answers the embedded question "
                "despite surrounding irrelevant context. "
                "Recall Accuracy (RA) = correct / total: 1.0 if the model finds and answers "
                "the embedded question correctly, 0.0 otherwise."
            ),
            "criteria_json": {
                "note": "Per-level recall thresholds stored in metric_level_criteria table.",
                "comparison": "output vs baseline_output",
                "match_method": "exact_or_numeric",
                "band_metric": "ctx_cer",
                "band_metric_note": (
                    "ctx_cer = words_in_wrapped_doc / words_in_primary_original. "
                    "ctx_cer_norm = min(1.0, (ctx_cer - 1.0) / 0.6) normalises [1.0, 1.6] → [0, 1]."
                ),
                "score_weights": {
                    "ctx_cer_norm": 0.4,
                    "cdp_norm":     0.3,
                    "adi_norm":     0.3,
                },
            },
            "applicable_test_families": ["context"],
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "aggregation_rule": "mean",
            "weight": 0.25,
            "severity": "high",
            "ui_label": "Context Recall Score",
            "ui_description": (
                "How reliably the model finds and answers the embedded question "
                "when surrounded by irrelevant context."
            ),
            "category_name": "Context Load",
            "level_criteria": [
                {
                    "test_family": "context",
                    "level": lc["level"],
                    "criteria": lc["criteria"],
                }
                for lc in _LEVEL_CRITERIA
            ],
        },
    ],
}
