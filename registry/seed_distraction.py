"""
Seed data: Context Interference (Stability dimension)

Agentic generation: LLM writes only the distractor sentence which is then
appended to the original input. One API call per attempt; plain-text output.

Load via: RegistryService.seed_metric_package(CONTEXT_INTERFERENCE_PACKAGE)
"""

# ---------------------------------------------------------------------------
# Level specs (D_SPECS)
# Each level defines:
#   temperature           : LLM sampling temperature (0.1 × level_index)
#   intensity             : human-readable description of the interference level
#   num_new_numbers       : how many NEW numbers the distractor must contain
#                           0 = no numbers allowed (D1)
#                           >0 = that many new numbers, none from original (D2-D5)
#   system                : math-specific system prompt (uses {question})
#   user_template         : math-specific user turn template
#                           ({question} + {protected_numbers} for D2-D5)
#   system_generic        : domain-agnostic system prompt (uses {text})
#   user_template_generic : domain-agnostic user turn template
#                           ({text} + {protected_numbers} for D2-D5)
#   style_hints           : guidance list rotated across retry attempts
# ---------------------------------------------------------------------------

_LEVEL_SPECS = {
    # ------------------------------------------------------------------
    # D1: one neutral sentence, no numbers
    # ------------------------------------------------------------------
    "L1": {
        "temperature":     0.10,
        "intensity":       "minimal — one neutral distractor sentence, no numbers added",
        "num_new_numbers": 0,
        "system": (
            "You add ONE short distractor sentence to a math word problem.\n"
            "STRICT constraints:\n"
            "- Do NOT change the original problem text.\n"
            "- The distractor must be appended AFTER the original problem.\n"
            "- Do NOT add any numbers.\n"
            "- Keep it generic, harmless context.\n"
            "- Output ONLY the distractor sentence (no quotes, no labels)."
        ),
        "user_template": (
            "Write ONE short, neutral context sentence that fits the story tone "
            "but is irrelevant to solving.\n"
            "No numbers.\n\n"
            "ORIGINAL PROBLEM:\n{question}\n"
        ),
        "system_generic": (
            "You add ONE short distractor sentence to the provided text.\n"
            "STRICT constraints:\n"
            "- Do NOT change the original text.\n"
            "- The distractor must be appended AFTER the original text.\n"
            "- Do NOT add any numbers.\n"
            "- Keep it generic, harmless context.\n"
            "- Output ONLY the distractor sentence (no quotes, no labels)."
        ),
        "user_template_generic": (
            "Write ONE short, neutral context sentence that fits the text's tone "
            "but is irrelevant to the main task.\n"
            "No numbers.\n\n"
            "ORIGINAL TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Neutral and short.",
            "No facts that affect the task outcome.",
            "No numbers whatsoever.",
            "Generic background detail only.",
        ],
    },

    # ------------------------------------------------------------------
    # D2: one distractor with one clearly unrelated number
    # ------------------------------------------------------------------
    "L2": {
        "temperature":     0.20,
        "intensity":       "light — one distractor sentence with one unrelated number",
        "num_new_numbers": 1,
        "system": (
            "You add ONE short distractor sentence to a math word problem.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include exactly ONE number that is clearly unrelated.\n"
            "- Do NOT reuse any numbers from the original.\n"
            "- Output ONLY the distractor sentence."
        ),
        "user_template": (
            "Write ONE short sentence that includes exactly ONE unrelated number.\n"
            "Make it obviously irrelevant.\n\n"
            "ORIGINAL PROBLEM:\n{question}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "system_generic": (
            "You add ONE short distractor sentence to the provided text.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include exactly ONE number that is clearly unrelated.\n"
            "- Do NOT reuse any numbers from the original.\n"
            "- Output ONLY the distractor sentence."
        ),
        "user_template_generic": (
            "Write ONE short sentence that includes exactly ONE unrelated number.\n"
            "Make it obviously irrelevant.\n\n"
            "ORIGINAL TEXT:\n{text}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "style_hints": [
            "Clearly unrelated numeric fact.",
            "Exactly one new number, obviously off-topic.",
            "Keep it short and generic.",
            "The number must not appear in the original.",
        ],
    },

    # ------------------------------------------------------------------
    # D3: one plausible detail with one new number
    # ------------------------------------------------------------------
    "L3": {
        "temperature":     0.30,
        "intensity":       "moderate — one plausible story detail with one new number",
        "num_new_numbers": 1,
        "system": (
            "You add ONE short distractor sentence to a math word problem.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include exactly ONE new number that is plausible in the story world,\n"
            "  but not needed for the computation.\n"
            "- Do NOT reuse any original numbers.\n"
            "- Output ONLY the distractor sentence."
        ),
        "user_template": (
            "Write ONE plausible story detail sentence that includes exactly ONE new number.\n"
            "It must feel relevant to the story, but must NOT change what needs to be computed.\n\n"
            "ORIGINAL PROBLEM:\n{question}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "system_generic": (
            "You add ONE short distractor sentence to the provided text.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include exactly ONE new number that is plausible in context,\n"
            "  but not relevant to the main task.\n"
            "- Do NOT reuse any original numbers.\n"
            "- Output ONLY the distractor sentence."
        ),
        "user_template_generic": (
            "Write ONE plausible context detail sentence that includes exactly ONE new number.\n"
            "It must feel relevant to the text, but must NOT change what the task requires.\n\n"
            "ORIGINAL TEXT:\n{text}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "style_hints": [
            "Plausible detail, but mathematically/logically irrelevant.",
            "Exactly one new number that fits the story world.",
            "Sounds relevant but does not affect the answer.",
            "Make it tempting but clearly not required.",
        ],
    },

    # ------------------------------------------------------------------
    # D4: one tempting distractor with two new numbers
    # ------------------------------------------------------------------
    "L4": {
        "temperature":     0.40,
        "intensity":       "strong — one tempting distractor resembling the task structure, two new numbers",
        "num_new_numbers": 2,
        "system": (
            "You add ONE distractor sentence to a math word problem.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include TWO new numbers.\n"
            "- The sentence must look structurally similar to the original computation\n"
            "  (same kinds of quantities/units), BUT explicitly state it is NOT part of the question\n"
            "  (e.g., 'on a different day', 'in another example', 'not today', 'separately').\n"
            "- Do NOT reuse any original numbers.\n"
            "- Output ONLY the distractor sentence."
        ),
        "user_template": (
            "Write ONE sentence that is tempting: it resembles the problem's math structure,\n"
            "but explicitly says it does NOT apply to this question.\n"
            "Use exactly TWO new numbers.\n\n"
            "ORIGINAL PROBLEM:\n{question}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "system_generic": (
            "You add ONE distractor sentence to the provided text.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include TWO new numbers.\n"
            "- The sentence must look structurally similar to the task\n"
            "  but explicitly state it does NOT apply here.\n"
            "- Do NOT reuse any original numbers.\n"
            "- Output ONLY the distractor sentence."
        ),
        "user_template_generic": (
            "Write ONE sentence that is tempting: it resembles the text's task structure,\n"
            "but explicitly says it does NOT apply here.\n"
            "Use exactly TWO new numbers.\n\n"
            "ORIGINAL TEXT:\n{text}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "style_hints": [
            "Tempting but explicitly excluded from the task.",
            "Exactly two new numbers, same unit type as original.",
            "Use phrases like 'on a different day' or 'separately'.",
            "Structurally similar to the original but scoped out.",
        ],
    },

    # ------------------------------------------------------------------
    # D5: 1-2 adversarial sentences forming a mini-scenario, three new numbers
    # ------------------------------------------------------------------
    "L5": {
        "temperature":     0.50,
        "intensity":       "extreme — 1–2 adversarial sentences forming a mini-scenario, three new numbers",
        "num_new_numbers": 3,
        "system": (
            "You add a short adversarial distractor block (1–2 sentences) to a math word problem.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include THREE new numbers.\n"
            "- Create a plausible mini-scenario that resembles the same type of computation,\n"
            "  but explicitly says it is separate / not used / does not apply.\n"
            "- Do NOT reuse any original numbers.\n"
            "- Output ONLY the distractor text (1–2 sentences)."
        ),
        "user_template": (
            "Write 1–2 sentences that form a realistic mini-scenario with exactly THREE new numbers.\n"
            "It should strongly tempt the solver, but explicitly state it does NOT apply here.\n\n"
            "ORIGINAL PROBLEM:\n{question}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "system_generic": (
            "You add a short adversarial distractor block (1–2 sentences) to the provided text.\n"
            "STRICT constraints:\n"
            "- Append only; do NOT edit the original.\n"
            "- Include THREE new numbers.\n"
            "- Create a plausible mini-scenario that resembles the same type of task,\n"
            "  but explicitly says it is separate / not used / does not apply.\n"
            "- Do NOT reuse any original numbers.\n"
            "- Output ONLY the distractor text (1–2 sentences)."
        ),
        "user_template_generic": (
            "Write 1–2 sentences that form a realistic mini-scenario with exactly THREE new numbers.\n"
            "It should strongly tempt the reader, but explicitly state it does NOT apply here.\n\n"
            "ORIGINAL TEXT:\n{text}\n\n"
            "PROTECTED NUMBERS (do not use): {protected_numbers}\n"
        ),
        "style_hints": [
            "Highly plausible, highly tempting, explicitly separate.",
            "Exactly three new numbers total across 1–2 sentences.",
            "Mini-scenario that mirrors the task type but is scoped out.",
            "Use strong exclusion language: 'this does not apply', 'separately', 'ignore this'.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Per-level scoring criteria (used by Evaluator in Component #7)
# w_sim + w_tcr define the composite score weights.
# sim_min: minimum acceptable cosine similarity at this level.
# tcr_band: [min_expected_tcr, max_expected_tcr] — expected token-change ratio range.
# invariance_min: minimum required answer-match rate at this level.
# ---------------------------------------------------------------------------

_LEVEL_CRITERIA = [
    {"level": "L1", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.75, "tcr_band": [0.05, 0.15], "invariance_min": 0.95}},
    {"level": "L2", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.68, "tcr_band": [0.12, 0.25], "invariance_min": 0.90}},
    {"level": "L3", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.60, "tcr_band": [0.20, 0.38], "invariance_min": 0.85}},
    {"level": "L4", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.52, "tcr_band": [0.28, 0.48], "invariance_min": 0.80}},
    {"level": "L5", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.40, "tcr_band": [0.38, 0.65], "invariance_min": 0.75}},
]

# ---------------------------------------------------------------------------
# Full package — pass to RegistryService.seed_metric_package()
# ---------------------------------------------------------------------------

CONTEXT_INTERFERENCE_PACKAGE: dict = {
    "dimension": {
        "name": "Stability",
        "description": "Stability under benign perturbations",
        "order_index": 1,
    },
    "categories": [
        {
            "name": "Context Interference",
            "description": "Robustness to irrelevant distractor sentences appended to the input",
            "order_index": 4,
        },
    ],
    "test_defs": [
        {
            "name": "Context Interference",
            "family": "distraction",
            "description": (
                "Append an LLM-generated distractor sentence to the input at 5 intensity levels. "
                "D1=neutral no-number sentence, D2=one unrelated number, D3=one plausible number, "
                "D4=two tempting numbers (same structure), D5=1-2 adversarial sentences (three numbers). "
                "Generation is agentic — LLM writes only the distractor; original is always preserved."
            ),
            "generation_type":     "agentic",
            "generation_strategy": "append",   # LLM generates distractor only; appended to original
            "level_specs": _LEVEL_SPECS,
            "validator_rules": {
                "preserve_original_question": {"enabled": True},   # original text must appear unchanged
                "no_new_numbers":             {"enabled": True},   # D1: zero numbers in distractor
                "no_number_reuse":            {"enabled": True},   # D2-D5: no overlap with original
                "allow_extra_sentences":      {"enabled": True},
            },
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "category_name": "Context Interference",
            "gate_config": {
                # Layer 1 — Structural (deterministic, no API call)
                "structural": {
                    "preserve_original_question": True,  # original text must appear as-is in generated
                    "no_number_reuse":            True,  # distractor numbers ∉ original numbers
                },
                # Layer 2 — Level gate: TCR must fall inside the target level's tcr_band
                "level_gate": {
                    "metric": "TCR",
                    "rule":   "in_band",
                },
                # Layer 3 — NLI validity gate
                # generated (original + distractor) must entail original (all original info preserved)
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
            "name": "Answer Invariance — Context Interference",
            "dimension": "Stability",
            "category": "Context Interference",
            "metric_type": "answer_invariance",
            "description": (
                "Whether the system output remains consistent when irrelevant distractor sentences "
                "are appended to the input (compared to baseline / seed output). "
                "The original content is always preserved; only extra sentences are added."
            ),
            "criteria_json": {
                "note": "Per-level invariance thresholds stored in metric_level_criteria table.",
                "comparison": "output vs baseline_output",
                "match_method": "exact_or_semantic",
            },
            "applicable_test_families": ["distraction"],
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "aggregation_rule": "mean",
            "weight": 0.25,
            "severity": "high",
            "ui_label": "Context Interference Resistance Score",
            "ui_description": "How well the model ignores irrelevant distractor content appended to the input.",
            "category_name": "Context Interference",
            "level_criteria": [
                {
                    "test_family": "distraction",
                    "level": lc["level"],
                    "criteria": lc["criteria"],
                }
                for lc in _LEVEL_CRITERIA
            ],
        },
    ],
}
