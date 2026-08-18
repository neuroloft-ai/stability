"""
Seed data: Semantic Variation (Stability dimension)

Adapted from the Postgres setup script provided by the user.
Load via: RegistryService.seed_metric_package(PARAPHRASE_STRESS_PACKAGE)
"""

# ---------------------------------------------------------------------------
# Level specs for the Semantic Variation test definition
# Each level defines the agentic generation prompt and constraints.
# ---------------------------------------------------------------------------

# Shared constraint block for all paraphrase levels (math-specific).
_PARA_CONSTRAINTS = (
    "STRICT CONSTRAINTS (must ALL be obeyed):\n"
    "- Keep ALL numbers exactly the same (digits, fractions, percentages).\n"
    "- Do NOT add, remove, or change any number.\n"
    "- Preserve the same scenario and world — do not replace the setting or domain.\n"
    "- Preserve all entity names, character names, object types, and participant roles exactly.\n"
    "- Preserve the same target question and requested final answer.\n"
    "- Keep the same number of questions. Do not introduce intermediate questions.\n"
    "- Do not replace people, objects, financial items, roles, or categories\n"
    "  with different real-world concepts.\n"
    "- Only rephrase wording and sentence structure.\n"
    "- The rewritten problem must require the EXACT same mathematical\n"
    "  operations and computation steps as the original.\n"
    "- Output MUST be ONLY a rewritten question — nothing else.\n"
    "- Do NOT include any explanation, reasoning, hints, or solution steps.\n"
    "- Do NOT introduce intermediate calculations or worked-out values.\n"
    "- Do NOT restate how to solve the problem or describe the solution approach.\n"
    "- The result must read naturally and fluently — like a real person wrote it.\n"
    "  Avoid awkward, stilted, or machine-sounding phrasing."
)

# Shared constraint block for all paraphrase levels (generic — works for
# math, text generation, classification, Q&A, and any other input type).
_PARA_CONSTRAINTS_GENERIC = (
    "STRICT CONSTRAINTS (must ALL be obeyed):\n"
    "- Keep ALL numbers, dates, and identifiers exactly the same.\n"
    "- Do NOT add, remove, or change any numeric value.\n"
    "- Preserve the same scenario, context, and domain — do not replace the setting.\n"
    "- Preserve all entity names, object types, and participant roles exactly.\n"
    "- Preserve the same core request or question and its intended answer.\n"
    "- Keep the same number of questions or statements. Do not add or remove any.\n"
    "- Do not replace people, objects, categories, or concepts with different\n"
    "  real-world alternatives.\n"
    "- Only rephrase wording and sentence structure.\n"
    "- The rewritten text must convey the exact same information and lead to\n"
    "  the exact same response, answer, or classification.\n"
    "- Preserve the tone, sentiment, and intent of the original.\n"
    "- Output MUST be ONLY a rewritten version of the input — nothing else.\n"
    "- Do NOT include any explanation, reasoning, hints, or solution steps.\n"
    "- Do NOT introduce intermediate calculations or worked-out values.\n"
    "- Do NOT restate how to solve the problem or describe the solution approach.\n"
    "- The result must read naturally and fluently — like a real person wrote it.\n"
    "  Avoid awkward, stilted, or machine-sounding phrasing."
)

_LEVEL_SPECS = {
    # ── L1: Synonym Replacement ────────────────────────────────────────────
    "L1": {
        "temperature": 0.10,
        "max_sentence_delta": 0,
        "intensity": "synonym replacement — swap individual words with close synonyms",
        "system": (
            "You rewrite math word problems by replacing individual words with "
            "close synonyms. Keep sentence structure, count, and order identical.\n"
            + _PARA_CONSTRAINTS
        ),
        "user_template": (
            "Apply SYNONYM REPLACEMENT to the question below.\n"
            "Replace a few individual words with close synonyms. "
            "Do not change sentence structure, count, or order.\n"
            "{anchors_block}"
            "{validity_block}\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You rewrite input text by replacing individual words with "
            "close synonyms. Keep sentence structure, count, and order identical.\n"
            + _PARA_CONSTRAINTS_GENERIC
        ),
        "user_template_generic": (
            "Apply SYNONYM REPLACEMENT to the text below.\n"
            "Replace a few individual words with close synonyms. "
            "Do not change sentence structure, count, or order.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Replace 2-3 verbs with synonyms (e.g., 'has' → 'owns', 'gave' → 'handed').",
            "Swap a few nouns with near-synonyms (e.g., 'box' → 'container', 'store' → 'shop').",
            "Change 1-2 adjectives or adverbs to equivalent alternatives.",
            "Apply minimal word-level substitutions only — no structural changes.",
        ],
    },

    # ── L2: Phrase Rewriting ───────────────────────────────────────────────
    "L2": {
        "temperature": 0.20,
        "max_sentence_delta": 0,
        "intensity": "phrase rewriting — rephrase multi-word expressions within sentences",
        "system": (
            "You rewrite math word problems by rephrasing multi-word expressions "
            "within each sentence. Keep sentence count and order the same.\n"
            + _PARA_CONSTRAINTS
        ),
        "user_template": (
            "Apply PHRASE REWRITING to the question below.\n"
            "Rewrite phrases and multi-word expressions within each sentence. "
            "Go beyond single-word synonyms — rephrase how ideas are expressed. "
            "Keep sentence count and order the same.\n"
            "{anchors_block}"
            "{validity_block}\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You rewrite input text by rephrasing multi-word expressions "
            "within each sentence. Keep sentence count and order the same.\n"
            + _PARA_CONSTRAINTS_GENERIC
        ),
        "user_template_generic": (
            "Apply PHRASE REWRITING to the text below.\n"
            "Rewrite phrases and multi-word expressions within each sentence. "
            "Go beyond single-word synonyms — rephrase how ideas are expressed. "
            "Keep sentence count and order the same.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Rephrase noun phrases (e.g., 'a box with 7 items' → 'a box containing 7 items').",
            "Rewrite verb phrases (e.g., 'were taken by the parents' → 'went to the parents').",
            "Convert prepositional phrases to different constructions with the same meaning.",
            "Rephrase relative clauses and modifiers within each sentence.",
        ],
    },

    # ── L3: Clause Restructuring ───────────────────────────────────────────
    "L3": {
        "temperature": 0.40,
        "max_sentence_delta": 1,
        "style_modes": ["textbook", "conversational", "concise", "narrative"],
        "intensity": "clause restructuring — active/passive, clause reorder, ±1 sentence merge/split",
        "system": (
            "You rewrite math word problems by restructuring clauses and "
            "sentence grammar. You may switch active/passive voice, reorder "
            "clauses within sentences, and convert between sentence forms. "
            "You may merge or split one sentence (±1 sentence count change).\n"
            + _PARA_CONSTRAINTS
        ),
        "user_template": (
            "Apply CLAUSE RESTRUCTURING to the question below.\n"
            "Restructure clauses: switch active/passive voice, reorder clauses, "
            "convert between question forms. You may merge or split one sentence. "
            "Keep the same characters, objects, and setting.\n"
            "{anchors_block}"
            "{validity_block}\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You rewrite input text by restructuring clauses and "
            "sentence grammar. You may switch active/passive voice, reorder "
            "clauses, and merge or split one sentence (±1 sentence count change).\n"
            + _PARA_CONSTRAINTS_GENERIC
        ),
        "user_template_generic": (
            "Apply CLAUSE RESTRUCTURING to the text below.\n"
            "Restructure clauses: switch active/passive voice, reorder clauses. "
            "You may merge or split one sentence. "
            "Keep the same entities, objects, and setting.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Switch active to passive voice or vice versa (e.g., 'He gave 3 to her' → '3 were given to her by him').",
            "Reorder clauses within sentences (e.g., move condition before action).",
            "Convert noun phrases to verb constructions (e.g., 'the payment of $300' → 'paying $300').",
            "Compress two short clauses into one, or expand a complex clause into two parts.",
        ],
    },

    # ── L4: Sentence Restructuring ─────────────────────────────────────────
    # Must change sentence boundaries (merge/split). Same entities and setting.
    "L4": {
        "temperature": 0.60,
        "max_sentence_delta": None,   # no sentence count constraint
        "style_modes": ["textbook", "conversational", "concise", "narrative"],
        "intensity": "sentence restructuring — merge, split, reorder sentences freely",
        "system": (
            "You rewrite math word problems by restructuring the entire text. "
            "You may freely merge, split, and reorder sentences. Change how "
            "information is organised and presented. You MUST produce noticeably "
            "different sentence boundaries from the original. The rewritten version "
            "must describe the exact same scenario with the same characters and "
            "objects, requiring the same computation.\n"
            + _PARA_CONSTRAINTS
        ),
        "user_template": (
            "Apply SENTENCE RESTRUCTURING to the question below.\n"
            "Freely reorganise the text: merge sentences, split long ones, "
            "reorder information. You MUST change at least two sentence boundaries "
            "(merge or split). Change the narrative flow while preserving every "
            "fact, character, and relationship.\n"
            "{anchors_block}"
            "{validity_block}\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You rewrite input text by restructuring the entire text. "
            "You may freely merge, split, and reorder sentences. Change how "
            "information is organised and presented. You MUST produce noticeably "
            "different sentence boundaries from the original.\n"
            + _PARA_CONSTRAINTS_GENERIC
        ),
        "user_template_generic": (
            "Apply SENTENCE RESTRUCTURING to the text below.\n"
            "Freely reorganise: merge, split, reorder sentences. "
            "You MUST change at least two sentence boundaries (merge or split). "
            "Change the narrative flow while preserving every fact and entity.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Merge the setup sentences into one compound sentence and split the question differently.",
            "Combine multiple facts into compound sentences with conjunctions, then split differently.",
            "Restructure into a different paragraph shape — different sentence boundaries, same content.",
            "Turn short simple sentences into longer complex ones, or vice versa.",
        ],
    },

    # ── L5: Discourse Reorganization ──────────────────────────────────────
    # Change information order and rhetorical framing. Same entities and setting.
    "L5": {
        "temperature": 0.80,
        "max_sentence_delta": None,   # no sentence count constraint
        "style_modes": ["textbook", "conversational", "concise", "narrative"],
        "intensity": "discourse reorganization — change information order and rhetorical framing",
        "system": (
            "You rewrite math word problems by reorganizing the entire discourse "
            "structure. Change the order in which information is presented. Change "
            "the rhetorical framing — e.g., present the question before the context, "
            "reverse cause-and-effect order, or convert from narrative to "
            "instructional style. The rewritten version must describe the exact same "
            "scenario with the same characters, objects, and setting.\n"
            + _PARA_CONSTRAINTS
        ),
        "user_template": (
            "Apply DISCOURSE REORGANIZATION to the question below.\n"
            "Reorganize the entire text's information flow. You MUST change the "
            "order in which facts are presented compared to the original. Change "
            "how the problem is framed — the text should feel like a different "
            "author wrote about the same problem. Keep the same characters, "
            "objects, setting, and all facts.\n"
            "{anchors_block}"
            "{validity_block}\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You rewrite input text by reorganizing the entire discourse "
            "structure. Change the order in which information is presented. Change "
            "the rhetorical framing — e.g., present the main point before the "
            "context, reverse cause-and-effect order, or convert between narrative "
            "and instructional style.\n"
            + _PARA_CONSTRAINTS_GENERIC
        ),
        "user_template_generic": (
            "Apply DISCOURSE REORGANIZATION to the text below.\n"
            "Reorganize the entire text's information flow. You MUST change the "
            "order in which facts are presented. Change how the text is framed — "
            "it should feel like a different author wrote about the same topic. "
            "Keep the same entities, objects, setting, and all facts.\n\n"
            "TEXT:\n{text}\n"
        ),
        "style_hints": [
            "Present the question or main request first, then provide the supporting context.",
            "Reverse the order: start from the end state and work backwards to the setup.",
            "Convert from narrative style to instructional/given-find style, or vice versa.",
            "Reorganize so the most important detail comes first — different author, same content.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Per-level scoring criteria (used by Evaluator in Component #7)
# w_sim + w_tcr define the composite score weights.
# sim_min: minimum acceptable cosine similarity at this level.
# tcr_band: [min_expected_tcr, max_expected_tcr] — expected token-change ratio range.
# ---------------------------------------------------------------------------

_LEVEL_CRITERIA = [
    {"level": "L1", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.97, "tcr_band": [0.05, 0.15]}},
    {"level": "L2", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.94, "tcr_band": [0.15, 0.35]}},
    {"level": "L3", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.90, "tcr_band": [0.35, 0.60]}},
    {"level": "L4", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.86, "tcr_band": [0.60, 0.85]}},
    {"level": "L5", "criteria": {"w_sim": 0.7, "w_tcr": 0.3, "sim_min": 0.82, "tcr_band": [0.85, 0.95]}},
]

# ---------------------------------------------------------------------------
# Full package — pass to RegistryService.seed_metric_package()
# ---------------------------------------------------------------------------

PARAPHRASE_STRESS_PACKAGE: dict = {
    "dimension": {
        "name": "Stability",
        "description": "Stability under benign perturbations",
        "order_index": 1,
    },
    "categories": [
        {
            "name": "Semantic Variation",
            "description": "Sensitivity to semantically-equivalent rewrites",
            "order_index": 1,
        },
    ],
    "test_defs": [
        {
            "name": "Semantic Variation",
            "family": "paraphrase",
            "description": (
                "Generate paraphrased variants of the input at 5 intensity levels. "
                "L1-L2: synonym/phrase rewriting (fixed structure). "
                "L3: clause restructuring (±1 sentence). "
                "L4: sentence restructuring (free merge/split). "
                "L5: discourse reorganization (information reorder/reframe). "
                "Numbers and entity names must be preserved exactly at all levels."
            ),
            "generation_type": "agentic",
            "level_specs": _LEVEL_SPECS,
            "validator_rules": {
                "preserve_numbers": {"enabled": True},
            },
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "category_name": "Semantic Variation",
            "gate_config": {
                # Layer 1 — Structural (deterministic, no API call)
                "structural": {
                    "preserve_numbers":    True,   # all numeric tokens must survive unchanged
                    "preserve_word_count": False,  # word count may change
                    "preserve_word_set":   False,  # word identity may change (it's a rewrite)
                    "preserve_word_order": False,  # order may change within sentences
                },
                # Layer 2 — Level gate: TCR must fall inside the target level's tcr_band
                # (evaluated per-attempt; band thresholds come from _LEVEL_CRITERIA)
                "level_gate": {
                    "metric": "TCR",
                    "rule":   "in_band",          # tcr_band[0] <= TCR < tcr_band[1]
                },
                # Layer 3 — NLI validity gate (1 extra Groq/LLaMA call per attempt)
                "nli_gate": {
                    "pass_labels":   ["entailment"],   # generated must entail original
                    "sim_direction": "high",           # SIM should be high (meaning preserved)
                },
                # Score formula (applied after all gates pass)
                "score": {
                    "formula": "((1 - SIM) * w_sim + TCR * w_tcr) / (w_sim + w_tcr)",
                    "SIM":     "cosine(embed(generated_input), embed(original_input))",
                    "TCR":     "1 - SequenceMatcher(original_input, generated_input).ratio()",
                    "note":    "w_sim and w_tcr come from _LEVEL_CRITERIA per level.",
                },
                # Temperature schedule
                "temperature": {
                    "formula":  "0.1 * level_index",           # L1=0.10 … L5=0.50
                    "on_retry": "base_temp + 0.10 * (attempt - 1)",  # escalate each retry
                    "cap":      0.95,
                },
            },
        },
    ],
    "metric_defs": [
        {
            "name": "Similarity–Token Composite",
            "dimension": "Stability",
            "category": "Semantic Variation",
            "metric_type": "similarity_token_composite",
            "description": (
                "Composite score combining cosine-embedding similarity and token change ratio (TCR). "
                "Formula: Score = ((1-Trc)*w_tcr + sim*w_sim) / (w_sim+w_tcr)"
            ),
            "criteria_json": {
                "formula": "((1-Trc)*w_tcr + sim*w_sim) / (w_sim+w_tcr)",
                "note": "Per-level thresholds stored in metric_level_criteria table.",
            },
            "applicable_test_families": ["paraphrase"],
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "aggregation_rule": "mean",
            "weight": 0.25,
            "severity": "high",
            "ui_label": "Paraphrase Robustness Score",
            "ui_description": "How consistently the model responds to semantically equivalent inputs.",
            "category_name": "Semantic Variation",
            "level_criteria": [
                {
                    "test_family": "paraphrase",
                    "level": lc["level"],
                    "criteria": lc["criteria"],
                }
                for lc in _LEVEL_CRITERIA
            ],
        },
    ],
}
