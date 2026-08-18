# MVP seed definitions — 3 test families + 5 metrics
# Inserted by seed_defaults() only if not already present (idempotent)

DEFAULT_METRIC_DEFS = [
    # ── ROBUSTNESS ──────────────────────────────────────────────────────────
    {
        "name": "Format Invariance",
        "dimension": "robustness",
        "category": "format_robustness",
        "metric_type": "invariance_check",
        "criteria_json": {
            "description": "Output should remain consistent when input formatting changes.",
            "thresholds": {"pass": 0.90, "warn": 0.75, "fail": 0.0},
            "direction": "higher_is_better",
            "level_thresholds": {
                "L1": {"pass": 0.95},
                "L2": {"pass": 0.92},
                "L3": {"pass": 0.88},
                "L4": {"pass": 0.80},
                "L5": {"pass": 0.70},
            },
        },
        "applicable_output_types": ["number", "text", "label", "short_text"],
        "applicable_test_families": ["format"],
        "aggregation_rule": "weighted_mean",
        "weight": 1.0,
        "severity": "high",
        "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
        "ui_label": "Format Stability",
        "ui_description": (
            "Measures whether the AI gives the same answer when the question formatting changes "
            "(spacing, punctuation, capitalization). L1 = minor tweaks, L5 = heavy reformatting."
        ),
    },
    {
        "name": "Paraphrase Invariance",
        "dimension": "robustness",
        "category": "semantic_robustness",
        "metric_type": "semantic_similarity",
        "criteria_json": {
            "description": "Output semantic meaning should remain consistent when input is paraphrased.",
            "thresholds": {"pass": 0.85, "warn": 0.70, "fail": 0.0},
            "direction": "higher_is_better",
            "similarity_model": "cosine_embedding",
            "level_thresholds": {
                "L1": {"pass": 0.92},
                "L2": {"pass": 0.88},
                "L3": {"pass": 0.82},
                "L4": {"pass": 0.75},
                "L5": {"pass": 0.65},
            },
        },
        "applicable_output_types": ["text", "short_text", "long_text"],
        "applicable_test_families": ["paraphrase"],
        "aggregation_rule": "mean",
        "weight": 1.0,
        "severity": "high",
        "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
        "ui_label": "Paraphrase Stability",
        "ui_description": (
            "Checks if the AI's meaning stays consistent when the question is reworded. "
            "L1 = simple synonym swap, L5 = heavy rewrite with same intent."
        ),
    },
    {
        "name": "Distraction Resistance",
        "dimension": "robustness",
        "category": "noise_robustness",
        "metric_type": "invariance_check",
        "criteria_json": {
            "description": "Output should not change significantly when irrelevant content is injected.",
            "thresholds": {"pass": 0.85, "warn": 0.70, "fail": 0.0},
            "direction": "higher_is_better",
            "level_thresholds": {
                "L1": {"pass": 0.92},
                "L2": {"pass": 0.88},
                "L3": {"pass": 0.80},
                "L4": {"pass": 0.72},
                "L5": {"pass": 0.60},
            },
        },
        "applicable_output_types": ["number", "text", "label", "short_text"],
        "applicable_test_families": ["distraction"],
        "aggregation_rule": "worst_k",
        "weight": 1.2,
        "severity": "high",
        "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
        "ui_label": "Distraction Resistance",
        "ui_description": (
            "Tests if the AI can ignore irrelevant or misleading context added to the question. "
            "L1 = unrelated sentence, L5 = adversarial misleading content."
        ),
    },
    # ── CONSISTENCY ─────────────────────────────────────────────────────────
    {
        "name": "Run-to-Run Consistency",
        "dimension": "consistency",
        "category": "stability",
        "metric_type": "variance",
        "criteria_json": {
            "description": "Same input run multiple times should yield consistent outputs.",
            "thresholds": {"max_variance": 0.10, "pass_rate_min": 0.90},
            "direction": "lower_variance_is_better",
        },
        "applicable_output_types": ["number", "text", "label", "short_text"],
        "applicable_test_families": ["format", "paraphrase", "distraction"],
        "aggregation_rule": "mean",
        "weight": 0.8,
        "severity": "medium",
        "profile_modes": ["builder", "certification"],
        "ui_label": "Answer Consistency",
        "ui_description": (
            "Runs the same question multiple times and checks if the AI gives consistent answers. "
            "High variance means unpredictable behavior."
        ),
    },
    # ── SENSITIVITY ─────────────────────────────────────────────────────────
    {
        "name": "Level Sensitivity Score",
        "dimension": "sensitivity",
        "category": "degradation",
        "metric_type": "threshold_rate",
        "criteria_json": {
            "description": "Tracks at which perturbation level (L1–L5) behavior degrades. Detects breakpoints.",
            "pass_rate_by_level": {
                "L1": 0.95,
                "L2": 0.90,
                "L3": 0.80,
                "L4": 0.65,
                "L5": 0.50,
            },
            "breakpoint_detection": True,
            "direction": "higher_is_better",
        },
        "applicable_output_types": ["number", "text", "label", "short_text"],
        "applicable_test_families": ["format", "paraphrase", "distraction"],
        "aggregation_rule": "auc",
        "weight": 1.0,
        "severity": "high",
        "profile_modes": ["builder", "inspection", "certification"],
        "ui_label": "Sensitivity Curve",
        "ui_description": (
            "Shows how the AI's performance degrades as tests get harder (L1→L5). "
            "A steep drop means the AI is fragile to small changes."
        ),
    },
    # ── SEMANTIC JUDGE ───────────────────────────────────────────────────────
    {
        "id": "reliability_semantic_judge",
        "name": "Semantic Response Quality",
        "dimension": "Reliability",
        "category": "reliability",
        "metric_type": "semantic_judge",
        "criteria_json": {
            "rubrics": [],           # empty — falls back to cosine/lexical similarity
            "score_threshold": 0.5,
            "judge_model": "gpt-4o-mini",
        },
        "applicable_output_types": [
            "short_text", "long_text", "text", "number", "label",
        ],
        "applicable_test_families": [
            "paraphrase", "noise", "format",
            "distraction", "conflict", "context",
        ],
        "aggregation_rule": "mean",
        "weight": 1.0,
        "severity": "medium",
        "is_active": 1,
        "profile_modes": ["builder", "inspection", "certification"],
        "ui_label": "Semantic Response Quality",
        "ui_description": (
            "LLM-as-judge evaluator for open-ended responses. "
            "Scores are rubric-weighted when rubrics are provided; "
            "falls back to embedding cosine similarity or lexical similarity."
        ),
    },
]

DEFAULT_TEST_DEFS = [
    {
        "name": "Format Perturbation",
        "family": "format",
        "description": (
            "Tests AI robustness by modifying input formatting (whitespace, case, punctuation, "
            "number formatting) while preserving semantic content and all numeric values."
        ),
        "generation_type": "transform",
        "level_specs": {
            "L1": {
                "description": "Minor whitespace and punctuation changes",
                "transforms": ["strip_extra_spaces", "remove_trailing_punctuation"],
                "temperature": 0.0,
                "sim_band": [0.95, 1.0],
                "trc_band": [0.0, 0.05],
            },
            "L2": {
                "description": "Case normalization and abbreviation substitution",
                "transforms": ["lowercase", "expand_abbreviations"],
                "temperature": 0.0,
                "sim_band": [0.90, 0.98],
                "trc_band": [0.0, 0.10],
            },
            "L3": {
                "description": "Number formatting and unit notation changes",
                "transforms": ["format_numbers", "unit_variations"],
                "temperature": 0.1,
                "sim_band": [0.85, 0.95],
                "trc_band": [0.05, 0.20],
            },
            "L4": {
                "description": "Mixed format distortions",
                "transforms": ["mixed_case", "random_spacing", "symbol_substitution"],
                "temperature": 0.2,
                "sim_band": [0.78, 0.92],
                "trc_band": [0.10, 0.30],
            },
            "L5": {
                "description": "Heavy format distortion while preserving meaning",
                "transforms": ["heavy_restructure", "all_caps_words", "compressed_format"],
                "temperature": 0.3,
                "sim_band": [0.70, 0.88],
                "trc_band": [0.20, 0.40],
            },
        },
        "validator_rules": {
            "preserve_numbers": True,
            "preserve_entities": True,
            "max_semantic_shift": 0.05,
            "must_be_answerable": True,
        },
        "applicable_output_types": ["number", "text", "label", "short_text"],
        "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
    },
    {
        "name": "Paraphrase Robustness",
        "family": "paraphrase",
        "description": (
            "Tests AI robustness by paraphrasing the input question at increasing semantic distance "
            "while preserving the intended meaning and all numeric values."
        ),
        "generation_type": "agentic",
        "level_specs": {
            "L1": {
                "description": "Near-synonym substitution — one or two word swaps",
                "paraphrase_distance": "minimal",
                "temperature": 0.3,
                "sim_band": [0.92, 1.0],
                "trc_band": [0.0, 0.10],
            },
            "L2": {
                "description": "Light paraphrase — phrase reordering and synonym clusters",
                "paraphrase_distance": "light",
                "temperature": 0.5,
                "sim_band": [0.85, 0.95],
                "trc_band": [0.05, 0.20],
            },
            "L3": {
                "description": "Moderate paraphrase — sentence restructuring, voice changes",
                "paraphrase_distance": "moderate",
                "temperature": 0.6,
                "sim_band": [0.78, 0.90],
                "trc_band": [0.10, 0.30],
            },
            "L4": {
                "description": "Strong paraphrase — full sentence rewrite, different style",
                "paraphrase_distance": "strong",
                "temperature": 0.7,
                "sim_band": [0.70, 0.85],
                "trc_band": [0.20, 0.50],
            },
            "L5": {
                "description": "Heavy paraphrase — completely different phrasing, same intent",
                "paraphrase_distance": "heavy",
                "temperature": 0.8,
                "sim_band": [0.60, 0.80],
                "trc_band": [0.30, 0.70],
            },
        },
        "validator_rules": {
            "preserve_intent": True,
            "preserve_numbers": True,
            "min_similarity": 0.60,
            "must_be_answerable": True,
            "no_answer_leakage": True,
        },
        "applicable_output_types": ["text", "short_text", "long_text", "number"],
        "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
    },
    {
        "name": "Distraction Injection",
        "family": "distraction",
        "description": (
            "Tests AI robustness by injecting irrelevant, distracting, or misleading content "
            "into the input to see if the model stays on task."
        ),
        "generation_type": "agentic",
        "level_specs": {
            "L1": {
                "description": "Add a single unrelated sentence at the end",
                "injection_type": "irrelevant_append",
                "injection_count": 1,
                "relatedness": "none",
                "temperature": 0.4,
                "sim_band": [0.90, 1.0],
                "trc_band": [0.10, 0.30],
            },
            "L2": {
                "description": "Insert a mildly related but misleading sentence",
                "injection_type": "mildly_related",
                "injection_count": 1,
                "relatedness": "low",
                "temperature": 0.5,
                "sim_band": [0.85, 0.95],
                "trc_band": [0.15, 0.40],
            },
            "L3": {
                "description": "Inject a moderately confusing distractor referencing the topic",
                "injection_type": "topic_related_confusion",
                "injection_count": 2,
                "relatedness": "medium",
                "temperature": 0.6,
                "sim_band": [0.75, 0.90],
                "trc_band": [0.20, 0.50],
            },
            "L4": {
                "description": "Inject strongly misleading content with plausible-sounding wrong framing",
                "injection_type": "misleading_context",
                "injection_count": 2,
                "relatedness": "high",
                "temperature": 0.7,
                "sim_band": [0.65, 0.85],
                "trc_band": [0.25, 0.60],
            },
            "L5": {
                "description": "Adversarial distraction — content designed to hijack reasoning",
                "injection_type": "adversarial",
                "injection_count": 3,
                "relatedness": "adversarial",
                "temperature": 0.8,
                "sim_band": [0.55, 0.80],
                "trc_band": [0.30, 0.70],
            },
        },
        "validator_rules": {
            "preserve_original_question": True,
            "injection_must_be_distinct": True,
            "no_answer_in_injection": True,
            "must_be_answerable_without_injection": True,
        },
        "applicable_output_types": ["number", "text", "label", "short_text"],
        "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
    },
]
