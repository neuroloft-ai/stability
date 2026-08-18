"""
Seed data: Structural/Format (Stability dimension)

Metric #3 in the Stability dimension.
Uses agentic generation via F_SPECS prompts.
Each level enforces a distinct layout disruption pattern.
STRICT rule: words and numbers are NEVER changed, only whitespace/symbols/wrappers.

Load via: RegistryService.seed_metric_package(FORMAT_STRESS_PACKAGE)
"""

# ---------------------------------------------------------------------------
# Level specs (F_SPECS)
# Each level defines:
#   temperature           : LLM sampling temperature (0.1 × level_index)
#   intensity             : human-readable description of the disruption level
#   system                : math-specific system prompt (uses {question})
#   user_template         : math-specific user turn template (uses {question})
#   system_generic        : domain-agnostic system prompt (uses {text})
#   user_template_generic : domain-agnostic user turn template (uses {text})
#   style_hints           : guidance list to help the model diversify outputs
# ---------------------------------------------------------------------------

_LEVEL_SPECS = {
    # ------------------------------------------------------------------
    # F1: whitespace only — blank lines, simple paragraph split
    # ------------------------------------------------------------------
    "L1": {
        "temperature": 0.30,
        "max_tokens":  300,
        "forbidden_chars": "#->*|`",
        "intensity":   "minimal — whitespace only: blank lines, simple paragraph split",
        "pattern_targets": {
            "line_break_target": {"min": 1, "max": 3},
            "tab_target":        {"min": 1, "max": 2},
            "blank_line_target": {"min": 0, "max": 1},
            "wrapper_mode":      "none",
            "separator_mode":    "light",
        },
        "system": (
            "You ONLY change the formatting/layout of math word problems.\n"
            "STRICT constraints:\n"
            "- Do NOT change meaning.\n"
            "- Do NOT add or remove any numbers.\n"
            "- Do NOT add new facts, examples, or irrelevant story.\n"
            "- Keep the same question content, but adjust formatting minimally.\n"
            "- Output ONLY the formatted question text. No commentary."
        ),
        "user_template": (
            "Apply VERY LIGHT formatting changes only.\n"
            "Examples: tiny punctuation spacing, one line break, very small layout tweaks.\n"
            "Do not add any new text besides formatting characters/whitespace.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You reformat text using ONLY whitespace changes.\n"
            "STRICT constraints:\n"
            "- Do NOT change, add, or remove ANY words.\n"
            "- Do NOT change, add, or remove ANY numbers.\n"
            "- Do NOT reorder words.\n"
            "- Do NOT use any markdown symbols (#, -, >, *, |, `).\n"
            "- Do NOT invent labels, titles, or section names.\n"
            "- You may ONLY add/remove spaces, tabs, newlines, and blank lines.\n"
            "- Output ONLY the modified text. No commentary."
        ),
        "user_template_generic": (
            "Reformat using whitespace only. You MUST make at least one visible change.\n"
            "Add blank lines between sentences or split into paragraphs.\n"
            "NOT allowed: any markdown symbols, labels, titles, new words.\n\n"
            "TEXT:\n{text}"
        ),
        "style_hints": [
            "Split into two paragraphs with a blank line.",
            "Add indentation at the start of each sentence.",
            "Insert a blank line before the question sentence.",
            "Add extra spacing after periods.",
        ],
    },

    # ------------------------------------------------------------------
    # F2: light formatting — bullets, one neutral heading, blockquote, code block
    # ------------------------------------------------------------------
    "L2": {
        "temperature": 0.35,
        "max_tokens":  400,
        "intensity":   "light — bullets, one neutral heading, blockquote, or code block",
        "pattern_targets": {
            "line_break_target":    {"min": 8, "max": 20},
            "indent_pattern":       "staircase",
            "mid_sentence_breaks":  True,
            "blank_line_target":    {"min": 1, "max": 4},
            "wrapper_mode":         "none",
            "separator_mode":       "medium",
        },
        "system": (
            "You ONLY change the formatting/layout of math word problems.\n"
            "STRICT constraints:\n"
            "- Do NOT change meaning.\n"
            "- Do NOT add or remove any numbers.\n"
            "- Do NOT add new facts or extra story.\n"
            "- Output ONLY the formatted question text. No commentary."
        ),
        "user_template": (
            "Apply LIGHT-to-MODERATE formatting changes.\n"
            "Allowed: line breaks, indentation, markdown symbols (#, -, >, *).\n"
            "Do NOT add any new words, labels, or headings with new text.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You reformat text using light markdown.\n"
            "STRICT constraints:\n"
            "- Do NOT change, add, or remove ANY words.\n"
            "- Do NOT change, add, or remove ANY numbers.\n"
            "- Do NOT reorder words.\n"
            "- Do NOT invent labels, titles, or section names.\n"
            "- You MAY use ONE of: a # heading (using words from the text), "
            "bullet points (- or *), a blockquote (>), or a fenced code block (```).\n"
            "- Headings must use existing words from the text, not new titles.\n"
            "- Output ONLY the modified text. No commentary."
        ),
        "user_template_generic": (
            "Reformat with light markdown.\n"
            "Pick ONE: put the first sentence as a # heading, convert sentences to "
            "bullets, wrap in a > blockquote, or put in a ``` code block.\n"
            "Do NOT invent labels or titles. Use only words from the text.\n\n"
            "TEXT:\n{text}"
        ),
        "style_hints": [
            "Put the first sentence as a # heading, rest as body.",
            "Convert sentences into a bullet list.",
            "Wrap the whole text in a > blockquote.",
            "Put the text inside a fenced ``` block.",
        ],
    },

    # ------------------------------------------------------------------
    # F3: document-style — markdown note, indented structure, nested bullets, clean snippet
    # ------------------------------------------------------------------
    "L3": {
        "temperature": 0.30,
        "max_tokens":  500,
        "intensity":   "moderate — markdown note, indented structure, nested bullets, clean copied snippet",
        "pattern_targets": {
            "wrapper_mode":          "markdown_structural",
            "require_header":        True,
            "require_nested_bullets": True,
            "require_blockquote":    True,
            "line_break_target":     {"min": 12, "max": 30},
            "fragmentation_mode":    "awkward_fragments",
        },
        "system": (
            "You ONLY change the formatting/layout of math word problems.\n"
            "STRICT constraints:\n"
            "- Do NOT change meaning.\n"
            "- Do NOT add or remove any numbers.\n"
            "- Do NOT add new facts or extra story.\n"
            "- You MAY reorder lines/paragraphs but must keep exact same wording.\n"
            "- Output ONLY the formatted question text. No commentary."
        ),
        "user_template": (
            "Apply STRONG formatting changes.\n"
            "Allowed: many line breaks, bullets (- *), blockquotes (>), code fences, extra spacing.\n"
            "Do NOT add any new words. Keep the exact same words in the same order.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You reformat text as a clean document-style layout.\n"
            "STRICT constraints:\n"
            "- Do NOT change, add, or remove ANY words.\n"
            "- Do NOT change, add, or remove ANY numbers.\n"
            "- Do NOT reorder words.\n"
            "- Do NOT invent labels, titles, or section names.\n"
            "- You MAY use: nested bullets, blockquotes, code blocks, indented structure.\n"
            "- Headings must use existing words from the text, not invented titles.\n"
            "- Output ONLY the modified text. No commentary."
        ),
        "user_template_generic": (
            "Reformat as a clean document or markdown note.\n"
            "Use nested bullets, indented structure, or a clean copied-snippet style.\n"
            "Do NOT invent labels or titles. Every word must come from the text.\n\n"
            "TEXT:\n{text}"
        ),
        "style_hints": [
            "Format as a markdown note with nested bullets.",
            "Use indented structure with blockquotes.",
            "Present as a clean copied snippet from a document.",
            "Combine bullets and a code block.",
        ],
    },

    # ------------------------------------------------------------------
    # F4: strong layout stress — hard wraps, quote-block wraps, narrow-column, simple ASCII
    # ------------------------------------------------------------------
    "L4": {
        "temperature": 0.40,
        "max_tokens":  600,
        "intensity":   "strong — hard wraps, quote-block wraps, narrow-column formatting, simple ASCII structure",
        "pattern_targets": {
            "wrapper_mode":      "table_or_form",
            "require_table":     True,
            "table_cols":        {"min": 5, "max": 9},
            "separator_mode":    "heavy",
            "line_break_target": {"min": 18, "max": 45},
            "indent_pattern":    "heavy",
        },
        "system": (
            "You ONLY change the formatting/layout of math word problems.\n"
            "STRICT constraints:\n"
            "- Do NOT change meaning.\n"
            "- Do NOT add or remove any numbers.\n"
            "- Do NOT add new facts or extra story.\n"
            "- Keep wording nearly identical.\n"
            "- Output ONLY the formatted question text. No commentary."
        ),
        "user_template": (
            "Apply VERY STRONG formatting perturbation.\n"
            "Make it hard to read using line breaks, spacing, and markdown symbols.\n"
            "Do NOT add any new words. Keep the exact same words in the same order.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You reformat text with strong layout stress.\n"
            "STRICT constraints:\n"
            "- Do NOT change, add, or remove ANY words.\n"
            "- Do NOT change, add, or remove ANY numbers.\n"
            "- Do NOT reorder words.\n"
            "- Do NOT invent labels, titles, or section names.\n"
            "- You MAY use: hard line wrapping (40-60 chars), quote-block wrapping (>), "
            "narrow-column formatting, simple ASCII dividers (--- or |).\n"
            "- Keep at least 3-5 words per line. Do NOT put one word per line.\n"
            "- Do NOT produce empty lines with only formatting symbols.\n"
            "- Do NOT repeat the text multiple times.\n"
            "- Output ONLY the modified text. No commentary."
        ),
        "user_template_generic": (
            "Reformat with strong layout stress.\n"
            "Use hard wraps at narrow column width, quote-block wrapping, "
            "or simple ASCII structure with dividers.\n"
            "Keep 3-5+ words per line. Do NOT invent labels or titles.\n"
            "Every word must come from the original text.\n\n"
            "TEXT:\n{text}"
        ),
        "style_hints": [
            "Hard-wrap at 40 characters like a narrow terminal.",
            "Use > prefix with short wrapped lines.",
            "Format with simple | dividers between chunks.",
            "Indent unevenly like pasted from a narrow window.",
        ],
    },

    # ------------------------------------------------------------------
    # F5: extreme but realistic — OCR wrap, markdown export, docs excerpt, ASCII table
    # ------------------------------------------------------------------
    "L5": {
        "temperature": 0.50,
        "max_tokens":  800,
        "intensity":   "extreme — OCR-like wrap, markdown export, documentation excerpt, ASCII table presentation",
        "pattern_targets": {
            "wrapper_mode":         "nested_markdown",
            "require_code_fence":   True,
            "require_blockquote":   True,
            "require_table":        True,
            "require_bullets":      True,
            "split_around_numbers": True,
            "indent_pattern":       "alternating",
            "separator_mode":       "max",
            "line_break_target":    {"min": 35, "max": 90},
        },
        "system": (
            "You ONLY change the formatting/layout of math word problems.\n"
            "STRICT constraints:\n"
            "- Do NOT change meaning.\n"
            "- Do NOT add or remove any numbers.\n"
            "- Do NOT add new facts or extra story.\n"
            "- Keep wording the same but make formatting EXTREME.\n"
            "- Output ONLY the formatted question text. No commentary."
        ),
        "user_template": (
            "Apply EXTREME formatting perturbation.\n"
            "Use maximum line breaks, markdown wrappers, spacing, and segmentation.\n"
            "Do NOT add any new words. Keep the exact same words in the same order.\n\n"
            "QUESTION:\n{question}\n"
        ),
        "system_generic": (
            "You reformat text as if it came from a realistic heavily-formatted source.\n"
            "STRICT constraints:\n"
            "- Do NOT change, add, or remove ANY words.\n"
            "- Do NOT change, add, or remove ANY numbers.\n"
            "- Do NOT reorder words.\n"
            "- Do NOT invent labels, titles, or section names.\n"
            "- You MAY use: OCR-like odd line wraps, markdown-export formatting, "
            "documentation excerpt style, or ASCII table with dividers.\n"
            "- Keep at least 3-5 words per line. Do NOT put one word per line.\n"
            "- Do NOT produce empty lines with only # or > symbols.\n"
            "- Do NOT repeat the text. Output it ONCE.\n"
            "- Do NOT split individual characters onto separate lines.\n"
            "- Output ONLY the modified text. No commentary."
        ),
        "user_template_generic": (
            "Reformat as if copied from a real-world heavily-formatted source.\n"
            "Pick a style: OCR-scanned text with odd line wraps, a markdown export, "
            "a documentation excerpt, or an ASCII table with dividers.\n"
            "Keep it READABLE. Output the text ONCE with 3-5+ words per line.\n"
            "Do NOT invent labels or titles. Every word must come from the text.\n\n"
            "TEXT:\n{text}"
        ),
        "style_hints": [
            "Mimic OCR output — odd line breaks mid-sentence, uneven spacing.",
            "Format like a markdown export with nested blockquotes and bullets.",
            "Present as a documentation excerpt with code blocks and indentation.",
            "Lay out in a simple ASCII table with --- dividers.",
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
    {"level": "L1", "criteria": {"w_sim": 0.3, "w_tcr": 0.7, "sim_min": 0.95, "tcr_band": [0.01, 0.06], "invariance_min": 0.95}},
    {"level": "L2", "criteria": {"w_sim": 0.3, "w_tcr": 0.7, "sim_min": 0.90, "tcr_band": [0.04, 0.10], "invariance_min": 0.90}},
    {"level": "L3", "criteria": {"w_sim": 0.3, "w_tcr": 0.7, "sim_min": 0.85, "tcr_band": [0.08, 0.15], "invariance_min": 0.85}},
    {"level": "L4", "criteria": {"w_sim": 0.3, "w_tcr": 0.7, "sim_min": 0.78, "tcr_band": [0.12, 0.22], "invariance_min": 0.80}},
    {"level": "L5", "criteria": {"w_sim": 0.3, "w_tcr": 0.7, "sim_min": 0.70, "tcr_band": [0.18, 0.35], "invariance_min": 0.75}},
]

# ---------------------------------------------------------------------------
# Full package — pass to RegistryService.seed_metric_package()
# ---------------------------------------------------------------------------

FORMAT_STRESS_PACKAGE: dict = {
    "dimension": {
        "name": "Stability",
        "description": "Stability under benign perturbations",
        "order_index": 1,
    },
    "categories": [
        {
            "name": "Structural/Format",
            "description": (
                "Robustness to formatting/layout-only perturbations "
                "(no word/number change, no reorder)"
            ),
            "order_index": 3,
        },
    ],
    "test_defs": [
        {
            "name": "Structural/Format",
            "family": "format",
            "description": (
                "Reformat the input at 5 intensity levels without changing any words or numbers. "
                "F1=whitespace only, F2=light formatting, F3=document-style, "
                "F4=strong layout stress, F5=extreme realistic rendering stress. "
                "Generation is agentic — LLM applies layout transforms guided by F_SPECS prompts."
            ),
            "generation_type":     "agentic",
            "generation_strategy": "per_field",    # one API call per field; plain text, no JSON
            "level_specs": _LEVEL_SPECS,
            "validator_rules": {
                "preserve_numbers":        {"enabled": True},
                "preserve_word_order":     {"enabled": True},
                "no_extra_words":          {"enabled": True},
                "max_length_ratio":        {"enabled": True, "max_ratio": 10},
                "allow_whitespace_changes": {"enabled": True},
                "allow_harmless_symbols":   {"enabled": True},
            },
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "category_name": "Structural/Format",
            "gate_config": {
                # Style hints go in system prompt to prevent LLM from echoing them
                "hint_placement": "system",
                # Layer 1 — Structural (deterministic, no API call)
                "structural": {
                    "preserve_numbers":    True,   # all numeric tokens must survive unchanged
                    "preserve_word_count": False,  # whitespace/symbols added — word count may change
                    "preserve_word_set":   False,  # markdown symbols added — word set may change
                    "preserve_word_order": True,   # word order must be preserved
                },
                # Layer 2 — Level gate: TCR must fall inside the target level's tcr_band
                # (evaluated per-attempt; band thresholds come from _LEVEL_CRITERIA)
                "level_gate": {
                    "metric": "TCR",
                    "rule":   "in_band",           # tcr_band[0] <= TCR < tcr_band[1]
                },
                # Layer 3 — NLI validity gate
                # Disabled for Format: preserve_word_order already guarantees meaning
                # preservation, and NLI is unreliable on heavily formatted text.
                "nli_gate": {
                    "enabled":       False,
                    "pass_labels":   ["entailment"],
                    "sim_direction": "high",
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
            "name": "Answer Invariance — Format",
            "dimension": "Stability",
            "category": "Structural/Format",
            "metric_type": "answer_invariance",
            "description": (
                "Whether the system output remains consistent under format/layout-only "
                "perturbations (compared to baseline / seed output). "
                "Words and numbers in the input are identical — only layout changes."
            ),
            "criteria_json": {
                "note": "Per-level invariance thresholds stored in metric_level_criteria table.",
                "comparison": "output vs baseline_output",
                "match_method": "exact_or_semantic",
            },
            "applicable_test_families": ["format"],
            "applicable_output_types": ["number", "text"],
            "profile_modes": ["builder", "inspection", "certification", "enterprise_low_code"],
            "aggregation_rule": "mean",
            "weight": 0.25,
            "severity": "high",
            "ui_label": "Format Invariance Score",
            "ui_description": "How consistently the model responds when only the layout of the input changes.",
            "category_name": "Structural/Format",
            "level_criteria": [
                {
                    "test_family": "format",
                    "level": lc["level"],
                    "criteria": lc["criteria"],
                }
                for lc in _LEVEL_CRITERIA
            ],
        },
    ],
}
