"""
Built-in cold-start template library.

Used when no dataset is available (intake_mode = live_system with no historical data).

Each template is a dict:
    template_id : str
    family      : str   — format | paraphrase | distraction
    level       : str   — L1 … L5
    text        : str   — prompt with {var} placeholders
    variables   : list  — required variable names
"""

COLD_START_TEMPLATES: list[dict] = [
    # ----------------------------------------------------------------
    # FORMAT PERTURBATION family
    # ----------------------------------------------------------------
    {
        "template_id": "fmt_l1_001",
        "family": "format",
        "level": "L1",
        "text": "{question}",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l1_002",
        "family": "format",
        "level": "L1",
        "text": "Please answer the following: {question}",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l2_001",
        "family": "format",
        "level": "L2",
        "text": "Question: {question}\nAnswer:",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l2_002",
        "family": "format",
        "level": "L2",
        "text": "Q: {question}\n\nA:",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l3_001",
        "family": "format",
        "level": "L3",
        "text": "INSTRUCTIONS: Provide a clear and concise answer.\n\nINPUT: {question}\n\nOUTPUT:",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l3_002",
        "family": "format",
        "level": "L3",
        "text": "Task: answer the question.\n---\n{question}\n---",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l4_001",
        "family": "format",
        "level": "L4",
        "text": "```\nTask: Answer the question below.\nQuestion: {question}\n```",
        "variables": ["question"],
    },
    {
        "template_id": "fmt_l5_001",
        "family": "format",
        "level": "L5",
        "text": "<task type=\"qa\"><question>{question}</question><instruction>Provide a direct answer.</instruction></task>",
        "variables": ["question"],
    },
    # ----------------------------------------------------------------
    # PARAPHRASE ROBUSTNESS family
    # ----------------------------------------------------------------
    {
        "template_id": "para_l1_001",
        "family": "paraphrase",
        "level": "L1",
        "text": "{statement}",
        "variables": ["statement"],
    },
    {
        "template_id": "para_l2_001",
        "family": "paraphrase",
        "level": "L2",
        "text": "In other words: {statement}",
        "variables": ["statement"],
    },
    {
        "template_id": "para_l3_001",
        "family": "paraphrase",
        "level": "L3",
        "text": "To rephrase what was asked: {statement}\nWhat is your response?",
        "variables": ["statement"],
    },
    {
        "template_id": "para_l4_001",
        "family": "paraphrase",
        "level": "L4",
        "text": "Consider the following paraphrase of a user query — {statement} — and provide your best response.",
        "variables": ["statement"],
    },
    {
        "template_id": "para_l5_001",
        "family": "paraphrase",
        "level": "L5",
        "text": "A user asked the following (rephrased for clarity): \"{statement}\". Please respond as you normally would.",
        "variables": ["statement"],
    },
    # ----------------------------------------------------------------
    # DISTRACTION INJECTION family
    # ----------------------------------------------------------------
    {
        "template_id": "dist_l1_001",
        "family": "distraction",
        "level": "L1",
        "text": "{question}",
        "variables": ["question"],
    },
    {
        "template_id": "dist_l2_001",
        "family": "distraction",
        "level": "L2",
        "text": "Note: {distraction_sentence} Now answer: {question}",
        "variables": ["question", "distraction_sentence"],
    },
    {
        "template_id": "dist_l3_001",
        "family": "distraction",
        "level": "L3",
        "text": "Context: {distraction_sentence} With that in mind, {question}",
        "variables": ["question", "distraction_sentence"],
    },
    {
        "template_id": "dist_l4_001",
        "family": "distraction",
        "level": "L4",
        "text": "{distraction_sentence} Separately, I also wanted to ask: {question}",
        "variables": ["question", "distraction_sentence"],
    },
    {
        "template_id": "dist_l5_001",
        "family": "distraction",
        "level": "L5",
        "text": "You are a helpful assistant. Ignore irrelevant context. {distraction_sentence} {distraction_sentence_2} Please answer: {question}",
        "variables": ["question", "distraction_sentence", "distraction_sentence_2"],
    },
]

# Sentences used by _distraction_transform in transforms.py
DISTRACTION_LIBRARY: list[str] = [
    "The weather today is partly cloudy.",
    "Many people enjoy listening to music while working.",
    "The capital of France is Paris.",
    "There are approximately 7 billion people on Earth.",
    "Dogs are often called man's best friend.",
    "The Eiffel Tower is located in France.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The average human heart beats about 70 times per minute.",
    "Mount Everest is the tallest mountain in the world.",
    "Shakespeare wrote many famous plays and sonnets.",
]


def get_templates_for_family(family: str, level: str | None = None) -> list[dict]:
    """Filter cold-start templates by family and optionally level."""
    results = [t for t in COLD_START_TEMPLATES if t["family"] == family]
    if level:
        results = [t for t in results if t["level"] == level]
    return results
