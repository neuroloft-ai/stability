from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class GeneratorConfig:
    max_attempts_per_case: int = 3       # used in standalone generate_test_case()
    max_pool_attempts: int = 3           # pool-level attempts in _generate_for_plan()
    temperature: float = 0.7             # base temperature (escalates each pool attempt)
    temperature_step: float = 0.10       # increment per pool attempt
    model: str = "gpt-4o-mini"
    fallback_to_transform: bool = True   # fall back to transform if agentic unavailable
    openai_client: Optional[Any] = None  # injected pre-built client
    system_description: str = ""          # optional: what the system under test does (user-provided)


@dataclass
class TestSuite:
    suite_id: str
    plan_id: str
    system_id: str
    snapshot_id: str
    total_cases: int
    status: str                           # complete | incomplete
    case_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created_at: Optional[str] = None


@dataclass
class TestCase:
    case_id: str
    suite_id: str
    plan_id: str
    test_id: str
    family: str
    level: str                            # L1 … L5
    seed_ref: dict                        # {seed_id, text (truncated)}
    input_payload: dict                   # {text, generation_type, level, …}
    generator_meta: dict                  # {generation_type, attempt_count, accepted, failure_history}
    validator_results: dict               # {validation_id, overall_status, failure_reasons}
    level_classified: Optional[str] = None  # level assigned by sim/TCR classification (may differ from level)
    expected: Optional[dict] = None
    tags: Optional[dict] = None
    created_at: Optional[str] = None
