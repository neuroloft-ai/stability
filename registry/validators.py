import json
from .errors import RegistryValidationError

VALID_METRIC_TYPES = {
    "exact_match",
    "contains_match",
    "invariance_check",
    "semantic_similarity",
    "schema_valid",
    "variance",
    "threshold_rate",
    "conflict_resistance",
    "token_change_rate",
    "recall_accuracy",
    "behavioral_compliance",
    "representation_fidelity",
}

VALID_AGGREGATION_RULES = {"mean", "weighted_mean", "worst_k", "percentile", "auc"}
VALID_SEVERITIES = {"low", "medium", "high", "critical", "info"}
VALID_GENERATION_TYPES = {"transform", "template", "agentic"}
VALID_PROFILE_MODES = {"builder", "inspection", "certification", "enterprise_low_code"}
VALID_OUTPUT_TYPES = {"number", "text", "json", "label", "short_text", "long_text", "tool_action"}
VALID_TEST_FAMILIES = {"format", "paraphrase", "distraction", "injection", "consistency", "tool_misuse",
                       "noise", "conflict", "context", "kb", "ri"}

METRIC_REQUIRED_FIELDS = [
    "name", "dimension", "metric_type", "criteria_json",
    "applicable_output_types", "applicable_test_families",
    "aggregation_rule", "weight", "severity", "profile_modes",
]

TEST_REQUIRED_FIELDS = [
    "name", "family", "generation_type", "level_specs",
    "validator_rules", "applicable_output_types", "profile_modes",
]


def _parse_json_field(value, field_name: str):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as e:
        raise RegistryValidationError(f"Field '{field_name}' is not valid JSON: {e}")


def _validate_list_field(value, field_name: str, valid_values: set = None) -> list:
    parsed = _parse_json_field(value, field_name)
    if not isinstance(parsed, list):
        raise RegistryValidationError(f"Field '{field_name}' must be a JSON array, got {type(parsed).__name__}")
    if valid_values:
        invalid = [v for v in parsed if v not in valid_values]
        if invalid:
            raise RegistryValidationError(
                f"Field '{field_name}' contains invalid values: {invalid}. Valid: {sorted(valid_values)}"
            )
    return parsed


def validate_metric_def(def_dict: dict) -> None:
    missing = [f for f in METRIC_REQUIRED_FIELDS if f not in def_dict or def_dict[f] is None]
    if missing:
        raise RegistryValidationError(f"MetricDef missing required fields: {missing}")

    if def_dict["metric_type"] not in VALID_METRIC_TYPES:
        raise RegistryValidationError(
            f"Invalid metric_type '{def_dict['metric_type']}'. Valid: {sorted(VALID_METRIC_TYPES)}"
        )
    if def_dict["aggregation_rule"] not in VALID_AGGREGATION_RULES:
        raise RegistryValidationError(
            f"Invalid aggregation_rule '{def_dict['aggregation_rule']}'. Valid: {sorted(VALID_AGGREGATION_RULES)}"
        )
    if def_dict["severity"] not in VALID_SEVERITIES:
        raise RegistryValidationError(
            f"Invalid severity '{def_dict['severity']}'. Valid: {sorted(VALID_SEVERITIES)}"
        )

    _parse_json_field(def_dict["criteria_json"], "criteria_json")
    _validate_list_field(def_dict["applicable_output_types"], "applicable_output_types", VALID_OUTPUT_TYPES)
    _validate_list_field(def_dict["applicable_test_families"], "applicable_test_families", VALID_TEST_FAMILIES)
    _validate_list_field(def_dict["profile_modes"], "profile_modes", VALID_PROFILE_MODES)

    if not isinstance(def_dict.get("weight", 0), (int, float)):
        raise RegistryValidationError("Field 'weight' must be a number (int or float)")


def validate_test_def(def_dict: dict) -> None:
    missing = [f for f in TEST_REQUIRED_FIELDS if f not in def_dict or def_dict[f] is None]
    if missing:
        raise RegistryValidationError(f"TestDef missing required fields: {missing}")

    if def_dict["generation_type"] not in VALID_GENERATION_TYPES:
        raise RegistryValidationError(
            f"Invalid generation_type '{def_dict['generation_type']}'. Valid: {sorted(VALID_GENERATION_TYPES)}"
        )
    if def_dict["family"] not in VALID_TEST_FAMILIES:
        raise RegistryValidationError(
            f"Invalid family '{def_dict['family']}'. Valid: {sorted(VALID_TEST_FAMILIES)}"
        )

    level_specs = _parse_json_field(def_dict["level_specs"], "level_specs")
    if not isinstance(level_specs, dict):
        raise RegistryValidationError("Field 'level_specs' must be a JSON object (dict) with L1–L5 keys")

    _parse_json_field(def_dict["validator_rules"], "validator_rules")
    _validate_list_field(def_dict["applicable_output_types"], "applicable_output_types", VALID_OUTPUT_TYPES)
    _validate_list_field(def_dict["profile_modes"], "profile_modes", VALID_PROFILE_MODES)
