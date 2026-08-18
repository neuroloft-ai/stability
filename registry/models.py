from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricDef:
    name: str
    dimension: str
    metric_type: str
    criteria_json: dict
    applicable_output_types: list[str]
    applicable_test_families: list[str]
    aggregation_rule: str
    weight: float
    severity: str
    profile_modes: list[str]
    id: Optional[str] = None
    category: Optional[str] = None
    version: Optional[int] = None
    is_active: bool = True
    ui_label: Optional[str] = None
    ui_description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class TestDef:
    name: str
    family: str
    description: str
    generation_type: str
    level_specs: dict
    validator_rules: dict
    applicable_output_types: list[str]
    profile_modes: list[str]
    id: Optional[str] = None
    version: Optional[int] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class RegistryVersion:
    version: int
    status: str
    notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class RegistrySnapshot:
    snapshot_id: str
    version: int
    profile_mode: str
    metric_ids: list[str]
    test_ids: list[str]
    output_type: Optional[str] = None
    filters: Optional[dict] = None
    created_at: Optional[str] = None


@dataclass
class SnapshotWithDefs:
    snapshot: RegistrySnapshot
    metric_defs: list[MetricDef]
    test_defs: list[TestDef]


@dataclass
class MetricDimension:
    name: str
    id: Optional[str] = None
    description: Optional[str] = None
    order_index: int = 0
    version: Optional[int] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class MetricCategory:
    dimension_id: str
    name: str
    id: Optional[str] = None
    description: Optional[str] = None
    order_index: int = 0
    version: Optional[int] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class MetricLevelCriteria:
    """Per-level scoring parameters for a MetricDef × test_family combination."""
    metric_id: str
    test_family: str
    level: str
    criteria: dict                # e.g. {w_sim, w_tcr, sim_min, tcr_band}
    id: Optional[str] = None
    version: Optional[int] = None
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class CertificationReadinessReport:
    is_ready: bool
    draft_metric_count: int
    draft_test_count: int
    missing_profile_modes: list[str]
    missing_output_types: list[str]
    warnings: list[str]
    errors: list[str]
