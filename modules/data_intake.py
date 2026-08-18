"""
data_intake.py — Pipeline module 1: Data loading & output definition.

load_data() accepts data from 3 sources, selects columns and rows,
auto-detects output types, and returns a DataIntakeResult that is
the sole input every downstream pipeline step needs.

Sources
-------
  csv_default   : read from a file path (hardcoded / config)
  csv_upload    : read from a Streamlit UploadedFile or any file path
  manual        : build DataFrame from a list of dicts supplied by the user

Row modes
---------
  all           : every row in the dataset
  first_n       : first n_rows rows
  random_n      : n_rows randomly sampled (reproducible via random_seed)
  indices       : specific row positions supplied in row_indices

Output types & similarity methods
----------------------------------
  multi_categorical : 2+ columns grouped together  →  Jaccard
  categorical       : single column, short labels  →  Exact match
  text              : single column, long strings  →  Cosine (embeddings)
  numeric           : numeric values               →  Normalised distance
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_TEXT_LEN_HARD  = 60   # chars: definitely text if longer than this
_TEXT_LEN_MIXED = 30   # chars: text if shorter but contains spaces

# Known prefix patterns for auto-grouping (tag_1 tag_2 … → "tags")
_KNOWN_PREFIXES = (
    "tag", "label", "category", "feature",
    "class", "topic", "skill", "keyword",
)
_PREFIX_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*?)_(\d+)$")


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class RiSourceConfig:
    """
    Source/KB data configuration for groundedness evaluation.
    Loaded at intake time; stored on DataIntakeResult.ri_source_config.
    Used by eval_groundedness._resolve_kb_source() for KB lookups.
    """

    source_df: Optional[pd.DataFrame]
    """Loaded KB/source DataFrame."""

    linking_mode: str = "row_order"
    """
    How test rows are linked to source rows:
      "row_order"   — row N in test → row N in source (1:1 by position)
      "key_column"  — match on a shared key column (1:1)
      "multi_chunk" — match on key column; concatenate multiple matching rows
    """

    source_key_col: Optional[str] = None
    """Column in source_df used as the join key (key_column / multi_chunk modes)."""

    test_key_col: Optional[str] = None
    """Column in the main test df used as the join key (key_column / multi_chunk modes)."""

    source_text_cols: Optional[list] = None
    """Which columns to concatenate as the source text. Defaults to all columns."""


@dataclass
class DataIntakeResult:
    """Output of load_data() — everything downstream modules need."""

    df: pd.DataFrame
    """Selected rows. Original CSV row positions preserved in index (named 'row')."""

    input_fields: list
    """Column names to feed to the agent. First = subject, rest = body."""

    output_defs: list
    """
    List of output definition dicts. Format per item:
      {
        "name":      "output_1",          # stable internal label
        "gt_cols":   ["queue"],           # CSV column(s) holding ground truth
        "agent_key": "queue",             # key in the agent result dict
        "type":      "categorical",       # multi_categorical | categorical | text | numeric
        "weight":    1.0,                 # relative weight in aggregated similarity
      }
    """

    meta: dict = field(default_factory=dict)
    """
    Diagnostic metadata:
      source_mode, total_rows, selected_rows, row_mode, available_cols
    """

    threshold_strategy: str = "fixed"
    """
    Strategy for computing the acceptance threshold:
      "fixed"    — use acceptance_threshold as-is
      "baseline" — compute from mean(Sim: outputs at L0) after baseline runs
    """

    acceptance_threshold: float = 0.8
    """
    Acceptance threshold for Sim: outputs.
    Used directly when threshold_strategy="fixed".
    Used as initial value when strategy="baseline" (overwritten by metrics module).
    """

    ri_source_config: Optional[RiSourceConfig] = None
    """
    Source/KB data for groundedness evaluation.
    None when ri_source_enabled=False (default).
    Set by passing ri_source_enabled=True to load_data().
    """

    cost_per_token: Optional[float] = None
    """
    Cost per token (USD) for Ops Performance measurement.
    If set, cost = (input_tokens + output_tokens) * cost_per_token per row.
    If None, cost feature is excluded from Ops Performance.
    """


# ── Type detection ────────────────────────────────────────────────────────────

def _detect_type(sample) -> str:
    """Infer output type from one representative sample value."""
    if isinstance(sample, (list, set, tuple)):
        return "multi_categorical"
    s = str(sample).strip()
    if s.lower() in ("", "nan", "n/a", "none", "null"):
        return "categorical"   # safest fallback when no data
    try:
        float(s.replace(",", "").replace("%", "").replace("$", ""))
        return "numeric"
    except ValueError:
        pass
    if len(s) > _TEXT_LEN_HARD or (len(s) > _TEXT_LEN_MIXED and " " in s):
        return "text"
    return "categorical"


def _sample_value(df: pd.DataFrame, col: str):
    """Return first non-null value from the first 5 rows of a column."""
    for v in df[col].head(5):
        s = str(v).strip()
        if s.lower() not in ("", "nan", "n/a", "none", "null"):
            return v
    return None


# ── Auto-group detection ──────────────────────────────────────────────────────

def _auto_groups(cols: list) -> dict:
    """
    Detect prefix_N groups in a list of column names.
    Returns {group_name: [col1, col2, ...]} for groups with >= 2 members.

    Examples
      tag_1, tag_2, tag_8  →  {"tags":  ["tag_1", "tag_2", "tag_8"]}
      label_1, label_2     →  {"labels": ["label_1", "label_2"]}
    """
    buckets: dict = {}
    for col in cols:
        m = _PREFIX_RE.match(col)
        if not m:
            continue
        prefix = m.group(1).lower()
        # pluralise group name: tag → tags, unless already ends with s
        group = prefix if prefix.endswith("s") else prefix + "s"
        buckets.setdefault(group, []).append(col)
    return {g: sorted(cs) for g, cs in buckets.items() if len(cs) >= 2}


# ── Row selection ─────────────────────────────────────────────────────────────

def _select_rows(
    df: pd.DataFrame,
    row_mode: str,
    n_rows: Optional[int],
    row_indices: Optional[list],
    random_seed: int,
) -> pd.DataFrame:
    if row_mode == "all":
        return df

    if row_mode == "first_n":
        n = n_rows if n_rows is not None else len(df)
        return df.head(n)

    if row_mode == "random_n":
        n = min(n_rows if n_rows is not None else len(df), len(df))
        return df.sample(n=n, random_state=random_seed)

    if row_mode == "indices":
        if not row_indices:
            raise ValueError("row_mode='indices' requires row_indices list")
        bad = [i for i in row_indices if i < 0 or i >= len(df)]
        if bad:
            raise ValueError(
                f"Out-of-range indices: {bad}  (valid 0–{len(df) - 1})"
            )
        return df.iloc[row_indices]

    raise ValueError(
        f"Unknown row_mode: {row_mode!r}. "
        "Use 'all', 'first_n', 'random_n', or 'indices'."
    )


# ── Output-definition builder ─────────────────────────────────────────────────

def _build_output_defs(
    df: pd.DataFrame,
    output_cols: list,
    output_groups: Optional[dict],
    type_overrides: Optional[dict],
    output_weights: Optional[dict] = None,
) -> list:
    """
    Build output_defs list from selected columns.

    Priority for grouping:
      1. User-supplied output_groups  (wins over auto-detect for same name)
      2. Auto-detected prefix_N groups

    agent_key for groups  = group name  (e.g. "tags")
    agent_key for singles = column name (e.g. "queue")
    """
    # Merge: auto groups first, then user overrides on top
    groups: dict = {}
    groups.update(_auto_groups(output_cols))
    if output_groups:
        groups.update(output_groups)

    grouped_cols: set = set()
    for cols in groups.values():
        grouped_cols.update(cols)

    output_defs = []
    idx = 1

    # 1 — Grouped outputs (always multi_categorical by default)
    for group_name, cols in groups.items():
        name = f"output_{idx}"
        output_defs.append({
            "name":      name,
            "gt_cols":   cols,
            "agent_key": group_name,
            "type":      (type_overrides or {}).get(name, "multi_categorical"),
            "weight":    0.0,   # filled in below
        })
        idx += 1

    # 2 — Individual (ungrouped) outputs
    for col in output_cols:
        if col in grouped_cols:
            continue
        name   = f"output_{idx}"
        sample = _sample_value(df, col) if col in df.columns else None
        typ    = _detect_type(sample) if sample is not None else "categorical"
        output_defs.append({
            "name":      name,
            "gt_cols":   [col],
            "agent_key": col,
            "type":      (type_overrides or {}).get(name, typ),
            "weight":    0.0,   # filled in below
        })
        idx += 1

    # 3 — Apply and normalize weights
    n = len(output_defs)
    if n == 0:
        return output_defs

    if output_weights:
        raw = [float(output_weights.get(od["agent_key"], 0.0)) for od in output_defs]
        total = sum(raw) or 1.0
        for od, w in zip(output_defs, raw):
            od["weight"] = round(w / total, 6)
    else:
        equal = round(1.0 / n, 6)
        for od in output_defs:
            od["weight"] = equal

    return output_defs


# ── Main entry point ──────────────────────────────────────────────────────────

def load_data(
    source: str,
    *,
    # Source-specific
    csv_path=None,
    file_obj=None,
    manual_rows=None,
    # Column selection
    input_cols: Optional[list] = None,
    output_cols: Optional[list] = None,
    output_groups: Optional[dict] = None,
    type_overrides: Optional[dict] = None,
    output_weights: Optional[dict] = None,
    # Acceptance threshold
    threshold_strategy: str = "fixed",
    acceptance_threshold: float = 0.8,
    # Row selection
    row_mode: str = "all",
    n_rows: Optional[int] = None,
    row_indices: Optional[list] = None,
    random_seed: int = 42,
    # KB/source data (for groundedness evaluation)
    ri_source_enabled: bool = False,
    ri_source_path: Any = None,
    ri_source_file_obj: Any = None,
    ri_source_linking: str = "row_order",
    ri_source_key_col: Optional[str] = None,
    ri_test_key_col: Optional[str] = None,
    ri_source_text_cols: Optional[list] = None,
    # Numeric output modes
    numeric_modes: Optional[dict] = None,
) -> DataIntakeResult:
    """
    Load ticket data, select columns and rows, detect output types.

    Parameters
    ----------
    source         : "csv_default" | "csv_upload" | "manual"
    csv_path       : Path or str — file path (csv_default / csv_upload)
    file_obj       : file-like object — Streamlit UploadedFile (csv_upload)
    manual_rows    : list[dict] — rows typed by user (manual)
    input_cols     : column names for agent input; first = subject, rest = body
    output_cols    : column names that are ground-truth outputs
    output_groups       : manual grouping override  {"tags": ["col_a", "col_b"]}
    type_overrides      : type override per output  {"output_2": "text"}
    output_weights      : weight per output field  {"queue": 60, "tags": 40}
                          values are percentages or any positive numbers; auto-normalized
                          to sum to 1.0.  Omitted fields get weight 0.  If None, equal
                          weights are applied across all output fields.
    threshold_strategy  : "fixed" (default) — use acceptance_threshold as-is
                          "baseline"         — compute from mean(Sim: outputs at L0)
    acceptance_threshold: float in [0, 1], default 0.8 — pass/fail cutoff for Sim: outputs
    row_mode            : "all" | "first_n" | "random_n" | "indices"
    n_rows         : row count for first_n / random_n
    row_indices    : row positions for indices mode
    random_seed    : random seed for random_n
    ri_source_enabled   : True to load a KB/source file for groundedness evaluation
    ri_source_path      : file path to the source/KB CSV (path or str)
    ri_source_file_obj  : file-like object (e.g. Streamlit UploadedFile)
    ri_source_linking   : "row_order" | "key_column" | "multi_chunk"
    ri_source_key_col   : column in source file used as join key
    ri_test_key_col     : column in main test data used as join key
    ri_source_text_cols : list of source columns to use as source text (default: all)
    numeric_modes       : per-field numeric mode override  {"field_name": "computed"}
                          values: "lookup" (default — number must exist in KB) or
                          "computed" (reserved — number derived from KB data via computation).
                          Only applies to numeric output fields. If None, all numeric fields
                          default to "lookup".

    Returns
    -------
    DataIntakeResult
    """

    # ── Step 1: Load raw DataFrame ────────────────────────────────────────────
    if source == "csv_default":
        if csv_path is None:
            raise ValueError("csv_default requires csv_path")
        df_raw = pd.read_csv(
            Path(csv_path), encoding="utf-8", encoding_errors="replace"
        )

    elif source == "csv_upload":
        if file_obj is not None:
            df_raw = pd.read_csv(file_obj, encoding="utf-8", encoding_errors="replace")
        elif csv_path is not None:
            df_raw = pd.read_csv(
                Path(csv_path), encoding="utf-8", encoding_errors="replace"
            )
        else:
            raise ValueError("csv_upload requires file_obj or csv_path")

    elif source == "manual":
        if not manual_rows:
            raise ValueError("manual requires manual_rows (non-empty list of dicts)")
        df_raw = pd.DataFrame(manual_rows)

    else:
        raise ValueError(
            f"Unknown source: {source!r}. "
            "Use 'csv_default', 'csv_upload', or 'manual'."
        )

    total_rows = len(df_raw)
    available  = list(df_raw.columns)

    # ── Step 2: Validate column selection ─────────────────────────────────────
    if input_cols is None:
        raise ValueError(
            f"input_cols is required.\nAvailable columns: {available}"
        )
    if output_cols is None:
        raise ValueError(
            f"output_cols is required.\nAvailable columns: {available}"
        )

    missing_in  = [c for c in input_cols  if c not in available]
    missing_out = [c for c in output_cols if c not in available]
    if missing_in:
        raise ValueError(f"input_cols not in dataset: {missing_in}")
    if missing_out:
        raise ValueError(f"output_cols not in dataset: {missing_out}")

    # ── Step 3: Select rows ───────────────────────────────────────────────────
    df_sel = _select_rows(df_raw, row_mode, n_rows, row_indices, random_seed).copy()
    df_sel.index.name = "row"

    # ── Step 4: Build output definitions ──────────────────────────────────────
    output_defs = _build_output_defs(
        df_sel, output_cols, output_groups, type_overrides, output_weights
    )

    # Apply numeric_modes to output_defs
    # "lookup" (default) = number must appear in KB; "computed" = reserved for future
    for od in output_defs:
        if od["type"] == "numeric":
            od["numeric_mode"] = (numeric_modes or {}).get(
                od["agent_key"], "lookup"
            )

    if threshold_strategy not in ("fixed", "baseline"):
        raise ValueError(
            f"threshold_strategy must be 'fixed' or 'baseline', got {threshold_strategy!r}"
        )

    # ── Step 5: Load RI source/KB data ────────────────────────────────────────
    ri_src_cfg: Optional[RiSourceConfig] = None
    if ri_source_enabled:
        if ri_source_file_obj is not None:
            src_df = pd.read_csv(
                ri_source_file_obj, encoding="utf-8", encoding_errors="replace"
            )
        elif ri_source_path is not None:
            src_df = pd.read_csv(
                Path(ri_source_path), encoding="utf-8", encoding_errors="replace"
            )
        else:
            raise ValueError(
                "ri_source_enabled=True requires ri_source_path or ri_source_file_obj"
            )

        valid_linking = ("row_order", "key_column", "multi_chunk")
        if ri_source_linking not in valid_linking:
            raise ValueError(
                f"ri_source_linking must be one of {valid_linking}, "
                f"got {ri_source_linking!r}"
            )

        ri_src_cfg = RiSourceConfig(
            source_df        = src_df,
            linking_mode     = ri_source_linking,
            source_key_col   = ri_source_key_col,
            test_key_col     = ri_test_key_col,
            source_text_cols = ri_source_text_cols or list(src_df.columns),
        )

    return DataIntakeResult(
        df                   = df_sel,
        input_fields         = list(input_cols),
        output_defs          = output_defs,
        meta                 = {
            "source_mode":    source,
            "total_rows":     total_rows,
            "selected_rows":  len(df_sel),
            "row_mode":       row_mode,
            "available_cols": available,
        },
        threshold_strategy   = threshold_strategy,
        acceptance_threshold = acceptance_threshold,
        ri_source_config     = ri_src_cfg,
    )


# ── Convenience: pretty-print a DataIntakeResult ──────────────────────────────

def describe(result: DataIntakeResult) -> None:
    """Print a human-readable summary of a DataIntakeResult."""
    m = result.meta
    print(f"Source      : {m['source_mode']}")
    print(f"Rows        : {m['selected_rows']} selected / {m['total_rows']} total  "
          f"(mode: {m['row_mode']})")
    print(f"Input fields: {result.input_fields}")
    print(f"\nOutput definitions ({len(result.output_defs)}):")
    sim_labels = {
        "multi_categorical": "Jaccard",
        "categorical":       "Exact match",
        "text":              "Cosine similarity",
        "numeric":           "Normalised distance",
    }
    for d in result.output_defs:
        sim  = sim_labels.get(d["type"], d["type"])
        cols = ", ".join(d["gt_cols"])
        w    = d.get("weight", 0.0)
        print(f"  {d['name']:<12} type={d['type']:<18} sim={sim:<20} "
              f"weight={w:.0%}  cols=[{cols}]  agent_key={d['agent_key']}")
    print(f"\nAcceptance threshold : {result.acceptance_threshold}  "
          f"(strategy: {result.threshold_strategy})")
    if result.ri_source_config is not None:
        cfg = result.ri_source_config
        src_rows = len(cfg.source_df) if cfg.source_df is not None else 0
        print(f"\nRI Source/KB        : {src_rows} rows  "
              f"(linking: {cfg.linking_mode}  "
              f"text_cols: {cfg.source_text_cols})")
