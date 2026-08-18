"""
test_runner.py — Pipeline module 3: Run generated stress tests against a target system.

Receives the DataFrame produced by test_generator.generate_tests(), parses each
generated input, calls the system under test (SUT), and returns the original
DataFrame extended with prediction columns and timing metadata.

Public functions
----------------
run_tests(generated_df, sut, intake, gate_only=True)  → DataFrame
make_agent_sut(agent)                                  → Callable[[dict], dict]
make_llm_sut(client, model, system_prompt, parser)     → Callable[[dict], dict]
describe_run(results_df, intake)                       → None

SUT contract
------------
Any callable with the signature:

    sut(input_fields: dict) -> dict

    input_fields : {field_name: value, ...}   — keys match intake.input_fields
    return       : {agent_key: value, ...}     — keys match output_def["agent_key"]

Extra keys in the return dict (e.g. "similar", "_embedding" from TagExtractorAgent)
are silently ignored.  Missing keys produce None in the output columns.

Output columns appended to generated_df
----------------------------------------
  Run At       — ISO 8601 UTC timestamp for each row  (e.g. "2026-03-07T14:23:01Z")
  Latency (ms) — integer milliseconds for the SUT call
  Run Status   — "ok"  or  "error: <message>"
  Pred: <key>  — one column per output_def agent_key
                 e.g. "Pred: queue", "Pred: tags"
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from modules.data_intake import DataIntakeResult


# ─────────────────────────────────────────────────────────────────────────────
# Input parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_generated_input(generated_text: str, input_fields: list) -> dict:
    """
    Reconstruct {field_name: value} from the [FIELD] marker format used by
    test_generator.

    Generated Input format (from test_generator.py):
        [SUBJECT] ticket subject line
        [BODY] full body text...

    Parses each field by locating its [FIELDNAME] marker and extracting content
    up to the next marker (or end of string).
    Falls back to empty string for any field not found in the text.
    """
    result = {}
    for i, field in enumerate(input_fields):
        label      = f"[{field.upper()}]"
        next_field = input_fields[i + 1] if i + 1 < len(input_fields) else None
        next_label = f"[{next_field.upper()}]" if next_field else None

        start = generated_text.find(label)
        if start == -1:
            result[field] = ""
            continue

        content_start = start + len(label)
        if next_label:
            end     = generated_text.find(next_label, content_start)
            content = (
                generated_text[content_start:end].strip()
                if end != -1
                else generated_text[content_start:].strip()
            )
        else:
            content = generated_text[content_start:].strip()

        result[field] = content

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUT adapter factories
# ─────────────────────────────────────────────────────────────────────────────

def make_agent_sut(agent) -> Callable[[dict], dict]:
    """
    Wrap a local agent into the SUT callable interface.

    Compatible with any agent that exposes:
        agent.extract(**input_fields) -> dict

    Example (SupportAgent):
        from agent import TagExtractorAgent
        sut = make_agent_sut(TagExtractorAgent())

    The returned callable accepts {field_name: value} and returns the agent
    result dict directly.  Extra keys like "similar" and "_embedding" are
    kept in the dict but only the output_def agent_keys are stored as columns.
    """
    def sut(input_fields: dict) -> dict:
        return agent.extract(**input_fields)
    return sut


def make_llm_sut(
    client,
    model: str,
    system_prompt: str,
    output_parser: Callable[[str], dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Callable[[dict], dict]:
    """
    Wrap an OpenAI-compatible LLM into the SUT callable interface.

    Parameters
    ----------
    client        : OpenAI-compatible client (openai.OpenAI, Groq, etc.)
    model         : model name  (e.g. "gpt-4o-mini", "llama-3.3-70b-versatile")
    system_prompt : instructions telling the LLM what to classify and what format
                    to return
    output_parser : Callable[[str], dict]
                    converts the LLM's raw text response into {agent_key: value}
                    Example:
                        def parser(raw):
                            import json
                            data = json.loads(raw)
                            return {"queue": data["queue"], "tags": data["tags"]}
    temperature   : sampling temperature (default 0.0 for deterministic output)
    max_tokens    : max response tokens

    Example:
        import json
        def my_parser(raw):
            data = json.loads(raw)
            return {"queue": data.get("queue", ""), "tags": data.get("tags", [])}

        sut = make_llm_sut(
            client        = openai_client,
            model         = "gpt-4o-mini",
            system_prompt = "Classify the ticket. Return JSON: {queue, tags}",
            output_parser = my_parser,
        )
    """
    def sut(input_fields: dict) -> dict:
        user_msg = "\n".join(
            f"{name.upper()}: {value}" for name, value in input_fields.items()
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return output_parser(raw)

    return sut


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def _call_sut(sut, input_fields: dict, agent_keys: list) -> tuple:
    """
    Call the SUT, return (result_dict, latency_ms, run_status).
    Isolates error handling so both baseline and stress paths share the same logic.
    """
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    try:
        result     = sut(input_fields)
        latency_ms = round((time.perf_counter() - t0) * 1000)
        run_status = "ok"
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        run_status = f"error: {exc}"
        result     = {}
    return result, latency_ms, run_status, run_at


def run_tests(
    generated_df: pd.DataFrame,
    sut: Callable[[dict], dict],
    intake: DataIntakeResult,
    gate_only: bool = True,
    n_baseline: int = 5,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    Run baseline rounds then each generated stress test against the SUT.

    Parameters
    ----------
    generated_df : DataFrame from test_generator.generate_tests()
    sut          : callable  (dict -> dict) — use make_agent_sut() or make_llm_sut()
    intake       : DataIntakeResult — provides input_fields and output_defs
    gate_only    : if True (default), only rows where Gate==True are executed;
                   skipped rows are dropped from the result.
                   Set to False to run all rows (useful for debugging).
    n_baseline   : number of baseline rounds to run before stress tests (default 5).
                   Each round calls the SUT on every original row in intake order.
                   Pattern for 3 rows, n_baseline=5:
                     round1(row0, row1, row2) → round2(row0, row1, row2) → ... × 5
                   Category = "Temporal Consistency", Level = "L0".
                   Pass 0 to skip baseline entirely.

    Returns
    -------
    DataFrame — baseline rows (if any) followed by stress test rows, with columns:
        Run At       : ISO 8601 UTC timestamp per row
        Latency (ms) : integer ms for SUT call
        Run Status   : "ok" | "error: <message>"
        Pred: <key>  : one column per output_def agent_key
    """
    agent_keys   = [od["agent_key"] for od in intake.output_defs]
    input_fields = intake.input_fields

    records = []

    # Pre-compute grand total for progress reporting
    _stress_df_preview = (
        generated_df[generated_df["Gate"] == True]
        if gate_only and "Gate" in generated_df.columns
        else generated_df
    )
    _grand_total = n_baseline * len(intake.df) + len(_stress_df_preview)
    _progress_done = 0

    # ── Baseline rounds ───────────────────────────────────────────────────────
    if n_baseline > 0:
        n_rows        = len(intake.df)
        total_baseline = n_baseline * n_rows
        print(f"── Baseline  ({n_baseline} round{'s' if n_baseline != 1 else ''} "
              f"× {n_rows} row{'s' if n_rows != 1 else ''}  = {total_baseline} calls)  "
              f"Category=Temporal Consistency")
        print("─" * 60)

        done = 0
        for round_num in range(1, n_baseline + 1):
            for row_idx, orig_row in intake.df.iterrows():
                done += 1
                orig_fields = {
                    f: str(orig_row.get(f, "")).strip()
                    for f in input_fields
                    if str(orig_row.get(f, "")).strip() not in ("", "nan", "NaN")
                }

                _progress_done += 1
                if progress_cb:
                    progress_cb(
                        _progress_done,
                        _grand_total,
                        f"Baseline L0 round {round_num} row {row_idx}",
                    )

                print(
                    f"  [{done:>3}/{total_baseline}]  "
                    f"Temporal Consistency  L0  round={round_num}  row={row_idx} ...",
                    end=" ", flush=True,
                )

                result, latency_ms, run_status, run_at = _call_sut(sut, orig_fields, agent_keys)

                record = {
                    "Dimension":        "Stability",
                    "Category":         "Temporal Consistency",
                    "Phase":            "Baseline",
                    "Target Level":     "L0",
                    "Classified Level": "L0",
                    "Attempt":          round_num,
                    "Temperature":      "-",
                    "Structural Gate":  "-",
                    "Struct Failures":  "",
                    "NLI Gate":         "-",
                    "NLI Label":        "-",
                    "Level Gate":       "-",
                    "Gate":             True,
                    "SIM":              1.0,
                    "TCR":              0.0,
                    "Score":            0.0,
                    "Source Row":       row_idx,
                    "Generated Input":  "\n".join(
                        f"[{f.upper()}] {v}" for f, v in orig_fields.items()
                    ),
                    "Run At":           run_at,
                    "Latency (ms)":     latency_ms,
                    "Run Status":       run_status,
                }
                for key in agent_keys:
                    record[f"Pred: {key}"] = result.get(key)

                pred_parts = [
                    f"{k}={str(result.get(k, '-'))[:40]}" for k in agent_keys
                ]
                print(f"{run_status}  {latency_ms}ms  {'  '.join(pred_parts)}")
                records.append(record)

        print("─" * 60)
        print(f"Baseline done.  "
              f"{sum(1 for r in records if r['Run Status'] == 'ok')} ok  "
              f"/ {sum(1 for r in records if r['Run Status'].startswith('error'))} errors\n")

    # ── Stress test rows ──────────────────────────────────────────────────────
    if gate_only:
        if "Gate" not in generated_df.columns:
            raise ValueError(
                "'Gate' column not found in generated_df. "
                "Pass gate_only=False to run all rows."
            )
        df = generated_df[generated_df["Gate"] == True].copy().reset_index(drop=True)
    else:
        df = generated_df.copy().reset_index(drop=True)

    if df.empty:
        print("  [!] No stress test rows to run (DataFrame is empty after filtering).")
    else:
        total  = len(df)
        print(f"── Stress tests  {total} row{'s' if total != 1 else ''}  "
              f"({'Gate=True only' if gate_only else 'all rows'})")
        print(f"   Output keys: {agent_keys}")
        print("─" * 60)

        stress_records = []

        for i, (_, row) in enumerate(df.iterrows()):
            level   = row.get("Target Level", "-")
            attempt = row.get("Attempt", "-")
            test    = row.get("Category", row.get("Phase", "-"))

            _progress_done += 1
            if progress_cb:
                progress_cb(
                    _progress_done,
                    _grand_total,
                    f"{test} {level}",
                )

            print(
                f"  [{i + 1:>3}/{total}]  {test}  {level}  attempt={attempt} ...",
                end=" ", flush=True,
            )

            generated_text = str(row.get("Generated Input", ""))
            parsed_input   = _parse_generated_input(generated_text, input_fields)

            result, latency_ms, run_status, run_at = _call_sut(sut, parsed_input, agent_keys)

            record = row.to_dict()
            record["Run At"]       = run_at
            record["Latency (ms)"] = latency_ms
            record["Run Status"]   = run_status

            pred_parts = []
            for key in agent_keys:
                val = result.get(key)
                record[f"Pred: {key}"] = val
                pred_parts.append(f"{key}={str(val)[:40] if val is not None else 'None'}")

            print(f"{run_status}  {latency_ms}ms  {'  '.join(pred_parts)}")
            stress_records.append(record)

        records.extend(stress_records)

        print("─" * 60)
        n_stress_ok  = sum(1 for r in stress_records if r["Run Status"] == "ok")
        n_stress_err = len(stress_records) - n_stress_ok
        print(f"Stress tests done.  {n_stress_ok} ok  / {n_stress_err} errors")

    # ── Combine and return ────────────────────────────────────────────────────
    n_ok  = sum(1 for r in records if r["Run Status"] == "ok")
    n_err = len(records) - n_ok
    print(f"\nTotal  {len(records)} rows  —  {n_ok} ok  / {n_err} errors")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience summary
# ─────────────────────────────────────────────────────────────────────────────

def describe_run(results_df: pd.DataFrame, intake: DataIntakeResult) -> None:
    """
    Print a human-readable summary of a run_tests() result DataFrame.

    Shows baseline (Temporal Consistency) and stress test sections separately,
    each with: row count, ok/error split, avg/P95 latency, per-classified-level
    breakdown, and sample predicted values.
    """
    if results_df.empty:
        print("No results to describe.")
        return

    agent_keys = [od["agent_key"] for od in intake.output_defs]

    is_baseline = results_df.get("Category", pd.Series()) == "Temporal Consistency"
    baseline_df = results_df[is_baseline]
    stress_df   = results_df[~is_baseline]

    def _section(df: pd.DataFrame, label: str) -> None:
        if df.empty:
            return
        total = len(df)
        n_ok  = (df["Run Status"] == "ok").sum()
        n_err = total - n_ok
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Total rows   : {total}")
        print(f"  OK           : {n_ok}")
        print(f"  Errors       : {n_err}")

        if "Latency (ms)" in df.columns:
            lat = df["Latency (ms)"].dropna()
            if len(lat):
                p95 = int(lat.quantile(0.95)) if len(lat) >= 2 else int(lat.iloc[0])
                print(f"  Avg latency  : {lat.mean():.0f} ms")
                print(f"  P95 latency  : {p95} ms")

        if "Classified Level" in df.columns:
            print(f"\n  Per level breakdown (Classified Level):")
            for lvl, grp in df.groupby("Classified Level"):
                ok_lvl  = (grp["Run Status"] == "ok").sum()
                avg_lat = grp["Latency (ms)"].mean() if "Latency (ms)" in grp.columns else 0
                print(f"    {lvl:<6}  rows={len(grp)}  ok={ok_lvl}  avg_lat={avg_lat:.0f}ms")

        for key in agent_keys:
            col = f"Pred: {key}"
            if col not in df.columns:
                continue
            sample = df[col].dropna().head(3).tolist()
            print(f"\n  Pred: {key}  (first 3 values)")
            for s in sample:
                print(f"    {str(s)[:80]}")

        print(f"{'='*60}")

    _section(baseline_df, "Baseline — Temporal Consistency")
    _section(stress_df,   "Stress Tests")
