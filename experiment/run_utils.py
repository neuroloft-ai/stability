"""
run_utils.py — timestamped, self-describing run folders so reruns never clobber.

Each run gets:  output/runs/<timestamp>__<dataset>_n<N>_<model>/
                ├── manifest.json   (full config + counts, updated as the run progresses)
                ├── generated.csv    (Step 2)
                ├── responses.csv    (Step 3)
                └── ... (metrics, figures)

Usage
-----
from run_utils import make_run, update_manifest
run_dir, manifest = make_run(OUT_DIR, dataset='gsm8k100', n_samples=1,
                             gen_model='gpt-4o-mini', sut_model='gpt-4o-mini',
                             families=FAMILIES, levels=[1,2,3,4,5],
                             max_attempts=2, threshold=0.80, notes='smoke test')
... write run_dir/'generated.csv' ...
update_manifest(run_dir, n_generated=123, n_gate_pass=98)
"""
from pathlib import Path
from datetime import datetime
import json


def make_run(out_dir, dataset, n_samples, gen_model, sut_model,
             families, levels, max_attempts, threshold, notes=""):
    """Create a timestamped run folder + manifest.json, return (run_dir, manifest)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = str(sut_model).replace("/", "-")
    run_id = f"{ts}__{dataset}_n{n_samples}_{safe_model}"
    run_dir = Path(out_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id":       run_id,
        "created":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":      dataset,
        "n_samples":    n_samples,
        "gen_model":    gen_model,
        "sut_model":    sut_model,
        "families":     families,
        "levels":       levels,
        "max_attempts": max_attempts,
        "threshold":    threshold,
        "notes":        notes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir, manifest


def update_manifest(run_dir, **updates):
    """Merge key/values into the run's manifest.json and return the updated dict."""
    p = Path(run_dir) / "manifest.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    m.update(updates)
    p.write_text(json.dumps(m, indent=2))
    return m


def latest_run(out_dir):
    """Return the most recent run_dir under output/runs/, or None."""
    runs = sorted((Path(out_dir) / "runs").glob("*__*"))
    return runs[-1] if runs else None


def filter_ambiguous_ci(generated_df):
    """Drop Context-Interference rows whose distractor makes the SAME subject buy something
    with a price (ambiguous 'total cost' -> changes the gold answer). Deterministic safety net
    on top of the tightened distractor prompt. Returns (clean_df, n_dropped)."""
    import re
    def _clean(t): return ' '.join(str(t).replace('[QUESTION]', '').split())
    def _bad(r):
        if r.get('Category') != 'Context Interference':
            return False
        added = _clean(r['Generated Input']).replace(_clean(r['Original Input']), '')
        same_subj  = bool(re.search(r'\b(she|he|they|also)\b[^.]{0,45}\b(bought|paid|picked up|purchas|spent|acquir|got)\b', added, re.I))
        disclaimed = bool(re.search(r'\b(if|imagine|instead|would not|does not apply|hypothetical|a friend|different (bakery|store|person)|yesterday|last week)\b', added, re.I))
        has_price  = bool(re.search(r'\$\s*\d', added))
        return same_subj and has_price and not disclaimed
    mask = generated_df.apply(_bad, axis=1)
    return generated_df[~mask].reset_index(drop=True), int(mask.sum())
