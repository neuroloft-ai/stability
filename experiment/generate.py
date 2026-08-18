"""
Stage 1 of the pipeline: load the seed dataset and GENERATE the validity-gated test suite.

Generation is LLM-based, so this needs an API key (OpenAI by default). The test suite is NOT
shipped with the repo — you build it here. It writes the generated, gated tests to
output/benchmark/gsm8k_symbolic100/generated.csv (resumable), which is the suite the run
scripts then read. Run this once before running any model.

Run from the experiment/ directory:
    python generate.py [n_questions]      # default 3 (a cheap demo); use 100 for the full benchmark

API key: put it in ../doc/Key_o.txt, or set the OPENAI_API_KEY environment variable.
"""
import sys, os, copy
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))                 # generator/ registry/ modules/ scorer/
sys.path.insert(0, str(REPO / 'experiment'))  # run_utils

from openai import OpenAI
from modules.data_intake import load_data
from modules.test_generator import generate_tests
from generator.llm_judge import LLMJudge
from run_utils import filter_ambiguous_ci

# ── config ──────────────────────────────────────────────────────────────────
GEN_MODEL    = 'gpt-4o-mini'      # model used for generation + the LLM gates/judges
FAMILIES     = ['Semantic Variation', 'Input Quality', 'Structural/Format',
                'Context Interference', 'Context Load', 'Conflict Instruction Stress',
                'Knowledge Boundary']
MAX_ATTEMPTS = 2                  # attempts per (question x family x level); temperature escalates
THRESHOLD    = 0.80
N_QUESTIONS  = int(sys.argv[1]) if len(sys.argv) > 1 else 3

DATA  = REPO / 'experiment' / 'data' / 'gsm8k_symbolic100.csv'
OUT   = REPO / 'experiment' / 'output'; OUT.mkdir(parents=True, exist_ok=True)
BENCH = OUT / 'benchmark' / 'gsm8k_symbolic100'; BENCH.mkdir(parents=True, exist_ok=True)
GEN_PATH = BENCH / 'generated.csv'     # canonical path that run_sut_model.py / phi_cot_run.py read

# ── API key ─────────────────────────────────────────────────────────────────
keyfile = REPO / 'doc' / 'Key_o.txt'
api_key = keyfile.read_text().strip() if keyfile.exists() else os.environ.get('OPENAI_API_KEY')
assert api_key, 'No API key: put one in doc/Key_o.txt or set OPENAI_API_KEY'
client = OpenAI(api_key=api_key)

# ── 1) load the seed dataset ────────────────────────────────────────────────
df = pd.read_csv(DATA)
run_base = OUT / 'run_base.csv'; df.to_csv(run_base, index=False)
intake = load_data(source='csv_upload', csv_path=str(run_base),
                   input_cols=['question'], output_cols=['ground_truth'],
                   type_overrides={'ground_truth': 'numeric'}, row_mode='all',
                   acceptance_threshold=THRESHOLD)
intake.df = intake.df.iloc[:N_QUESTIONS].reset_index(drop=True)
print(f'[1] loaded {len(intake.df)} seed problems from {DATA.name}')

# ── 2) generate the validity-gated suite (7 families x 5 levels), resumable ──
judge = LLMJudge(client=client, model=GEN_MODEL)
if GEN_PATH.exists():
    generated_df = pd.read_csv(GEN_PATH); done = set(generated_df['Source Row'].unique())
    print(f'[2] resume: {len(generated_df)} rows, {len(done)} questions already generated')
else:
    generated_df = pd.DataFrame(); done = set()

for row_idx in range(len(intake.df)):
    if row_idx in done:
        continue
    ri = copy.copy(intake); ri.df = intake.df.iloc[[row_idx]].reset_index(drop=True)
    batch = generate_tests(
        test_names=FAMILIES, intake=ri, max_attempts=MAX_ATTEMPTS,
        llm_client=client, gen_model=GEN_MODEL, use_nli=True,
        local_nli_gate=None, llm_judge=judge,
        system_description='A math problem solver that answers grade school math questions')
    batch['Source Row'] = row_idx
    generated_df = pd.concat([generated_df, batch], ignore_index=True)
    generated_df.to_csv(GEN_PATH, index=False)
    print(f'    question {row_idx}: +{len(batch)} candidates (total {len(generated_df)})')

generated_df, n_ci = filter_ambiguous_ci(generated_df)   # drop ambiguous distractors
if n_ci:
    generated_df.to_csv(GEN_PATH, index=False)

# ── 3) quick QC ─────────────────────────────────────────────────────────────
gp = generated_df[generated_df['Gate'] == True]
print(f'\n[3] gate pass: {len(gp)}/{len(generated_df)} '
      f'({100 * len(gp) / max(len(generated_df), 1):.0f}%) admitted -> {GEN_PATH.name}')
print('admitted per family:')
print(gp['Category'].value_counts().to_string())
print('\nadmitted per measured (classified) level:')
print(gp['Classified Level'].value_counts().sort_index().to_string())
print('\nnext: test the generated tests with a model ->')
print('    python run_sut_model.py <run_name> <provider> <model>')
