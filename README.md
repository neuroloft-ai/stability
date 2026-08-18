# Stability — Stress-Testing Reasoning Models under Structured Perturbations

Code, prompts, and data for the paper ***Measuring Stability and Failure Behavior in
Language Models Under Structured Perturbations.*** The framework perturbs each problem
along a **five-level severity ladder** across **seven families** — six that preserve the
answer and one, **Knowledge Boundary**, that removes answerability so refusal becomes the
correct response — validity-gates every generated test, and summarizes each model by
per-level **Accuracy**, magnitude-weighted **Stability**, and a per-family **Collapse Point**
defined relative to the model's own baseline.

It is instantiated on the **100 GSM-Symbolic seed problems**, expanded into **4,473 gated
tests**.

---

## Perturbation families

| Family | Code seed | What it stresses | Severity metric (level assignment) |
|---|---|---|---|
| Semantic Variation (SV) | `seed_paraphrase` | invariance to rewording | SIM (lexical similarity, lower = harder) |
| Input Quality (IQ) | `seed_noise` | tolerance to noisy input | CER + WCR (char-edit / word-corruption rate) |
| Structural / Format (F) | `seed_format` | independence from formatting | CCR + LCS (char-change rate / layout score) |
| Context Interference (CI) | `seed_distraction` | ignoring irrelevant context | ALE (added-content load) |
| Context Load (CL) | `seed_context` | robustness to context length | CTX_CER (word-expansion ratio) |
| Conflict Instruction Stress (CIS) | `seed_conflict` | holding the task under conflict | CES (conflict-explicitness tier) |
| Knowledge Boundary (KB) | `seed_kb` | recognizing unanswerability | none — five categorical boundary types |

Per-test **perturbation magnitude** is the unified blend `score = ½(1 − SIM) + ½·TCR`.

---

## Repository layout

```
generator/    stress-test generation engine
  level_specs.py   per-level specs and bands for every family (L1–L5)
  metrics.py       severity metrics (SIM, CER, WCR, CCR, LCS, ALE, CTX_CER, CES) + magnitude
  gates.py         structural / entailment(NLI) / level gates
  nli_gate.py      optional local cross-encoder NLI gate
  llm_judge.py     LLM judges (meaning-preservation, distractor/context/conflict, KB epistemic-validity)
registry/     per-family seed definitions (the seven families above)
modules/      test generation + evaluation
  test_generator.py   generate + validity-gate candidate tests
  test_runner.py      run a system-under-test over a set of tests
  data_intake.py      load the seed dataset
  eval_all.py / eval_output.py / eval_kb.py   scoring (numeric match; KB epistemic judge)
scorer/       scoring utilities
experiment/
  data/gsm8k_symbolic100.csv                          100 GSM-Symbolic seed problems
  output/run_base.csv
  output/benchmark/gsm8k_symbolic100/generated.csv    the test suite — BUILT by Stage 1 (not shipped)
  generate.py             Stage 1: load seeds + generate the validity-gated test suite
  run_sut_model.py        Stage 2: run a hosted model (OpenAI / Google / …) over the suite
  phi_cot_run.py          run a local model via Ollama
  run_utils.py
  results_crossmodel.ipynb   compute Accuracy / Stability / Collapse Point + the paper figures
results/      pre-rendered figures and the stability table from the paper
```

---

## Install

```bash
pip install -r requirements.txt        # Python 3.10+
```

The optional **local** NLI gate (`generator/nli_gate.py`) additionally needs `transformers`
and `torch`; the paper used an **LLM judge** for meaning preservation, so these are not
required to reproduce the results.

## API keys

The run scripts call hosted models and an LLM judge — you'll need at least an OpenAI key
(generation, judging, and the `gpt-4o-mini`/`o4-mini` SUTs all use it). Set it up **one of
two ways**:

1. **Environment variable** (preferred) — `export OPENAI_API_KEY="sk-..."` before running.
   Used by `example.py`, `generate.py`, and the notebooks.
2. **`doc/` folder** — create a `doc/` directory at the repo root (it's git-ignored, so
   anything you put there never gets committed) and drop in a plain-text file with just the
   key string, no quotes or newline needed beyond the key itself:

   | Provider | File | Used by |
   |---|---|---|
   | OpenAI | `doc/Key_o.txt` | generation, judging, `gpt-4o-mini` / `o4-mini` SUTs |
   | Groq | `doc/Key_g.txt` | `llama-3.3-70b` (or any Groq-hosted SUT) |
   | Gemini | `doc/Key_go.txt` | `gemini-2.5-flash` (or any Gemini SUT) |

   You only need the files for the providers/models you actually run — e.g. running the
   OpenAI models only needs `doc/Key_o.txt`. `phi4-mini` runs locally via Ollama and needs
   no key at all.

`experiment/run_sut_model.py`'s `PROVIDERS` dict is where these are wired up if you want to
add another OpenAI-compatible provider. **Never commit keys** — `doc/` and `.env` are both
git-ignored, so keep credentials there and nowhere else in the repo.

---

## Pipeline

The benchmark is built and used in three stages — all run from the `experiment/` directory.
The repository ships only the **100 seed problems** (`data/`) and the **generation code** — the
test suite itself is **not bundled**; you build it locally in Stage 1 (it is generated once and
then read from disk). So **run Stage 1 before Stage 2.**

**Stage 1 — Generate the test suite** (load seeds → perturb across 7 families × 5 levels →
validity-gate). LLM-based, so it needs an API key. Run once.
```bash
cd experiment
python generate.py 3        # demo: 3 seeds (cheap); use 100 for the full benchmark
```
It loads `data/gsm8k_symbolic100.csv`, generates and gates candidates, prints the gate-pass rate
and per-family / per-level coverage, and writes the suite to
`output/benchmark/gsm8k_symbolic100/generated.csv` (resumable; the run scripts read it from there).

**Stage 2 — Run a model over the suite** (test the generated tests).
```bash
python run_sut_model.py <run_name> <provider> <model> [throttle_seconds]
# e.g.
python run_sut_model.py gpt-4o-mini openai gpt-4o-mini
```
A local model via Ollama: `python phi_cot_run.py`. Each run scores every test (numeric match for
the six answer-preserving families; an LLM epistemic judge for Knowledge Boundary), writes to
`output/runs/<run_name>__gsm8k_symbolic100/`, and is resumable per seed problem.

**Stage 3 — Analyze** (Accuracy / Stability / Collapse Point + figures). Open
`results_crossmodel.ipynb` from `experiment/`; it reads every model's `eval.csv` from
`output/runs/` and produces the accuracy-degradation curve, the behavioral-quadrant figure, and
the per-family Stability and Collapse-Point tables. Pre-rendered versions are in `results/`.

**Quick offline check (no API key):** `python ../example.py` computes the per-test severity
metrics on a worked example and cross-checks them against the shipped suite.

Each admitted row of `generated.csv` is one stress test, with its `Category`, `Classified Level`,
`SIM`, `TCR`, `Score`, gate flags, and `Generated Input`. The gate pipeline is described in
§4.2 of the paper.

---

## Metrics (paper §3.2)

For a single test *i*, `sim_i ∈ [0,1]` is output correctness and `score_i ∈ [0,1]` is the
perturbation magnitude. Over any set *S* of tests:

- **Accuracy(S)** = mean `sim_i`
- **Magnitude(S)** = mean `score_i`
- **Stability(S)** = `Σ sim_i·score_i / Σ score_i` (magnitude-weighted correctness)
- **Collapse Point** = lowest level *k* with `acc(L_k) < 0.80 · acc(L_0)`

---

## Citation

```bibtex
@misc{stability2026,
  title  = {Measuring Stability and Failure Behavior in Language Models Under Structured Perturbations},
  author = {<authors>},
  year   = {2026},
  note   = {arXiv preprint}
}
```

---

## License

Released under the [MIT License](LICENSE).
