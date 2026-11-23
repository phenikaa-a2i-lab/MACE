# Runner and evaluation guide

For installation and a quick start, see the [main README](../README.md). Run commands below from `code/`; dataset paths such as `data/dev.jsonl` are resolved against the project root.

## Layout

```text
data/                    VM14K and MedQA-US inputs (project root)
results_USMLE/           Generated MedQA-US results (project root)
results/                 Generated VM14K results (project root)
results_ablation/        Generated VM14K-subset results (project root)
results_test/            Generated non-USMLE pilot results (project root)
code/
  dataio/                Dataset loading and normalization, subset and data preparation scripts
  frameworks/tier1/      Zero-shot, CoT, CoT-SC, Self-Refine
  frameworks/tier2/      Shuffled Voting Ensemble
  frameworks/tier3/      MedAgents, MDAgents, their _vi variants, MAD debate
  infrastructure/        Model configuration, HTTP clients, label helpers, parser
  runner/                Experiment dispatch, per-seed execution, manifests
  metrics/               Pass@1 / Pass@k, mean and std across seeds, significance tests, result readers
  scripts/               Model-server setup, smoke tests, backend comparisons
  tests/                 Offline regression tests
```

## Input schemas

### MedQA-US

The active `data/dev.jsonl` contains 1,272 four-option development questions from `questions/US/4_options/phrases_no_exclude_dev.jsonl` in the [MedQA release](https://github.com/jind11/MedQA). One source line has this shape:

```json
{
  "question": "A medical multiple-choice question",
  "options": {"A": "First option", "B": "Second option", "C": "Third option", "D": "Fourth option"},
  "answer": "Second option",
  "answer_idx": "B",
  "meta_info": "step1"
}
```

The loader sorts dictionary keys A-D (or A-E for five choices), converts options to a list, and takes the gold label from `answer_idx`. It checks that any answer text agrees with that label. Missing IDs become `medqa_us_<file_stem>_<source_line>`, such as `medqa_us_dev_000001`.

Existing IDs are preserved. `data/pilot_usmle_10.jsonl` holds the first ten normalized development questions with their original development IDs. Repeated seeds and resumed runs therefore use the same question IDs.

The source JSONL is never modified by loading. Duplicate IDs, missing/invalid gold labels, malformed option keys, and option counts other than four or five are rejected before inference. `meta_info` identifies exam sections; it is not an Easy/Medium/Hard difficulty label.

### VM14K

VM14K already uses the normalized schema:

```json
{
  "id": "question-id",
  "question": "A Vietnamese medical question",
  "options": ["First option", "Second option", "Third option", "Fourth option"],
  "answer": "B",
  "difficulty_level": "Medium",
  "medical_topic": ["Cardiology"]
}
```

The 11,094 existing records and their IDs are preserved. The current VM14K file contains two empty question strings; validation reports these without filtering them or changing the benchmark population.

## Run experiments

Validate inputs without model calls or output files:

```bash
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only
```

Run a framework/model pair across all configured seeds:

```bash
python -m runner.run_experiments --dataset data/dev.jsonl --framework zero_shot --model sglang-llama3.1-8b-awq --workers 8
```

Run only the ten-question pilot at seed 42:

```bash
python -c "from runner.run_experiments import run_one_seed; run_one_seed('zero_shot', 'sglang-llama3.1-8b-awq', 'data/pilot_usmle_10.jsonl', seed=42, workers=1)"
```

The model server must be running and serving the exact model ID configured for the selected key. For `sglang-llama3.1-8b-awq`, that ID is `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. Server setup is in the [main README](../README.md#serving-the-models).

| CLI option | Behavior |
|---|---|
| `--dataset PATH` | JSONL input; defaults to `data/pilot_100.jsonl` |
| `--framework NAME` | Framework to execute; defaults to `zero_shot` |
| `--model KEY` | Key from `MODELS`; defaults to `sglang-qwen3-32b` |
| `--workers N` | Concurrent requests; default 32 for SGLang, overridden by `MACE_NUM_WORKERS` |
| `--seeds S [S ...]` | Seeds to run for a single framework/model; defaults to 42 43 44 |
| `--validate-only` | Validate and summarize without inference |
| `--all` | Run every framework in `ALL_FRAMEWORKS` on every model in `config.py`; ignores `--framework` and `--model` |
| `--no-resume` | Ignore completed IDs; currently still appends to the existing file |

Re-running the same normal command resumes by skipping completed question IDs. To get a fresh experiment, use a separate dataset/run name or archive the existing outputs first; `--no-resume` does not truncate an existing result file.

The configured seeds are 42, 43, and 44. They govern Python-side operations such as shuffling. They are not passed as sampling seeds to the model server, so generation is not guaranteed to be deterministic.

## Frameworks and missing metadata

| Tier | Framework | LLM calls per question |
|---|---|---:|
| 1 | `zero_shot` | 1 |
| 1 | `cot` | 1 |
| 1 | `cot_sc` | 5 |
| 1 | `self_refine` | 2-5 |
| 2 | `voting_ensemble` | 3 |
| 3 | `medagents` | 9 with three selected experts |
| 3 | `medagents_vi` | 8 with three experts |
| 3 | `mdagents` | 2 / 4 / 7 by routed complexity |
| 3 | `mdagents_vi` | 1 / 3 / 6 by difficulty label |
| 3 | `mad_debate` | Up to 9 |
| 3 (variant) | `medagents_hetero` | 9; the three experts are played by three different models |
| 3 (variant) | `medagents_noreport` | 8; no report, consultants read the three analyses |
| 3 (re-execution) | `medagents_reextract` | 5 live calls; steps 1-2 replayed from a stored `medagents` result |

The framework implementations follow the repository's existing protocols; they are re-implementations rather than exact reproductions of the original papers' code. Four-choice prompt text is preserved when expanding the choice count to five.

MedQA-US does not include `difficulty_level` or `medical_topic`:

- `mdagents` uses a model call to select Low/Moderate/High complexity.
- `medagents` uses a model call to recruit experts.
- `mdagents_vi` falls back to Medium when no difficulty label is supplied.
- `medagents_vi` falls back to its generic expert roles when no topic is supplied.

The runner prints warnings for the `_vi` fallbacks. These fallbacks are not inferred dataset labels. Result `difficulty` and `topic` remain `null`, and metric summaries use `Unknown`. Use the original `mdagents` and `medagents` variants for the main MedQA comparison unless an additional labeling protocol is defined.

## Answer extraction

`infrastructure.llm_client.parse_answer()` receives each question's option count and delegates to
`infrastructure.vote_extraction.extract_vote()` (policy `last-explicit-v1`, recorded in every manifest as
`answer_parser_version`). It implements the vote-extraction rule of the paper (Section 3.6): the option
label is the **last explicit, non-negated statement of an answer** in the output. Thinking blocks and
Markdown emphasis are ignored.

Statements that count:

1. A declaration: `the (correct/best/final) answer is X`, `my final answer remains X`, `Answer: X`,
   `đáp án (đúng) là X`, `Conclusion X`, `I choose/maintain/select X`.
2. A verdict on a letter: `option X is the correct answer / the best choice / the most appropriate`.
3. A letter followed by its option text in an answering sentence (`the most likely pathogen is B. Coxsackie A virus`).
4. A line that is only a label (`B`, `(B)`, `B. text`, `Option B`).

Not statements: entries of a restated option list, `(option D) should be considered`, abbreviations such as
`C. difficile`, the article `a` (`the answer is a beta-blocker`), and anything negated (`option A is incorrect`).
When the last statement names two options (`A or B`), a label outside the option range, or no statement exists,
the result is `None` (`null` in JSON; counted as incorrect in accuracy). The gold answer is never consulted.

`extract_vote(text, n_options, options)` can additionally fall back to the option *text* when no letter statement
exists. The framework code is frozen and calls `parse_answer(raw, n_options=...)` without options, so this fallback is
not part of the run-time parser; it is only available to offline analysis and to new framework files.

Coverage figures of the parser are reported in the paper (Section 3.6 and Appendix A).

Changing the parser changes scoring, and runs made with `explicit-answer-v1` are not comparable with
`last-explicit-v1` runs at the vote level. Retain raw responses and record any offline reprocessing. For multi-agent
methods, offline parsing changes cannot recreate the later model messages that different intermediate labels would
have produced.

Zero-shot requests a single character, but its prompt is not a decoding constraint; raw output may include explanations.

## Result routing and files

`infrastructure.config.get_results_dir()` is shared by the runner, metric readers, and `scripts/compare_results.py`.

| Dataset stem | Root |
|---|---|
| `dev`, `train`, `test` | `results_USMLE/` |
| `phrases_no_exclude_dev`, `phrases_no_exclude_train`, `phrases_no_exclude_test` | `results_USMLE/` |
| Prefixes `usmle`, `medqa_us`, `pilot_usmle`, `pilot_medqa_us` | `results_USMLE/` |
| Stems starting with `vm14k_sub` (the VM14K subset) | `results_ablation/` (the folder name is historical) |
| Other stems starting with `pilot` | `results_test/` |
| Other datasets, including `vm14k_clean` | `results/` |

Result directories sit at the project root, next to `data/`. The generic stems `dev`, `train`, and `test` are reserved for MedQA-US inputs.

```text
<root>/tier<N>/<framework>/<framework>__<safe_model>__<dataset_stem>__s<seed>.jsonl
<root>/tier<N>/<framework>/<framework>__<safe_model>__<dataset_stem>__s<seed>.jsonl.meta.json
```

Slashes and periods in model keys are replaced with underscores. JSONL records are appended as worker requests complete, so their order need not match input order; join by `id` and `seed`.

| Result fields | Meaning |
|---|---|
| `id`, `seed`, `framework`, `model` | Question and run identity |
| `gold`, `answer`, `correct` | Gold letter, parsed prediction, and correctness |
| `difficulty`, `topic` | Dataset labels, or null when absent |
| `api_calls`, `n_llm_calls` | Framework and client call counters |
| `tokens_in`, `tokens_out`, `n_usage_missing` | Token usage and missing-usage count |
| `finish_reasons` | Why each model call stopped, in call order (`stop`, `length` = hit the output limit, or null); same length as `n_llm_calls`. Present from parser `last-explicit-v1` on; older files lack it |
| `individual_votes`, `round1_answers` | Sampling/agent votes where applicable |
| `audit_trail` | Step details and preserved raw model output |
| `timestamp` | Original result timestamp |

The manifest stores dataset path and size, model/provider/endpoint, seed, temperature, token budgets, code revision, parser version, and restart history. Recorded offline reparsing history is preserved when the manifest is updated.

After the configured seeds, the terminal prints mean accuracy and sample standard deviation. The runner does not create CSV, Excel, or Markdown reports.

## Metrics and checks

Check that each seed has the expected question count, unique IDs, `api_calls == n_llm_calls`, `len(finish_reasons) == n_llm_calls` (newer files) and `n_usage_missing == 0`. If usage is missing, the recorded token total is only a lower bound. Do not count repeated seeds for a question as independent questions in significance tests.

Accuracy, sampling Pass@k and the collaboration metrics in `metrics/compute.py` accept A-E. Agent votes within one multi-agent run are not independent sampling attempts for Pass@k. For datasets outside the built-in VM14K sizes, metric readers use `n_samples` from the runner manifest to determine complete seeds.

Compare two model keys on the same USMLE dataset:

```bash
python scripts/compare_results.py --framework zero_shot --dataset dev --a sglang-llama3.3-70b --b sglang-qwen3-32b
```

The legacy analysis CLI takes an explicit results directory:

```bash
python -m analysis.analyze_results --dataset dev --results-dir ../results_USMLE
```

## Configuration and validation

Model keys, token budgets, seed lists, and defaults live in `infrastructure/config.py`.

| Environment variable | Purpose |
|---|---|
| `SGLANG_BASE_URL` | SGLang endpoint; default `http://127.0.0.1:30000/v1` |
| `SGLANG_API_KEY` | API key if the SGLang server requires one |
| `MACE_TOKEN_CAP` | One output cap for all free-text steps (the paper runs use 1024); the one-letter decision and the complexity label keep their own caps |
| `MACE_FORCE_TEMPERATURE` | One temperature for every call; unset keeps the per-step defaults (0.3, and 0.5 for CoT-SC) |
| `MACE_REQUEST_TIMEOUT` | Per-request timeout in seconds; default 300 |
| `MACE_NUM_WORKERS` | Override default concurrency |

Keep credentials in the environment. Generated results, logs, and environment files are excluded by `.gitignore`.

```bash
python -m pytest tests -q
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only
python -m runner.run_experiments --dataset data/vm14k_clean.jsonl --validate-only
```

The offline tests exercise vote extraction and parser precedence, both option counts, MedQA normalization, all framework routes including the protocol variants and `medagents_reextract`, shuffled votes, output routing, result and manifest readers, and resume behavior. They mock model calls and require no inference server.
