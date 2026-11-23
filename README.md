<h1 align="center">Consensus Is Not Confidence</h1>
<h3 align="center">Multi-Agent Consultation Erases the Deferral Signal in Medical Question Answering</h3>

<p align="center">
  <a href="#citation"><img alt="Paper" src="https://img.shields.io/badge/paper-TODO-b31b1b"></a>
  <a href="LICENSE"><img alt="Code license" src="https://img.shields.io/badge/code-MIT-green"></a>
  <a href="data/LICENSES.md"><img alt="Data licenses" src="https://img.shields.io/badge/data-MIT%20%2F%20CC%20BY--NC%204.0-blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
</p>

This repository (**MACE**) contains the code, data and evaluation pipeline for the paper *Consensus Is Not Confidence: Multi-Agent Consultation Erases the Deferral Signal in Medical Question Answering*. It runs single-model and multi-agent reasoning frameworks on two medical multiple-choice benchmarks, MedQA-USMLE (English) and VM14K (Vietnamese), logs every model call, and computes the metrics used in the paper.

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Serving the models](#serving-the-models)
- [Data](#data)
- [Running experiments](#running-experiments)
- [Output format](#output-format)
- [Repository structure](#repository-structure)
- [Tests](#tests)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

| Role in the paper | Frameworks (`--framework`) |
|---|---|
| Main comparison | `zero_shot`, `cot`, `cot_sc` (5 samples), `medagents` (the consultation protocol, 9 calls per question) |
| Additional strategies | `self_refine`, `voting_ensemble`, `mdagents`, `medagents_vi` |
| Protocol variants (VM14K subset, Qwen3-32B) | `medagents_hetero`, `medagents_noreport` |
| Re-execution of steps 3-5 | `medagents_reextract` (replays the stored steps 1-2 of a `medagents` run) |
| Implemented, not reported | `mad_debate`, `mdagents_vi` |

The multi-agent frameworks are re-implementations in the spirit of MedAgents (Tang et al., ACL 2024), MDAgents (Kim et al., NeurIPS 2024) and multi-agent debate (Du et al., ICML 2024), not line-by-line replicas.

| Setting | Value |
|---|---|
| Serving | SGLang, AWQ 4-bit weights |
| Models | Llama-3.3-70B, Qwen3-32B, Llama-3.1-8B |
| Seeds | 42, 43, 44 on MedQA-USMLE; 42 and 43 on the VM14K subset |
| Temperature | 0.3; 0.5 for the CoT-SC samples |
| Token cap | 1,024 per call (`MACE_TOKEN_CAP=1024`), except the one-letter decision step |
| Answer parsing | last explicit, non-negated answer (`last-explicit-v1`) |
| Logging | raw output and `finish_reason` of every call |

## Installation

Python 3.10 or newer. The runner calls a model server over HTTP, so it needs no GPU or PyTorch.

```bash
git clone https://github.com/phenikaa-a2i-lab/MACE.git
cd MACE/code
python -m pip install -r requirements.txt
```

All commands below run from `code/`.

## Serving the models

Each experiment needs an [SGLang](https://github.com/sgl-project/sglang) server with an OpenAI-compatible API. One server serves one model, on any GPU setup SGLang supports.

| Model | Config key | Hugging Face weights |
|---|---|---|
| Llama-3.3-70B | `sglang-llama3.3-70b` | `casperhansen/llama-3.3-70b-instruct-awq` |
| Qwen3-32B | `sglang-qwen3-32b` | `Qwen/Qwen3-32B-AWQ` |
| Llama-3.1-8B | `sglang-llama3.1-8b-awq` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |

```bash
python -m sglang.launch_server \
    --model-path casperhansen/llama-3.3-70b-instruct-awq \
    --host 127.0.0.1 --port 30000 \
    --context-length 8192 --mem-fraction-static 0.85 \
    --max-running-requests 32 --trust-remote-code
```

Adjust the flags to your hardware: lower `--mem-fraction-static` on a shared GPU, set `--max-running-requests` to the runner's `--workers`, use `--tp N` to split a model over N GPUs. The 4-bit 70B weights take roughly 40 GB plus KV cache. The client switches Qwen3 reasoning off in each request, so no server flag is needed.

The client uses `http://127.0.0.1:30000/v1`. Override it with `SGLANG_BASE_URL`, or per model with `SGLANG_BASE_URL__<MODEL_KEY>` (upper case, non-alphanumeric characters replaced by `_`), for example `SGLANG_BASE_URL__SGLANG_LLAMA3_1_8B_AWQ`.

## Data

| File in `data/` | Questions | Description |
|---|---:|---|
| `dev.jsonl` | 1,272 | MedQA-USMLE, four-option development split |
| `vm14k_sub2000.jsonl` | 2,000 | VM14K subset of the primary analysis; `vm14k_sub2000.manifest.json` records the sampling rule and checksums |
| `vm14k_clean.jsonl` | 11,094 | VM14K full set, used for the replication |
| `pilot_usmle_10.jsonl` | 10 | First ten questions of `dev.jsonl`, for smoke tests |
| `pilot_100.jsonl` | 100 | VM14K pilot |

MedQA is released under the MIT license and VM14K under CC BY-NC 4.0 (research use only). Citations and license texts are in [`data/LICENSES.md`](data/LICENSES.md).

## Running experiments

```bash
# check the data without calling a model
python -m runner.run_experiments --dataset data/dev.jsonl --validate-only

# one framework on one model, all seeds
MACE_TOKEN_CAP=1024 python -m runner.run_experiments \
    --dataset data/dev.jsonl --framework medagents --model sglang-llama3.3-70b --workers 32

# the VM14K subset of the primary analysis
MACE_TOKEN_CAP=1024 python -m runner.run_experiments \
    --dataset data/vm14k_sub2000.jsonl --framework cot_sc --model sglang-llama3.1-8b-awq
```

`--seeds 42 43` limits the seeds, `--no-resume` ignores existing results, `--workers N` sets request concurrency. A run resumes from the rows already written and stops a seed after more than 10 errors. `--all` runs every framework on every model in `config.py` and ignores `--model`, so loop over `--framework` instead.

`medagents_hetero` needs one endpoint per model (see above). `medagents_reextract` needs the `medagents` result file for the same model, dataset and seed.

| Environment variable | Effect |
|---|---|
| `MACE_TOKEN_CAP` | One cap for all free-text steps |
| `MACE_FORCE_TEMPERATURE` | One temperature for every call; unset keeps the per-step defaults |
| `MACE_REQUEST_TIMEOUT` | Per-request timeout in seconds (default 300) |
| `MACE_NUM_WORKERS` | Default request concurrency |

## Output format

Results are written next to `data/`: `results_USMLE/` for MedQA, `results_ablation/` for the VM14K subset (a historical folder name) and `results/` for the full VM14K set, as `<tier>/<framework>/<framework>__<model>__<dataset>__s<seed>.jsonl`, each with a `.meta.json` manifest (model, endpoint, temperature, token caps, parser and code version).

| Field | Meaning |
|---|---|
| `id`, `seed`, `framework`, `model` | Identification |
| `gold`, `answer`, `correct` | Gold label, extracted answer, correctness |
| `individual_votes`, `round1_answers` | Votes of the individual agents or samples |
| `api_calls`, `n_llm_calls` | Number of model calls; the two counts must agree |
| `tokens_in`, `tokens_out`, `n_usage_missing` | Token usage; a nonzero `n_usage_missing` makes the totals a lower bound |
| `finish_reasons` | Why each call stopped; `"length"` means the token cap was hit |
| `audit_trail` | Raw output of every step |

The full guide (input schemas, answer extraction, result routing) is in [`code/README.md`](code/README.md).

## Repository structure

```text
MACE/
├── data/                  MedQA-US and VM14K inputs, subset manifest, LICENSES.md
└── code/
    ├── dataio/            Dataset loading and normalization, subset and preparation scripts
    ├── frameworks/        tier1 (single model), tier2 (voting ensemble), tier3 (multi-agent)
    ├── infrastructure/    Model configuration, HTTP client, vote extraction
    ├── runner/            Experiment dispatch, per-seed execution, manifests
    ├── metrics/           Pass@1, Pass@k, mean and std, significance tests, result readers
    ├── analysis/          Result analysis and table generation
    ├── scripts/           Model-server setup and smoke tests
    └── tests/             Offline regression tests
```

## Tests

The tests run offline and need no model server. They use `pytest`, which is not in `requirements.txt`.

```bash
python -m pip install pytest
python -m pytest tests -q
```

## Citation

If you use this code or data, please cite the paper (BibTeX entry to be completed):

```bibtex
@misc{mace,
  title  = {Consensus Is Not Confidence: Multi-Agent Consultation Erases the Deferral Signal in Medical Question Answering},
  author = {TODO},
  year   = {TODO},
  note   = {TODO: arXiv identifier or venue}
}
```

Please also cite the datasets: MedQA (Jin et al., 2021) and VM14K (arXiv:2506.01305).

## Acknowledgements

We thank the authors of [MedQA](https://github.com/jind11/MedQA) (Jin et al.) and [VM14K](https://venera-ai.github.io/VM14K) for releasing their data. We thank the authors of MedAgents, MDAgents and multi-agent debate, whose protocols the frameworks here build on. We thank the [SGLang](https://github.com/sgl-project/sglang) team for the inference engine used to serve all models.

## License

The code is released under the MIT license; see [`LICENSE`](LICENSE). The datasets in `data/` keep their own licenses (MedQA: MIT, VM14K: CC BY-NC 4.0, research use only); see [`data/LICENSES.md`](data/LICENSES.md).
