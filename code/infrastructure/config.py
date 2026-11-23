"""
MACE — Central configuration

v2 adds the "sglang" provider (https://github.com/sgl-project/sglang): an
OpenAI-compatible server with continuous batching + RadixAttention prefix
caching. Same frameworks, same prompts, same result format as v1 — only the
inference backend (and the client-side concurrency) changes.

Edit N_RUNS and SEEDS to control how many times each experiment runs.
All other settings are referenced by the rest of the codebase via imports.
"""

import os
import re


MODELS: dict[str, dict] = {
    "sglang-qwen3-32b": {
        "provider": "sglang",
        "model_id": "Qwen/Qwen3-32B-AWQ",
    },
    "sglang-llama3.1-8b-awq": {
        "provider": "sglang",
        "model_id": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    },
    "sglang-llama3.3-70b": {
        "provider": "sglang",
        "model_id": "casperhansen/llama-3.3-70b-instruct-awq",
    },
}


SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")
SGLANG_API_KEY  = os.environ.get("SGLANG_API_KEY", "sglang")


def get_api_key(provider: str) -> str:
    if provider == "sglang":
        return SGLANG_API_KEY
    raise ValueError(f"Unknown provider: {provider}")


def get_base_url(provider: str) -> str:
    if provider == "sglang":
        return SGLANG_BASE_URL
    raise ValueError(f"Unknown provider: {provider}")


def model_url_env(model_name: str) -> str:
    """Env var naming a per-model SGLang endpoint, e.g. SGLANG_BASE_URL__SGLANG_LLAMA3_3_70B."""
    return "SGLANG_BASE_URL__" + re.sub(r"[^A-Za-z0-9]", "_", model_name).upper()


def get_model_base_url(model_name: str) -> str:
    """Endpoint for one model key.

    Ablation A (medagents_hetero) calls three SGLang models in the same question, so each
    needs its own server. If the per-model env var (see model_url_env) is set it wins;
    otherwise the provider default applies, so every existing run behaves exactly as before.
    """
    provider = MODELS[model_name]["provider"]
    if provider == "sglang":
        override = os.environ.get(model_url_env(model_name))
        if override:
            return override
    return get_base_url(provider)


ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(ROOT, "data")
PILOT_FILE  = os.path.join(DATA_DIR, "pilot_100.jsonl")
FULL_FILE   = os.path.join(DATA_DIR, "vm14k_clean.jsonl")
RESULTS_DIR      = os.path.join(ROOT, "results")
RESULTS_TEST_DIR = os.path.join(ROOT, "results_test")
RESULTS_USMLE_DIR = os.path.join(ROOT, "results_USMLE")
RESULTS_ABLATION_DIR = os.path.join(ROOT, "results_ablation")


def resolve_data_path(path: str) -> str:
    """A dataset path that does not exist as given is looked up under the project root."""
    if os.path.exists(path):
        return path
    candidate = os.path.join(ROOT, path)
    return candidate if os.path.exists(candidate) else path


def get_results_dir(dataset: str) -> str:
    """Shared output routing for the runner, metrics and backend comparisons."""
    stem = os.path.splitext(os.path.basename(dataset))[0].lower()
    if stem.startswith("vm14k_sub"):
        return RESULTS_ABLATION_DIR
    usmle_stems = {
        "dev", "train", "test",
        "phrases_no_exclude_dev", "phrases_no_exclude_train", "phrases_no_exclude_test",
    }
    if stem in usmle_stems or stem.startswith(("usmle", "medqa_us", "pilot_usmle", "pilot_medqa_us")):
        return RESULTS_USMLE_DIR
    return RESULTS_TEST_DIR if stem.startswith("pilot") else RESULTS_DIR

DEFAULT_TEMPERATURE = 0.3
COT_SC_TEMPERATURE  = 0.5

_FORCE_T = os.environ.get("MACE_FORCE_TEMPERATURE")
FORCE_TEMPERATURE = float(_FORCE_T) if _FORCE_T not in (None, "") else None
MAX_RETRIES         = 3
RETRY_DELAY         = 10

NUM_WORKERS = 4

NUM_WORKERS_BY_PROVIDER = {
    "sglang": 32,
}


def get_num_workers(model_name: str) -> int:
    env = os.environ.get("MACE_NUM_WORKERS")
    if env:
        return max(1, int(env))
    provider = MODELS[model_name]["provider"]
    return NUM_WORKERS_BY_PROVIDER.get(provider, NUM_WORKERS)

REQUEST_TIMEOUT = int(os.environ.get("MACE_REQUEST_TIMEOUT", "300"))

MAX_TOKENS_SHORT   = 16
MAX_TOKENS_ROLES   = 512
MAX_TOKENS_COMPLEX = 8
MAX_TOKENS_MEDIUM  = 300
MAX_TOKENS_LONG    = 512
MAX_TOKENS_AGENT   = 600
MAX_TOKENS_DEBATE  = 400

_TOKEN_CAP = os.environ.get("MACE_TOKEN_CAP")
if _TOKEN_CAP:
    MAX_TOKENS_ROLES = MAX_TOKENS_MEDIUM = MAX_TOKENS_LONG = MAX_TOKENS_AGENT = MAX_TOKENS_DEBATE = int(_TOKEN_CAP)

N_RUNS = 3
SEEDS  = [42, 43, 44]

TIER1_FRAMEWORKS = ["zero_shot", "cot", "cot_sc", "self_refine"]

TIER2_FRAMEWORKS = ["voting_ensemble"]

TIER3_FRAMEWORKS = [
    "medagents", "medagents_vi",
    "mdagents",  "mdagents_vi",
    "mad_debate",
]

ALL_FRAMEWORKS = TIER1_FRAMEWORKS + TIER2_FRAMEWORKS + TIER3_FRAMEWORKS

ABLATION_FRAMEWORKS = ["medagents_hetero", "medagents_noreport", "medagents_reextract"]

FRAMEWORK_TIER = {
    **{fw: "tier1" for fw in TIER1_FRAMEWORKS},
    **{fw: "tier2" for fw in TIER2_FRAMEWORKS},
    **{fw: "tier3" for fw in TIER3_FRAMEWORKS},
    **{fw: "tier3" for fw in ABLATION_FRAMEWORKS},
}

MODEL_NAMES = list(MODELS.keys())

EXPERIMENT_MATRIX: list[tuple[str, str]] = [
    (fw, model)
    for fw in ALL_FRAMEWORKS
    for model in MODEL_NAMES
]

EN_REFERENCE = {
    "medagents":       {"nmar": 18.3, "icr": 42.1},
    "medagents_vi":    {"nmar": 18.3, "icr": 42.1},
    "mdagents":        {"nmar": 17.1, "icr": 38.4},
    "mdagents_vi":     {"nmar": 17.1, "icr": 38.4},
    "mad_debate":      {"nmar": 16.9, "icr": 44.2},
    "voting_ensemble": {"nmar": None, "icr": None},
}
