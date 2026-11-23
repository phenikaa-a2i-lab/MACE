"""
MACE — LLM client for SGLang.

SGLang serves an OpenAI-compatible API (default http://127.0.0.1:30000/v1,
https://github.com/sgl-project/sglang) and is called through the OpenAI SDK.
"""

import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infrastructure.choices import choice_letters
from infrastructure.config import (
    MODELS, get_api_key, get_base_url, get_model_base_url, REQUEST_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY, DEFAULT_TEMPERATURE, MAX_TOKENS_LONG, FORCE_TEMPERATURE,
)


_usage = threading.local()


def reset_usage() -> None:
    """Clear the counters of the current thread. The runner calls this at the start of each question."""
    _usage.tokens_in = 0
    _usage.tokens_out = 0
    _usage.n_calls = 0
    _usage.n_missing = 0
    _usage.finish_reasons = []


def get_usage() -> dict:
    """Token totals of the current thread since reset_usage().

    n_missing counts calls that came back WITHOUT usage (old server, network error). If it
    is > 0, tokens_in/out are only a lower bound; check it before reporting cost.
    """
    return {
        "tokens_in":  getattr(_usage, "tokens_in", 0),
        "tokens_out": getattr(_usage, "tokens_out", 0),
        "n_llm_calls": getattr(_usage, "n_calls", 0),
        "n_usage_missing": getattr(_usage, "n_missing", 0),
        "finish_reasons": list(getattr(_usage, "finish_reasons", [])),
    }


def _add_usage(tin, tout, finish_reason=None) -> None:
    _usage.n_calls = getattr(_usage, "n_calls", 0) + 1
    if not hasattr(_usage, "finish_reasons"):
        _usage.finish_reasons = []
    _usage.finish_reasons.append(finish_reason)
    if tin is None and tout is None:
        _usage.n_missing = getattr(_usage, "n_missing", 0) + 1
        return
    _usage.tokens_in  = getattr(_usage, "tokens_in", 0)  + int(tin or 0)
    _usage.tokens_out = getattr(_usage, "tokens_out", 0) + int(tout or 0)


_last = threading.local()


def get_last_call() -> dict:
    """Details of the most recent call_llm in this thread: model, model_id, temperature,
    max_tokens, tokens_in, tokens_out, finish_reason (None if the server did not say)."""
    return dict(getattr(_last, "info", None) or {})


_CLIENTS: dict = {}
_CLIENTS_LOCK = threading.Lock()


def _get_client(api_key: str, base_url: str):
    key = (api_key, base_url)
    client = _CLIENTS.get(key)
    if client is None:
        with _CLIENTS_LOCK:
            client = _CLIENTS.get(key)
            if client is None:
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key, base_url=base_url,
                    timeout=REQUEST_TIMEOUT, max_retries=0,
                )
                _CLIENTS[key] = client
    return client


def _sglang_extra_body(think: bool | None) -> dict:
    """
    SGLang honours HF chat-template kwargs per request. For hybrid-reasoning
    models (Qwen3) `enable_thinking=False` makes the template emit an empty
    <think></think> block so the model answers directly. Models whose template
    has no such kwarg (Llama) simply ignore it.
    """
    if think is None:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": bool(think)}}


def call_llm(
    model_name: str,
    messages: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = MAX_TOKENS_LONG,
    think: bool | None = None,
) -> str:
    """
    Call the SGLang server. Returns text response.
    Retries with backoff on connection errors and rate limits.

    think: only meaningful for hybrid-reasoning models (Qwen3).
    None (default) leaves the server's default thinking behavior untouched.
    Pass False to force a direct answer with no hidden <think> pass
    (extra_body.chat_template_kwargs.enable_thinking=False).
    """
    cfg      = MODELS[model_name]
    model_id = cfg["model_id"]
    provider = cfg["provider"]
    api_key  = get_api_key(provider)
    base_url = get_model_base_url(model_name)
    if FORCE_TEMPERATURE is not None:
        temperature = FORCE_TEMPERATURE
    call_info = {"model": model_name, "model_id": model_id,
                 "temperature": temperature, "max_tokens": max_tokens}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _last.info = None
            client = _get_client(api_key, base_url)
            kwargs = dict(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if provider == "sglang":
                extra = _sglang_extra_body(think)
                if extra:
                    kwargs["extra_body"] = extra
            resp = client.chat.completions.create(**kwargs)
            u = getattr(resp, "usage", None)
            tin = getattr(u, "prompt_tokens", None) if u else None
            tout = getattr(u, "completion_tokens", None) if u else None
            finish = getattr(resp.choices[0], "finish_reason", None)
            _add_usage(tin, tout, finish)
            _last.info = {**call_info, "tokens_in": tin, "tokens_out": tout,
                          "finish_reason": finish}
            content = resp.choices[0].message.content
            return (content or "").strip()

        except Exception as e:
            err  = str(e)
            wait = RETRY_DELAY * attempt

            if "rate_limit" in err.lower() or "429" in err:
                wait = 30 * attempt
                print(f"  [{provider}] rate-limited — waiting {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
            else:
                print(f"  [{provider}] {model_name} attempt {attempt}/{MAX_RETRIES}: "
                      f"{err[:100]}")

            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise

    return ""


from infrastructure.vote_extraction import extract_vote, VERSION as ANSWER_PARSER_VERSION


def parse_answer(text: str, n_options: int = 4) -> str | None:
    """
    Option label of a model output: the last explicit, non-negated statement of an answer
    (paper Section 3.6; see infrastructure/vote_extraction.py). Unresolved "A or B", labels
    outside the option range and outputs that state no answer return None. Never consults the gold.
    """
    return extract_vote(text, n_options)
