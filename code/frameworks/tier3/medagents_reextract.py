"""MedAgents with steps 1-2 taken from a stored run and steps 3-5 run live.

The stored expert-gathering output and the three analyses are reused and the analysis labels are
extracted again with parse_answer. Report, three revisions and decision are generated live
(5 calls), so api_calls and tokens cover those calls only.
"""

from infrastructure.llm_client import call_llm, parse_answer
from infrastructure.config import MAX_TOKENS_SHORT, MAX_TOKENS_MEDIUM
from frameworks.utils import extract_reasoning
from frameworks.tier3.medagents import (
    parse_roles, prompt_report, prompt_consult, prompt_decision,
)

N_EXPERTS = 3
REUSED_STEPS = ("expert_gathering", "analysis")
LIVE_CALLS = 5
REUSED_CALLS = 4

_EXPECTED = ("expert_gathering", "analysis", "analysis", "analysis",
             "report", "consultation", "consultation", "consultation", "decision")


def check_source_trail(trail: list[dict]) -> None:
    steps = tuple(t.get("step") for t in trail or [])
    if steps != _EXPECTED:
        raise ValueError(f"source audit_trail is not a MedAgents row: steps={steps}")
    if any(not isinstance(t.get("raw"), str) for t in trail[:4]):
        raise ValueError("source row has a missing raw output in steps 1-2")


def run_medagents_reextract(question: str, options: list[str], model: str,
                            source_trail: list[dict]) -> dict:
    check_source_trail(source_trail)
    n = len(options)
    audit_trail = []

    raw_roles = source_trail[0]["raw"]
    roles = parse_roles(raw_roles, fallback_n=N_EXPERTS)
    stored_roles = [t["role"] for t in source_trail[1:4]]
    if roles != stored_roles:
        raise ValueError(f"roles parsed from the stored gathering output {roles} "
                         f"differ from the stored analysis roles {stored_roles}")
    audit_trail.append({**source_trail[0], "reused": True})

    analyses = []
    for role, step in zip(roles, source_trail[1:4]):
        raw = step["raw"]
        ans = parse_answer(raw, n_options=n)
        analyses.append({"role": role, "analysis": extract_reasoning(raw), "answer": ans, "raw": raw})
        audit_trail.append({"step": "analysis", "role": role, "answer": ans,
                            "answer_at_run_time": step.get("answer"),
                            "reasoning": analyses[-1]["analysis"], "raw": raw, "reused": True})

    api_calls = 0

    msgs = prompt_report(question, options, analyses)
    report = call_llm(model, msgs, max_tokens=MAX_TOKENS_MEDIUM, think=False)
    api_calls += 1
    audit_trail.append({"step": "report", "raw": report})

    final_opinions = []
    for a in analyses:
        msgs = prompt_consult(question, options, a["role"], report, a["analysis"])
        raw = call_llm(model, msgs, temperature=0.3, max_tokens=MAX_TOKENS_MEDIUM, think=False)
        ans = parse_answer(raw, n_options=n)
        api_calls += 1
        final_opinions.append({"role": a["role"], "answer": ans,
                               "reasoning": extract_reasoning(raw), "raw": raw})
        audit_trail.append({"step": "consultation", "role": a["role"], "answer": ans,
                            "reasoning": final_opinions[-1]["reasoning"], "raw": raw})

    msgs = prompt_decision(question, options, final_opinions)
    raw_dec = call_llm(model, msgs, max_tokens=MAX_TOKENS_SHORT, think=False)
    ans_dec = parse_answer(raw_dec, n_options=n)
    api_calls += 1
    audit_trail.append({"step": "decision", "answer": ans_dec, "raw": raw_dec})

    return {
        "answer":           ans_dec,
        "roles":            roles,
        "round1_answers":   [a["answer"] for a in analyses],
        "round2_answers":   [op["answer"] for op in final_opinions],
        "individual_votes": [op["answer"] for op in final_opinions],
        "audit_trail":      audit_trail,
        "api_calls":        api_calls,
    }
