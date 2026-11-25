"""Offline mocks: a rule-based "judge" and a fake Groq transport. Used by --mock and all tests.

The mock judge reads the judge prompt itself (constraints line + numbered candidate lines) and
applies the rubric's rules literally. `noise` flips deterministic fractions of labels so mock
judges disagree a little, exercising the agreement code. Zero network, zero LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from evals.retrieval.groq_judge import HttpResult

_CAND_RE = re.compile(
    r"^(\d+)\. (.*?) \| price (?:₹(\d+)|n/a) \| category (.*?) \| colour (.*?)(?: \| .*)?$"
)


def _rule_label(constraints: dict[str, str], title: str, price: int | None, colour: str) -> int:
    t = title.lower()
    gender = constraints.get("gender", "not stated")
    if gender == "women" and re.search(r"\bmen's\b|\bmens\b", t):
        return 0
    if gender == "men" and re.search(r"\bwomen's\b|\bwomens\b|\bsaree\b|\bgown\b", t):
        return 0
    cat = constraints.get("category", "not stated").lower()
    if cat != "not stated" and not any(w in t for w in cat.split()):
        return 0
    max_price = constraints.get("max price", "not stated")
    label = 2
    if max_price != "not stated" and price is not None:
        cap = int(max_price.replace("₹", ""))
        if price > cap * 1.25:
            return 0
        if price > cap:
            label = 1
    col = constraints.get("colour", "not stated").lower()
    if col != "not stated" and col not in (colour.lower() + " " + t):
        label = min(label, 1)
    return label


def mock_judge_text(user: str, noise: float = 0.0, salt: int = 0) -> str:
    """Judge response text for a judge prompt `user` (rule-based, deterministic)."""
    cons_line = next(ln for ln in user.splitlines() if ln.startswith("Expected constraints"))
    cons = dict(p.strip().split(": ", 1) for p in cons_line.split("): ", 1)[1].split(";"))
    labels: dict[str, int] = {}
    for ln in user.splitlines():
        m = _CAND_RE.match(ln)
        if not m:
            continue
        n, title, price, _cat, colour = m.groups()
        lab = _rule_label(cons, title, int(price) if price else None, colour)
        if noise and ((int(n) * 7919 + salt * 104729) % 1000) / 1000 < noise:
            lab = (lab + 1) % 3
        labels[n] = lab
    return json.dumps({"labels": labels, "reason": "rule-based mock"})


# deterministic mock disagreement per model so agreement code sees non-trivial data
MOCK_NOISE = {"openai/gpt-oss-120b": 0.0, "qwen/qwen3.8-27b": 0.04}


def _default_responder(system: str, user: str, model: str) -> str:
    return mock_judge_text(user, MOCK_NOISE.get(model, 0.0), salt=len(model))


def mock_transport(
    responder: Callable[[str, str, str], str] = _default_responder,
    tokens_per_call: tuple[int, int] = (2000, 150),
) -> Callable[[dict[str, Any]], HttpResult]:
    """A fake Groq transport: answers every payload through `responder(system, user, model)`."""

    def post(payload: dict[str, Any]) -> HttpResult:
        msgs = payload["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        p, c = tokens_per_call
        body = {
            "choices": [
                {
                    "message": {"content": responder(system, user, payload["model"])},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": p,
                "completion_tokens": c,
                "total_tokens": p + c,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        return HttpResult(200, {"x-ratelimit-remaining-tokens": "7000"}, body)

    return post
