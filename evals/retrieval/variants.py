"""The four reranking variants compared on the same 20-candidate pools.

A  current production reranker: the REAL `src.agents.reranker.rerank` (production prompt, 150
   max_tokens, production parser and fallbacks) behind an injectable, recording LLM.
A2 variant A with the truncation fixed: same prompt/parser/fallbacks, cap A2_MAX_TOKENS (600),
   finish_reason == "length" never trusted (explicit retrieval-order fallback, counted).
B  no reranker: hybrid RRF order, i.e. the candidate order itself.
C  compact reranker: candidates numbered 1..N (never long article ids), one short line each,
   JSON-free comma-separated answer, a cap sized so a normal answer cannot truncate, and a
   byte-stable instruction prefix (instructions first, candidates after) for prompt caching.

What reaches the user is the first 5 of the returned list ("shown cards"). Downstream steps in
graph.py (post-rerank dedup by name+colour, beach per-type cap, price re-sort) are NOT replayed:
the pool is already deduped by the retriever, and those steps are variant-independent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, cast

from src.agents.reranker import _parse_selected, rerank

logger = logging.getLogger(__name__)

TOP_K = 5
A_MAX_TOKENS = 150  # hard-coded in src/agents/reranker.py::rerank; asserted against in tests
# A2 cap, from the stage-2 day-1 variant A ledger (53 live calls, reasoning_effort low, cap 150,
# reports/wave1b/v4e/ledger.jsonl): the 22 calls that finished (stop) used 22-107 reasoning tokens
# (median 77) and 26-64 visible tokens; the 31 calls cut at the cap (finish_reason length) show
# reasoning of 98-148, and 21 of them 148 = the cap minus two visible tokens, so the reasoning
# distribution is CENSORED at ~148 and its true tail is unknown. 600 = ~64 visible + ~536 of
# reasoning room, 3.6x the censoring point. Smaller (400) would still leave 2.3x but this is the
# first run that can show where the tail ends; the stage asserts finish_reason is never "length"
# and counts any hit (assert_no_length_hits). Cost: the budget pre-check reserves prompt + 600.
A2_MAX_TOKENS = 600
# C: visible output is 8 numbers (~3 tokens each incl. separators, so ~24-30 tokens). gpt-oss
# reasoning counts against max_tokens even at low effort and was censored at ~148 on 40% of day-1
# variant A calls (see above), so the old 200 cap (visible 30 + 170 reasoning) would be at risk of
# the same truncation. 400 leaves ~370 for reasoning (2.5x the censoring point) at a small cost
# (C prompts are ~500 tokens). Stage 2 still reads finish_reason from the ledger and counts hits.
C_MAX_TOKENS = 400
C_ASK_N = 8
C_SYSTEM = (
    "You rank fashion products for a shopper's query. Choose the 8 best matches, best first.\n"
    "Rules: a stated gender, colour, category and budget are hard requirements; the occasion "
    "must fit; never pick sleepwear or swimwear for a formal occasion. Each line is: number. "
    "title | price band | colour.\n"
    "Reply with ONLY the product numbers, comma-separated, best first, for example: 4,1,9,2,7,3,"
    "12,5. No words, no JSON."
)
PRICE_BANDS = (500, 1000, 1500, 2000, 3000, 5000, 10000)


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str: ...


@dataclass
class RecordingLLM:
    """Wraps any `generate`-style client; records each call's prompt/output/params/usage.

    `inner` may expose `last_meta` (finish_reason, usage, latency_ms) after a call: the Groq
    adapter and the test mocks do. Exceptions propagate (rerank() turns them into a fallback)."""

    inner: SupportsGenerate
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        rec: dict[str, Any] = {"prompt": prompt, "system": system, "kwargs": dict(kwargs)}
        self.calls.append(rec)
        try:
            out = self.inner.generate(prompt, system=system, **kwargs)
        except Exception as exc:
            rec["error"] = type(exc).__name__  # lets the outcome say budget vs http_error
            raise
        rec["output"] = out
        rec["meta"] = dict(getattr(self.inner, "last_meta", {}) or {})
        return out


# Distinct, counted reasons a rerank turn ends the way it does. This is also the SPEC for the
# production fix (src/agents/reranker.py currently logs everything it swallows as "timeout" and an
# empty selection as "parse_error"). "partial" is a flavour of ok (valid but < 5 picks, topped up).
OUTCOMES = (
    "ok",
    "partial",
    "length",  # finish_reason == "length" and nothing usable (A2: never trusted at all)
    "parse_error",  # complete reply that is not the expected shape
    "empty_selection",  # valid `selected: []` with finish_reason stop = "no exact match"
    "http_error",  # the provider call raised (timeout, 5xx, transport)
    "budget",  # day budget / 429 stop: never a row in practice, the stage aborts
    "no_llm_call",
)
_BUDGET_ERRORS = ("BudgetExhaustedError", "RateLimitStopError", "BusyStopError")


def is_empty_selection(text: str) -> bool:
    """True when the reply is a well-formed JSON object whose `selected` list is empty."""
    import json

    blobs = [text.strip()]
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        blobs.append(m.group())
    for blob in blobs:
        try:
            parsed = json.loads(blob)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("selected"), list):
            return len(parsed["selected"]) == 0
    return False


def classify_reply(
    call: Mapping[str, Any], valid: Sequence[Any] | None, text_empty_ok: bool = True
) -> str:
    """Outcome of one call that was made. `valid` = the parsed valid picks (None/[] = unusable)."""
    if "output" not in call:
        return "budget" if call.get("error") in _BUDGET_ERRORS else "http_error"
    meta = call.get("meta") or {}
    if valid:
        return "partial" if len(valid) < TOP_K else "ok"
    if meta.get("finish_reason") == "length":
        return "length"
    if text_empty_ok and is_empty_selection(str(call["output"])):
        return "empty_selection"
    return "parse_error"


def _outcome(call: Mapping[str, Any] | None, n: int) -> str:
    """Classify variant A's call: no_llm_call / http_error / budget / length / empty_selection /
    parse_error / partial / ok (see OUTCOMES)."""
    if call is None:
        return "no_llm_call"  # <= top_k candidates: rerank() returns them untouched
    if "output" not in call:
        return classify_reply(call, None)  # generate() raised
    return classify_reply(call, _parse_selected(call["output"], n))


class CapOverrideLLM:
    """Forces `max_tokens` on every call of a wrapped client.

    `rerank()` hard-codes max_tokens=150, so variant A2 (same prompt, same parser, same
    fallbacks, larger cap) overrides the cap here instead of forking the production function."""

    def __init__(self, inner: SupportsGenerate, max_tokens: int) -> None:
        self.inner, self.max_tokens = inner, max_tokens

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        kwargs["max_tokens"] = self.max_tokens
        return self.inner.generate(prompt, system=system, **kwargs)


def run_variant_a(
    query: str,
    items: list[dict[str, Any]],
    llm: SupportsGenerate,
) -> dict[str, Any]:
    """Variant A: production `rerank()` on `items`; returns the shown top-5 and call telemetry."""
    return _run_production_rerank("A", query, items, llm, None)


def run_variant_a2(
    query: str,
    items: list[dict[str, Any]],
    llm: SupportsGenerate,
) -> dict[str, Any]:
    """Variant A2: production `rerank()` with the truncation fixed (cap A2_MAX_TOKENS).

    Everything else is variant A: production system prompt, candidate block, parser, partial
    top-up and colour-diversity swap; reasoning_effort stays low (set by the live adapter).
    Contract: a call that still ends with finish_reason == "length" is NEVER trusted: the turn
    falls back EXPLICITLY to retrieval order (variant B's cards), outcome "length" and
    `length_hit` True, a warning is logged, and `assert_no_length_hits` fails the stage check.
    """
    row = _run_production_rerank("A2", query, items, llm, A2_MAX_TOKENS)
    row["length_hit"] = bool(row["hit_cap"])
    if row["length_hit"]:
        logger.warning(
            "[A2] finish_reason=length at cap %s for %r: explicit fallback to retrieval order",
            A2_MAX_TOKENS,
            query,
        )
        row["final_ids"] = [it["article_id"] for it in items[:TOP_K]]
        row["outcome"] = "length"
        row["no_exact_match"] = False
    return row


def assert_no_length_hits(rows: Sequence[Mapping[str, Any]]) -> None:
    """A2's sizing check: the cap must be large enough that finish_reason is never 'length'.

    Raises AssertionError naming the offending queries (the stage reports it, never hides it)."""
    bad = [f"{r.get('query_id', '?')}#{r.get('rep', 1)}" for r in rows if r.get("length_hit")]
    assert not bad, (
        f"A2 hit its {A2_MAX_TOKENS}-token cap (finish_reason=length) on {len(bad)} call(s): "
        f"{bad}; raise A2_MAX_TOKENS from the measured completion_tokens and rerun those queries"
    )


def _run_production_rerank(
    variant: str,
    query: str,
    items: list[dict[str, Any]],
    llm: SupportsGenerate,
    cap: int | None,
) -> dict[str, Any]:
    rec = RecordingLLM(llm)
    # the cap override sits OUTSIDE the recorder so the recorded kwargs show the cap actually sent
    front: SupportsGenerate = rec if cap is None else CapOverrideLLM(rec, cap)
    shown = rerank(query, items, cast(Any, front), top_k=TOP_K)  # duck-types LLMClient.generate
    call = rec.calls[0] if rec.calls else None
    meta = (call or {}).get("meta", {})
    usage = meta.get("usage") or {}
    max_tokens = (call or {}).get("kwargs", {}).get("max_tokens")
    comp = usage.get("completion_tokens")
    hit_cap = bool(
        meta.get("finish_reason") == "length"
        or (comp is not None and max_tokens is not None and comp >= max_tokens)
    )
    outcome = _outcome(call, len(items))
    return {
        "variant": variant,
        "final_ids": [it["article_id"] for it in shown],
        "raw_output": (call or {}).get("output"),
        "outcome": outcome,
        # "no exact match": the model said nothing fits (valid empty list, finish_reason stop).
        # The turn returns retrieval order; it is a valid answer, not a parse failure.
        "no_exact_match": outcome == "empty_selection",
        "max_tokens": max_tokens,
        "finish_reason": meta.get("finish_reason"),
        "hit_cap": hit_cap,
        "parsed": _parse_selected(call["output"], len(items))
        if call and "output" in call
        else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp,
        "latency_ms": meta.get("latency_ms"),
    }


def run_variant_b(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Variant B: RRF order (the candidate order) is the shown order."""
    return {
        "variant": "B",
        "final_ids": [it["article_id"] for it in items[:TOP_K]],
        "outcome": "no_llm_call",  # nothing is called; its zeros are not usage measurements
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0,
    }


def price_band(price: float | None) -> str:
    """Coarse price band label, aligned with the budget thresholds used in the queries."""
    if not isinstance(price, (int, float)):
        return "price n/a"
    for edge in PRICE_BANDS:
        if price <= edge:
            return f"<=₹{edge}"
    return f">₹{PRICE_BANDS[-1]}"


def short_title(title: str, words: int = 12) -> str:
    return " ".join((title or "").split()[:words])


def build_compact_prompt(query: str, items: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """(system, user) for variant C. The system prompt is constant (cacheable prefix)."""
    lines = "\n".join(
        f"{i}. {short_title(it.get('display_name', ''))} | {price_band(it.get('price_inr'))} | "
        f"{it.get('colour') or 'n/a'}"
        for i, it in enumerate(items, 1)
    )
    return C_SYSTEM, f'Query: "{query}"\nProducts:\n{lines}\nBest {C_ASK_N}, numbers only:'


def parse_compact(text: str, n: int) -> list[int]:
    """Valid, distinct product numbers (1..n) in the order given; [] when none are usable."""
    seen: list[int] = []
    for tok in re.findall(r"\d+", text or ""):
        v = int(tok)
        if 1 <= v <= n and v not in seen:
            seen.append(v)
    return seen


def run_variant_c(
    query: str,
    items: list[dict[str, Any]],
    llm: SupportsGenerate,
) -> dict[str, Any]:
    """Variant C: compact reranker. Invalid/empty output falls back to RRF order; fewer than 5
    valid numbers are topped up from RRF order (never duplicates)."""
    if len(items) <= TOP_K:  # same shortcut as production rerank(): nothing to choose between
        return {
            "variant": "C",
            "final_ids": [it["article_id"] for it in items[:TOP_K]],
            "raw_output": None,
            "outcome": "no_llm_call",
            "parsed": [],
            "hit_cap": False,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
        }
    system, user = build_compact_prompt(query, items)
    rec = RecordingLLM(llm)
    try:
        raw = rec.generate(user, system=system, max_tokens=C_MAX_TOKENS, temperature=0)
        picked = parse_compact(raw, len(items))
    except Exception:  # noqa: BLE001 - any provider failure falls back like production does
        raw, picked = None, []
    call = rec.calls[0]
    # C's reply has no JSON, so a reply without numbers is a parse_error (text_empty_ok=False)
    outcome = classify_reply(call, picked, text_empty_ok=False)
    order = [items[i - 1] for i in picked]
    seen = {it["article_id"] for it in order}
    order += [it for it in items if it["article_id"] not in seen]
    meta = call.get("meta", {})
    usage = meta.get("usage") or {}
    comp = usage.get("completion_tokens")
    return {
        "variant": "C",
        "final_ids": [it["article_id"] for it in order[:TOP_K]],
        "raw_output": raw,
        "outcome": outcome,
        "parsed": picked,
        "max_tokens": C_MAX_TOKENS,
        "finish_reason": meta.get("finish_reason"),
        "hit_cap": bool(
            meta.get("finish_reason") == "length" or (comp is not None and comp >= C_MAX_TOKENS)
        ),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": comp,
        "latency_ms": meta.get("latency_ms"),
    }
