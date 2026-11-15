"""The three reranking variants compared on the same 20-candidate pools.

A  current production reranker: the REAL `src.agents.reranker.rerank` (production prompt, 150
   max_tokens, production parser and fallbacks) behind an injectable, recording LLM.
B  no reranker: hybrid RRF order, i.e. the candidate order itself.
C  compact reranker: candidates numbered 1..N (never long article ids), one short line each,
   JSON-free comma-separated answer, a cap sized so a normal answer cannot truncate, and a
   byte-stable instruction prefix (instructions first, candidates after) for prompt caching.

What reaches the user is the first 5 of the returned list ("shown cards"). Downstream steps in
graph.py (post-rerank dedup by name+colour, beach per-type cap, price re-sort) are NOT replayed:
the pool is already deduped by the retriever, and those steps are variant-independent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, cast

from src.agents.reranker import _parse_selected, rerank

TOP_K = 5
A_MAX_TOKENS = 150  # hard-coded in src/agents/reranker.py::rerank; asserted against in tests
# C: visible output is 8 numbers (~3 tokens each incl. separators, so ~24-30 tokens). gpt-oss
# reasoning tokens count against max_tokens even at low effort (prod's 150-token JSON answers hit
# the 150 cap in 8/15 logged calls), so the cap carries ~150 tokens of reasoning slack. Stage 2
# must check completion_tokens / finish_reason from the ledger and raise it if any call hits it.
C_MAX_TOKENS = 200
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
        out = self.inner.generate(prompt, system=system, **kwargs)
        rec["output"] = out
        rec["meta"] = dict(getattr(self.inner, "last_meta", {}) or {})
        return out


def _outcome(call: Mapping[str, Any] | None, n: int) -> str:
    """Classify variant A's call the way rerank() logs it: none / timeout / parse_error / partial / ok."""
    if call is None:
        return "no_llm_call"  # <= top_k candidates: rerank() returns them untouched
    if "output" not in call:
        return "timeout"  # generate() raised
    parsed = _parse_selected(call["output"], n)
    if parsed is None:
        return "parse_error"
    return "partial" if len(parsed) < TOP_K else "ok"


def run_variant_a(
    query: str,
    items: list[dict[str, Any]],
    llm: SupportsGenerate,
) -> dict[str, Any]:
    """Variant A: production `rerank()` on `items`; returns the shown top-5 and call telemetry."""
    rec = RecordingLLM(llm)
    shown = rerank(query, items, cast(Any, rec), top_k=TOP_K)  # rec duck-types LLMClient.generate
    call = rec.calls[0] if rec.calls else None
    meta = (call or {}).get("meta", {})
    usage = meta.get("usage") or {}
    max_tokens = (call or {}).get("kwargs", {}).get("max_tokens")
    comp = usage.get("completion_tokens")
    hit_cap = bool(
        meta.get("finish_reason") == "length"
        or (comp is not None and max_tokens is not None and comp >= max_tokens)
    )
    return {
        "variant": "A",
        "final_ids": [it["article_id"] for it in shown],
        "raw_output": (call or {}).get("output"),
        "outcome": _outcome(call, len(items)),
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
        "outcome": "ok",
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
    system, user = build_compact_prompt(query, items)
    rec = RecordingLLM(llm)
    outcome = "ok"
    try:
        raw = rec.generate(user, system=system, max_tokens=C_MAX_TOKENS, temperature=0)
        picked = parse_compact(raw, len(items))
    except Exception:  # noqa: BLE001 - any provider failure falls back like production does
        raw, picked, outcome = None, [], "timeout"
    if outcome == "ok" and not picked:
        outcome = "parse_error"
    elif outcome == "ok" and len(picked) < TOP_K:
        outcome = "partial"
    order = [items[i - 1] for i in picked]
    seen = {it["article_id"] for it in order}
    order += [it for it in items if it["article_id"] not in seen]
    call = rec.calls[0]
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
