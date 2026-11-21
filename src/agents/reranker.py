"""LLM reranker: stage-2 eval variant "C" (compact reranker), promoted to production.

Candidates are numbered 1..N (never long article ids), one short line each; the model answers
with the best 8 numbers, best first. The prompt text, the line format and the 400-token cap are
byte-for-byte the eval's variant C (evals/retrieval/variants.py on feat/v4-retrieval-eval);
tests/test_reranker.py pins them. Retired with the old reranker: the long production prompt, its
JSON answer, the colour-diversity swap (the eval measured C without it) and the 150-token cap
(the cause of the truncations the stage-2 eval found).

Every turn ends in exactly one logged outcome (see OUTCOMES); none is swallowed:
  ok               >= top_k valid distinct picks
  partial          1..top_k-1 valid picks; the rest is filled from retrieval order
  empty_selection  empty reply / `[]` with finish_reason=stop: the model says nothing fits
                   ("no exact match"), a VALID answer; retrieval order is shown
  parse_error      a complete reply that is not a list of integers, or has no valid candidate
  length           finish_reason=length: the reply may be cut mid-number, so none of it is trusted
  http_error       the provider call failed (timeout, 429/5xx, transport, malformed payload)
  budget           the provider's daily quota (TPD/RPD) is spent
  no_llm_call      <= top_k candidates, nothing to choose between (not logged)
Every non-pick outcome returns the first top_k candidates in retrieval order.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from src.llm.client import LLMCallError, LLMClient, LLMCompletion

logger = logging.getLogger(__name__)

OUTCOMES = (
    "ok",
    "partial",
    "empty_selection",
    "parse_error",
    "length",
    "http_error",
    "budget",
    "no_llm_call",
)

# Completion cap. gpt-oss reasoning tokens count against it. Sized from the stage-2 variant C
# ledger (53 live calls at reasoning_effort=low, cap 400, reports/wave1b/v4e/ledger.jsonl):
# completion_tokens min 39, p50 123, p95 286.6, p99 312.9, max 316, finish_reason=stop on all 53.
# 400 = 1.27x the observed max, so length = 0 on those 53. It is a ceiling, not a spend (the mean
# completion was far below it).
RERANK_MAX_TOKENS = 400
# Groq accepts temperature 0 (the eval sent 0 on all 53 calls and got HTTP 200); it is the lowest
# value the API allows, so this is what is sent.
RERANK_TEMPERATURE = 0
RERANK_REASONING_EFFORT = "low"
ASK_N = 8
PRICE_BANDS = (500, 1000, 1500, 2000, 3000, 5000, 10000)

# Byte-stable instruction prefix (prompt caching); the candidates and query go in the user turn.
RERANK_SYSTEM = (
    "You rank fashion products for a shopper's query. Choose the 8 best matches, best first.\n"
    "Rules: a stated gender, colour, category and budget are hard requirements; the occasion "
    "must fit; never pick sleepwear or swimwear for a formal occasion. Each line is: number. "
    "title | price band | colour.\n"
    "Reply with ONLY the product numbers, comma-separated, best first, for example: 4,1,9,2,7,3,"
    "12,5. No words, no JSON."
)


def _price_band(price: Any) -> str:
    """Coarse price band label (the eval's `price_band`)."""
    if not isinstance(price, (int, float)):
        return "price n/a"
    for edge in PRICE_BANDS:
        if price <= edge:
            return f"<=₹{edge}"
    return f">₹{PRICE_BANDS[-1]}"


def _short_title(title: str, words: int = 12) -> str:
    return " ".join((title or "").split()[:words])


def build_prompt(query: str, items: list[dict]) -> tuple[str, str]:
    """(system, user) messages for the reranker. Candidates are numbered 1..len(items)."""
    # `or "n/a"` guards explicit None values (a present key with value None), not just missing keys.
    lines = "\n".join(
        f"{i}. {_short_title(it.get('display_name') or '')} | "
        f"{_price_band(it.get('price_inr'))} | {it.get('colour') or 'n/a'}"
        for i, it in enumerate(items, 1)
    )
    return RERANK_SYSTEM, f'Query: "{query}"\nProducts:\n{lines}\nBest {ASK_N}, numbers only:'


class ReplyFormatError(ValueError):
    """The reply is not a list of integers (bare `4,1,9` or a JSON array)."""


def parse_reply(text: str) -> list[int]:
    """Integers in the reply, in order. [] for an empty reply or `[]`.

    Strict: either a JSON array of integers or comma/space separated integers (a trailing
    period is tolerated). Words, objects or any non-integer token raise ReplyFormatError, so
    a chatty or JSON-object answer is a parse_error rather than a guessed selection."""
    s = text.strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
        except ValueError as exc:
            raise ReplyFormatError(f"invalid JSON list: {exc}") from exc
        if not isinstance(parsed, list) or any(
            isinstance(x, bool) or not isinstance(x, int) for x in parsed
        ):
            raise ReplyFormatError("JSON list must contain only integers")
        return list(parsed)
    s = s.rstrip(".")
    tokens = [t for t in re.split(r"[,\s]+", s) if t]
    if any(not re.fullmatch(r"\d+", t) for t in tokens):
        raise ReplyFormatError("reply contains non-integer tokens")
    return [int(t) for t in tokens]


def validate_ids(raw: list[int], n: int) -> tuple[list[int], int, int]:
    """(valid ids in order, n_unknown, n_duplicate). Valid = 1..n and not seen before."""
    valid: list[int] = []
    seen: set[int] = set()
    n_unknown = n_duplicate = 0
    for x in raw:
        if not 1 <= x <= n:
            n_unknown += 1
        elif x in seen:
            n_duplicate += 1
        else:
            seen.add(x)
            valid.append(x)
    return valid, n_unknown, n_duplicate


def _call_model(llm: LLMClient, system: str, user: str) -> LLMCompletion:
    """One reranker call. Clients with `complete()` (GroqClient) give finish_reason, tokens and
    typed errors; plain `generate()` clients (Ollama, test stubs) give text only (finish_reason
    unknown, so an empty reply there is a parse_error, never a trusted empty selection)."""
    complete = getattr(llm, "complete", None)
    if callable(complete):
        result: LLMCompletion = complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=RERANK_TEMPERATURE,
            max_tokens=RERANK_MAX_TOKENS,
            reasoning_effort=RERANK_REASONING_EFFORT,
        )
        return result
    t0 = time.monotonic()
    try:
        text = llm.generate(
            user, system=system, max_tokens=RERANK_MAX_TOKENS, temperature=RERANK_TEMPERATURE
        )
    except Exception as exc:  # noqa: BLE001 - generate() has no typed errors; typed here, logged by rerank()
        raise LLMCallError("fatal", None, f"{type(exc).__name__}: {exc}") from exc
    return LLMCompletion(text, None, None, None, None, round((time.monotonic() - t0) * 1000))


def _classify(reply: LLMCompletion, n: int, top_k: int) -> tuple[str, list[int], int, int]:
    """(outcome, valid 1-based ids, n_unknown, n_duplicate) for a completed call."""
    if reply.finish_reason == "length":
        return "length", [], 0, 0
    try:
        raw = parse_reply(reply.text)
    except ReplyFormatError:
        return "parse_error", [], 0, 0
    if not raw:
        return ("empty_selection" if reply.finish_reason == "stop" else "parse_error"), [], 0, 0
    valid, n_unknown, n_duplicate = validate_ids(raw, n)
    if not valid:
        return "parse_error", [], n_unknown, n_duplicate
    return ("ok" if len(valid) >= top_k else "partial"), valid, n_unknown, n_duplicate


def rerank(
    query: str,
    items: list[dict],
    llm: LLMClient,
    top_k: int = 5,
) -> list[dict]:
    """Rerank retrieved candidates with the LLM; retrieval order on every non-pick outcome.

    Takes the model's valid candidate numbers in its order, then fills the remainder from
    retrieval order. Returns `top_k` items."""
    if len(items) <= top_k:
        return items

    system, user = build_prompt(query, items)
    t0 = time.monotonic()
    reply: LLMCompletion | None = None
    error_kind: str | None = None
    picks: list[int] = []
    n_unknown = n_duplicate = 0
    try:
        reply = _call_model(llm, system, user)
    except LLMCallError as exc:
        error_kind = exc.kind
        outcome = "budget" if exc.kind == "daily_quota" else "http_error"
    else:
        outcome, picks, n_unknown, n_duplicate = _classify(reply, len(items), top_k)

    selected = [items[i - 1] for i in picks]
    seen = {it["article_id"] for it in selected}
    selected += [it for it in items if it["article_id"] not in seen]

    _log(outcome, reply, error_kind, len(items), len(picks), n_unknown, n_duplicate,
         round((time.monotonic() - t0) * 1000))
    return selected[:top_k]


def _log(
    outcome: str,
    reply: LLMCompletion | None,
    error_kind: str | None,
    n_candidates: int,
    n_selected: int,
    n_unknown: int,
    n_duplicate: int,
    wall_ms: int,
) -> None:
    """One structured line per turn. No query text; `n_selected` = valid model picks before the
    retrieval-order fill."""
    level = logging.INFO if outcome in ("ok", "partial", "empty_selection") else logging.WARNING
    logger.log(
        level,
        json.dumps({
            "event": "rerank",
            "outcome": outcome,
            "no_exact_match": outcome == "empty_selection",
            "finish_reason": reply.finish_reason if reply else None,
            "prompt_tokens": reply.prompt_tokens if reply else None,
            "completion_tokens": reply.completion_tokens if reply else None,
            "reasoning_tokens": reply.reasoning_tokens if reply else None,
            "latency_ms": reply.latency_ms if reply else wall_ms,
            "n_candidates": n_candidates,
            "n_selected": n_selected,
            "n_unknown": n_unknown,
            "n_duplicate": n_duplicate,
            "error_kind": error_kind,
            "model": reply.model if reply else None,
            "max_tokens": RERANK_MAX_TOKENS,
            "temperature": RERANK_TEMPERATURE,
            "reasoning_effort": RERANK_REASONING_EFFORT,
        }),
    )
