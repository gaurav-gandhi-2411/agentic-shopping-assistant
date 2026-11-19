"""Throttled, budgeted, resumable Groq chat client for the judges and the reranker variants.

Limits baked in as defaults (architect, stage 2): <= 1 request per 8 s GLOBALLY (across models), a hard
DAILY TOKEN BUDGET per model (judges 150,000; reranker model 100,000 because prod canary traffic
already spends ~100K of its 200K TPD), stop after 3 consecutive 429s, honour Retry-After and the
`x-ratelimit-remaining-*` headers, and a resumable JSONL cache keyed by
(judge, query_id, prompt_hash) so a rerun never repeats a call. Every call lands in a JSONL usage
ledger. Nothing here touches the network unless a live transport is built, which requires the
`--live` flag AND STAGE2_LIVE_OK=1 (see `assert_live_allowed`). The API key is read by name from
GROQ_API_KEY only inside `httpx_transport` and is never printed or logged.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from evals.retrieval.common import (
    JUDGE_DAILY_TOKEN_BUDGET,
    MIN_REQUEST_INTERVAL_S,
    count_tokens,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_CONSECUTIVE_429 = 3
LIVE_ENV_FLAG = "STAGE2_LIVE_OK"


class LiveNotAllowedError(RuntimeError):
    """Raised when a live API call is attempted without --live AND STAGE2_LIVE_OK=1."""


class BudgetExhaustedError(RuntimeError):
    """The daily token budget for a model would be exceeded; the run stops (hard stop)."""


class RateLimitStopError(RuntimeError):
    """Three consecutive 429 responses: stop rather than hammer the provider."""


def assert_live_allowed(live: bool, environ: Mapping[str, str] | None = None) -> None:
    """Refuse live calls unless `live` is set AND STAGE2_LIVE_OK == "1"."""
    environ = os.environ if environ is None else environ
    if not live:
        raise LiveNotAllowedError("live calls require the --live flag")
    if environ.get(LIVE_ENV_FLAG) != "1":
        raise LiveNotAllowedError(f"live calls require {LIVE_ENV_FLAG}=1 in the process settings")


@dataclass
class HttpResult:
    """Minimal HTTP response view (status, lower-cased headers, JSON body)."""

    status: int
    headers: dict[str, str]
    body: dict[str, Any]


Transport = Callable[[dict[str, Any]], HttpResult]  # POST one chat-completions payload


def httpx_transport(timeout_s: float = 60.0) -> Transport:
    """Live transport. Reads GROQ_API_KEY by name; the value is never printed."""
    import httpx

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")

    def post(payload: dict[str, Any]) -> HttpResult:
        r = httpx.post(
            GROQ_URL,
            json=payload,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {key}"},
        )
        try:
            body = r.json()
        except ValueError:
            body = {}
        return HttpResult(r.status_code, {k.lower(): v for k, v in r.headers.items()}, body)

    return post


@dataclass(frozen=True)
class ModelSpec:
    """One Groq model plus its fixed request parameters and daily token budget."""

    name: str  # short judge/variant name used in cache keys
    model: str
    params: Mapping[str, Any] = field(default_factory=dict)
    max_tokens: int = 1500
    daily_budget: int = JUDGE_DAILY_TOKEN_BUDGET


def parse_reset_seconds(value: str | None) -> float:
    """Parse Groq reset headers such as '2m59.56s', '7.66s', '250ms' into seconds."""
    if not value:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|m|s|h)", value):
        total += float(num) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total


class Ledger:
    """Append-only JSONL usage ledger; also answers "tokens used today by model"."""

    def __init__(self, path: Path, now: Callable[[], dt.datetime] | None = None) -> None:
        self.path = path
        self._now = now or (lambda: dt.datetime.now(dt.timezone.utc))
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: dict[str, Any]) -> None:
        rec = {"ts": self._now().isoformat(), **rec}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def tokens_today(self, model: str) -> int:
        if not self.path.exists():
            return 0
        today = self._now().date().isoformat()
        used = 0
        for line in self.path.read_text("utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("model") == model and str(rec.get("ts", ""))[:10] == today:
                used += int(rec.get("total_tokens") or 0)
        return used


class ResultCache:
    """Resumable cache of completed responses keyed by (judge, query_id, prompt_hash).

    A record keeps the response text AND its call metadata (finish_reason, usage, latency_ms) so
    a resumed run reports the same token / latency / length-hit numbers as the original call.
    Empty-content replies (the model spent its whole cap on hidden reasoning) are cached too: the
    tokens were already spent, and repeating the identical request would only spend them again.
    Records written before metadata existed (text only) load with `meta` empty."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._d: dict[str, dict[str, Any]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text("utf-8").splitlines():
                rec = json.loads(line)
                self._d[rec["key"]] = rec

    @staticmethod
    def key(judge: str, query_id: str, prompt_hash: str) -> str:
        return f"{judge}|{query_id}|{prompt_hash}"

    def get(self, key: str) -> str | None:
        """Cached response text, or None when the key was never cached."""
        rec = self._d.get(key)
        return None if rec is None else str(rec["text"])

    def get_record(self, key: str) -> dict[str, Any] | None:
        """Full cached record {"key", "text", "meta"} or None."""
        return self._d.get(key)

    def put(self, key: str, text: str, meta: Mapping[str, Any] | None = None) -> None:
        rec = {"key": key, "text": text, "meta": dict(meta or {})}
        self._d[key] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class GroqRunner:
    """Runs calls through throttle + budget + 429 policy + cache + ledger."""

    def __init__(
        self,
        transport: Transport,
        ledger: Ledger,
        cache: ResultCache | None = None,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_sleep_s: float = 120.0,
    ) -> None:
        self.transport, self.ledger, self.cache = transport, ledger, cache
        self.min_interval_s, self._clock, self._sleep = min_interval_s, clock, sleep
        self.max_sleep_s = max_sleep_s
        self._last: float | None = None
        self.consecutive_429 = 0
        self.n_calls = 0

    def _throttle(self) -> None:
        """GLOBAL pacing: one request per `min_interval_s` across ALL models sharing this runner
        (the architect limit is 1 request / 8 s in total, not per model, so the two judge
        models interleave on a single clock)."""
        if self._last is not None:
            wait = self.min_interval_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def call(
        self,
        spec: ModelSpec,
        query_id: str,
        system: str,
        user: str,
        prompt_hash: str,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """One cached/throttled/budgeted call. Returns {"text", "finish_reason", "usage",
        "latency_ms", "cached", "headers"} or None if the call failed (non-429 error)."""
        key = ResultCache.key(spec.name, query_id, prompt_hash)
        if self.cache and (hit := self.cache.get_record(key)) is not None:
            meta = hit.get("meta") or {}
            return {
                "text": hit["text"],
                "finish_reason": meta.get("finish_reason"),
                "usage": meta.get("usage") or {},
                "latency_ms": meta.get("latency_ms", 0),
                "cached": True,
                "headers": {},
            }
        # Worst case: the full completion cap (reasoning included) is spent. Estimating with a
        # typical completion instead let a run overshoot the daily budget by up to
        # max_tokens - typical (~800 tokens for the 1,500-token judge cap).
        est = count_tokens(system) + count_tokens(user) + spec.max_tokens
        used = self.ledger.tokens_today(spec.model)
        if used + est > spec.daily_budget:
            raise BudgetExhaustedError(
                f"{spec.model}: {used} tokens used today + ~{est} for this call exceeds the "
                f"daily budget {spec.daily_budget}; rerun tomorrow (the cache resumes)"
            )
        payload = {
            "model": spec.model,
            "temperature": 0,
            "max_tokens": spec.max_tokens,
            "seed": 42,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **spec.params,
            **(extra_payload or {}),
        }
        attempt = 0
        while True:
            self._throttle()
            t0 = self._clock()
            res = self.transport(payload)
            latency_ms = round((self._clock() - t0) * 1000)
            self.n_calls += 1
            base = {
                "judge": spec.name,
                "model": spec.model,
                "query_id": query_id,
                "prompt_hash": prompt_hash,
                "status": res.status,
                "latency_ms": latency_ms,
                "remaining_tokens": res.headers.get("x-ratelimit-remaining-tokens"),
                "remaining_requests": res.headers.get("x-ratelimit-remaining-requests"),
            }
            if res.status == 429:
                self.consecutive_429 += 1
                self.ledger.append({**base, "total_tokens": 0, "event": "429"})
                if self.consecutive_429 >= MAX_CONSECUTIVE_429:
                    raise RateLimitStopError(f"{MAX_CONSECUTIVE_429} consecutive 429s; stopping")
                wait = max(float(res.headers.get("retry-after", 0) or 0), 2.0 ** (attempt + 1))
                self._sleep(min(wait, self.max_sleep_s))
                attempt += 1
                continue
            self.consecutive_429 = 0
            if res.status != 200:
                self.ledger.append({**base, "total_tokens": 0, "event": "error"})
                return None
            choice = (res.body.get("choices") or [{}])[0]
            usage = res.body.get("usage") or {}
            text = (choice.get("message") or {}).get("content") or ""
            details = usage.get("completion_tokens_details") or {}
            pdetails = usage.get("prompt_tokens_details") or {}
            self.ledger.append(
                {
                    **base,
                    "event": "ok",
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens") or 0,
                    "reasoning_tokens": details.get("reasoning_tokens"),
                    "cached_tokens": pdetails.get("cached_tokens"),
                }
            )
            # Header-aware slowdown: if the provider says the TPM window is nearly spent,
            # wait for its reset before the next call.
            rem = res.headers.get("x-ratelimit-remaining-tokens")
            if rem is not None and rem.isdigit() and int(rem) < est:
                self._sleep(
                    min(
                        parse_reset_seconds(res.headers.get("x-ratelimit-reset-tokens")),
                        self.max_sleep_s,
                    )
                )
            if self.cache:  # empty replies are cached too (their tokens are already spent)
                self.cache.put(
                    key,
                    text,
                    {
                        "finish_reason": choice.get("finish_reason"),
                        "usage": usage,
                        "latency_ms": latency_ms,
                    },
                )
            return {
                "text": text,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "latency_ms": latency_ms,
                "cached": False,
                "headers": res.headers,
            }
