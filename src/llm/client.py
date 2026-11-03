from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

from src.llm.context import llm_user_id_var
from src.llm.cost import TurnCost
from src.llm.resilience import LLM_BUSY_MESSAGE, LLMStartupError, check_models_listed

logger = logging.getLogger(__name__)

# Sentinel yielded by chat_stream() on unrecoverable error.
# The WebSocket handler in api/routes/chat.py checks for this value and
# converts it to a WSErrorMessage frame instead of forwarding it as a token.
STREAM_ERROR_SENTINEL = "\x00STREAM_ERROR\x00"


def _parse_retry_after(exc_str: str) -> float:
    """Extract 'Please try again in Xm Ys' from a Groq rate-limit error string."""
    m = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", exc_str)
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return 0.0


@dataclass(frozen=True)
class ErrorInfo:
    """Classification of a provider exception.

    kind: "daily_quota" (TPD/RPD 429: retrying the same model is pointless),
    "rate_limit" (other 429), "server" (5xx), "timeout" (timeouts and
    connection errors), or "fatal" (404 model_not_found, 400, 401, ...).
    """

    kind: str
    status: int | None
    retry_after: float | None


_TIMEOUT_CLASS_NAMES = {"APITimeoutError", "APIConnectionError", "TimeoutError", "ReadTimeout", "ConnectTimeout"}


def classify_error(exc: BaseException) -> ErrorInfo:
    """Map a Groq SDK exception (or a plain Exception carrying the API's text)
    to an ErrorInfo. Falls back to parsing 'Error code: NNN' because tests and
    some wrappers raise bare Exceptions with only the message."""
    text = str(exc)
    status = getattr(exc, "status_code", None)
    if status is None:
        m = re.search(r"Error code:\s*(\d{3})", text)
        status = int(m.group(1)) if m else None
    retry_after: float | None = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
    if retry_after is None:
        retry_after = _parse_retry_after(text) or None
    if status == 429:
        daily = "per day" in text or "(TPD)" in text or "(RPD)" in text
        kind = "daily_quota" if daily else "rate_limit"
    elif status is not None and status >= 500:
        kind = "server"
    elif status == 408 or _TIMEOUT_CLASS_NAMES & {c.__name__ for c in type(exc).__mro__}:
        kind = "timeout"
    else:
        kind = "fatal"
    return ErrorInfo(kind, status, retry_after)


def retry_wait_seconds(info: ErrorInfo, cap_s: float) -> float | None:
    """Seconds to sleep before the single retry, or None if retrying is pointless.

    Backoff is 1-2 s with jitter, raised to Retry-After when the provider sent
    one. If the required wait exceeds `cap_s` there is no retry (the next model
    is a better use of the turn's 90 s WS deadline than sleeping).
    """
    if info.kind not in ("rate_limit", "server", "timeout"):
        return None
    wait = max(1.0 + random.uniform(0.0, 1.0), info.retry_after or 0.0)
    return wait if wait <= cap_s else None


_BOUNDED_SLOTS = threading.BoundedSemaphore(4)


def run_bounded(fn: Callable[[], object], timeout_s: float) -> object:
    """Run fn in a daemon thread and give up after timeout_s (raises TimeoutError).

    Hard wall-clock cap for fallback tiers: whatever sleeps or SDK-internal retries
    they contain, the caller is never held longer than timeout_s. The abandoned
    thread finishes on its own (same trade-off as the WS turn deadline).
    """
    # Cap on simultaneously live fallback threads (abandoned ones included): a
    # sustained provider hang must not grow threads without limit.
    if not _BOUNDED_SLOTS.acquire(blocking=False):
        raise RuntimeError("fallback tier skipped: too many in-flight bounded calls")
    box: dict[str, object] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc
        finally:
            _BOUNDED_SLOTS.release()

    try:
        t = threading.Thread(target=contextvars.copy_context().run, args=(_run,), daemon=True)
        t.start()
    except BaseException:
        # The thread never ran, so its finally will never release: do it here, or
        # repeated start failures would permanently disable the fallback tiers.
        _BOUNDED_SLOTS.release()
        raise
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"fallback tier exceeded {timeout_s:.0f}s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str = None, **kwargs) -> str: ...
    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]: ...
    def chat(self, messages: list[dict], **kwargs) -> str: ...
    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]: ...


class OllamaClient:
    """Wraps ollama.Client. Messages use OpenAI-style {role, content}."""

    def __init__(self, config: dict):
        try:
            import ollama as _ollama_lib
        except ImportError:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        self.config = config
        llm_cfg = config["llm"]
        self.model = llm_cfg["model"]
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        self.timeout = llm_cfg["timeout_seconds"]
        self._client = _ollama_lib.Client(
            host=llm_cfg["host"],
            timeout=self.timeout,
        )
        self.cost_reporter: Callable[[float], None] | None = None

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        # Collect the stream rather than using stream=False — the non-streaming
        # endpoint triggers a runner crash on some Ollama builds (Windows).
        return "".join(self.chat_stream(messages, temperature=temperature, max_tokens=max_tokens))

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        t0 = time.monotonic()
        input_tokens = 0
        output_tokens = 0
        turn_id = str(uuid.uuid4())
        try:
            options = {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
            }
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=True,
            )
            for chunk in stream:
                content = chunk.message.content
                if content:
                    yield content
                # Final chunk carries token counts.
                if getattr(chunk, "done", False):
                    input_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
                    output_tokens = getattr(chunk, "eval_count", 0) or 0
        except Exception as exc:
            logger.error("[ollama] chat_stream error: %s", exc, exc_info=True)
            yield STREAM_ERROR_SENTINEL
        finally:
            latency_ms = round((time.monotonic() - t0) * 1000)
            cost = TurnCost(self.model, input_tokens, output_tokens).usd_cost
            logger.info(
                json.dumps({
                    "event": "llm_call",
                    "provider": "ollama",
                    "model": self.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_tokens": 0,
                    "latency_ms": latency_ms,
                    "usd_cost": round(cost, 8),
                    "user_id": llm_user_id_var.get(""),
                    "turn_id": turn_id,
                })
            )
            if self.cost_reporter is not None:
                self.cost_reporter(cost)

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(messages, **kwargs)


class GroqClient:
    """Wraps the Groq API. Drop-in replacement for OllamaClient on HF Spaces.
    Requires: pip install groq  and  GROQ_API_KEY env var."""

    def __init__(self, config: dict):
        try:
            import groq as _groq_lib
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        llm_cfg = config["llm"]
        self.model = llm_cfg.get("groq_model", "openai/gpt-oss-20b")
        # Ordered Groq model chain: primary, then groq_fallback_models (each Groq
        # model has its own free-tier quota, so a 429/404 on one does not imply
        # the next is down). Duplicates of the primary are dropped.
        self._chain: list[str] = [self.model] + [
            m for m in (llm_cfg.get("groq_fallback_models") or []) if m != self.model
        ]
        # Total wall-clock after which no NEW Groq attempt starts (the WS turn
        # deadline in api/routes/chat.py is 90 s and one turn makes several LLM
        # calls), and the longest Retry-After we are willing to sleep through.
        self._chain_budget_s = float(llm_cfg.get("groq_chain_budget_seconds", 30))
        self._retry_wait_cap_s = float(llm_cfg.get("groq_retry_wait_cap_seconds", 8))
        # Hard cap per OpenRouter/Gemini fallback call (no rate-limit sleeps either):
        # two tiers + the Groq budget must stay well inside the 90 s WS deadline.
        self._fallback_tier_s = float(llm_cfg.get("fallback_tier_budget_seconds", 8))
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]
        # Explicit per-request HTTP timeout, matching OllamaClient's existing
        # timeout_seconds convention above — the groq SDK already defaults to a
        # bounded ~60s read timeout, but making it explicit here closes the audit
        # gap (every LLMClient should have a documented, config-driven timeout)
        # and keeps all four providers consistent. This is NOT the fix for the
        # Wave 7 hang (that was an unbounded application-level retry loop, fixed
        # at the WS turn-deadline in api/routes/chat.py::ws_chat) — a single HTTP
        # call here was already bounded.
        # max_retries=0: the SDK's hidden retries (default 2) would compound with
        # this class's own bounded retry/fallback policy and blow the time budget.
        self._client = _groq_lib.Groq(
            api_key=api_key,
            timeout=llm_cfg.get("groq_timeout_seconds", llm_cfg.get("timeout_seconds", 60)),
            max_retries=0,
        )
        self.cost_reporter: Callable[[float], None] | None = None
        # Lazy fallback chain for TPD (daily quota) exhaustion — see
        # _iter_fallback_tiers(). Config is retained (not the clients
        # themselves) so construction — and each tier's own API-key check —
        # is deferred until actually needed, and re-attempted at most once
        # per GroqClient instance per tier.
        self._config = config
        self._openrouter_client = None
        self._openrouter_unavailable = False
        self._gemini_client = None
        self._gemini_unavailable = False

    def _fallback_config(self) -> dict:
        """Config copy for fallback-tier clients: SDK request timeout = the tier
        budget, so an abandoned run_bounded thread dies close to the cap. The
        primary-provider path never sees this copy."""
        return {**self._config, "llm": {**self._config["llm"], "timeout_seconds": self._fallback_tier_s}}

    def _get_openrouter_fallback(self) -> "object | None":
        """Return a lazily-constructed OpenRouterClient, or None if unavailable.

        Incident (2026-07-29/30): Groq's 500k TPD budget gets exhausted by
        cumulative traffic (eval runs + real usage), and the old behaviour —
        wait up to 10x for 60s+ each on the SAME exhausted provider — produced
        multi-minute hangs that the WS turn-deadline (90s) then had to cut off
        as a user-facing timeout. Falling over to OpenRouter immediately avoids
        that hang. Construction failure (e.g. missing/invalid key) is cached as
        `_openrouter_unavailable` so a dead fallback doesn't add per-call retry
        overhead — see 2026-07-30 finding that the OpenRouter key was dead.

        First tier of the fallback chain — see _iter_fallback_tiers(). Gemini
        (_get_gemini_fallback) was added 2026-07-31 as a second tier once
        OpenRouter's free-tier model (a reasoning model) was found to
        sometimes return empty content when max_tokens is too tight for it
        to finish its hidden reasoning.
        """
        if self._openrouter_unavailable:
            return None
        if self._openrouter_client is None:
            try:
                from src.llm.openrouter_client import OpenRouterClient
                self._openrouter_client = OpenRouterClient(self._fallback_config())
                self._openrouter_client.cost_reporter = self.cost_reporter
            except Exception as exc:
                logger.error("[groq] OpenRouter fallback unavailable: %r", exc)
                self._openrouter_unavailable = True
                return None
        return self._openrouter_client

    def _get_gemini_fallback(self) -> "object | None":
        """Return a lazily-constructed GeminiClient, or None if unavailable.

        Second tier of the fallback chain, tried only when OpenRouter is
        itself unavailable or its call fails/returns empty — see
        _iter_fallback_tiers(). Same lazy-construct-and-cache pattern as
        _get_openrouter_fallback (construction failure cached in
        `_gemini_unavailable` so a dead/missing key isn't retried every call).
        """
        if self._gemini_unavailable:
            return None
        if self._gemini_client is None:
            try:
                from src.llm.gemini_client import GeminiClient
                self._gemini_client = GeminiClient(self._fallback_config())
                self._gemini_client.cost_reporter = self.cost_reporter
            except Exception as exc:
                logger.error("[groq] Gemini fallback unavailable: %r", exc)
                self._gemini_unavailable = True
                return None
        return self._gemini_client

    def _iter_fallback_tiers(self):
        """Yield (provider_name, client) for each fallback tier that
        successfully constructs, in priority order (OpenRouter first,
        Gemini second)."""
        for name, getter in (
            ("openrouter", self._get_openrouter_fallback),
            ("gemini", self._get_gemini_fallback),
        ):
            client = getter()
            if client is not None:
                yield name, client

    def _log_llm_call(
        self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int,
        latency_ms: int, turn_id: str,
    ) -> None:
        cost = TurnCost(model, input_tokens, output_tokens, cached_tokens).usd_cost
        logger.info(
            json.dumps({
                "event": "llm_call",
                "provider": "groq",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "latency_ms": latency_ms,
                "usd_cost": round(cost, 8),
                "user_id": llm_user_id_var.get(""),
                "turn_id": turn_id,
            })
        )
        if self.cost_reporter is not None:
            self.cost_reporter(cost)

    def _log_attempt_failed(
        self, model: str, attempt: int, kind: str, status: int | None,
        latency_ms: int, wait_s: float | None, turn_id: str,
    ) -> None:
        logger.warning(
            json.dumps({
                "event": "llm_attempt_failed",
                "provider": "groq",
                "model": model,
                "attempt": attempt,
                "error_kind": kind,
                "status": status,
                "latency_ms": latency_ms,
                "retry_in_s": None if wait_s is None else round(wait_s, 2),
                "turn_id": turn_id,
            })
        )

    def _log_fallback_served(self, to_provider: str, model: str | None, turn_id: str) -> None:
        logger.warning(
            json.dumps({
                "event": "llm_fallback_served",
                "from_provider": "groq",
                "from_model": self._chain[0],
                "to_provider": to_provider,
                "to_model": model,
                "turn_id": turn_id,
            })
        )

    def _retry_delay(self, info: ErrorInfo, attempt: int, deadline: float) -> float | None:
        """Seconds to wait before the single retry, or None to move to the next model."""
        if attempt != 1:
            return None
        wait = retry_wait_seconds(info, self._retry_wait_cap_s)
        if wait is not None and time.monotonic() + wait >= deadline:
            return None
        return wait

    def _chat_one(
        self, model: str, messages: list[dict], temperature: float | None,
        max_tokens: int | None, turn_id: str, deadline: float, *,
        retry: bool = True, allow_empty: bool = False,
    ) -> str | None:
        """Call one Groq model: initial attempt plus (if retry) one backoff retry on
        429/5xx/timeout. Returns the content, or None if this model failed or
        returned empty content (the caller then moves to the next model)."""
        for attempt in (1, 2):
            t_start = time.monotonic()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                )
            except Exception as exc:
                info = classify_error(exc)
                wait = self._retry_delay(info, attempt, deadline) if retry else None
                self._log_attempt_failed(
                    model, attempt, info.kind, info.status,
                    round((time.monotonic() - t_start) * 1000), wait, turn_id,
                )
                if wait is None:
                    return None
                time.sleep(wait)
                continue
            latency_ms = round((time.monotonic() - t_start) * 1000)
            try:
                content = resp.choices[0].message.content
                usage = resp.usage
                cached = 0
                if usage and getattr(usage, "prompt_tokens_details", None):
                    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                self._log_llm_call(
                    model, usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0, cached, latency_ms, turn_id,
                )
            except (AttributeError, IndexError, TypeError):
                # Empty `choices` or an unexpected payload shape: a failed model,
                # never an exception out of chat().
                self._log_attempt_failed(model, attempt, "malformed", None, latency_ms, None, turn_id)
                return None
            if content and content.strip():
                return content
            if allow_empty:
                return ""
            # Reasoning models can spend the whole max_tokens budget thinking and
            # return content=None: that is a failed model, not a blank reply.
            self._log_attempt_failed(model, attempt, "empty", None, latency_ms, None, turn_id)
            return None
        return None

    def _fallback_chat(
        self, messages: list[dict], temperature: float | None, max_tokens: int | None,
        turn_id: str, *, accept_empty: bool = False,
    ) -> "tuple[str, str] | None":
        """Walk the OpenRouter -> Gemini tiers. Returns (tier_name, text) or None."""
        tried: list[str] = []
        for name, fb_client in self._iter_fallback_tiers():
            tried.append(name)
            try:
                result = run_bounded(
                    lambda c=fb_client: c.chat(
                        messages, temperature=temperature, max_tokens=max_tokens, bounded=True
                    ),
                    self._fallback_tier_s,
                )
            except Exception as fb_exc:
                logger.warning("[groq->%s fallback] failed: %r, trying next tier", name, fb_exc)
                continue
            # Empty/None content is not an exception on either provider (OpenRouter's
            # free reasoning model can burn max_tokens on hidden reasoning; Gemini
            # returns empty on safety-filtered output): treat as a failed tier.
            if (not result or not result.strip()) and not accept_empty:
                logger.warning("[groq->%s fallback] returned empty response, trying next tier", name)
                continue
            self._log_fallback_served(name, None, turn_id)
            return name, result or ""
        logger.error(json.dumps({
            "event": "llm_fallback_exhausted", "provider_attempts": tried, "turn_id": turn_id,
        }))
        return None

    def chat(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Groq models in order (each: attempt + one backoff retry) -> OpenRouter ->
        Gemini -> LLM_BUSY_MESSAGE. Never returns empty and never waits unboundedly:
        the Groq portion stops starting new attempts after groq_chain_budget_seconds
        (the WS turn deadline is 90s, api/routes/chat.py)."""
        turn_id = str(uuid.uuid4())
        deadline = time.monotonic() + self._chain_budget_s
        for i, model in enumerate(self._chain):
            if time.monotonic() >= deadline:
                break
            result = self._chat_one(model, messages, temperature, max_tokens, turn_id, deadline)
            if result is not None:
                if i > 0:
                    self._log_fallback_served("groq", model, turn_id)
                return result
        served = self._fallback_chat(messages, temperature, max_tokens, turn_id)
        if served is not None:
            return served[1]
        logger.error(json.dumps({"event": "llm_busy_message", "turn_id": turn_id}))
        return LLM_BUSY_MESSAGE

    def probe(self) -> dict:
        """Minimal real LLM call (max_tokens=1) through the whole chain, for /readyz.

        No retry sleeps. Success means "the provider answered", so empty content is
        accepted (a reasoning model may spend the single token on reasoning).
        Returns {"tier", "model", "latency_ms"}; raises RuntimeError if nothing answered.
        """
        msgs = [{"role": "user", "content": "ping"}]
        turn_id = str(uuid.uuid4())
        for model in self._chain:
            t0 = time.monotonic()
            res = self._chat_one(
                model, msgs, 0, 1, turn_id, time.monotonic() + 10, retry=False, allow_empty=True
            )
            if res is not None:
                return {"tier": "groq", "model": model,
                        "latency_ms": round((time.monotonic() - t0) * 1000)}
        t0 = time.monotonic()
        served = self._fallback_chat(msgs, 0, 1, turn_id, accept_empty=True)
        if served is not None:
            return {"tier": served[0], "model": None,
                    "latency_ms": round((time.monotonic() - t0) * 1000)}
        raise RuntimeError(f"all LLM tiers failed (groq models tried: {self._chain})")

    def verify_models(self, timeout_s: float = 5.0) -> str:
        """Startup check against Groq's live GET /models (not a chat call).

        - primary not listed             -> raises LLMStartupError (revision fails).
        - Groq rejects the key (401/403) -> raises LLMStartupError (fail closed: this
          revision could never serve).
        - /models unreachable (timeout, 5xx, 429, network) -> ERROR log and continue,
          returning "unverified". Fail-open on purpose: a Groq outage at deploy time
          must not make deploys impossible, and /readyz's real LLM call still reports
          an actually broken chain.
        - unlisted fallbacks             -> warning, dropped from the chain.
        Returns "verified" or "unverified".
        """
        try:
            page = self._client.with_options(timeout=timeout_s, max_retries=0).models.list()
            listed = {m.id for m in page.data}
        except Exception as exc:
            info = classify_error(exc)
            if info.status in (401, 403):
                raise LLMStartupError(
                    f"Groq /models rejected the API key (HTTP {info.status}); "
                    "this revision could not serve chat."
                ) from exc
            logger.error(json.dumps({
                "event": "llm_startup_check_unverified", "error_kind": info.kind,
                "status": info.status, "primary": self._chain[0],
            }))
            return "unverified"
        kept = check_models_listed(listed, self._chain[0], self._chain[1:])
        for dropped in self._chain[1:]:
            if dropped not in kept:
                logger.warning(json.dumps({"event": "llm_fallback_model_unlisted", "model": dropped}))
        self._chain = [self._chain[0], *kept]
        logger.info(json.dumps({"event": "llm_startup_check_ok", "chain": self._chain}))
        return "verified"

    def chat_stream(
        self,
        messages: list[dict],
        temperature: float = None,
        max_tokens: int = None,
    ) -> Iterator[str]:
        """Streaming twin of chat(). Failover only happens before the first token
        reaches the caller; an empty stream is a failure, and chain exhaustion yields
        LLM_BUSY_MESSAGE as ordinary text. A failure after tokens were already sent
        still yields STREAM_ERROR_SENTINEL (the partial reply cannot be unsent)."""
        turn_id = str(uuid.uuid4())
        deadline = time.monotonic() + self._chain_budget_s
        model_used = self._chain[0]
        input_tokens = output_tokens = cached_tokens = 0
        t_start = time.monotonic()
        yielded_any = False
        try:
            for i, model in enumerate(self._chain):
                for attempt in (1, 2):
                    if time.monotonic() >= deadline:
                        break
                    model_used, t_start = model, time.monotonic()
                    input_tokens = output_tokens = cached_tokens = 0
                    try:
                        stream = self._client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature if temperature is not None else self.default_temperature,
                            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                            stream=True,
                            # stream_options is not in the groq SDK's TypedDict; pass via
                            # extra_body so the API returns a usage chunk for cost tracking.
                            extra_body={"stream_options": {"include_usage": True}},
                        )
                        for chunk in stream:
                            if chunk.choices:
                                content = chunk.choices[0].delta.content
                                if content:
                                    if not yielded_any and i > 0:
                                        self._log_fallback_served("groq", model, turn_id)
                                    yielded_any = True
                                    yield content
                            if chunk.usage:
                                input_tokens = chunk.usage.prompt_tokens or 0
                                output_tokens = chunk.usage.completion_tokens or 0
                                details = getattr(chunk.usage, "prompt_tokens_details", None)
                                if details:
                                    cached_tokens = getattr(details, "cached_tokens", 0) or 0
                    except Exception as exc:
                        if yielded_any:
                            logger.error("[groq] chat_stream error mid-stream: %s", exc, exc_info=True)
                            yield STREAM_ERROR_SENTINEL
                            return
                        info = classify_error(exc)
                        wait = self._retry_delay(info, attempt, deadline)
                        self._log_attempt_failed(
                            model, attempt, info.kind, info.status,
                            round((time.monotonic() - t_start) * 1000), wait, turn_id,
                        )
                        if wait is None:
                            break
                        time.sleep(wait)
                        continue
                    if yielded_any:
                        return
                    # Stream ended without a single content token: failed model.
                    self._log_attempt_failed(
                        model, attempt, "empty", None,
                        round((time.monotonic() - t_start) * 1000), None, turn_id,
                    )
                    break
            # Groq models exhausted. Fully buffer fallback tiers: their clients catch
            # their own errors and yield STREAM_ERROR_SENTINEL instead of raising, and
            # buffering lets an empty/errored tier be rejected before committing to it.
            for name, fb_client in self._iter_fallback_tiers():
                try:
                    buffer = run_bounded(
                        lambda c=fb_client: list(c.chat_stream(
                            messages, temperature=temperature, max_tokens=max_tokens,
                            bounded=True,
                        )),
                        self._fallback_tier_s,
                    )
                except Exception as fb_exc:
                    logger.warning("[groq->%s fallback] failed: %r, trying next tier", name, fb_exc)
                    continue
                if STREAM_ERROR_SENTINEL in buffer or not "".join(buffer).strip():
                    logger.warning("[groq->%s fallback] returned empty response, trying next tier", name)
                    continue
                self._log_fallback_served(name, None, turn_id)
                yield from buffer
                return
            logger.error(json.dumps({"event": "llm_busy_message", "turn_id": turn_id}))
            yield LLM_BUSY_MESSAGE
        finally:
            self._log_llm_call(
                model_used, input_tokens, output_tokens, cached_tokens,
                round((time.monotonic() - t_start) * 1000), turn_id,
            )

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(messages, **kwargs)


def get_llm_client(config: dict) -> "OllamaClient | GroqClient":
    provider = os.environ.get("LLM_PROVIDER") or config["llm"]["provider"]
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "groq":
        return GroqClient(config)
    if provider == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient(config)
    if provider == "openrouter":
        from src.llm.openrouter_client import OpenRouterClient
        return OpenRouterClient(config)
    raise NotImplementedError(f"Unknown LLM provider: {provider!r}. Supported: ollama, groq, gemini, openrouter")
