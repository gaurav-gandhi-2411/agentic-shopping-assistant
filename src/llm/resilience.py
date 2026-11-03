"""LLM resilience helpers: user-facing busy message, startup check, readiness probe cache."""
from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any, Callable

# Shown to the user when every tier of the chain failed or returned nothing.
# Plain ASCII on purpose: it is streamed as a normal assistant token.
LLM_BUSY_MESSAGE = (
    "I'm getting a lot of requests right now and couldn't put together a reply. "
    "Please try again in a minute."
)


class LLMStartupError(RuntimeError):
    """Raised at startup when the configured primary LLM model cannot be used."""


def check_models_listed(
    listed: set[str], primary: str, fallbacks: list[str]
) -> list[str]:
    """Validate the configured chain against the provider's live model list.

    Raises LLMStartupError if `primary` is not listed. Returns the fallbacks that
    ARE listed; unlisted ones are dropped by the caller (with a warning).
    """
    if primary not in listed:
        raise LLMStartupError(
            f"Configured primary Groq model {primary!r} is not in the live /models list "
            f"({len(listed)} models). Refusing to start: every chat call would fail "
            "(see Groq's 2026-08-16 llama-3.1-8b-instant deprecation). Fix llm.groq_model "
            "in config.yaml."
        )
    return [m for m in fallbacks if m in listed]


class ProbeCache:
    """Cache one expensive health probe for `ttl_s`, never hanging the caller.

    The probe runs on a single worker thread; the caller waits at most
    `timeout_s`. Success and failure are both cached for `ttl_s` so a flapping
    provider is not hammered by repeated /readyz hits. At most one probe is ever
    in flight (a stuck probe is not stacked on).
    """

    def __init__(self, ttl_s: float = 60.0, timeout_s: float = 20.0) -> None:
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._inflight: concurrent.futures.Future | None = None
        self._cached: tuple[float, dict[str, Any] | None, str | None] | None = None

    def get(self, probe: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Return the probe result dict, or raise RuntimeError(reason) if degraded."""
        with self._lock:
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self.ttl_s:
                return self._unpack(self._cached)
            if self._inflight is None or self._inflight.done():
                self._inflight = self._pool.submit(probe)
            fut = self._inflight
        try:
            result: dict[str, Any] | None = fut.result(timeout=self.timeout_s)
            err = None
        except concurrent.futures.TimeoutError:
            # Do not cache: the probe is still running and may yet succeed.
            raise RuntimeError(f"LLM probe timed out after {self.timeout_s:.0f}s") from None
        except Exception as exc:
            result, err = None, f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._cached = (time.monotonic(), result, err)
            return self._unpack(self._cached)

    @staticmethod
    def _unpack(entry: tuple[float, dict[str, Any] | None, str | None]) -> dict[str, Any]:
        _, result, err = entry
        if err is not None or result is None:
            raise RuntimeError(err or "LLM probe failed")
        return result
