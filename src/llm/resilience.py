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
        self._late_cb_for: concurrent.futures.Future | None = None  # callback attached once

    def get(self, probe: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Return the probe result dict, or raise RuntimeError(reason) if degraded."""
        with self._lock:
            now = time.monotonic()
            if self._cached and now - self._cached[0] < self.ttl_s:
                return self._unpack(self._cached)
            if self._inflight is None or self._inflight.done():
                self._inflight = self._pool.submit(self._run, probe)
            fut = self._inflight
        try:
            entry = fut.result(timeout=self.timeout_s)
        except concurrent.futures.TimeoutError:
            # Not cached now; if the probe finishes later its outcome is cached then.
            with self._lock:
                attach = self._late_cb_for is not fut
                self._late_cb_for = fut
            # Outside the lock: if fut already finished, add_done_callback runs the
            # callback inline, and _store takes the (non-reentrant) lock itself.
            if attach:
                fut.add_done_callback(self._cache_late_result)
            raise RuntimeError(f"LLM probe timed out after {self.timeout_s:.0f}s") from None
        self._store(fut, entry)
        with self._lock:
            return self._unpack(entry)

    @staticmethod
    def _run(
        probe: Callable[[], dict[str, Any]],
    ) -> tuple[float, dict[str, Any] | None, str | None]:
        """Run the probe and stamp the entry with its COMPLETION time."""
        try:
            result, err = probe(), None
        except Exception as exc:
            result, err = None, f"{type(exc).__name__}: {exc}"
        return time.monotonic(), result, err

    def _store(
        self, fut: concurrent.futures.Future,
        entry: tuple[float, dict[str, Any] | None, str | None],
    ) -> None:
        # Only the most recently started probe may write: an older probe finishing
        # late must not overwrite a newer cached result.
        with self._lock:
            if self._inflight is fut:
                self._cached = entry

    def _cache_late_result(self, fut: concurrent.futures.Future) -> None:
        self._store(fut, fut.result())

    @staticmethod
    def _unpack(entry: tuple[float, dict[str, Any] | None, str | None]) -> dict[str, Any]:
        _, result, err = entry
        if err is not None or result is None:
            raise RuntimeError(err or "LLM probe failed")
        return result
