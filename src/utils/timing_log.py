"""Structured startup and first-call timing logs (stdlib only).

Cold-start cost on Cloud Run is paid by whoever arrives first, so it has to be visible in the
logs, per stage, with the same field names every time:

    {"msg": "startup_stage index_load ms=2310", "event": "startup_stage", "stage": "index_load",
     "ms": 2310}
    {"msg": "first_call dense_encode ms=412", "event": "first_call", "stage": "dense_encode",
     "ms": 412}

``stage()`` logs every time it is used (startup, run once). ``first_call()`` logs only the first
use of a name per process, so hot paths cost one set lookup afterwards.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger("timing")

_SEEN: set[str] = set()
_LOCK = threading.Lock()


def _emit(event: str, name: str, ms: int) -> None:
    logger.info("%s %s ms=%d", event, name, ms, extra={"event": event, "stage": name, "ms": ms})


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time a startup stage and log it (always)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _emit("startup_stage", name, round((time.perf_counter() - t0) * 1000))


@contextmanager
def first_call(name: str) -> Iterator[None]:
    """Time the FIRST use of ``name`` in this process and log it; later uses are untimed."""
    with _LOCK:
        first = name not in _SEEN
        _SEEN.add(name)
    if not first:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _emit("first_call", name, round((time.perf_counter() - t0) * 1000))


def mark(name: str, t0: float) -> float:
    """Log ``name`` as a startup stage that ran from ``t0`` (a ``perf_counter`` value) to now.

    Returns the new reference time, so consecutive stages can be chained:
    ``t = mark("config_load", t)``.
    """
    now = time.perf_counter()
    _emit("startup_stage", name, round((now - t0) * 1000))
    return now


def reset_for_tests() -> None:
    """Forget which first calls were logged (tests only)."""
    with _LOCK:
        _SEEN.clear()
