"""Startup warm-up: pay the first-call costs before the instance reports ready.

On a cold Cloud Run instance the first request used to pay for lazy imports, first-call caches
(footwear floors, BM25 postings pages, mask construction), thread-pool/BLAS spin-up and the
first run of every graph node. ``run_warmup`` runs one real, budgeted outfit turn through the
compiled-graph code path, plus a plain and a masked search per gender, with a stub LLM that
raises: rationales take the deterministic template path, so there is NO LLM call, NO network and
NO API key involved.

Contract (tested):
  * never raises: any failure is caught and logged once, and startup continues;
  * bounded: the work runs on a daemon thread and startup waits at most
    ``STARTUP_WARMUP_TIMEOUT_S`` (default 30 s); on timeout startup continues and the thread
    keeps warming in the background;
  * ``STARTUP_WARMUP=0`` (or false/off/no) disables it.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.utils import timing_log

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0
# The canary's own query (budgeted, women, sangeet): the exact path whose first call was slow.
WARMUP_OUTFIT_QUERY = "sangeet outfit under ₹3000 for women"
# One plain search and one masked search per gender covers the dense encoder, FAISS (plain and
# IDSelector-masked), BM25 and the mask construction for both genders.
WARMUP_SEARCHES: tuple[tuple[str, dict | None], ...] = (
    ("saree for wedding", None),
    ("kurta set", {"gender": "women"}),
    ("kurta set", {"gender": "men"}),
)


class WarmupLLM:
    """Stub LLM for the warm-up turn: every call raises, so callers use their template paths."""

    model = "warmup-stub"

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("warm-up: no LLM")

    def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("warm-up: no LLM")


@dataclass
class WarmupResult:
    ok: bool = False
    timed_out: bool = False
    elapsed_ms: int = 0
    steps_ms: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def warmup_enabled(env: dict[str, str] | None = None) -> bool:
    """True unless STARTUP_WARMUP is 0/false/off/no."""
    value = (os.environ if env is None else env).get("STARTUP_WARMUP", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def warmup_timeout_s(env: dict[str, str] | None = None) -> float:
    raw = (os.environ if env is None else env).get("STARTUP_WARMUP_TIMEOUT_S", "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S
    return value if value > 0 else DEFAULT_TIMEOUT_S


def _outfit_turn(retriever: Any, catalogue_df: Any, config: dict) -> None:
    """One real outfit turn through the graph code path with the stub LLM (template rationales)."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    llm = WarmupLLM()
    agent = build_graph(
        retriever,
        catalogue_df,
        llm,  # type: ignore[arg-type]  # structural stub: every call raises -> template paths
        config,
        streaming_mode=False,
    )
    memory = ConversationMemory(llm, config)  # type: ignore[arg-type]
    agent.invoke(
        {
            "messages": [{"role": "user", "content": WARMUP_OUTFIT_QUERY}],
            "user_query": WARMUP_OUTFIT_QUERY,
            "current_plan": None,
            "tool_calls": [],
            "retrieved_items": [],
            "filters": {},
            "final_answer": None,
            "iteration": 0,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
            "anchor_article_id": None,
            "outfit_rationale": None,
            "outfit_variants": None,
            "_memory": memory,
        }
    )


def _run_steps(
    retriever: Any,
    catalogue_df: Any,
    config: dict,
    result: WarmupResult,
    extra_steps: dict[str, Callable[[], None]] | None = None,
) -> None:
    for query, filters in WARMUP_SEARCHES:
        name = f"search_{(filters or {}).get('gender', 'plain')}"
        t0 = time.perf_counter()
        retriever.search(query, top_k=10, filters=filters)
        result.steps_ms[name] = round((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    _outfit_turn(retriever, catalogue_df, config)
    result.steps_ms["outfit_turn"] = round((time.perf_counter() - t0) * 1000)
    for name, step in (extra_steps or {}).items():  # e.g. the CLIP stack (loads index + encoder)
        t0 = time.perf_counter()
        step()
        result.steps_ms[name] = round((time.perf_counter() - t0) * 1000)


def run_warmup(
    retriever: Any,
    catalogue_df: Any,
    config: dict,
    *,
    timeout_s: float | None = None,
    extra_steps: dict[str, Callable[[], None]] | None = None,
) -> WarmupResult:
    """Run the warm-up, bounded by ``timeout_s``. Never raises; returns what happened."""
    limit = warmup_timeout_s() if timeout_s is None else timeout_s
    result = WarmupResult()
    t0 = time.perf_counter()

    def _target() -> None:
        try:
            _run_steps(retriever, catalogue_df, config, result, extra_steps)
            result.ok = True
        except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
            result.error = f"{type(exc).__name__}: {exc}"
            logger.warning("startup warm-up failed (continuing, ready anyway): %s", result.error)

    worker = threading.Thread(target=_target, name="startup-warmup", daemon=True)
    worker.start()
    worker.join(limit)
    result.elapsed_ms = round((time.perf_counter() - t0) * 1000)
    if worker.is_alive():
        result.timed_out = True
        logger.warning(
            "startup warm-up still running after %.0fs; continuing startup "
            "(it keeps warming in the background)",
            limit,
        )
    timing_log.mark("warmup", t0)
    logger.info(
        "startup warm-up finished ok=%s timed_out=%s elapsed_ms=%d steps=%s",
        result.ok,
        result.timed_out,
        result.elapsed_ms,
        result.steps_ms,
    )
    return result
