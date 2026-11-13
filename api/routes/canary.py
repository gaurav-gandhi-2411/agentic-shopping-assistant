"""GET /canary and GET /canary/status: production canary for an external 5-minute pinger.

Each run sends ONE query, rotating over the 3 fixed canary queries (same list and assertion
semantics as scripts/ops/canary.py, PR #80), through ``api.routes.chat.run_chat_turn`` -- the
exact function the WebSocket handler uses for a turn -- but without a socket, so the
cards/prices/latency a real user would get are what is asserted. The rotation is a pure function
of wall-clock time (``_query_for``), so it survives cold starts and agrees across instances.

``GET /canary`` (token): 200 ``{"status":"pass",...}`` or 500 ``{"status":"fail",
"failing_assertion":...}``; the detailed body, used by the deploy smoke.
``GET /canary/status`` (public, no token): 200 ``{"status":"pass"}``, 503
``{"status":"fail","assertion":<name>,"detail":<fixed string>}`` or 503 ``{"status":"unknown"}``.
It shares the cache, single-flight lock and daily run counter with ``/canary``. The monitor
keyword is ``"status":"pass"`` (compact JSON, no spaces). Bodies never contain
tokens, tickets, session ids, user text or exception messages.

Auth: env ``CANARY_TOKEN`` (Secret Manager secret ``CANARY_TOKEN``). Unset, missing or wrong token
all return 404, so the endpoint is invisible. The token is accepted as ``X-Canary-Token`` header
(preferred: never lands in a URL) or a plain ``?token=`` (a percent-encoded param name is
rejected; if the header is present it alone decides). Cloud Run's own request log records the full
request URL including the query string; that cannot be masked from this code (see DEPLOY.md).

Budget (per instance; Cloud Run maxScale is low, so the fleet-wide bound is
``instances x`` these numbers): a pass is cached for 30 min (-> at most 48 runs/day in steady
state, 1 LLM-backed query each; a failure only for FAIL_CACHE_TTL_S = 300 s so recovery shows
within one pinger window) and a UTC-day counter caps real runs at 60/day so restarts (which
empty the cache) cannot cause unbounded runs. When the cap is exhausted ``/canary/status``
serves the LAST KNOWN result flagged ``X-Canary-Stale: true`` (never a pass it did not observe).
See docs/ops/canary-budget.md. A run bypasses the per-IP and daily-request demo limits (it uses a
dedicated internal user, never touches ``demo_rate_limits``/``demo_daily_stats``), so it cannot
block real users; the global daily cost guard still applies and its exhaustion is reported as
``cost_guard``. LLM spend of canary runs is still recorded by the cost reporter like any turn.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.demo.guards import check_daily_cost
from api.logging_config import conversation_id_var
from api.routes import chat as chat_routes
from api.schemas import WSDoneMessage, WSErrorMessage, WSItemsMessage, WSTokenMessage
from src.agents.grounding import GROUNDED_FALLBACK_EMPTY, GROUNDED_FALLBACK_PREFIX
from src.llm.context import llm_user_id_var
from src.llm.resilience import LLM_BUSY_MESSAGE
from src.retrieval.index_store import resolve_brand as _resolve_brand

logger = logging.getLogger(__name__)
router = APIRouter(tags=["canary"], include_in_schema=False)

# Same queries as scripts/ops/canary.py (PR #80).
QUERIES: tuple[tuple[str, str], ...] = (
    ("q1", "sangeet outfit under ₹3000 for women"),
    ("q2", "black kurta set for haldi"),
    ("q3", "formal shirts for office under 1500"),
)
LATENCY_BUDGET_S = 15.0  # send -> done, strictly less (scripts/ops/canary.py LATENCY_BUDGET_MS)
DEADLINE_S = 25.0  # the one query; the pinger is assumed to time out at 30 s
CACHE_TTL_S = 30 * 60  # a PASS is reused this long (bounds spend: <= 48 runs/day)
# A FAIL is reused only briefly: after recovery the monitor must go green within one pinger
# window, not stay red for up to 30 min. 300 s (not 60) because a failing run costs one LLM
# query and a 5-min pinger would otherwise re-run on nearly every hit; the 60/day run cap and
# the cost guard still bound spend.
FAIL_CACHE_TTL_S = 300
MAX_RUNS_PER_DAY = 60  # hard per-instance cap; 30 min cache alone gives <= 48
ROTATION_SLOT_S = 30 * 60  # one query per wall-clock slot; matches the pass TTL
_CANARY_USER_ID = "anon:canary-internal"  # anon: -> in-memory session path, no Postgres user row

_monotonic = time.monotonic  # patched in tests
_wall = time.time  # patched in tests; epoch seconds drive the query rotation


def _query_for(epoch_s: float) -> tuple[str, str]:
    """The query for the wall-clock slot containing ``epoch_s``.

    Pure function of time, not a per-process counter: a cold start (or a second instance)
    lands on the same query for the same slot instead of restarting at q1.
    """
    return QUERIES[int(epoch_s // ROTATION_SLOT_S) % len(QUERIES)]


def _utcnow() -> datetime:
    """Current UTC time (patched in tests)."""
    return datetime.now(timezone.utc)


@dataclass
class _State:
    """Process-local cache + daily counter. Guarded by ``lock``."""

    lock: asyncio.Lock | None = None
    lock_loop: Any = None
    cached_at: float | None = None
    cached_body: dict | None = None
    cached_status: int = 200
    day: str = ""
    runs_today: int = 0
    runs_total: int = 0  # never reset; lets tests/ops see how many real runs happened


_state = _State()


def _get_lock() -> asyncio.Lock:
    """asyncio.Lock bound to the running loop (recreated if the loop changed, e.g. in tests)."""
    loop = asyncio.get_running_loop()
    if _state.lock is None or _state.lock_loop is not loop:
        _state.lock = asyncio.Lock()
        _state.lock_loop = loop
    return _state.lock


def _commit() -> str:
    return os.environ.get("COMMIT_SHA") or os.environ.get("K_REVISION") or "unknown"


def _plain_query_token(raw_query: bytes) -> str | None:
    """Value of the first query param whose RAW name is exactly ``token``.

    Starlette percent-decodes param names, so ``?%74oken=x`` would also authenticate; that
    form is rejected here so the secret can only travel in a shape the access-log filter
    and a human reading the URL recognise.
    """
    for part in raw_query.split(b"&"):
        name, eq, value = part.partition(b"=")
        if eq and name == b"token":
            return unquote_plus(value.decode("latin-1"))
    return None


def _authorised(header_token: str | None, query_token: str | None) -> bool:
    """Header wins: if ``X-Canary-Token`` is present it alone decides (no fallthrough to the
    query string, so a wrong header with a right ``?token=`` is refused); else the query token.
    """
    expected = os.environ.get("CANARY_TOKEN", "")
    if not expected:
        return False  # fail closed: unset secret -> endpoint does not exist
    supplied = header_token if header_token else query_token
    if not supplied:
        return False
    return hmac.compare_digest(supplied.encode(), expected.encode())


class _CountingExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that knows how many submitted calls have not yet returned.

    A timed-out canary turn abandons its worker thread (it cannot be killed); counting
    calls until they truly return tells the next run whether the pool is still wedged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._inflight = 0
        self._count_lock = threading.Lock()

    @property
    def inflight(self) -> int:
        with self._count_lock:
            return self._inflight

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        with self._count_lock:
            self._inflight += 1
        try:
            fut = super().submit(fn, *args, **kwargs)
        except BaseException:
            with self._count_lock:
                self._inflight -= 1
            raise
        fut.add_done_callback(self._done)
        return fut

    def _done(self, _fut: Any) -> None:
        with self._count_lock:
            self._inflight -= 1


# Dedicated pool: canary agent calls never touch the default executor real chat turns share.
WORKERS = 1  # one query per run
_executor = _CountingExecutor(max_workers=WORKERS, thread_name_prefix="canary")


# --------------------------------------------------------------------------- assertions


class _Failure(Exception):
    """An assertion failed; ``name`` is the failing_assertion, ``detail`` a fixed safe string."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(name)
        self.name = name
        self.detail = detail


def _valid_price(price: Any) -> bool:
    return (
        isinstance(price, (int, float))
        and not isinstance(price, bool)
        and math.isfinite(price)
        and price > 0
    )


def _check_cards(cards: list[dict]) -> None:
    for card in cards:
        if card.get("is_owned"):
            continue  # the user's own garment: not for sale, no price
        if not str(card.get("article_id") or "").strip():
            raise _Failure("price", "card with empty article_id")
        if not _valid_price(card.get("price_inr")):
            raise _Failure("price", "card with missing or non-positive price_inr")


def evaluate_turn(
    frames: list[BaseModel], latency_s: float, degraded: str | None = None
) -> dict[str, Any]:
    """Apply the canary assertions to the frames of one turn; return its summary or raise.

    Order matches scripts/ops/canary.py: transport (a ``done`` frame arrived), answer,
    cards, price, latency. ``answer`` is the streamed token text -- what the user sees; with
    the empty-reply guard of PR #82 a blank model reply is replaced by a fallback token, and
    this check still fires if a turn ever reaches ``done`` with no token text.

    ``llm_degraded`` fails the turn when the user got LLM_BUSY_MESSAGE (chain exhausted) or a
    grounded fallback in place of a model reply: retrieval cards do not need the LLM, so
    without it a dead Groq chain would still look healthy. Primary signal is ``degraded``
    (``TurnOutcome.degraded``, set in run_chat_turn); the answer-text match is a backstop.
    """
    errors = [f for f in frames if isinstance(f, WSErrorMessage)]
    done = next((f for f in frames if isinstance(f, WSDoneMessage)), None)
    if errors or done is None:
        raise _Failure("transport", "no done frame" if not errors else f"error frame: {errors[0].code}")

    answer = "".join(f.text for f in frames if isinstance(f, WSTokenMessage))
    if not answer.strip():
        raise _Failure("answer", "empty answer")
    if (
        degraded
        or LLM_BUSY_MESSAGE in answer
        or answer.strip() == GROUNDED_FALLBACK_EMPTY
        or answer.lstrip().startswith(GROUNDED_FALLBACK_PREFIX)
    ):
        raise _Failure("llm_degraded", f"reply not from the model ({degraded or 'text match'})")

    cards: list[dict] = []
    for f in frames:
        if isinstance(f, WSItemsMessage):
            cards.extend(it.model_dump() for it in f.items)
    if not cards:  # fall back to outfit boards, as the frontend renders them
        for variant in done.final_state.get("outfit_variants") or []:
            cards.extend(variant.get("items") or [])
            if cards:
                break
    if not cards:
        raise _Failure("cards", "no cards")
    _check_cards(cards)

    if not latency_s < LATENCY_BUDGET_S:
        raise _Failure("latency", f"latency {latency_s:.1f}s >= {LATENCY_BUDGET_S:.0f}s")

    return {"latency_s": round(latency_s, 2), "cards": len(cards), "answer_chars": len(answer)}


# --------------------------------------------------------------------------- run


async def _run_query(query_id: str, text: str) -> dict[str, Any]:
    """Run one query through the shared turn runner; return summary dict (raises _Failure)."""
    conversation_id = str(uuid.uuid4())
    cid_token = conversation_id_var.set(conversation_id)
    llm_user_id_var.set(_CANARY_USER_ID)
    frames: list[BaseModel] = []

    async def _collect(msg: BaseModel) -> None:
        frames.append(msg)

    t0 = _monotonic()
    try:
        outcome = await chat_routes.run_chat_turn(
            user_id=_CANARY_USER_ID,
            message=text,
            conversation_id=conversation_id,
            send=_collect,
            cancel_event=asyncio.Event(),
            executor=_executor,
        )
    except _Failure:
        raise
    except Exception as exc:
        logger.error("canary %s turn raised %s", query_id, type(exc).__name__)
        raise _Failure("transport", f"turn raised {type(exc).__name__}") from exc
    finally:
        latency_s = _monotonic() - t0
        chat_routes._DEMO_SESSIONS.pop(conversation_id, None)  # don't accumulate canary state
        conversation_id_var.reset(cid_token)

    if outcome.status != "done":
        raise _Failure("transport", f"turn ended: {outcome.status}")
    summary = evaluate_turn(frames, latency_s, outcome.degraded)
    return {"id": query_id, **summary}


async def _run_all() -> tuple[int, dict[str, Any]]:
    """Run the current slot's query under DEADLINE_S; return (http_status, body)."""
    qid, text = _query_for(_wall())
    tasks = {qid: asyncio.create_task(_run_query(qid, text))}
    done, pending = await asyncio.wait(tasks.values(), timeout=DEADLINE_S)
    for t in pending:
        t.cancel()  # agent worker threads can't be killed; we just stop waiting
    results: list[dict] = []
    failure: tuple[str, str, str] | None = None
    for qid, task in tasks.items():  # fixed order -> deterministic "first failure"
        if task in pending:
            failure = failure or ("timeout", qid, f"no result within {DEADLINE_S:.0f}s")
            continue
        exc = task.exception()
        if isinstance(exc, _Failure):
            failure = failure or (exc.name, qid, exc.detail)
        elif exc is not None:  # includes CancelledError; type name only, never the message
            failure = failure or ("transport", qid, f"turn raised {type(exc).__name__}")
        else:
            results.append(task.result())
    base = {"checked_at": _utcnow().isoformat(timespec="seconds"), "commit": _commit()}
    if failure:
        name, qid, detail = failure
        return 500, {
            "status": "fail", "failing_assertion": name, "query_id": qid, "detail": detail,
            "queries": results, **base,
        }
    return 200, {"status": "pass", "queries": results, **base}


def _fail_body(name: str, detail: str) -> dict[str, Any]:
    return {
        "status": "fail", "failing_assertion": name, "query_id": None, "detail": detail,
        "queries": [], "cached": False,
        "checked_at": _utcnow().isoformat(timespec="seconds"), "commit": _commit(),
    }


def _respond(status: int, body: dict[str, Any], cached: bool) -> JSONResponse:
    return JSONResponse(status_code=status, content={**body, "cached": cached})


def _fresh() -> bool:
    ttl = CACHE_TTL_S if _state.cached_status == 200 else FAIL_CACHE_TTL_S
    return (
        _state.cached_body is not None
        and _state.cached_at is not None
        and _monotonic() - _state.cached_at < ttl
    )


async def _result() -> tuple[str, int, dict[str, Any]]:
    """Single-flight refresh shared by /canary and /canary/status.

    Returns ``(kind, status, body)``: ``"cached"`` (fresh cache hit), ``"ran"`` (this call did
    the one run) or ``"blocked"`` (no run could start: daily cap, cost guard or busy pool; the
    body is the explanatory failure, NOT stored as a result).
    """
    if _fresh():
        return "cached", _state.cached_status, _state.cached_body or {}

    async with _get_lock():
        if _fresh():  # a concurrent caller finished the run while we waited
            return "cached", _state.cached_status, _state.cached_body or {}

        today = _utcnow().strftime("%Y-%m-%d")
        if _state.day != today:
            _state.day, _state.runs_today = today, 0
        if _state.runs_today >= MAX_RUNS_PER_DAY:
            logger.error("canary daily cap reached (%d runs)", MAX_RUNS_PER_DAY)
            return "blocked", 500, _fail_body("daily_cap", "canary daily run cap reached")
        if not check_daily_cost(_resolve_brand()):
            logger.error("canary not run: global daily cost guard exhausted")
            return "blocked", 500, _fail_body("cost_guard", "daily cost guard exhausted")

        if _executor.inflight + 1 > WORKERS:
            # Earlier timed-out turns are still holding the dedicated pool's thread (a stalled
            # provider call). Fail fast rather than queue behind them; not cached, not counted.
            logger.error("canary not run: dedicated executor busy (%d stuck)", _executor.inflight)
            return "blocked", 500, _fail_body("executor_busy", "canary workers still busy")

        _state.runs_today += 1
        _state.runs_total += 1
        status, body = await _run_all()
        _state.cached_at, _state.cached_body, _state.cached_status = _monotonic(), body, status
        if status != 200:
            logger.error(
                "canary FAIL assertion=%s query=%s", body["failing_assertion"], body["query_id"]
            )
        return "ran", status, body


@router.get("/canary")
async def canary(
    request: Request,
    x_canary_token: str | None = Header(default=None),
) -> JSONResponse:
    """Run (or return the cached result of) the production canary. See module docstring."""
    token = _plain_query_token(request.scope.get("query_string", b""))
    if not _authorised(x_canary_token, token):
        raise HTTPException(status_code=404, detail="Not Found")
    kind, status, body = await _result()
    return _respond(status, body, cached=kind == "cached")


# Public status body: the detail text is a FIXED string chosen by assertion name, never the
# stored detail, so nothing from a result (answers, errors, ids) can ever reach this endpoint.
_PUBLIC_DETAIL: dict[str, str] = {
    "transport": "turn did not complete",
    "answer": "empty answer",
    "llm_degraded": "reply not from the model",
    "cards": "no product cards",
    "price": "card with invalid price",
    "latency": "latency over budget",
    "timeout": "no result within the deadline",
}


def _public_status(status: int, body: dict[str, Any]) -> tuple[int, dict[str, str]]:
    """Reduce a stored result to the public body. Unknown assertion names collapse to
    ``unknown`` so a malformed result can never put free text on a public endpoint."""
    if status == 200 and body.get("status") == "pass":
        return 200, {"status": "pass"}
    name = body.get("failing_assertion")
    if not isinstance(name, str) or name not in _PUBLIC_DETAIL:
        return 503, {"status": "fail", "assertion": "unknown", "detail": "unrecognised failure"}
    return 503, {"status": "fail", "assertion": name, "detail": _PUBLIC_DETAIL[name]}


@router.get("/canary/status")
async def canary_status() -> JSONResponse:
    """Public pass/fail for the external monitor (no token). See module docstring.

    Fresh cache -> it; stale and a run can start -> one coalesced run; stale and no run can
    start -> the LAST KNOWN result flagged ``X-Canary-Stale: true`` (never a fabricated pass),
    or ``503 {"status":"unknown"}`` when nothing has ever been observed.
    """
    kind, status, body = await _result()
    if kind != "blocked":
        code, public = _public_status(status, body)
        return JSONResponse(status_code=code, content=public)
    if _state.cached_body is None or _state.cached_at is None:
        return JSONResponse(status_code=503, content={"status": "unknown"})
    code, public = _public_status(_state.cached_status, _state.cached_body)
    headers: dict[str, str] = {}
    if not _fresh():
        headers = {
            "X-Canary-Stale": "true",
            "X-Canary-Age-Seconds": str(int(_monotonic() - _state.cached_at)),
        }
    return JSONResponse(status_code=code, content=public, headers=headers)
