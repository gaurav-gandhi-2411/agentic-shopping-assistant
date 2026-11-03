"""GET /canary: production canary for an external 5-minute pinger (UptimeRobot keyword monitor).

Runs the 3 fixed canary queries (same list and assertion semantics as scripts/ops/canary.py,
PR #80) through ``api.routes.chat.run_chat_turn`` -- the exact function the WebSocket handler
uses for a turn -- but without a socket, so the cards/prices/latency a real user would get are
what is asserted.

Response: 200 ``{"status":"pass",...}`` or 500 ``{"status":"fail","failing_assertion":...}``.
The monitor keyword is ``"status":"pass"`` (compact JSON, no spaces). Bodies never contain
tokens, tickets, session ids, user text or exception messages.

Auth: env ``CANARY_TOKEN`` (Secret Manager ``asa-canary-token``). Unset, missing or wrong token
all return 404, so the endpoint is invisible. The token is accepted as ``X-Canary-Token`` header
(preferred: never lands in a URL) or ``?token=``. Cloud Run's own request log records the full
request URL including the query string; that cannot be masked from this code (see DEPLOY.md).

Budget (per instance; Cloud Run maxScale is low, so the fleet-wide bound is
``instances x`` these numbers): a result is cached for 30 min (-> at most 48 runs/day in steady
state) and a UTC-day counter caps real runs at 60/day so restarts (which empty the cache) cannot
cause unbounded runs. A run bypasses the per-IP and daily-request demo limits (it uses a
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
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.demo.guards import check_daily_cost
from api.logging_config import conversation_id_var
from api.routes import chat as chat_routes
from api.schemas import WSDoneMessage, WSErrorMessage, WSItemsMessage, WSTokenMessage
from src.llm.context import llm_user_id_var
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
DEADLINE_S = 25.0  # all queries; the pinger is assumed to time out at 30 s
CACHE_TTL_S = 30 * 60
MAX_RUNS_PER_DAY = 60  # hard per-instance cap; 30 min cache alone gives <= 48
_CANARY_USER_ID = "anon:canary-internal"  # anon: -> in-memory session path, no Postgres user row

_monotonic = time.monotonic  # patched in tests
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


def _authorised(header_token: str | None, query_token: str | None) -> bool:
    expected = os.environ.get("CANARY_TOKEN", "")
    if not expected:
        return False  # fail closed: unset secret -> endpoint does not exist
    exp = expected.encode()
    ok = False
    for supplied in (header_token, query_token):
        if supplied and hmac.compare_digest(supplied.encode(), exp):
            ok = True
    return ok


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
    frames: list[BaseModel], latency_s: float
) -> dict[str, Any]:
    """Apply the canary assertions to the frames of one turn; return its summary or raise.

    Order matches scripts/ops/canary.py: transport (a ``done`` frame arrived), answer,
    cards, price, latency. ``answer`` is the streamed token text -- what the user sees; with
    the empty-reply guard of PR #82 a blank model reply is replaced by a fallback token, and
    this check still fires if a turn ever reaches ``done`` with no token text.
    """
    errors = [f for f in frames if isinstance(f, WSErrorMessage)]
    done = next((f for f in frames if isinstance(f, WSDoneMessage)), None)
    if errors or done is None:
        raise _Failure("transport", "no done frame" if not errors else f"error frame: {errors[0].code}")

    answer = "".join(f.text for f in frames if isinstance(f, WSTokenMessage))
    if not answer.strip():
        raise _Failure("answer", "empty answer")

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
    summary = evaluate_turn(frames, latency_s)
    return {"id": query_id, **summary}


async def _run_all() -> tuple[int, dict[str, Any]]:
    """Run all queries concurrently under DEADLINE_S; return (http_status, body)."""
    tasks = {qid: asyncio.create_task(_run_query(qid, text)) for qid, text in QUERIES}
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
    return (
        _state.cached_body is not None
        and _state.cached_at is not None
        and _monotonic() - _state.cached_at < CACHE_TTL_S
    )


@router.get("/canary")
async def canary(
    token: str | None = Query(default=None),
    x_canary_token: str | None = Header(default=None),
) -> JSONResponse:
    """Run (or return the cached result of) the production canary. See module docstring."""
    if not _authorised(x_canary_token, token):
        raise HTTPException(status_code=404, detail="Not Found")

    if _fresh():
        return _respond(_state.cached_status, _state.cached_body or {}, cached=True)

    async with _get_lock():
        if _fresh():  # a concurrent caller finished the run while we waited
            return _respond(_state.cached_status, _state.cached_body or {}, cached=True)

        today = _utcnow().strftime("%Y-%m-%d")
        if _state.day != today:
            _state.day, _state.runs_today = today, 0
        if _state.runs_today >= MAX_RUNS_PER_DAY:
            logger.error("canary daily cap reached (%d runs)", MAX_RUNS_PER_DAY)
            return _respond(500, _fail_body("daily_cap", "canary daily run cap reached"), False)
        if not check_daily_cost(_resolve_brand()):
            logger.error("canary not run: global daily cost guard exhausted")
            return _respond(500, _fail_body("cost_guard", "daily cost guard exhausted"), False)

        _state.runs_today += 1
        _state.runs_total += 1
        status, body = await _run_all()
        _state.cached_at, _state.cached_body, _state.cached_status = _monotonic(), body, status
        if status != 200:
            logger.error(
                "canary FAIL assertion=%s query=%s", body["failing_assertion"], body["query_id"]
            )
        return _respond(status, body, cached=False)
