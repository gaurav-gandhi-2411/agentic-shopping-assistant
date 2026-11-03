"""Hermetic tests for GET /canary (fake agent + mock LLM, no network, no real index)."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
import api.routes.canary as canary
import api.routes.chat as chat_routes
from api.logging_config import _TokenRedactFilter
from api.main import app
from api.session import InMemorySessionStore

TOKEN = "canary-test-token-0123456789"
_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}


def _item(**over: Any) -> dict:
    base = {
        "article_id": "A1",
        "prod_name": "Kurta",
        "display_name": "Kurta",
        "colour": "Black",
        "product_type": "Kurta",
        "department": "Women",
        "image_url": "https://example.com/a1.jpg",
        "price_inr": 1299.0,
    }
    base.update(over)
    return base


class _Behaviour:
    """Mutable knobs + counters shared with the fake agent/LLM."""

    def __init__(self) -> None:
        self.items: list[dict] = [_item()]
        self.new_items = True
        self.reply = "A lovely black kurta set."
        self.delay_s = 0.0
        self.invocations = 0
        self._lock = threading.Lock()

    def bump(self) -> None:
        with self._lock:
            self.invocations += 1


class _Agent:
    def __init__(self, b: _Behaviour) -> None:
        self._b = b

    def invoke(self, state: dict, **kwargs: Any) -> dict:
        self._b.bump()
        if self._b.delay_s:
            time.sleep(self._b.delay_s)
        return {
            "messages": state["messages"],
            "retrieved_items": list(self._b.items),
            "filters": {},
            "tool_calls": [{"router_decision": {"action": "search"}}],
            "current_plan": json.dumps({"action": "pending_respond", "prompt": "describe"}),
            "final_answer": None,
            "new_items_this_turn": self._b.new_items,
            "out_of_catalogue": False,
            "excluded_colours": None,
        }


class _LLM:
    def __init__(self, b: _Behaviour) -> None:
        self._b = b

    def generate_stream(self, prompt: str, system: str | None = None, **kw: Any) -> Iterator[str]:
        yield self._b.reply

    def generate(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        return self._b.reply


@pytest.fixture
def b(monkeypatch: pytest.MonkeyPatch) -> _Behaviour:
    beh = _Behaviour()
    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.setattr(deps, "_llm", _LLM(beh))
    monkeypatch.setattr(deps, "_config", _CONFIG)
    monkeypatch.setattr(deps, "_db_engine", None)
    monkeypatch.setattr(deps, "get_agent_factory", lambda: lambda memory, streaming=False: _Agent(beh))
    monkeypatch.setattr(chat_routes, "_DEMO_SESSIONS", {})
    monkeypatch.setenv("CANARY_TOKEN", TOKEN)
    monkeypatch.setattr(canary, "_state", canary._State())
    return beh


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)  # no lifespan: no index needed


def _has_token(text: str) -> bool:
    """Boolean so a failing assert never echoes the token value."""
    return TOKEN in text


def _get(client: TestClient, **kw: Any) -> httpx.Response:
    return client.get("/canary", params={"token": TOKEN}, **kw)


# ----------------------------------------------------------------------------- pass / fail


def test_pass_returns_200_and_keyword(client: TestClient, b: _Behaviour) -> None:
    r = _get(client)
    assert r.status_code == 200
    assert '"status":"pass"' in r.text  # the UptimeRobot keyword, compact JSON
    body = r.json()
    assert body["cached"] is False
    assert [q["id"] for q in body["queries"]] == ["q1", "q2", "q3"]
    for q in body["queries"]:
        assert q["cards"] == 1 and q["answer_chars"] > 0 and 0 <= q["latency_s"] < 15
    assert body["checked_at"] and "commit" in body
    assert not _has_token(r.text), "token leaked into response"
    assert b.invocations == 3


def test_header_token_also_accepted(client: TestClient, b: _Behaviour) -> None:
    r = client.get("/canary", headers={"X-Canary-Token": TOKEN})
    assert r.status_code == 200


@pytest.mark.parametrize(
    ("mutate", "assertion"),
    [
        (lambda b: setattr(b, "reply", "   "), "answer"),
        (lambda b: setattr(b, "new_items", False), "cards"),
        (lambda b: setattr(b, "items", []), "cards"),
        (lambda b: setattr(b, "items", [_item(price_inr=0)]), "price"),
        (lambda b: setattr(b, "items", [_item(price_inr=None)]), "price"),
        (lambda b: setattr(b, "items", [_item(price_inr=float("nan"))]), "price"),
        (lambda b: setattr(b, "items", [_item(price_inr=-5)]), "price"),
        (lambda b: setattr(b, "items", [_item(article_id="")]), "price"),
    ],
)
def test_assertion_failures_return_500(
    client: TestClient, b: _Behaviour, mutate: Any, assertion: str
) -> None:
    mutate(b)
    r = _get(client)
    assert r.status_code == 500
    body = r.json()
    assert body["status"] == "fail"
    assert body["failing_assertion"] == assertion
    assert body["query_id"] == "q1"
    assert '"status":"pass"' not in r.text
    assert not _has_token(r.text), "token leaked into response"


def test_owned_item_exempt_from_price(client: TestClient, b: _Behaviour) -> None:
    b.items = [_item(), {**_item(article_id="own"), "price_inr": None, "_owned": True}]
    assert _get(client).status_code == 200


def test_latency_budget_failure(
    client: TestClient, b: _Behaviour, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(canary, "LATENCY_BUDGET_S", 0.0)  # nothing is strictly < 0
    r = _get(client)
    assert r.status_code == 500 and r.json()["failing_assertion"] == "latency"


def test_deadline_returns_timeout(b: _Behaviour, monkeypatch: pytest.MonkeyPatch) -> None:
    b.delay_s = 1.0
    monkeypatch.setattr(canary, "DEADLINE_S", 0.2)

    async def go() -> tuple[httpx.Response, float]:
        # Timed inside the running loop: TestClient / asyncio.run both wait for the
        # abandoned worker threads when they tear their loop down, which a long-lived
        # uvicorn loop never does.
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            t0 = time.monotonic()
            r = await c.get("/canary", params={"token": TOKEN})
            return r, time.monotonic() - t0

    r, elapsed = asyncio.run(go())
    assert elapsed < 0.8, "must answer at the deadline, not wait for the agents"
    assert r.status_code == 500
    assert r.json()["failing_assertion"] == "timeout"
    assert r.json()["query_id"] == "q1"


def test_cost_guard_exhausted(
    client: TestClient, b: _Behaviour, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(canary, "check_daily_cost", lambda brand: False)
    r = _get(client)
    assert r.status_code == 500
    assert r.json()["failing_assertion"] == "cost_guard"
    assert b.invocations == 0  # no LLM work when the guard is exhausted


# ----------------------------------------------------------------------------- cache / caps


def test_second_call_is_served_from_cache(client: TestClient, b: _Behaviour) -> None:
    first, second = _get(client), _get(client)
    assert first.json()["cached"] is False and second.json()["cached"] is True
    assert second.json()["checked_at"] == first.json()["checked_at"]
    assert b.invocations == 3 and canary._state.runs_total == 1


def test_cached_failure_is_also_served_without_rerun(client: TestClient, b: _Behaviour) -> None:
    b.new_items = False
    assert _get(client).status_code == 500
    r = _get(client)
    assert r.status_code == 500 and r.json()["cached"] is True
    assert b.invocations == 3


def test_expiry_reruns(
    client: TestClient, b: _Behaviour, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(canary, "_monotonic", lambda: now[0])
    _get(client)
    now[0] += canary.CACHE_TTL_S - 1
    assert _get(client).json()["cached"] is True
    now[0] += 2
    assert _get(client).json()["cached"] is False
    assert b.invocations == 6


def test_concurrent_calls_run_once(b: _Behaviour) -> None:
    b.delay_s = 0.2

    async def go() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            return await asyncio.gather(*[c.get("/canary", params={"token": TOKEN}) for _ in range(6)])

    responses = asyncio.run(go())
    assert all(r.status_code == 200 for r in responses)
    assert sorted(r.json()["cached"] for r in responses) == [False] + [True] * 5
    assert b.invocations == 3 and canary._state.runs_total == 1


def test_daily_cap_and_utc_rollover(
    client: TestClient, b: _Behaviour, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timezone

    monkeypatch.setattr(canary, "CACHE_TTL_S", 0)  # every call wants a fresh run
    monkeypatch.setattr(canary, "MAX_RUNS_PER_DAY", 2)
    day = [datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(canary, "_utcnow", lambda: day[0])
    assert _get(client).status_code == 200
    assert _get(client).status_code == 200
    capped = _get(client)
    assert capped.status_code == 500 and capped.json()["failing_assertion"] == "daily_cap"
    assert b.invocations == 6  # the capped call ran nothing
    day[0] = datetime(2026, 11, 4, 0, 1, tzinfo=timezone.utc)
    assert _get(client).status_code == 200
    assert b.invocations == 9


# ----------------------------------------------------------------------------- auth / logs


def test_missing_and_wrong_token_are_404(client: TestClient, b: _Behaviour) -> None:
    assert client.get("/canary").status_code == 404
    assert client.get("/canary", params={"token": "nope"}).status_code == 404
    assert client.get("/canary", headers={"X-Canary-Token": "nope"}).status_code == 404
    assert b.invocations == 0


def test_unset_token_is_404_even_with_empty_token(
    client: TestClient, b: _Behaviour, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CANARY_TOKEN")
    assert _get(client).status_code == 404
    assert client.get("/canary", params={"token": ""}).status_code == 404
    monkeypatch.setenv("CANARY_TOKEN", "")
    assert client.get("/canary", params={"token": ""}).status_code == 404
    assert b.invocations == 0


def test_token_never_logged(
    client: TestClient, b: _Behaviour, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    _get(client)
    b.new_items = False
    canary._state.cached_at = None  # force a failing run so error paths log too
    _get(client)
    client.get("/canary", params={"token": "wrong-" + TOKEN})
    # Scope to SERVER loggers: the in-process httpx client logs its own request URL (with the
    # query string) at INFO; that is a test-harness artefact, not something the API emits.
    server = [r for r in caplog.records if not r.name.startswith("httpx")]
    leaked = any(TOKEN in r.getMessage() for r in server)
    assert not leaked, "canary token appeared in a server log record"  # never echo the value
    assert any(r.getMessage().startswith("GET /canary") for r in server)


def test_uvicorn_access_log_masks_token() -> None:
    """uvicorn.access args carry the full path+query; the filter must mask token=."""
    rec = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", f"/canary?token={TOKEN}&x=1", "1.1", 200), None,
    )
    assert _TokenRedactFilter().filter(rec)
    assert not _has_token(rec.getMessage()), "token not masked"
    assert "token=[REDACTED]" in rec.getMessage() and "x=1" in rec.getMessage()


# ----------------------------------------------------------------------------- shared code path


def test_ws_and_canary_use_the_same_turn_function(
    client: TestClient, b: _Behaviour, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real = chat_routes.run_chat_turn

    async def spy(**kwargs: Any) -> Any:
        calls.append(kwargs["message"])
        return await real(**kwargs)

    monkeypatch.setattr(chat_routes, "run_chat_turn", spy)
    monkeypatch.setenv("JWT_VERIFICATION_DISABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

    with client.websocket_connect("/chat/stream") as ws:
        ws.send_json({"type": "user_message", "message": "ws-hello"})
        while ws.receive_json()["type"] != "done":
            pass
    assert calls == ["ws-hello"]

    assert _get(client).status_code == 200
    assert sorted(calls[1:]) == sorted(
        text for _, text in canary.QUERIES
    )
