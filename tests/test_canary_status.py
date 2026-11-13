"""Hermetic tests for the rotating query and the public GET /canary/status.

Reuses the fake agent/LLM fixtures of test_canary_endpoint.py (no network, no real index).
"""
# ruff: noqa: F811  (pytest fixtures `b`/`client` are imported, then used as parameters)

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import api.demo.guards as guards
import api.routes.canary as canary
from api.main import app
from tests.test_canary_endpoint import TOKEN, _Behaviour, b, client  # noqa: F401

PASS_BODY = '{"status":"pass"}'


def _status(c: TestClient, **kw: Any) -> httpx.Response:
    return c.get("/canary/status", **kw)


def _clock(monkeypatch: pytest.MonkeyPatch, start: float = 1000.0) -> list[float]:
    now = [start]
    monkeypatch.setattr(canary, "_monotonic", lambda: now[0])
    return now


# ----------------------------------------------------------------------------- rotation


def test_rotation_same_slot_same_query_and_covers_all_queries() -> None:
    slot = canary.ROTATION_SLOT_S
    base = 5_000_000 * slot
    seen = []
    for i in range(len(canary.QUERIES) * 2):
        start = base + i * slot
        q = canary._query_for(start)
        assert canary._query_for(start + slot - 1) == q  # same slot -> same query
        assert canary._query_for(start + slot) != q  # next slot -> next query
        seen.append(q)
    assert {qid for qid, _ in seen} == {qid for qid, _ in canary.QUERIES}  # all covered
    assert seen[: len(canary.QUERIES)] == seen[len(canary.QUERIES) :]  # period == len(QUERIES)


def test_rotation_survives_cold_start_and_runs_one_query(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    ids = []
    for slot in (1, 2, 3, 4):  # slot 1 = q2: a cold start (empty state) must NOT fall back to q1
        monkeypatch.setattr(canary, "_state", canary._State())  # simulates a fresh process
        monkeypatch.setattr(canary, "_wall", lambda s=slot: s * canary.ROTATION_SLOT_S + 7.0)
        r = client.get("/canary", params={"token": TOKEN})
        assert r.status_code == 200 and len(r.json()["queries"]) == 1
        ids.append(r.json()["queries"][0]["id"])
    assert ids == ["q2", "q3", "q1", "q2"]
    assert b.invocations == 4  # exactly one LLM-backed turn per run


# ----------------------------------------------------------------------------- bodies / auth


def test_pass_body_is_exactly_the_keyword_body(client: TestClient, b: _Behaviour) -> None:  # noqa: F811
    r = _status(client)
    assert r.status_code == 200 and r.text == PASS_BODY
    assert "x-canary-stale" not in r.headers


def test_fail_body_is_exactly_specified(client: TestClient, b: _Behaviour) -> None:  # noqa: F811
    b.new_items = False  # -> "cards" assertion
    r = _status(client)
    assert r.status_code == 503
    assert r.json() == {"status": "fail", "assertion": "cards", "detail": "no product cards"}
    assert '"status":"pass"' not in r.text and "x-canary-stale" not in r.headers


def test_public_status_needs_no_token_and_token_route_still_404(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    assert _status(client).status_code == 200  # no token, no auth
    assert client.get("/canary").status_code == 404  # token route stays protected
    assert client.get("/canary", params={"token": "nope"}).status_code == 404
    monkeypatch.delenv("CANARY_TOKEN")
    assert _status(client).status_code == 200  # public even with the secret unset
    assert client.get("/canary", params={"token": TOKEN}).status_code == 404
    assert _status(client, params={"token": "ignored"}).status_code == 200


def test_status_is_hidden_from_openapi_and_not_demo_rate_limited(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    assert "/canary/status" not in app.openapi()["paths"]

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("demo IP limiter must not be consulted")

    monkeypatch.setattr(guards, "check_ip_rate_limit", boom)
    assert all(_status(client).status_code == 200 for _ in range(40))  # far over any per-IP cap
    assert b.invocations == 1


# ----------------------------------------------------------------------------- shared cache


def test_status_and_canary_share_cache_and_run_counter(
    client: TestClient,
    b: _Behaviour,  # noqa: F811
) -> None:
    assert client.get("/canary", params={"token": TOKEN}).json()["cached"] is False
    assert _status(client).text == PASS_BODY  # served from the run /canary made
    assert canary._state.runs_total == 1 and canary._state.runs_today == 1 and b.invocations == 1


def test_daily_counter_decrements_budget_per_real_run_only(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    now = _clock(monkeypatch)
    for expected in (1, 2, 3):
        assert _status(client).status_code == 200
        assert _status(client).status_code == 200  # cache hit: not counted
        assert canary._state.runs_today == expected
        now[0] += canary.CACHE_TTL_S + 1
    assert b.invocations == 3


# ----------------------------------------------------------------------------- TTLs


def test_pass_ttl_is_30_minutes(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    assert canary.CACHE_TTL_S == 30 * 60
    now = _clock(monkeypatch)
    assert _status(client).status_code == 200
    now[0] += 30 * 60 - 1
    _status(client)
    assert b.invocations == 1  # still inside the window
    now[0] += 2
    _status(client)
    assert b.invocations == 2  # expired -> exactly one more run


def test_fail_ttl_is_5_minutes_then_recovery_shows(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    assert canary.FAIL_CACHE_TTL_S == 5 * 60
    now = _clock(monkeypatch)
    b.new_items = False
    assert _status(client).status_code == 503
    b.new_items = True  # dependency recovers
    now[0] += 5 * 60 - 1
    assert _status(client).status_code == 503 and b.invocations == 1  # no re-run yet
    now[0] += 2
    ok = _status(client)
    assert ok.status_code == 200 and ok.text == PASS_BODY and b.invocations == 2


# ----------------------------------------------------------------------------- stale / unknown


def test_stale_with_cap_exhausted_returns_last_known_pass_with_headers(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.setattr(canary, "MAX_RUNS_PER_DAY", 1)
    now = _clock(monkeypatch)
    assert _status(client).text == PASS_BODY
    now[0] += canary.CACHE_TTL_S + 90
    r = _status(client)
    assert r.status_code == 200 and r.text == PASS_BODY  # last OBSERVED result, flagged stale
    assert r.headers["x-canary-stale"] == "true"
    assert r.headers["x-canary-age-seconds"] == str(canary.CACHE_TTL_S + 90)
    assert b.invocations == 1  # the cap held: no extra run


def test_stale_with_cap_exhausted_never_turns_a_fail_into_a_pass(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.setattr(canary, "MAX_RUNS_PER_DAY", 1)
    now = _clock(monkeypatch)
    b.new_items = False
    assert _status(client).status_code == 503
    b.new_items = True  # healthy now, but the cap forbids observing it
    now[0] += canary.FAIL_CACHE_TTL_S + 10
    r = _status(client)
    assert r.status_code == 503 and r.json()["assertion"] == "cards"  # last known stays a fail
    assert r.headers["x-canary-stale"] == "true"
    assert '"status":"pass"' not in r.text and b.invocations == 1


def test_cap_exhausted_with_fresh_cache_is_not_flagged_stale(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.setattr(canary, "MAX_RUNS_PER_DAY", 1)
    _clock(monkeypatch)
    _status(client)
    r = _status(client)
    assert r.text == PASS_BODY and "x-canary-stale" not in r.headers


@pytest.mark.parametrize("reason", ["cap", "cost_guard", "executor_busy"])
def test_unknown_when_nothing_observed_and_no_run_can_start(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,  # noqa: F811
) -> None:
    if reason == "cap":
        monkeypatch.setattr(canary, "MAX_RUNS_PER_DAY", 0)
    elif reason == "cost_guard":
        monkeypatch.setattr(canary, "check_daily_cost", lambda brand: False)
    else:
        monkeypatch.setattr(canary, "WORKERS", 0)  # pool "full" before any run
    r = _status(client)
    assert r.status_code == 503 and r.json() == {"status": "unknown"}
    assert "x-canary-stale" not in r.headers and b.invocations == 0


def test_stale_when_cost_guard_blocks_returns_last_known(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    now = _clock(monkeypatch)
    assert _status(client).text == PASS_BODY
    monkeypatch.setattr(canary, "check_daily_cost", lambda brand: False)
    now[0] += canary.CACHE_TTL_S + 1
    r = _status(client)
    assert r.status_code == 200 and r.headers["x-canary-stale"] == "true"


def test_utc_rollover_allows_a_new_run_after_cap(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    monkeypatch.setattr(canary, "MAX_RUNS_PER_DAY", 1)
    day = [datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)]
    monkeypatch.setattr(canary, "_utcnow", lambda: day[0])
    now = _clock(monkeypatch)
    _status(client)
    now[0] += canary.CACHE_TTL_S + 1
    assert _status(client).headers.get("x-canary-stale") == "true"
    day[0] = datetime(2026, 11, 4, 0, 1, tzinfo=timezone.utc)
    r = _status(client)
    assert r.status_code == 200 and "x-canary-stale" not in r.headers and b.invocations == 2


# ----------------------------------------------------------------------------- concurrency


def test_concurrent_status_calls_coalesce_into_one_run(b: _Behaviour) -> None:  # noqa: F811
    b.delay_s = 0.2

    async def go() -> list[httpx.Response]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            return await asyncio.gather(*[c.get("/canary/status") for _ in range(12)])

    responses = asyncio.run(go())
    assert all(r.status_code == 200 and r.text == PASS_BODY for r in responses)
    assert b.invocations == 1 and canary._state.runs_total == 1


# ----------------------------------------------------------------------------- no leakage


SECRET = "sk-FAKESECRET-0123456789 Bearer abcdef postgresql://u:hunter2@h/db"


def test_stored_result_with_secret_strings_never_leaks(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    now = _clock(monkeypatch)
    canary._state.cached_at = now[0]
    canary._state.cached_status = 500
    canary._state.cached_body = {
        "status": "fail",
        "failing_assertion": "cards",
        "query_id": "q1",
        "detail": SECRET,
        "queries": [{"answer": SECRET, "cards": [SECRET]}],
        "checked_at": SECRET,
        "commit": SECRET,
    }
    r = _status(client)
    assert r.status_code == 503
    assert r.json() == {"status": "fail", "assertion": "cards", "detail": "no product cards"}
    for frag in ("FAKESECRET", "hunter2", "Bearer", "postgresql", "q1"):
        assert frag not in r.text
    assert not any(SECRET in v for v in r.headers.values())


def test_unrecognised_assertion_name_collapses_to_unknown_without_echo(
    client: TestClient,
    b: _Behaviour,
    monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    now = _clock(monkeypatch)
    canary._state.cached_at = now[0]
    canary._state.cached_status = 500
    canary._state.cached_body = {"status": "fail", "failing_assertion": SECRET, "detail": SECRET}
    r = _status(client)
    assert r.status_code == 503 and "FAKESECRET" not in r.text
    assert r.json()["assertion"] == "unknown"


def test_run_exception_text_and_token_never_reach_public_body(
    client: TestClient,
    b: _Behaviour,  # noqa: F811
) -> None:
    b.factory_error = RuntimeError("secret-detail-xyz")
    r = _status(client)
    assert r.status_code == 503 and r.json()["assertion"] == "transport"
    assert "secret-detail-xyz" not in r.text and "RuntimeError" not in r.text
    assert TOKEN not in r.text
