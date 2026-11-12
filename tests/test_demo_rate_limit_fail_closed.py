"""W5 / E9: demo rate limits fail CLOSED when the DB limiter is unavailable.

Covers api/demo/guards.py fallback, api/demo/fallback_limiter.py, api/demo/client_ip.py and
the /chat, /chat/stream (WebSocket) and /demo/session call sites.  No network, no real DB.
"""
from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import api.demo.guards as guards
import api.deps as deps
from api.auth import get_current_user_id_or_demo
from api.demo.client_ip import get_client_ip
from api.demo.fallback_limiter import FallbackLimiter
from api.main import app
from api.session import InMemorySessionStore

_ANON = "anon:w5-test"


def _down_engine() -> MagicMock:
    engine = MagicMock()
    engine.begin.side_effect = ConnectionError("db unreachable")
    return engine


def _scope(xff: str | None, peer: str = "10.0.0.1") -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_PER_IP_HOUR_LIMIT", "3")
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "1000")
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)


@pytest.fixture
def rest(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setitem(app.dependency_overrides, get_current_user_id_or_demo, lambda: _ANON)
    # Allowed requests reach the (unconfigured) agent and may 500; only 429-ness matters here.
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Guards: DB unreachable -> still enforced
# ---------------------------------------------------------------------------


def test_db_error_enforces_per_ip_limit() -> None:
    engine = _down_engine()
    results = [guards.check_ip_rate_limit("1.1.1.1", "b", engine)[0] for _ in range(5)]
    assert results == [True, True, True, False, False]
    allowed, retry = guards.check_ip_rate_limit("1.1.1.1", "b", engine)
    assert not allowed and retry > 0
    # A different IP is unaffected.
    assert guards.check_ip_rate_limit("2.2.2.2", "b", engine)[0] is True


def test_missing_engine_enforces_per_ip_limit() -> None:
    results = [guards.check_ip_rate_limit("1.1.1.1", "b", None)[0] for _ in range(4)]
    assert results == [True, True, True, False]


def test_fallback_limit_derived_from_same_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_PER_IP_HOUR_LIMIT", "7")
    results = [guards.check_ip_rate_limit("1.1.1.1", "b", None)[0] for _ in range(9)]
    assert results.count(True) == 7


def test_global_daily_cap_enforced_when_db_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "2")
    engine = _down_engine()
    for _ in range(2):
        assert guards.check_daily_cap("b", engine) is True
        guards.record_request("b", engine)
    assert guards.check_daily_cap("b", engine) is False


def test_global_cap_with_no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "1")
    assert guards.check_daily_cap("b", None) is True
    guards.record_request("b", None)
    assert guards.check_daily_cap("b", None) is False


def test_transition_logged_once_and_recovery_returns_to_db(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(guards, "_DB_RETRY_S", 0.0)  # retry DB on every call
    caplog.set_level(logging.INFO, logger="api.demo.guards")
    engine = MagicMock()
    healthy_conn = MagicMock()
    healthy_conn.execute.return_value.fetchone.return_value = (0,)
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=healthy_conn)
    ctx.__exit__ = MagicMock(return_value=False)

    engine.begin.side_effect = ConnectionError("down")
    for _ in range(3):
        assert guards.check_ip_rate_limit("1.1.1.1", "b", engine)[0] is True
    assert sum("state=memory" in r.getMessage() for r in caplog.records) == 1

    engine.begin.side_effect = None
    engine.begin.return_value = ctx
    assert guards.check_ip_rate_limit("1.1.1.1", "b", engine) == (True, 0)
    assert sum("state=db" in r.getMessage() for r in caplog.records) == 1
    # DB limiter is authoritative again: a DB row at the limit now blocks, memory is not used.
    healthy_conn.execute.return_value.fetchone.return_value = (3,)
    assert guards.check_ip_rate_limit("9.9.9.9", "b", engine)[0] is False


def test_db_skipped_during_retry_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guards, "_DB_RETRY_S", 3600.0)
    engine = _down_engine()
    guards.check_ip_rate_limit("1.1.1.1", "b", engine)
    guards.check_ip_rate_limit("1.1.1.1", "b", engine)
    assert engine.begin.call_count == 1  # second call went straight to memory


# ---------------------------------------------------------------------------
# FallbackLimiter: window, memory bound, concurrency
# ---------------------------------------------------------------------------


def test_sliding_window_expires_with_clock() -> None:
    now = [1_000_000.0]
    lim = FallbackLimiter(lambda: 2, lambda: 100, clock=lambda: now[0])
    assert lim.check_ip("a")[0] and lim.check_ip("a")[0]
    assert lim.check_ip("a")[0] is False
    now[0] += 3601
    assert lim.check_ip("a")[0] is True


def test_per_ip_map_is_bounded_lru() -> None:
    lim = FallbackLimiter(lambda: 5, lambda: 100, max_ips=lambda: 100)
    for i in range(5000):
        lim.check_ip(f"ip-{i}")
    assert lim.tracked_ips() == 100


def test_max_ips_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_FALLBACK_MAX_IPS", "10")
    lim = FallbackLimiter(lambda: 5, lambda: 100)
    for i in range(50):
        lim.check_ip(f"ip-{i}")
    assert lim.tracked_ips() == 10


def test_concurrent_requests_never_exceed_limit() -> None:
    lim = FallbackLimiter(lambda: 10, lambda: 100)
    allowed: list[bool] = []

    def worker() -> None:
        allowed.append(lim.check_ip("same")[0])

    threads = [threading.Thread(target=worker) for _ in range(60)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert allowed.count(True) == 10


# ---------------------------------------------------------------------------
# IP extraction
# ---------------------------------------------------------------------------


def test_client_ip_uses_rightmost_hop_not_spoofable_leftmost() -> None:
    assert get_client_ip(_scope("6.6.6.6, 203.0.113.9")) == "203.0.113.9"
    assert get_client_ip(_scope("7.7.7.7, 203.0.113.9")) == "203.0.113.9"


def test_client_ip_trusted_hops_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert get_client_ip(_scope("6.6.6.6, 203.0.113.9, 35.1.1.1")) == "203.0.113.9"


def test_client_ip_falls_back_to_peer_and_rejects_garbage() -> None:
    assert get_client_ip(_scope(None, peer="198.51.100.4")) == "198.51.100.4"
    assert get_client_ip(_scope("not-an-ip", peer="198.51.100.4")) == "198.51.100.4"


def test_client_ip_ipv6_collapsed_to_slash64() -> None:
    a = get_client_ip(_scope("2001:db8:1:2:aaaa::1"))
    b = get_client_ip(_scope("2001:db8:1:2:bbbb::9"))
    assert a == b == "2001:db8:1:2::"


# ---------------------------------------------------------------------------
# Endpoints with DB unreachable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine_kind", ["down", "none"])
def test_rest_chat_capped_when_db_unreachable(
    monkeypatch: pytest.MonkeyPatch, rest: TestClient, engine_kind: str
) -> None:
    monkeypatch.setattr(deps, "_db_engine", _down_engine() if engine_kind == "down" else None)
    codes = [
        rest.post(
            "/chat", json={"message": "hi"}, headers={"X-Forwarded-For": f"9.9.9.{i}, 203.0.113.9"}
        ).status_code
        for i in range(5)
    ]
    assert all(c != 429 for c in codes[:3])
    assert codes[3:] == [429, 429]  # rotating the spoofable leftmost hop does not help


def test_rest_chat_429_shape_has_retry_after(
    monkeypatch: pytest.MonkeyPatch, rest: TestClient
) -> None:
    monkeypatch.setattr(deps, "_db_engine", None)
    for _ in range(3):
        rest.post("/chat", json={"message": "hi"})
    resp = rest.post("/chat", json={"message": "hi"})
    assert resp.status_code == 429
    assert resp.json()["detail"] == "Rate limit exceeded"
    assert int(resp.headers["Retry-After"]) > 0


def test_rest_chat_global_cap_when_db_down(
    monkeypatch: pytest.MonkeyPatch, rest: TestClient
) -> None:
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "2")
    monkeypatch.setenv("DEMO_PER_IP_HOUR_LIMIT", "1000")
    monkeypatch.setattr(deps, "_db_engine", _down_engine())
    codes = [
        rest.post(
            "/chat", json={"message": "hi"}, headers={"X-Forwarded-For": f"203.0.113.{i}"}
        ).status_code
        for i in range(4)
    ]
    assert codes[:2] != [429, 429] and codes[2:] == [429, 429]


def test_demo_session_capped_when_db_down(
    monkeypatch: pytest.MonkeyPatch, rest: TestClient
) -> None:
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "1")
    monkeypatch.setattr(deps, "_db_engine", None)
    guards.record_request("b", None)
    monkeypatch.setenv("BRAND", "b")
    assert rest.post("/demo/session").status_code == 429


@pytest.mark.parametrize("engine_kind", ["down", "none"])
def test_ws_capped_when_db_unreachable(
    monkeypatch: pytest.MonkeyPatch, engine_kind: str
) -> None:
    monkeypatch.setattr("api.routes.chat.get_current_user_id_ws", lambda ws: _ANON)
    monkeypatch.setattr(deps, "_db_engine", _down_engine() if engine_kind == "down" else None)
    client = TestClient(app, raise_server_exceptions=False)
    hdr = {"X-Forwarded-For": "6.6.6.6, 203.0.113.9"}

    for _ in range(3):  # admitted: server waits for the first frame, we just hang up
        with client.websocket_connect("/chat/stream", headers=hdr):
            pass

    with client.websocket_connect("/chat/stream", headers=hdr) as ws:
        # If (wrongly) admitted, the bad frame makes the server answer "bad_request" instead of
        # blocking forever on receive, so a regression fails the assertion rather than hanging.
        ws.send_text("{}")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["code"] == "rate_limited"
