"""POST /events hardening: auth, identity, rate limits, bounds, body cap, event loop, errors.

Hermetic: a mini FastAPI app mounts only the events router; the DB is a recorder, never real.
Golden payloads are copied from the real call sites in frontend/components/chat/OutfitBoard.tsx
(look_shown, variant_selected, add_the_look, save_look).
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.deps as deps
import api.routes.events as events_mod
from api.demo import guards
from api.demo.fallback_limiter import FallbackLimiter
from api.demo.session import create_demo_token
from api.rate_limit import get_rate_limiter
from src.flywheel.events import EVENT_TYPES, StylingEvent, log_event

# ---------------------------------------------------------------------------
# Golden payloads: field sets exactly as OutfitBoard.tsx sends them today.
# ---------------------------------------------------------------------------

_BASE = {
    "session_id": "sess-1",
    "look_id": "look-1",
    "anchor_item_id": "0123456789",
    "anchor_category": "Trousers",
}

GOLDEN: dict[str, dict[str, Any]] = {
    "look_shown": {
        **_BASE,
        "event_type": "look_shown",
        "occasion": "sangeet_look",
        "brand": None,  # unified cross-store mode sends null
        "look_total_inr": 4599,
        "filled_slots": [
            {"slot": "top", "item_id": "a1", "price_inr": 1299.0},
            {"slot": "shoes", "item_id": "a2", "brand": "snitch", "price_inr": None},
            {"item_id": "a3", "price_inr": 999},  # slot undefined -> key omitted by JSON
        ],
    },
    "variant_selected": {
        **_BASE,
        "event_type": "variant_selected",
        "variant_id": "dressier",  # extra keys: ignored, must not 422
        "variant_label": "Dressed Up",
        "occasion": "sangeet_look",
        "brand": None,
    },
    "add_the_look": {
        **_BASE,
        "event_type": "add_the_look",
        "occasion": None,
        "brand": "snitch",
        "look_total_inr": 0,
        "filled_slots": [],
    },
    "save_look": {
        **_BASE,
        "event_type": "save_look",
        "occasion": "sangeet_look",
        "brand": None,
        "look_total_inr": None,
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "events-test-secret")
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    for name in (
        "EVENTS_RATE_LIMIT_PER_IDENTITY_PER_MIN",
        "EVENTS_RATE_LIMIT_PER_IP_PER_MIN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Fresh limiters on a fake clock so tests never share or wait out real windows."""
    c = _Clock()
    for attr, limit in (
        ("_identity_limiter", events_mod._per_identity_limit),
        ("_ip_limiter", events_mod._per_ip_limit),
    ):
        monkeypatch.setattr(
            events_mod,
            attr,
            FallbackLimiter(
                limit,
                events_mod._no_daily_cap,
                events_mod._max_keys,
                clock=c,
                window_s=events_mod._RATE_WINDOW_S,
            ),
        )
    return c


class _Recorder:
    """Stands in for log_event; records the StylingEvent and the calling thread."""

    def __init__(self) -> None:
        self.events: list[StylingEvent] = []
        self.on_loop: list[bool] = []
        self.result: bool = True
        self.raises: Exception | None = None

    def __call__(self, engine: Any, event: StylingEvent) -> bool:
        try:
            asyncio.get_running_loop()
            self.on_loop.append(True)
        except RuntimeError:
            self.on_loop.append(False)
        if self.raises is not None:
            raise self.raises
        self.events.append(event)
        return self.result


@pytest.fixture
def rec(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> _Recorder:
    r = _Recorder()
    monkeypatch.setattr(events_mod, "log_event", r)
    monkeypatch.setattr(deps, "get_db_engine", lambda: object())
    return r


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(events_mod.router)
    with TestClient(app) as c:
        yield c


def _auth(anon_id: str = "anon:alice") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_demo_token(anon_id, 'snitch')}"}


def _post(client: TestClient, payload: dict[str, Any], anon_id: str = "anon:alice") -> Any:
    return client.post("/events", json=payload, headers=_auth(anon_id))


# ---------------------------------------------------------------------------
# Auth + identity
# ---------------------------------------------------------------------------


def test_unauthenticated_is_401(client: TestClient, rec: _Recorder) -> None:
    r = client.post("/events", json=GOLDEN["look_shown"])
    assert r.status_code == 401
    assert rec.events == []


def test_demo_session_token_accepted(client: TestClient, rec: _Recorder) -> None:
    r = _post(client, GOLDEN["look_shown"])
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert len(rec.events) == 1


def test_client_supplied_user_id_cannot_spoof_identity(client: TestClient, rec: _Recorder) -> None:
    payload = {**GOLDEN["look_shown"], "user_id": "victim-user-id"}
    r = _post(client, payload, anon_id="anon:mallory")
    assert r.status_code == 200
    assert [e.user_id for e in rec.events] == ["anon:mallory"]


def test_stored_row_carries_authenticated_user_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    """End-to-end through the real log_event: the INSERT row's user_id is the token's sub."""
    engine = MagicMock()
    conn = engine.begin.return_value.__enter__.return_value
    monkeypatch.setattr(deps, "get_db_engine", lambda: engine)
    r = _post(client, {**GOLDEN["look_shown"], "user_id": "victim"}, anon_id="anon:bob")
    assert r.status_code == 200
    row = conn.execute.call_args.args[1]
    assert row["user_id"] == "anon:bob"
    assert row["price_band"] == "budget_3k_5k"
    assert json.loads(row["filled_slots"])[0]["item_id"] == "a1"


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_frontend_payloads_still_accepted(
    client: TestClient, rec: _Recorder, name: str
) -> None:
    r = _post(client, copy.deepcopy(GOLDEN[name]))
    assert r.status_code == 200, r.text
    assert rec.events[-1].event_type == name


def test_unknown_extra_fields_are_ignored_not_stored(client: TestClient, rec: _Recorder) -> None:
    r = _post(client, GOLDEN["variant_selected"])
    assert r.status_code == 200
    assert not hasattr(rec.events[0], "variant_id")


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_every_allowlisted_event_type_accepted(
    client: TestClient, rec: _Recorder, event_type: str
) -> None:
    assert _post(client, {**_BASE, "event_type": event_type}).status_code == 200


def test_no_db_engine_still_ok_true(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, clock: _Clock
) -> None:
    monkeypatch.setattr(deps, "get_db_engine", lambda: None)
    r = _post(client, GOLDEN["look_shown"])
    assert (r.status_code, r.json()) == (200, {"ok": True})


# ---------------------------------------------------------------------------
# Validation bounds
# ---------------------------------------------------------------------------


def _bad(**override: Any) -> dict[str, Any]:
    return {**GOLDEN["look_shown"], **override}


@pytest.mark.parametrize(
    "override",
    [
        {"session_id": "s" * 129},
        {"look_id": "l" * 129},
        {"anchor_item_id": "a" * 129},
        {"anchor_category": "c" * 65},
        {"occasion": "o" * 65},
        {"brand": "b" * 65},
        {"look_total_inr": -1},
        {"look_total_inr": 10_000_001},
        {"filled_slots": [{"slot": "top"}] * 21},
        {"filled_slots": [{f"k{i}": 1 for i in range(17)}]},
        {"filled_slots": [{"slot": "x" * 129}]},
        {"filled_slots": [{"slot": {"nested": 1}}]},
        {"filled_slots": [{"slot": [1, 2]}]},
        {"filled_slots": [{"k" * 65: 1}]},
        {"metadata": {f"k{i}": 1 for i in range(17)}},
        {"metadata": {"a": {"nested": "dict"}}},
        {"metadata": {"a": [1]}},
        {"metadata": {"a": "x" * 129}},
        {"metadata": {f"k{i}": "x" * 120 for i in range(16)}},  # each ok, total > 2 KB
        {"event_type": "not_a_real_event"},
        {"event_type": ["look_shown"]},
    ],
    ids=lambda o: next(iter(o)) + ":" + str(list(o.values())[0])[:20],
)
def test_bounds_rejected_with_422(client: TestClient, rec: _Recorder, override: dict) -> None:
    r = _post(client, _bad(**override))
    assert r.status_code == 422
    assert rec.events == []


def test_boundary_values_accepted(client: TestClient, rec: _Recorder) -> None:
    payload = _bad(
        session_id="s" * 128,
        anchor_category="c" * 64,
        look_total_inr=10_000_000,
        filled_slots=[{"a": "x" * 128, "b": "y" * 128}] * 20,  # max slots, max string length
        metadata={"a": 1, "b": "text", "c": None, "d": True, "e": 1.5},
    )
    assert _post(client, payload).status_code == 200
    payload["filled_slots"] = [{f"k{i}": 1 for i in range(16)}]  # max keys per slot
    assert _post(client, payload).status_code == 200


def test_total_payload_cap(client: TestClient, rec: _Recorder) -> None:
    # Every per-field bound holds, raw body (~8.8 KB) is under the 16 KB cap, but the validated
    # payload exceeds the 8 KB whole-request cap.
    slots = [{f"k{i}": "x" * 100 for i in range(4)} for _ in range(20)]
    r = _post(client, _bad(filled_slots=slots))
    assert r.status_code == 422
    assert rec.events == []


def test_422_does_not_echo_submitted_values(client: TestClient, rec: _Recorder) -> None:
    secret = "TOPSECRETVALUE"
    for payload in (
        _bad(session_id=secret * 20),
        _bad(event_type=secret),
        _bad(filled_slots=[{"slot": {secret: secret}}]),
    ):
        r = _post(client, payload)
        assert r.status_code == 422
        assert secret not in r.text
        assert "Traceback" not in r.text and "Unknown event_type" not in r.text


def test_malformed_json_is_422_without_echo(client: TestClient, rec: _Recorder) -> None:
    r = client.post(
        "/events",
        content=b'{"session_id": SECRETJUNK',
        headers={**_auth(), "Content-Type": "application/json"},
    )
    assert r.status_code == 422
    assert "SECRETJUNK" not in r.text


# ---------------------------------------------------------------------------
# Body size cap
# ---------------------------------------------------------------------------


def test_oversized_content_length_is_413(client: TestClient, rec: _Recorder) -> None:
    r = _post(client, {**GOLDEN["look_shown"], "pad": "x" * 20_000})
    assert r.status_code == 413
    assert r.json() == {"detail": "Request body too large"}
    assert rec.events == []


def test_chunked_body_without_content_length_is_capped(client: TestClient, rec: _Recorder) -> None:
    def gen() -> Iterator[bytes]:
        for _ in range(40):  # 40 KB, no Content-Length header
            yield b" " * 1024

    r = client.post(
        "/events", content=gen(), headers={**_auth(), "Content-Type": "application/json"}
    )
    assert r.status_code == 413
    assert rec.events == []


def test_lying_content_length_is_capped_by_real_byte_count(rec: _Recorder) -> None:
    """Content-Length claims 10 bytes but the body streams 20 KB: counted, not trusted."""
    app = FastAPI()
    app.include_router(events_mod.router)
    sent: list[dict[str, Any]] = []
    chunks = [b" " * 10_000, b" " * 10_000, b""]

    async def receive() -> dict[str, Any]:
        body = chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    token = create_demo_token("anon:alice", "snitch")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/events",
        "raw_path": b"/events",
        "query_string": b"",
        "root_path": "",
        "client": ("1.2.3.4", 1),
        "server": ("test", 80),
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", b"10"),
            (b"authorization", f"Bearer {token}".encode()),
        ],
    }
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert rec.events == []


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


def test_per_identity_limit_429_with_retry_after(
    client: TestClient, rec: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVENTS_RATE_LIMIT_PER_IDENTITY_PER_MIN", "3")
    codes = [_post(client, GOLDEN["look_shown"], "anon:a").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = _post(client, GOLDEN["look_shown"], "anon:a")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1
    assert r.json() == {"detail": "Too many events"}
    # Another identity is unaffected.
    assert _post(client, GOLDEN["look_shown"], "anon:b").status_code == 200


def test_per_ip_limit_applies_across_identities_and_to_unauthenticated(
    client: TestClient, rec: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVENTS_RATE_LIMIT_PER_IP_PER_MIN", "3")
    ip = {"v": "9.9.9.9"}
    monkeypatch.setattr(events_mod, "get_client_ip", lambda request: ip["v"])
    assert [_post(client, GOLDEN["look_shown"], f"anon:{i}").status_code for i in range(3)] == [
        200,
        200,
        200,
    ]
    assert _post(client, GOLDEN["look_shown"], "anon:fresh").status_code == 429
    # Unauthenticated floods are throttled before auth runs: 429, not 401.
    assert client.post("/events", json=GOLDEN["look_shown"]).status_code == 429
    ip["v"] = "8.8.8.8"
    assert _post(client, GOLDEN["look_shown"], "anon:fresh").status_code == 200


def test_window_expires_with_fake_clock(
    client: TestClient, rec: _Recorder, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVENTS_RATE_LIMIT_PER_IDENTITY_PER_MIN", "2")
    assert [_post(client, GOLDEN["look_shown"]).status_code for _ in range(3)] == [200, 200, 429]
    clock.now += 59
    assert _post(client, GOLDEN["look_shown"]).status_code == 429
    clock.now += 2
    assert _post(client, GOLDEN["look_shown"]).status_code == 200


def test_events_limits_are_independent_of_chat_limiters(
    client: TestClient, rec: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVENTS_RATE_LIMIT_PER_IDENTITY_PER_MIN", "2")
    assert events_mod._ip_limiter is not guards._fallback
    assert events_mod._identity_limiter is not guards._fallback
    chat_before = guards._fallback.tracked_ips()
    chat_windows = dict(get_rate_limiter()._windows)
    for _ in range(5):
        _post(client, GOLDEN["look_shown"])
    assert guards._fallback.tracked_ips() == chat_before
    assert get_rate_limiter()._windows == chat_windows


def test_limiter_memory_is_lru_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVENTS_RATE_LIMIT_MAX_KEYS", "50")
    lim = FallbackLimiter(
        events_mod._per_identity_limit,
        events_mod._no_daily_cap,
        events_mod._max_keys,
        window_s=60.0,
    )
    for i in range(500):
        lim.check_ip(f"user-{i}")
    assert lim.tracked_ips() == 50


def test_concurrent_requests_respect_the_limit(
    rec: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVENTS_RATE_LIMIT_PER_IDENTITY_PER_MIN", "25")
    app = FastAPI()
    app.include_router(events_mod.router)
    headers = _auth("anon:burst")

    def worker(_: int) -> list[int]:
        with TestClient(app) as c:
            return [
                c.post("/events", json=GOLDEN["look_shown"], headers=headers).status_code
                for _ in range(10)
            ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = [code for batch in pool.map(worker, range(8)) for code in batch]
    assert codes.count(200) == 25
    assert codes.count(429) == 55
    assert len(rec.events) == 25


# ---------------------------------------------------------------------------
# Event loop + failure behaviour
# ---------------------------------------------------------------------------


def test_log_event_does_not_run_on_the_event_loop(client: TestClient, rec: _Recorder) -> None:
    assert _post(client, GOLDEN["look_shown"]).status_code == 200
    assert rec.on_loop == [False]


def test_failing_log_event_is_generic_503_without_leak(client: TestClient, rec: _Recorder) -> None:
    rec.raises = RuntimeError("connection to db failed password=hunter2")
    r = _post(client, GOLDEN["look_shown"])
    assert r.status_code == 503
    assert r.json() == {"detail": "Event logging unavailable"}
    assert "hunter2" not in r.text


def test_log_event_reporting_failure_is_503(client: TestClient, rec: _Recorder) -> None:
    rec.result = False
    assert _post(client, GOLDEN["look_shown"]).status_code == 503


def test_log_event_returns_false_when_insert_fails() -> None:
    engine = MagicMock()
    engine.begin.side_effect = RuntimeError("boom")
    event = StylingEvent(event_type="look_shown", **_BASE)
    assert log_event(engine, event) is False
    ok_engine = MagicMock()
    assert log_event(ok_engine, event) is True


# ---------------------------------------------------------------------------
# Wiring in the real app
# ---------------------------------------------------------------------------


def test_real_app_events_route_requires_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import app

    # No `with`: skips the lifespan (index loading); the route table is what is under test.
    assert TestClient(app).post("/events", json=GOLDEN["look_shown"]).status_code == 401
