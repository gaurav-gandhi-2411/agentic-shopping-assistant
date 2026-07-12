from __future__ import annotations

import datetime
import time
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

import api.demo.guards as guards_module
import api.deps as deps
from api.demo.session import create_demo_token, validate_demo_token
from api.main import app
from api.session import InMemorySessionStore

# ---------------------------------------------------------------------------
# Section A: api/demo/session.py
# ---------------------------------------------------------------------------


def test_create_and_validate_demo_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_JWT_SECRET", "test-secret-for-session")
    token = create_demo_token("anon:test-user", "snitch")
    assert isinstance(token, str) and len(token) > 0
    result = validate_demo_token(token)
    assert result == "anon:test-user"


def test_validate_demo_token_rejects_bad_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_JWT_SECRET", "correct-secret")
    token = create_demo_token("anon:bad-sig-user", "snitch")
    # Now change secret so the token's signature is wrong
    monkeypatch.setenv("DEMO_JWT_SECRET", "wrong-secret")
    result = validate_demo_token(token)
    assert result is None


def test_validate_demo_token_wrong_type(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "type-test-secret"
    monkeypatch.setenv("DEMO_JWT_SECRET", secret)
    now = int(time.time())
    payload = {
        "sub": "anon:wrong-type",
        "brand": "snitch",
        "type": "not_demo",
        "iat": now,
        "exp": now + 3600,
    }
    bad_token = pyjwt.encode(payload, secret, algorithm="HS256")
    result = validate_demo_token(bad_token)
    assert result is None


# ---------------------------------------------------------------------------
# Section B: api/demo/guards.py
# ---------------------------------------------------------------------------


def _make_engine_mock(fetchone_return: object) -> MagicMock:
    """Build a MagicMock engine whose begin() context manager returns a conn
    whose execute().fetchone() returns the given value."""
    engine = MagicMock()
    mock_conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchone.return_value = fetchone_return
    return engine


def test_check_daily_cap_under_limit() -> None:
    engine = _make_engine_mock((50,))
    result = guards_module.check_daily_cap("snitch", engine)
    assert result is True


def test_check_daily_cap_over_limit() -> None:
    engine = _make_engine_mock((700,))  # matches DEMO_DAILY_REQUEST_CAP default
    result = guards_module.check_daily_cap("snitch", engine)
    assert result is False


def test_check_daily_cap_no_row() -> None:
    engine = _make_engine_mock(None)
    result = guards_module.check_daily_cap("snitch", engine)
    assert result is True


def test_check_daily_cap_db_error() -> None:
    engine = MagicMock()
    engine.begin.side_effect = Exception("DB down")
    result = guards_module.check_daily_cap("snitch", engine)
    assert result is True


def test_check_ip_rate_limit_under_limit() -> None:
    engine = _make_engine_mock((5,))
    mock_conn = engine.begin.return_value.__enter__.return_value
    allowed, retry_after = guards_module.check_ip_rate_limit("1.2.3.4", "snitch", engine)
    assert allowed is True
    assert retry_after == 0
    # The INSERT/upsert should have been called (two execute calls: SELECT + INSERT)
    assert mock_conn.execute.call_count == 2


def test_check_ip_rate_limit_at_limit() -> None:
    engine = _make_engine_mock((35,))  # matches DEMO_PER_IP_HOUR_LIMIT default
    allowed, retry_after = guards_module.check_ip_rate_limit("1.2.3.4", "snitch", engine)
    assert allowed is False
    assert retry_after > 0


def test_check_ip_rate_limit_db_error() -> None:
    engine = MagicMock()
    engine.begin.side_effect = Exception("DB gone")
    allowed, retry_after = guards_module.check_ip_rate_limit("1.2.3.4", "snitch", engine)
    assert allowed is True
    assert retry_after == 0


def test_check_daily_cost_under_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    today = datetime.date.today().isoformat()
    monkeypatch.setattr(guards_module, "_daily_cost_accumulated", 0.20)
    monkeypatch.setattr(guards_module, "_current_day", today)
    result = guards_module.check_daily_cost("snitch")
    assert result is True


def test_check_daily_cost_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    today = datetime.date.today().isoformat()
    monkeypatch.setattr(guards_module, "_daily_cost_accumulated", 0.55)
    monkeypatch.setattr(guards_module, "_current_day", today)
    result = guards_module.check_daily_cost("snitch")
    assert result is False


def test_check_daily_cost_day_rollover(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stale day triggers fail-open (return True, delegate to DB cap)
    monkeypatch.setattr(guards_module, "_daily_cost_accumulated", 0.55)
    monkeypatch.setattr(guards_module, "_current_day", "2000-01-01")
    result = guards_module.check_daily_cost("snitch")
    assert result is True


# ---------------------------------------------------------------------------
# Fixtures shared by Section C endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_base_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject minimal deps so the app module-level singletons don't need real data."""
    store = InMemorySessionStore()
    monkeypatch.setattr(deps, "_session_store", store)
    # Raise rate limit to avoid interference with demo tests
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    # Demo JWT secret for predictable tokens in endpoint tests
    monkeypatch.setenv("DEMO_JWT_SECRET", "endpoint-test-secret")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Section C: api/routes/demo.py — endpoint tests
# ---------------------------------------------------------------------------


def test_demo_session_returns_200_when_demo_mode_true(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("BRAND", "snitch")
    monkeypatch.setattr(deps, "_db_engine", None)
    monkeypatch.setattr("api.auth.mint_ws_ticket", lambda user_id: "fake-ticket")

    resp = client.post("/demo/session")
    assert resp.status_code == 200
    body = resp.json()
    assert "session_token" in body
    assert body["ws_ticket"] == "fake-ticket"
    assert body["expires_in"] == 3600
    assert body["brand"] == "snitch"


def test_demo_session_returns_404_when_demo_mode_false(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.setenv("DEMO_MODE", "false")

    resp = client.post("/demo/session")
    assert resp.status_code == 404


def test_demo_session_returns_429_when_daily_cap_hit(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("BRAND", "snitch")
    # Use a real MagicMock engine so engine is not None (cap check is triggered)
    mock_engine = MagicMock()
    monkeypatch.setattr(deps, "_db_engine", mock_engine)
    monkeypatch.setattr("api.routes.demo.check_daily_cap", lambda brand, engine: False)

    resp = client.post("/demo/session")
    assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Section D: api/auth.get_current_user_id_or_demo
# ---------------------------------------------------------------------------


def test_get_current_user_id_or_demo_passes_through_when_demo_mode_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setattr(
        "api.auth.get_current_user_id", lambda authorization: "real-user-id"
    )
    from api.auth import get_current_user_id_or_demo

    result = get_current_user_id_or_demo("Bearer some-supabase-jwt")
    assert result == "real-user-id"


def test_get_current_user_id_or_demo_accepts_demo_token_when_demo_mode_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "demo-direct-test")
    demo_token = create_demo_token("anon:direct-test-user", "snitch")

    from api.auth import get_current_user_id_or_demo

    result = get_current_user_id_or_demo(f"Bearer {demo_token}")
    assert result == "anon:direct-test-user"


def test_get_current_user_id_or_demo_falls_through_on_bad_demo_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "some-secret")
    # validate_demo_token will return None → falls through to get_current_user_id
    # which raises 401 because the token is not a valid Supabase RS256 JWT
    monkeypatch.setattr(
        "api.auth.get_current_user_id",
        lambda authorization: (_ for _ in ()).throw(
            __import__("fastapi").HTTPException(status_code=401, detail="Unauthorized")
        ),
    )

    from fastapi import HTTPException

    from api.auth import get_current_user_id_or_demo

    with pytest.raises(HTTPException) as exc_info:
        get_current_user_id_or_demo("Bearer not-a-valid-demo-token")
    assert exc_info.value.status_code == 401
