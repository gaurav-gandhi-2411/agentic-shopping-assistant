"""Tests for JWT authentication middleware (api/auth.py).

Uses RS256 with a module-level generated RSA key pair so tests are entirely
self-contained — no Supabase connection, no network requests.

JWT_TEST_PUBLIC_KEY is set per-test via monkeypatch; api.auth reads it at call
time so the monkeypatch takes effect immediately without reloading the module.
"""
from __future__ import annotations

import time
from typing import Any, Iterator

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import api.auth as auth_module
import api.deps as deps
from api.auth import exchange_ws_ticket, mint_ws_ticket
from api.main import app
from api.session import InMemorySessionStore

# ---------------------------------------------------------------------------
# Module-level RSA key pair — generated once on import, reused across all tests.
# 2048-bit to match realistic production verification cost.
# ---------------------------------------------------------------------------
_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PRIVATE_PEM: str = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
).decode()
_TEST_PUBLIC_PEM: str = _PRIVATE_KEY.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
).decode()

_MINIMAL_CONFIG = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockLLM:
    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return "ok"

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        yield "ok"

    def chat(self, messages: list[dict], **kwargs) -> str:
        return "ok"

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield "ok"


class _MockAgent:
    def invoke(self, state: dict, **kwargs) -> dict[str, Any]:
        result = {
            "retrieved_items": [],
            "filters": {},
            "tool_calls": [{"router_decision": {"action": "search", "query": "test"}}],
            "final_answer": "Auth test response.",
            "iteration": 1,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
        }
        result.setdefault("messages", state.get("messages", []))
        return result


def _make_agent_factory() -> Any:
    agent = _MockAgent()

    def factory(memory: Any, streaming: bool = False) -> _MockAgent:
        return agent

    def get_factory() -> Any:
        return factory

    return get_factory


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _mint_token(
    sub: str = deps.DEV_USER_ID,
    aud: str = "authenticated",
    exp_offset: int = 3600,
    email: str | None = "dev@example.com",
) -> str:
    """Return a signed RS256 JWT using the module-level test private key."""
    now = int(time.time())
    claims: dict = {
        "sub": sub,
        "aud": aud,
        "iss": "test-issuer",
        "iat": now,
        "exp": now + exp_offset,
    }
    if email is not None:
        claims["email"] = email
    return jwt.encode(claims, _TEST_PRIVATE_PEM, algorithm="RS256")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject test deps and wire JWT_TEST_PUBLIC_KEY for all tests in this module.

    JWT_VERIFICATION_DISABLED is explicitly NOT set — these tests exercise the
    real verification path.

    _check_allowlist defaults to allow-all so existing tests are unaffected;
    individual tests can override it via monkeypatch.setattr.
    """
    store = InMemorySessionStore()
    monkeypatch.setattr(deps, "_session_store", store)
    monkeypatch.setattr(deps, "_llm", _MockLLM())
    monkeypatch.setattr(deps, "_config", _MINIMAL_CONFIG)
    monkeypatch.setattr(deps, "get_agent_factory", _make_agent_factory())
    # Wire test public key so verify_jwt uses it instead of fetching JWKS.
    monkeypatch.setenv("JWT_TEST_PUBLIC_KEY", _TEST_PUBLIC_PEM)
    monkeypatch.setenv("SUPABASE_JWT_AUD", "authenticated")
    # Ensure verification is enabled for all auth tests.
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    # Allow all emails by default; individual tests can tighten this.
    monkeypatch.setattr(auth_module, "_check_allowlist", lambda email: True)
    # Raise rate limit high so tests never trip the sliding-window limiter.
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture
def client() -> TestClient:
    # Instantiate without context manager so the lifespan is never triggered
    # and no real index files are required.
    yield TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# POST /chat — auth tests
# ---------------------------------------------------------------------------

def test_chat_rejects_missing_token(client: TestClient) -> None:
    """POST /chat with no Authorization header must return 401."""
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 401


def test_chat_rejects_invalid_token(client: TestClient) -> None:
    """POST /chat with a malformed token must return 401."""
    resp = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": "Bearer not.a.valid.jwt"},
    )
    assert resp.status_code == 401


def test_chat_rejects_expired_token(client: TestClient) -> None:
    """POST /chat with an already-expired JWT must return 401."""
    token = _mint_token(exp_offset=-1)  # exp in the past
    resp = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401


def test_chat_accepts_valid_token(client: TestClient) -> None:
    """POST /chat with a valid signed JWT must return 200."""
    token = _mint_token()
    resp = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["response"] == "Auth test response."


# ---------------------------------------------------------------------------
# WS /chat/stream — auth test
# ---------------------------------------------------------------------------

def test_ws_rejects_missing_token(client: TestClient) -> None:
    """WebSocket connect with no ?token= query param must be closed with code 1008."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/chat/stream") as ws:
            ws.receive_text()  # server closes immediately after accept
    assert exc_info.value.code == 1008


def test_chat_rejects_non_allowlisted_user(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """POST /chat with a valid JWT whose email is not on the allow-list must return 401.

    This test exercises the defense-in-depth path: the token is cryptographically
    valid but the email claim fails the allow-list check.
    """
    monkeypatch.setattr(auth_module, "_check_allowlist", lambda email: False)
    token = _mint_token(email="unlisted@example.com")
    resp = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 401
    assert "allow-list" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# POST /auth/ws-ticket
# ---------------------------------------------------------------------------

def test_ws_ticket_endpoint_returns_ticket(client: TestClient) -> None:
    """POST /auth/ws-ticket with a valid JWT must return a non-empty ticket string."""
    token = _mint_token()
    resp = client.post(
        "/auth/ws-ticket",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "ticket" in data
    assert isinstance(data["ticket"], str)
    assert len(data["ticket"]) > 10  # URL-safe base64, 32 bytes → ~43 chars


def test_ws_ticket_endpoint_rejects_missing_auth(client: TestClient) -> None:
    """POST /auth/ws-ticket without Authorization header must return 401."""
    resp = client.post("/auth/ws-ticket")
    assert resp.status_code == 401


def test_ws_ticket_endpoint_rejects_invalid_jwt(client: TestClient) -> None:
    """POST /auth/ws-ticket with a bad token must return 401."""
    resp = client.post(
        "/auth/ws-ticket",
        headers={"Authorization": "Bearer not.a.jwt"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# WS /chat/stream — ticket auth unit tests (ticket store functions)
# ---------------------------------------------------------------------------

def test_mint_and_exchange_ticket() -> None:
    """mint_ws_ticket + exchange_ws_ticket round-trip must return the original user_id."""
    user_id = "test-user-abc123"
    nonce = mint_ws_ticket(user_id)
    assert isinstance(nonce, str) and len(nonce) > 10
    result = exchange_ws_ticket(nonce)
    assert result == user_id


def test_exchange_ticket_is_single_use() -> None:
    """A ticket must be consumed on first exchange; second exchange must return None."""
    nonce = mint_ws_ticket("user-single-use")
    assert exchange_ws_ticket(nonce) == "user-single-use"
    assert exchange_ws_ticket(nonce) is None


def test_exchange_unknown_ticket_returns_none() -> None:
    """Exchanging a nonce that was never minted must return None."""
    assert exchange_ws_ticket("totally-bogus-nonce-xyz") is None


def test_exchange_expired_ticket_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ticket that has passed its TTL must not be exchangeable.

    We monkeypatch time.time inside api.auth to simulate expiry without
    actually waiting 60 seconds.
    """
    nonce = mint_ws_ticket("user-expiry")
    # Advance time past expiry by patching time.time in the auth module.
    future_time = time.time() + 120
    monkeypatch.setattr(auth_module.time, "time", lambda: future_time)
    result = exchange_ws_ticket(nonce)
    assert result is None


# ---------------------------------------------------------------------------
# WS /chat/stream — ticket-based connection (integration)
# ---------------------------------------------------------------------------

def test_ws_connects_via_ticket(client: TestClient) -> None:
    """WebSocket opened with a valid ?ticket= must be accepted (session frame sent)."""
    token = _mint_token()
    resp = client.post(
        "/auth/ws-ticket",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    ticket = resp.json()["ticket"]

    with client.websocket_connect(f"/chat/stream?ticket={ticket}") as ws:
        ws.send_json({"type": "user_message", "message": "hello"})
        msg = ws.receive_json()
        assert msg["type"] == "session"


def test_ws_rejects_invalid_ticket(client: TestClient) -> None:
    """WebSocket opened with an unknown ?ticket= must be closed with code 1008."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/chat/stream?ticket=not-a-real-ticket") as ws:
            ws.receive_text()
    assert exc_info.value.code == 1008


def test_ws_still_accepts_legacy_token(client: TestClient) -> None:
    """Legacy ?token= path must still work (backward-compat for Streamlit Spaces)."""
    token = _mint_token()
    with client.websocket_connect(f"/chat/stream?token={token}") as ws:
        ws.send_json({"type": "user_message", "message": "hello"})
        msg = ws.receive_json()
        assert msg["type"] == "session"
