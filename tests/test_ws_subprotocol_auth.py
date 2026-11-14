"""WS auth ticket via the Sec-WebSocket-Protocol subprotocol (keeps the ticket out of the URL).

Cloud Run's platform request log records the full request URL, so a ``?ticket=`` query param is
persisted. The browser instead offers ``["stylemaitri.ticket.v1", "<ticket>"]``; the server
selects ONLY the marker. The URL ``?ticket=`` and legacy ``?token=`` paths stay for one release.
No network, no real DB.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import api.auth as auth_module
import api.deps as deps
from api.auth import WS_TICKET_SUBPROTOCOL, mint_ws_ticket
from api.main import app
from api.session import InMemorySessionStore

MARKER = WS_TICKET_SUBPROTOCOL
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_PEM = _KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
).decode()
_PUBLIC_PEM = (
    _KEY.public_key()
    .public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode()
)


class _MockLLM:
    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> str:
        return "ok"

    def generate_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield "ok"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_session_store", InMemorySessionStore())
    monkeypatch.setattr(deps, "_llm", _MockLLM())
    monkeypatch.setenv("JWT_TEST_PUBLIC_KEY", _PUBLIC_PEM)
    monkeypatch.setenv("SUPABASE_JWT_AUD", "authenticated")
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setattr(auth_module, "_check_allowlist", lambda email: True)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def seen_users(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the user id the real WS auth resolves for each connection."""
    seen: list[str] = []
    real = auth_module.get_current_user_id_ws

    def recording(ws: Any) -> str:
        uid = real(ws)
        seen.append(uid)
        return uid

    monkeypatch.setattr("api.routes.chat.get_current_user_id_ws", recording)
    return seen


def _jwt(sub: str) -> str:
    now = int(time.time())
    claims = {"sub": sub, "aud": "authenticated", "iat": now, "exp": now + 3600}
    return jwt.encode(claims, _PRIVATE_PEM, algorithm="RS256")


def _closes_1008(client: TestClient, url: str, subprotocols: list[str] | None) -> str:
    """Connect, expect a server close with 1008; return the close reason."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(url, subprotocols=subprotocols) as ws:
            # A wrongly admitted connection answers this bad frame (so a regression fails the
            # assertion instead of blocking forever on receive); a rejected one is already closed.
            ws.send_text("{}")
            ws.receive_text()
    assert exc_info.value.code == 1008
    return exc_info.value.reason


def _first_frame(client: TestClient, url: str, subprotocols: list[str] | None) -> dict[str, Any]:
    with client.websocket_connect(url, subprotocols=subprotocols) as ws:
        ws.send_json({"type": "user_message", "message": "hello"})
        return ws.receive_json()


# ---------------------------------------------------------------------------
# Success paths + identity
# ---------------------------------------------------------------------------


def test_subprotocol_and_url_ticket_yield_same_identity(
    client: TestClient, seen_users: list[str]
) -> None:
    t_sub = mint_ws_ticket("user-same")
    t_url = mint_ws_ticket("user-same")
    assert _first_frame(client, "/chat/stream", [MARKER, t_sub])["type"] == "session"
    assert _first_frame(client, f"/chat/stream?ticket={t_url}", None)["type"] == "session"
    assert seen_users == ["user-same", "user-same"]


def test_marker_is_selected_and_ticket_never_echoed(client: TestClient) -> None:
    ticket = mint_ws_ticket("user-echo")
    with client.websocket_connect("/chat/stream", subprotocols=[MARKER, ticket]) as ws:
        assert ws.accepted_subprotocol == MARKER
        assert ws.accepted_subprotocol != ticket
        ws.send_json({"type": "user_message", "message": "hello"})
        assert ticket not in ws.receive_text()


def test_url_ticket_path_selects_no_subprotocol(client: TestClient) -> None:
    ticket = mint_ws_ticket("user-url")
    with client.websocket_connect(f"/chat/stream?ticket={ticket}") as ws:
        assert ws.accepted_subprotocol is None


def test_legacy_token_path_still_works(client: TestClient, seen_users: list[str]) -> None:
    assert _first_frame(client, f"/chat/stream?token={_jwt('user-legacy')}", None)["type"] == (
        "session"
    )
    assert seen_users == ["user-legacy"]


# ---------------------------------------------------------------------------
# Failure paths: identical close behaviour on both ticket paths
# ---------------------------------------------------------------------------


def _expired_ticket() -> str:
    """A ticket whose 60 s TTL has already elapsed (no global time patching needed)."""
    ticket = mint_ws_ticket("user-expired")
    auth_module._ticket_store[ticket] = ("user-expired", time.time() - 1)
    return ticket


@pytest.mark.parametrize("kind", ["unknown", "expired", "reused", "empty_after_marker"])
def test_invalid_ticket_rejected_identically_on_both_paths(client: TestClient, kind: str) -> None:
    def make() -> str:
        if kind == "unknown":
            return "not-a-real-ticket"
        if kind == "expired":
            return _expired_ticket()
        if kind == "reused":
            t = mint_ws_ticket("user-reuse")
            assert _first_frame(client, f"/chat/stream?ticket={t}", None)["type"] == "session"
            return t
        return ""

    ticket = make()
    sub_protocols = [MARKER, ticket] if ticket else [MARKER]
    sub_reason = _closes_1008(client, "/chat/stream", sub_protocols)
    if ticket:
        url_reason = _closes_1008(client, f"/chat/stream?ticket={ticket}", None)
        assert sub_reason == url_reason  # no oracle for well-formed vs expired vs reused
    assert sub_reason == "Policy violation: invalid or missing token"


def test_subprotocol_ticket_is_single_use(client: TestClient) -> None:
    ticket = mint_ws_ticket("user-once")
    assert _first_frame(client, "/chat/stream", [MARKER, ticket])["type"] == "session"
    _closes_1008(client, "/chat/stream", [MARKER, ticket])


def test_marker_without_ticket_is_rejected(client: TestClient) -> None:
    assert _closes_1008(client, "/chat/stream", [MARKER]) == (
        "Policy violation: invalid or missing token"
    )


def test_marker_is_selected_even_when_auth_fails(client: TestClient) -> None:
    """A browser aborts the handshake if it offered subprotocols and none was selected."""
    with client.websocket_connect("/chat/stream", subprotocols=[MARKER, "bogus"]) as ws:
        assert ws.accepted_subprotocol == MARKER
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
    assert exc_info.value.code == 1008


def test_ticket_before_marker_is_not_a_credential(client: TestClient) -> None:
    ticket = mint_ws_ticket("user-order")
    _closes_1008(client, "/chat/stream", [ticket, MARKER])
    # The ticket was never consumed by the failed attempt.
    assert auth_module.exchange_ws_ticket(ticket) == "user-order"


# ---------------------------------------------------------------------------
# Precedence: subprotocol > ?ticket > ?token (first present wins, no fall-through)
# ---------------------------------------------------------------------------


def test_subprotocol_wins_over_url_ticket_and_token(
    client: TestClient, seen_users: list[str]
) -> None:
    t_sub = mint_ws_ticket("user-sub")
    t_url = mint_ws_ticket("user-url")
    url = f"/chat/stream?ticket={t_url}&token={_jwt('user-jwt')}"
    assert _first_frame(client, url, [MARKER, t_sub])["type"] == "session"
    assert seen_users == ["user-sub"]
    assert auth_module.exchange_ws_ticket(t_url) == "user-url"  # URL ticket left untouched


def test_bad_subprotocol_does_not_fall_through_to_valid_url_ticket(client: TestClient) -> None:
    t_url = mint_ws_ticket("user-url")
    _closes_1008(client, f"/chat/stream?ticket={t_url}", [MARKER, "bad"])
    assert auth_module.exchange_ws_ticket(t_url) == "user-url"


def test_url_ticket_wins_over_legacy_token(client: TestClient, seen_users: list[str]) -> None:
    t_url = mint_ws_ticket("user-url")
    url = f"/chat/stream?ticket={t_url}&token={_jwt('user-jwt')}"
    assert _first_frame(client, url, None)["type"] == "session"
    assert seen_users == ["user-url"]


# ---------------------------------------------------------------------------
# Rate limit / daily cap still run before any message on the subprotocol path (E9)
# ---------------------------------------------------------------------------


def test_demo_ip_rate_limit_applies_on_subprotocol_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMO_PER_IP_HOUR_LIMIT", "2")
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "1000")
    monkeypatch.setattr(deps, "_db_engine", None)

    def hdr() -> dict[str, str]:
        # fresh dict per call: TestClient setdefault()s sec-websocket-protocol into it
        return {"X-Forwarded-For": "198.51.100.77"}

    for _ in range(2):  # admitted: the server waits for a first frame; just hang up
        with client.websocket_connect(
            "/chat/stream", headers=hdr(), subprotocols=[MARKER, mint_ws_ticket("anon:sp-1")]
        ):
            pass

    ticket = mint_ws_ticket("anon:sp-1")
    with client.websocket_connect(
        "/chat/stream", headers=hdr(), subprotocols=[MARKER, ticket]
    ) as ws:
        ws.send_text("{}")  # a wrongly admitted connection answers "bad_request" instead
        msg = ws.receive_json()
    assert (msg["type"], msg["code"]) == ("error", "rate_limited")


def test_demo_daily_cap_applies_on_subprotocol_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEMO_PER_IP_HOUR_LIMIT", "1000")
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "1")
    monkeypatch.setattr(deps, "_db_engine", None)

    def hdr() -> dict[str, str]:
        return {"X-Forwarded-For": "198.51.100.78"}

    with client.websocket_connect(
        "/chat/stream", headers=hdr(), subprotocols=[MARKER, mint_ws_ticket("anon:sp-2")]
    ):
        pass
    with client.websocket_connect(
        "/chat/stream", headers=hdr(), subprotocols=[MARKER, mint_ws_ticket("anon:sp-2")]
    ) as ws:
        ws.send_text("{}")
        msg = ws.receive_json()
    assert (msg["type"], msg["code"]) == ("error", "demo_limit")


# ---------------------------------------------------------------------------
# Observability: one ws_auth_path log line per connection, never a credential
# ---------------------------------------------------------------------------


def _ws_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if hasattr(r, "ws_auth_path")]


def _assert_no_secret_logged(caplog: pytest.LogCaptureFixture, *secrets: str) -> None:
    for record in caplog.records:
        blob = record.getMessage() + repr(record.__dict__)
        for secret in secrets:
            assert secret not in blob


def test_ws_auth_path_logged_per_path_without_credentials(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    t_sub = mint_ws_ticket("user-log")
    t_url = mint_ws_ticket("user-log")
    legacy = _jwt("user-log")
    _first_frame(client, "/chat/stream", [MARKER, t_sub])
    _first_frame(client, f"/chat/stream?ticket={t_url}", None)
    _first_frame(client, f"/chat/stream?token={legacy}", None)
    _closes_1008(client, "/chat/stream", None)
    t_bad = "rejected-ticket-value"
    _closes_1008(client, "/chat/stream", [MARKER, t_bad])
    _closes_1008(client, f"/chat/stream?ticket={t_bad}", None)

    got = [(r.ws_auth_path, r.ws_auth_ok) for r in _ws_records(caplog)]
    assert got == [
        ("subprotocol", True),
        ("url_ticket", True),
        ("legacy_token", True),
        ("none", False),
        ("subprotocol", False),
        ("url_ticket", False),
    ]
    _assert_no_secret_logged(caplog, t_sub, t_url, t_bad, legacy)
