"""Log/Sentry redaction (hermetic)."""
from __future__ import annotations

import logging

from api.main import _sentry_before_send
from src.utils.redact import RedactionFilter, Redactor, collect_secrets, install_log_redaction

DB_PW = "Alpha@Bravo:Charlie/Delta"


def _record(msg: str, *args: object, exc: BaseException | None = None) -> logging.LogRecord:
    exc_info = (type(exc), exc, exc.__traceback__) if exc else None
    return logging.LogRecord("t", logging.ERROR, __file__, 1, msg, args or None, exc_info)


def test_userinfo_masked_even_with_at_in_password() -> None:
    out = Redactor().redact(f"connect postgresql://postgres.ref:{DB_PW}@db.host:5432/postgres")
    assert out == "connect postgresql://postgres.ref:***@db.host:5432/postgres"


def test_bearer_and_jwt_masked() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.abcDEF123_-xyz"
    out = Redactor().redact(f"Authorization: Bearer abcdef1234567890 and token {jwt}")
    assert "abcdef1234567890" not in out
    assert "eyJhbGci" not in out
    assert "Bearer ***" in out


def test_normal_messages_untouched() -> None:
    msg = "Retrieval ready: 44,654 items; GET /api/brand?x=1 -> 200 at http://localhost:8080/a"
    assert Redactor().redact(msg) == msg


def test_env_secrets_collected_and_masked(monkeypatch) -> None:
    env = {
        "DATABASE_URL": f"postgresql://u:{DB_PW}@h.invalid/db",
        "GROQ_API_KEY": "gsk_live_0123456789",
        "DEMO_JWT_SECRET": "short",  # < min length: not masked (would hit ordinary words)
    }
    red = Redactor(collect_secrets(env))
    text = f"k=gsk_live_0123456789 pw={DB_PW} frag=Charlie@h.invalid enc=Alpha%40Bravo%3ACharlie%2FDelta"
    out = red.redact(text)
    for leaked in ("gsk_live_0123456789", "Alpha", "Charlie", "Delta"):
        assert leaked not in out
    assert "short" not in collect_secrets(env)


def test_filter_masks_message_args_and_traceback() -> None:
    red = Redactor([DB_PW])
    try:
        raise RuntimeError(f"failed to resolve host '{DB_PW}@x.invalid'")
    except RuntimeError as e:
        rec = _record("db error %s", f"pw={DB_PW}", exc=e)
    assert RedactionFilter(red).filter(rec)
    assert rec.exc_info is None
    blob = rec.getMessage() + (rec.exc_text or "")
    assert DB_PW not in blob
    assert "RuntimeError" in blob  # traceback still present, just masked


def test_installed_filter_covers_propagated_child_logger(caplog) -> None:
    install_log_redaction(Redactor([DB_PW]))
    logging.getLogger("some.child").error("boom %s", DB_PW)
    assert DB_PW not in caplog.text


def test_sentry_before_send_redacts_exception_values(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"postgresql://u:{DB_PW}@h.invalid/db")
    event = {"exception": {"values": [{"value": f"failed host '{DB_PW}@h.invalid'"}]}}
    out = _sentry_before_send(event, {})
    assert DB_PW not in str(out)
    assert "Charlie" not in str(out)
