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
    assert rec.exc_info is not None  # kept for Sentry; redacted copy lives in exc_text
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


def test_factory_redacts_at_creation_for_late_handlers_and_keeps_exc_info() -> None:
    install_log_redaction(Redactor([DB_PW]))
    seen: list[logging.LogRecord] = []

    class Late(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record)

    late = Late()  # added after install: no filter on it
    logging.getLogger().addHandler(late)
    try:
        try:
            raise RuntimeError(f"bad {DB_PW}")
        except RuntimeError:
            logging.getLogger("late").exception("failed %s", DB_PW)
    finally:
        logging.getLogger().removeHandler(late)
    (rec,) = seen
    assert DB_PW not in rec.getMessage()
    assert rec.exc_info is not None
    assert DB_PW not in (rec.exc_text or "")


def test_sentry_logger_exception_event_has_exception_and_no_credential(monkeypatch) -> None:
    import json

    import sentry_sdk
    from sentry_sdk.transport import Transport

    from api.main import _sentry_before_breadcrumb

    monkeypatch.setenv("DATABASE_URL", f"postgresql://u:{DB_PW}@h.invalid/db")
    install_log_redaction()
    events: list[dict] = []

    class Capture(Transport):  # in-memory transport: no network
        def capture_envelope(self, envelope) -> None:
            events.extend(i.payload.json for i in envelope.items if i.type == "event")

    # A handler carrying the filter is what cleared exc_info before the fix.
    handler = logging.StreamHandler()
    handler.addFilter(RedactionFilter(Redactor.from_env()))
    logging.getLogger().addHandler(handler)
    sentry_sdk.init(
        dsn="http://k@localhost:1/1", transport=Capture, traces_sample_rate=0,
        before_send=_sentry_before_send, before_breadcrumb=_sentry_before_breadcrumb,
        include_local_variables=False,
    )
    try:
        try:
            url = f"postgresql://u:{DB_PW}@h.invalid/db"  # noqa: F841 - frame local
            raise RuntimeError(f"failed to resolve host '{DB_PW}@h.invalid'")
        except RuntimeError:
            logging.getLogger("probe").exception("db failure %s", DB_PW)
        sentry_sdk.flush()
    finally:
        logging.getLogger().removeHandler(handler)
        sentry_sdk.init(dsn=None)
    assert events
    assert any(e.get("exception", {}).get("values") for e in events)
    blob = json.dumps(events, default=str)
    for leaked in (DB_PW, "Alpha", "Bravo", "Charlie", "Delta"):
        assert leaked not in blob
