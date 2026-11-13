"""Log redaction must not break uvicorn's AccessFormatter (needs record.args as a 5-tuple)."""
from __future__ import annotations

import copy
import io
import logging
import logging.config
from collections.abc import Iterator

import pytest
import uvicorn.config

from api.logging_config import setup_logging
from src.utils import redact
from src.utils.redact import Redactor

ACCESS_FMT = '%s - "%s %s HTTP/%s" %d'
SECRET = "hunter2-db-password"  # invented test constant


@pytest.fixture
def access_stream() -> Iterator[io.StringIO]:
    """uvicorn access handler writing to a buffer, with the app's logging setup installed."""
    saved_factory = logging.getLogRecordFactory()
    saved_redactor = redact._active_redactor
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    access = logging.getLogger("uvicorn.access")
    saved_filters = list(access.filters)
    logging.config.dictConfig(copy.deepcopy(uvicorn.config.LOGGING_CONFIG))
    stream = io.StringIO()
    access.handlers[0].setStream(stream)
    setup_logging()
    redact.install_log_redaction(Redactor([SECRET]))  # also attaches filters to uvicorn's handler
    try:
        yield stream
    finally:
        logging.setLogRecordFactory(saved_factory)
        redact._active_redactor = saved_redactor
        access.filters[:] = saved_filters
        access.handlers.clear()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _emit(path: str, status: int = 200) -> None:
    logging.getLogger("uvicorn.access").info(
        ACCESS_FMT, "127.0.0.1:1234", "GET", path, "1.1", status
    )


def test_access_line_formats_without_error(
    access_stream: io.StringIO, capsys: pytest.CaptureFixture[str]
) -> None:
    _emit("/healthz")
    # a formatter failure is reported by logging on stderr, not raised
    assert "Logging error" not in capsys.readouterr().err
    line = access_stream.getvalue().strip()
    assert line.endswith('127.0.0.1:1234 - "GET /healthz HTTP/1.1" 200 OK')


def test_secret_in_path_and_token_in_query_are_masked(access_stream: io.StringIO) -> None:
    _emit(f"/ws?token=opaque-not-a-jwt&x={SECRET}")
    out = access_stream.getvalue()
    assert SECRET not in out
    assert "opaque-not-a-jwt" not in out
    assert "GET /ws?token=[REDACTED]&x=*** HTTP/1.1" in out


def test_http_access_line_masks_ticket_and_other_credential_params(
    access_stream: io.StringIO,
) -> None:
    _emit("/canary?q=saree&access_token=aaa111&code=bbb222&Sig=ccc333&page=2")
    out = access_stream.getvalue()
    for leaked in ("aaa111", "bbb222", "ccc333"):
        assert leaked not in out
    assert "GET /canary?q=saree&access_token=[REDACTED]&code=[REDACTED]" in out
    assert "&Sig=[REDACTED]&page=2 HTTP/1.1" in out


def test_websocket_line_via_uvicorn_error_logger_masks_ticket(
    access_stream: io.StringIO,
) -> None:
    """uvicorn logs the WS upgrade on uvicorn.error as '%s - "WebSocket %s" [accepted]'."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    err = logging.getLogger("uvicorn.error")
    err.addHandler(handler)
    try:
        err.info(
            '%s - "WebSocket %s" [accepted]', "127.0.0.1:1234", "/chat/stream?ticket=one-time-nonce"
        )
    finally:
        err.removeHandler(handler)
    line = stream.getvalue().strip()
    assert "one-time-nonce" not in line
    assert line == '127.0.0.1:1234 - "WebSocket /chat/stream?ticket=[REDACTED]" [accepted]'


def test_status_code_stays_an_int_for_percent_d(access_stream: io.StringIO) -> None:
    record = logging.getLogger("uvicorn.access").makeRecord(
        "uvicorn.access", logging.INFO, "f", 1, ACCESS_FMT,
        ("127.0.0.1:1234", "GET", f"/x/{SECRET}", "1.1", 404), None,
    )
    assert isinstance(record.args, tuple)
    assert len(record.args) == 5
    assert record.args[4] == 404
    assert SECRET not in record.getMessage()
    assert record.getMessage().endswith("404")


def test_secret_split_across_template_and_args_falls_back_to_masked_message() -> None:
    redactor = Redactor([SECRET])
    record = logging.LogRecord("n", logging.INFO, "f", 1, "pw=hunter2-%s", ("db-password",), None)
    redact.redact_record(record, redactor)
    assert SECRET not in record.getMessage()
