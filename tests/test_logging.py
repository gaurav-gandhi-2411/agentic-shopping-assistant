"""Tests for structured JSON logging and conversation_id contextvar plumbing."""
from __future__ import annotations

import io
import json
import logging

from api.logging_config import _ConversationIdFilter, _JsonFormatter, conversation_id_var


def _make_logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_ConversationIdFilter())
    log = logging.getLogger(name)
    log.propagate = False
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    return log, buf


def _cleanup(log: logging.Logger) -> None:
    log.handlers.clear()


def test_conversation_id_appears_in_json_log():
    """Set the contextvar, emit a log line, verify conversation_id is in the JSON."""
    log, buf = _make_logger("test_contextvar_set")
    token = conversation_id_var.set("test-conv-id-abc")
    try:
        log.info("hello from test")
    finally:
        conversation_id_var.reset(token)
        _cleanup(log)

    record = json.loads(buf.getvalue().strip())
    assert record["conversation_id"] == "test-conv-id-abc"
    assert record["msg"] == "hello from test"
    assert record["level"] == "INFO"
    assert "ts" in record
    assert record["logger"] == "test_contextvar_set"


def test_no_conversation_id_when_contextvar_empty():
    """When the contextvar is empty (default), conversation_id must NOT appear."""
    log, buf = _make_logger("test_contextvar_unset")
    token = conversation_id_var.set("")
    try:
        log.info("no id here")
    finally:
        conversation_id_var.reset(token)
        _cleanup(log)

    record = json.loads(buf.getvalue().strip())
    assert "conversation_id" not in record
    assert record["msg"] == "no id here"


def test_extra_fields_forwarded_to_json():
    """Fields passed via extra={} (latency_ms, n_items, etc.) appear in the JSON."""
    log, buf = _make_logger("test_extra_fields")
    try:
        log.info("with extras", extra={"latency_ms": 42, "n_items": 5})
    finally:
        _cleanup(log)

    record = json.loads(buf.getvalue().strip())
    assert record["latency_ms"] == 42
    assert record["n_items"] == 5


def test_contextvar_reset_after_request():
    """Verify that resetting the token clears the conversation_id for the next log."""
    log, buf = _make_logger("test_contextvar_reset")
    token = conversation_id_var.set("request-1")
    try:
        log.info("during request")
    finally:
        conversation_id_var.reset(token)

    # Second log — no id set
    log.info("after reset")
    _cleanup(log)

    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    assert lines[0]["conversation_id"] == "request-1"
    assert "conversation_id" not in lines[1]
