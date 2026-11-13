"""Credential-bearing query params (?ticket=, ?token=, ?code= ...) are masked in any log text."""

from __future__ import annotations

import logging

import pytest

from src.utils.redact import SENSITIVE_QUERY_PARAMS, Redactor, redact_query_params, redact_record

VAL = "abc123XYZ"  # invented test constant


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # websocket access-log shape (value ends at whitespace)
        (
            f'127.0.0.1:1 - "WebSocket /chat/stream?ticket={VAL} [accepted]"',
            '127.0.0.1:1 - "WebSocket /chat/stream?ticket=[REDACTED] [accepted]"',
        ),
        # HTTP access-log shape
        (
            f'"GET /canary?token={VAL} HTTP/1.1" 200',
            '"GET /canary?token=[REDACTED] HTTP/1.1" 200',
        ),
        (f"/x?ticket={VAL}&x=1", "/x?ticket=[REDACTED]&x=1"),
        (f"/x?x=1&token={VAL}", "/x?x=1&token=[REDACTED]"),  # param at end of line
        (f"/x?x=1&token={VAL}&y=2", "/x?x=1&token=[REDACTED]&y=2"),
        (f"/x?TICKET={VAL}", "/x?TICKET=[REDACTED]"),  # case-insensitive
        (f"/x?Access_Token={VAL}", "/x?Access_Token=[REDACTED]"),
        ("/x?ticket=a%2Fb%3D%3D%26c&x=1", "/x?ticket=[REDACTED]&x=1"),  # URL-encoded value
        (f"/x?%74icket={VAL}", "/x?%74icket=[REDACTED]"),  # URL-encoded name
        (f"/x?a=1&ticket={VAL}#frag", "/x?a=1&ticket=[REDACTED]#frag"),
        (f"/x?ticket={VAL}&token={VAL}", "/x?ticket=[REDACTED]&token=[REDACTED]"),
        (f'url="/x?code={VAL}"', 'url="/x?code=[REDACTED]"'),
        # unrelated params are untouched (incl. names that merely contain a sensitive word)
        ("/search?q=red+saree&brand=fabindia&page=2", "/search?q=red+saree&brand=fabindia&page=2"),
        (f"/x?monkey={VAL}&tokens={VAL}&codec=1", f"/x?monkey={VAL}&tokens={VAL}&codec=1"),
        ("no query here, tickets are fine", "no query here, tickets are fine"),
    ],
)
def test_redact_query_params(raw: str, expected: str) -> None:
    assert redact_query_params(raw) == expected


@pytest.mark.parametrize("name", sorted(SENSITIVE_QUERY_PARAMS))
def test_every_listed_param_is_masked(name: str) -> None:
    assert redact_query_params(f"/x?q=1&{name}={VAL}&z=2") == f"/x?q=1&{name}=[REDACTED]&z=2"


def test_param_list_matches_the_documented_set() -> None:
    assert SENSITIVE_QUERY_PARAMS == {
        "ticket",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "jwt",
        "api_key",
        "apikey",
        "key",
        "secret",
        "signature",
        "sig",
        "code",
        "password",
    }


def test_redactor_applies_it_to_message_and_args() -> None:
    redactor = Redactor()
    record = logging.LogRecord(
        "x", logging.INFO, "f", 1, "GET %s HTTP/1.1", (f"/chat/stream?ticket={VAL}&q=1",), None
    )
    redact_record(record, redactor)
    assert VAL not in record.getMessage()
    assert record.getMessage() == "GET /chat/stream?ticket=[REDACTED]&q=1 HTTP/1.1"
    msg = f"ws /s?ticket={VAL} [accepted]"
    inline = logging.LogRecord("x", logging.INFO, "f", 1, msg, None, None)
    redact_record(inline, redactor)
    assert inline.getMessage() == "ws /s?ticket=[REDACTED] [accepted]"
