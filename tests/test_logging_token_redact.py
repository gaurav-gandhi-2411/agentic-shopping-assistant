"""uvicorn access-log token redaction, including percent-encoded param names."""
from __future__ import annotations

import logging

import pytest

from api.logging_config import _redact_token_params, _TokenRedactFilter

SECRET = "s3cr3t-value"  # invented test constant


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (f"/canary?token={SECRET}", "/canary?token=[REDACTED]"),
        (f"/canary?x=1&token={SECRET}&y=2", "/canary?x=1&token=[REDACTED]&y=2"),
        (f"/canary?%74oken={SECRET}", "/canary?%74oken=[REDACTED]"),
        (f"/canary?tok%65n={SECRET}", "/canary?tok%65n=[REDACTED]"),
        (f"/canary?%54oKeN={SECRET}", "/canary?%54oKeN=[REDACTED]"),
        (f"/canary?TOKEN={SECRET}", "/canary?TOKEN=[REDACTED]"),
        (
            f"/canary?token={SECRET}&%74oken={SECRET}",
            "/canary?token=[REDACTED]&%74oken=[REDACTED]",
        ),
        (
            f"/canary?a=1&%74oken={SECRET}&token={SECRET}",
            "/canary?a=1&%74oken=[REDACTED]&token=[REDACTED]",
        ),
        (f"/canary?%74oken={SECRET}#frag", "/canary?%74oken=[REDACTED]#frag"),
        # encoded '&' (%26) inside a value stays part of that value and is fully masked
        ("/canary?%74oken=ab%26cd&x=1", "/canary?%74oken=[REDACTED]&x=1"),
        (
            "/canary?x=a%26token%3Dkeep&%74oken=zz",
            "/canary?x=a%26token%3Dkeep&%74oken=[REDACTED]",
        ),
        (f"/canary?token%3D{SECRET}", "/canary?token=[REDACTED]"),
        (f"/canary?a=1&%54oken%3D{SECRET}&b=2", "/canary?a=1&token=[REDACTED]&b=2"),
        ("/health?x=%41", "/health?x=%41"),
        ("/no-query", "/no-query"),
    ],
)
def test_redact(raw: str, expected: str) -> None:
    assert _redact_token_params(raw) == expected


def test_filter_masks_uvicorn_access_args() -> None:
    rec = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", f"/canary?%74oken={SECRET}", "1.1", 200), None,
    )
    assert _TokenRedactFilter().filter(rec)
    assert SECRET not in rec.getMessage()
    assert "[REDACTED]" in rec.getMessage()
