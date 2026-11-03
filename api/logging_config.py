"""Structured JSON logging with conversation_id propagation via contextvars."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from contextvars import ContextVar
from typing import Any
from urllib.parse import unquote_plus

# Module-level contextvar: set at the start of each request, read by the log filter.
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="")

_TOKEN_PARAM_RE = re.compile(r"(\?|&)(token)=[^&\s\"']*", re.IGNORECASE)
_QUERY_END_RE = re.compile(r"[#\s]")  # a fragment or whitespace ends the query string


def _redact_token_params(text: str) -> str:
    """Mask the value of every ``token`` query param in ``text`` (a request path/URL).

    Plain strings take the regex fast path. If a ``%`` is present the param NAME may be
    percent-encoded (``?%74oken=x``): Starlette decodes names, so such a request still
    authenticates and the raw value would reach the access log. In that case the query is
    split on ``&``, each name is percent-decoded and compared case-insensitively.
    """
    if "?" not in text:
        return text
    if "%" not in text:
        return _TOKEN_PARAM_RE.sub(r"\1\2=[REDACTED]", text)
    base, _, rest = text.partition("?")
    end = _QUERY_END_RE.search(rest)
    query, tail = (rest[: end.start()], rest[end.start():]) if end else (rest, "")
    parts = []
    for part in query.split("&"):
        name, eq, _value = part.partition("=")
        if eq and unquote_plus(name).lower() == "token":
            part = f"{name}=[REDACTED]"
        parts.append(part)
    return f"{base}?{'&'.join(parts)}{tail}"


class _TokenRedactFilter(logging.Filter):
    """Strips ?token=<jwt> from uvicorn access log request lines (incl. encoded names)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                _redact_token_params(a) if isinstance(a, str) else a for a in record.args
            )
        return True


class _ConversationIdFilter(logging.Filter):
    """Injects conversation_id from the contextvar into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.conversation_id = conversation_id_var.get("")  # type: ignore[attr-defined]
        return True


class _JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        cid = getattr(record, "conversation_id", "")
        if cid:
            data["conversation_id"] = cid
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        # Include any extra fields attached via logging.getLogger().info(..., extra={...})
        for key in ("latency_ms", "tool", "action", "n_items", "provider"):
            val = record.__dict__.get(key)
            if val is not None:
                data[key] = val
        return json.dumps(data, default=str)


def setup_logging(log_level: str | None = None) -> None:
    """Configure root logger with JSON output.  Call once at application startup."""
    level_name = (log_level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_ConversationIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any default handlers to avoid duplicate output.
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "sentence_transformers", "faiss", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Redact ?token=<jwt> from uvicorn's built-in access log (WS upgrade lines).
    logging.getLogger("uvicorn.access").addFilter(_TokenRedactFilter())
