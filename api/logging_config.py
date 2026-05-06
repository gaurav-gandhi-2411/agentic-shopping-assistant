"""Structured JSON logging with conversation_id propagation via contextvars."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextvars import ContextVar
from typing import Any

# Module-level contextvar: set at the start of each request, read by the log filter.
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="")


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
