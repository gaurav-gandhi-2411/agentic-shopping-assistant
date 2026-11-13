"""Secret redaction for logs, error text and Sentry events.

Masks (1) ``scheme://user:password@`` userinfo, (2) the literal values of
configured secrets read from the environment (plus their percent-encoded forms
and, for the DB password, its long segments), (3) Bearer / JWT-looking
tokens, and (4) the values of credential-bearing query parameters (``?ticket=``,
``?token=``, ``?code=`` ...).  Secret values are never logged by this module.
"""
from __future__ import annotations

import logging
import os
import re
import weakref
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import quote, quote_plus, unquote_plus

MASK = "***"

# Env vars whose values must never appear in output.
SECRET_ENV_VARS = (
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "LANGSMITH_API_KEY",
    "DEMO_JWT_SECRET",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_ANON_KEY",
    "SUPABASE_KEY",
    "SENTRY_DSN",
    "CANARY_TOKEN",  # GET /canary auth secret (Secret Manager secret CANARY_TOKEN)
)
# Shorter values would mask ordinary words; real secrets are far longer.
_MIN_SECRET_LEN = 6
_MIN_SEGMENT_LEN = 4
_SEGMENT_SPLIT_RE = re.compile(r"[@:/?#\[\]\s%]+")

# Password may itself contain '@', so run greedily to the last '@' in the token.
_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*://[^\s:/@'\"]*:)[^\s'\"]*@")
_BEARER_RE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{8,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}(?:\.[A-Za-z0-9_\-]*)?")


# Query-string parameters whose VALUES are credentials (WS tickets, OAuth codes, API keys...).
# Matched case-insensitively and after percent-decoding the name (Starlette decodes names, so
# ``?%74icket=x`` still authenticates).
SENSITIVE_QUERY_PARAMS = frozenset(
    {
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
        "x_api_key",
        "client_secret",
    }
)
# Names are compared lower-cased with "-" folded to "_", so access-token / x-api-key /
# client-secret are covered by the underscore entries above.
# Exact ``key`` and ``code`` are masked even when innocuous (e.g. ``?key=2``): deliberately
# fail-safe, a harmless value lost from a log line beats a leaked credential.
REDACTED = "[REDACTED]"
# Values that are already masks (ours, Sentry's, userinfo's) must be left byte-for-byte alone,
# so redaction is idempotent and an earlier scrubber's placeholder (Sentry's ``[Filtered]``) is
# not overwritten by ours.
_ALREADY_MASKED = frozenset({"[Filtered]", "[REDACTED]", "***"})
# ``?name=value`` / ``&name=value`` (``=`` may itself be encoded as %3D).  The value runs to the
# next ``&``, whitespace, quote or fragment, so it also stops correctly in an access-log line
# such as ``"WebSocket /chat/stream?ticket=abc [accepted]"``.  A percent-encoded ``&`` (%26)
# is not a separator, so it stays inside the value and is masked with it.
_QUERY_PARAM_RE = re.compile(
    r"(?P<sep>[?&])(?P<name>[^=&\s\"'#?]*?)(?P<eq>=|%3[dD])(?P<val>[^&\s\"'#]*)"
)


def _mask_query_param(match: re.Match[str]) -> str:
    name = unquote_plus(match["name"]).lower().replace("-", "_")
    if name not in SENSITIVE_QUERY_PARAMS or unquote_plus(match["val"]) in _ALREADY_MASKED:
        return match[0]
    return f"{match['sep']}{match['name']}={REDACTED}"


def redact_query_params(text: str) -> str:
    """Replace the value of every sensitive query parameter in ``text`` with ``[REDACTED]``."""
    if "?" not in text and "&" not in text:
        return text
    return _QUERY_PARAM_RE.sub(_mask_query_param, text)


def _variants(value: str) -> set[str]:
    return {value, quote(value, safe=""), quote_plus(value)}


def collect_secrets(environ: Mapping[str, str] | None = None) -> list[str]:
    """Secret strings to mask, taken from the environment (longest first)."""
    env = os.environ if environ is None else environ
    found: set[str] = set()
    for name in SECRET_ENV_VARS:
        val = env.get(name, "").strip()
        if len(val) >= _MIN_SECRET_LEN:
            found |= _variants(val)

    raw_url = env.get("DATABASE_URL", "").strip()
    if raw_url:
        from src.utils.db_url import parse_database_url

        try:
            password = parse_database_url(raw_url).password
        except ValueError:
            password = None
        if password:
            found |= _variants(password)
            for seg in _SEGMENT_SPLIT_RE.split(password):
                if len(seg) >= _MIN_SEGMENT_LEN:
                    found |= _variants(seg)
    return sorted((s for s in found if len(s) >= _MIN_SEGMENT_LEN), key=len, reverse=True)


class Redactor:
    """Applies pattern and literal-secret masking to text and nested objects."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = sorted({s for s in secrets if s}, key=len, reverse=True)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Redactor:
        return cls(collect_secrets(environ))

    def redact(self, text: str) -> str:
        # Literal secrets first: a password with spaces or a raw '@' would otherwise be only
        # partly consumed by the userinfo regex, leaving the literal unmatched afterwards.
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        text = _USERINFO_RE.sub(r"\1" + MASK + "@", text)
        text = redact_query_params(text)
        text = _BEARER_RE.sub(r"\1" + MASK, text)
        return _JWT_RE.sub(MASK, text)

    def redact_obj(self, obj: Any) -> Any:
        """Recursively redact strings inside dicts / lists / tuples (Sentry events)."""
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {k: self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_obj(v) for v in obj)
        return obj

    def redact_inplace(self, obj: Any) -> Any:
        """Like redact_obj but mutates dicts/lists in place (Sentry hooks return the same object)."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self.redact_inplace(v)
            return obj
        if isinstance(obj, list):
            obj[:] = [self.redact_inplace(v) for v in obj]
            return obj
        return self.redact_obj(obj)


def _redact_message(record: logging.LogRecord, message: str, redactor: Redactor) -> None:
    """Redact record.msg/args, keeping the args container shape when that is safe.

    Formatters such as uvicorn's AccessFormatter unpack ``record.args`` as a fixed tuple, so
    blanking it (the old behaviour) raised on every access-log line.  Nothing to mask: leave
    the record untouched.  Otherwise mask the template and each string arg in place and keep
    the result only if the re-rendered message is clean (a secret split across template and
    args would not be); else fall back to the pre-rendered, masked message with args=None.
    """
    masked = redactor.redact(message)
    args = record.args
    if masked == message and (args is None or isinstance(args, tuple | dict)):
        return
    if isinstance(args, tuple | dict) and args and isinstance(record.msg, str):
        new_msg = redactor.redact(record.msg)
        new_args = redactor.redact_obj(args)
        probe = logging.LogRecord("x", 0, "", 0, new_msg, new_args, None)
        try:
            rendered: str | None = probe.getMessage()
        except Exception:  # noqa: BLE001 - fall through to the pre-rendered fallback
            rendered = None
        if rendered is not None and redactor.redact(rendered) == rendered:
            record.msg, record.args = new_msg, new_args
            return
    record.msg = masked
    record.args = None


def redact_record(record: logging.LogRecord, redactor: Redactor) -> None:
    """Mask a LogRecord in place.

    exc_info is deliberately KEPT: sentry_sdk's logging integration runs after the
    handlers and reads record.exc_info to build the event's exception section.  The
    redacted traceback goes in exc_text instead; stdlib Formatter.format (and our
    _JsonFormatter) use a pre-set exc_text rather than re-rendering exc_info.
    """
    try:
        message = record.getMessage()
    except Exception:  # noqa: BLE001 - a bad format string must not hide the record
        record.msg = redactor.redact(str(record.msg))
        record.args = None
    else:
        _redact_message(record, message, redactor)
    if record.exc_info and record.exc_info[0] is not None and not record.exc_text:
        record.exc_text = logging.Formatter().formatException(record.exc_info)
    if record.exc_text:
        record.exc_text = redactor.redact(record.exc_text)
    if record.stack_info:
        record.stack_info = redactor.redact(record.stack_info)


class RedactionFilter(logging.Filter):
    """Handler-level safety net: redacts records that bypassed the record factory."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        redact_record(record, self._redactor)
        return True


# Swapped by install_log_redaction so the factory (installed once) always uses the latest.
_active_redactor: Redactor | None = None
# Record factories we installed (idempotency marker; a typed replacement for tagging the
# function with an attribute). Weak so a replaced factory can still be collected.
_redacting_factories: weakref.WeakSet[Callable[..., logging.LogRecord]] = weakref.WeakSet()


def install_log_redaction(redactor: Redactor | None = None) -> Redactor:
    """Redact every LogRecord at creation, and filter every existing handler.

    The record factory is the primary mechanism: it runs for every record before any
    handler (ours, uvicorn's, Sentry's, or ones added later), so there is no handler
    ordering or late-registration gap.  The handler filter is a second layer.  Both
    paths are idempotent; calling this again just swaps the active redactor.
    """
    global _active_redactor
    redactor = redactor or Redactor.from_env()
    _active_redactor = redactor

    current = logging.getLogRecordFactory()
    if current not in _redacting_factories:
        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = current(*args, **kwargs)
            if _active_redactor is not None:
                redact_record(record, _active_redactor)
            return record

        _redacting_factories.add(factory)
        logging.setLogRecordFactory(factory)

    loggers = [logging.getLogger()] + [
        lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)
    ]
    for lg in loggers:
        for handler in lg.handlers:
            if not any(isinstance(f, RedactionFilter) for f in handler.filters):
                handler.addFilter(RedactionFilter(redactor))
    return redactor
