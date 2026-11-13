"""Safe parsing of DATABASE_URL.

Why this exists: the production secret held a URL whose password contained raw
reserved characters (notably '@').  A generic URL parser then split userinfo at
the wrong '@', treated part of the password as the host, and the driver's
connect error echoed that "host" -- i.e. a password fragment -- into the logs.

Rules (single source of truth, tested in tests/test_db_url.py):

* The userinfo/host boundary is the LAST '@' in the URL.  A password may
  therefore contain raw '@', ':', '/', '?', '#', '[', ']' or spaces.  A raw '@'
  in the path or query string is NOT supported (percent-encode it).
* User and password are decoded exactly once, and only when the component looks
  percent-encoded: it contains at least one '%' and EVERY '%' is followed by two
  hex digits.  So ``p%40ss`` -> ``p@ss`` (already encoded, not double-encoded),
  while a raw ``100%`` or ``a%zz`` is kept literally.  Known ambiguity: a raw
  password that literally contains e.g. ``%41`` is read as encoded; store such
  passwords percent-encoded (``%2541``).
* The result is a ``sqlalchemy.engine.URL`` built from parts, so there is no
  string re-parse, and ``str(url)`` masks the password.
* Error messages here never echo any part of the input.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from sqlalchemy.engine import URL, Engine

_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://(.*)$", re.DOTALL)
# A '%' that is NOT followed by two hex digits => the value is raw, not encoded.
_STRAY_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")

# SQLAlchemy 2.x + psycopg3 needs the explicit +psycopg driver qualifier.
_PG_SCHEMES = {"postgresql", "postgres"}


def _decode_component(value: str) -> str:
    """Decode a userinfo component once, only if it is percent-encoded."""
    if "%" in value and not _STRAY_PERCENT_RE.search(value):
        return unquote(value)
    return value


def parse_database_url(raw: str) -> URL:
    """Parse a DATABASE_URL (password may hold raw reserved chars) into a URL.

    Raises ValueError with a generic message (never echoing the input).
    """
    match = _SCHEME_RE.match(raw.strip())
    if match is None:
        raise ValueError("DATABASE_URL is not a valid URL (expected scheme://user:pass@host/db)")
    scheme, rest = match.groups()

    at = rest.rfind("@")
    username: str | None = None
    password: str | None = None
    tail = rest
    if at != -1:
        userinfo, tail = rest[:at], rest[at + 1:]
        user_part, sep, pass_part = userinfo.partition(":")
        username = _decode_component(user_part) or None
        password = _decode_component(pass_part) if sep else None

    try:
        parts = urlsplit("//" + tail)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise ValueError("DATABASE_URL has an invalid host or port") from None

    query: dict[str, Any] = {}
    for key, val in parse_qsl(parts.query, keep_blank_values=True):
        if key in query:
            existing = query[key]
            query[key] = (*existing, val) if isinstance(existing, tuple) else (existing, val)
        else:
            query[key] = val

    driver = "postgresql+psycopg" if scheme.lower() in _PG_SCHEMES else scheme
    return URL.create(
        driver,
        username=username,
        password=password,
        host=host or None,
        port=port,
        database=unquote(parts.path.lstrip("/")) or None,
        query=query,
    )


def normalize_database_url(raw: str) -> str:
    """Return a fully percent-encoded URL string (includes the password!).

    Only for handing to tools that require a string; never log the result.
    """
    return parse_database_url(raw).render_as_string(hide_password=False)


def pooler_connect_args(connect_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """connect_args safe for the Supabase transaction pooler (Supavisor, port 6543).

    psycopg3 auto-prepares statements after 5 executions; a transaction-mode pooler hands
    out different server connections, so the next PREPARE collides
    (``DuplicatePreparedStatement: prepared statement "_pg3_N" already exists``).
    ``prepare_threshold=None`` disables auto-prepare.  Caller-supplied keys win.
    """
    return {"prepare_threshold": None, **(connect_args or {})}


def create_db_engine(raw: str, **kwargs: Any) -> Engine:
    """create_engine() over a safely parsed URL.  Every call site uses this.

    Always disables psycopg3 auto-prepare (see pooler_connect_args); a caller-supplied
    ``connect_args`` is merged, not replaced.
    """
    from sqlalchemy import create_engine

    kwargs["connect_args"] = pooler_connect_args(kwargs.get("connect_args"))
    return create_engine(parse_database_url(raw), **kwargs)
