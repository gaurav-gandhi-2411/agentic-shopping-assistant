"""Health and readiness endpoints."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Response

import api.deps as deps
from api.auth import _is_verification_disabled
from src.llm.resilience import ProbeCache
from src.retrieval.snapshot import describe_index
from src.utils.redact import Redactor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

# W4: /readyz makes a real minimal LLM call (max_tokens=1) through the whole chain.
# Cached so a burst of probes costs one call. The only scheduled caller is the uptime-alert
# workflow (cron */15 -> at most 96 calls/day, ~6 at the measured GitHub cadence), plus 2 per
# deploy smoke; the external canary monitor uses /canary/status and must NOT hit this path.
# Budget vs Groq's free 1K requests/day per model: docs/ops/canary-budget.md. Failures are
# cached too.
_llm_probe_cache = ProbeCache(
    ttl_s=float(os.environ.get("READYZ_LLM_CACHE_SECONDS", "60")), timeout_s=20.0
)

# Item 257: readiness previously checked retriever/catalogue/LLM but never the
# database, despite looks.py/feedback.py/events.py/auth.py all depending on it.
# The existing 15-minute uptime cron hits /api/brand, which never touches
# Supabase either -- found while investigating Item 255's min-instances fix,
# this service had ZERO scheduled DB activity, real risk of Supabase's 7-day-
# inactivity auto-pause going unnoticed on the one product with a confirmed
# real user. A trivial `SELECT 1` costs nothing and definitively answers "is
# the database actually reachable," not just "is the app process alive."

# Schema-head gate: the Dockerfile copies alembic.ini + alembic/ next to api/, so the image
# knows which revision(s) it was built for. /readyz compares that with the database's
# alembic_version and fails closed on ANY difference (behind, ahead, unreadable), so the
# tag-URL deploy smoke stops a rollout BEFORE the traffic shift when migrations were not
# applied. The gate only reads: one SELECT, no DDL, no migration is ever run from here.
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


@lru_cache(maxsize=1)
def _image_heads() -> tuple[str, ...]:
    """Head revision id(s) shipped in this image (sorted; static for the process lifetime)."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    heads = ScriptDirectory.from_config(Config(str(_ALEMBIC_INI))).get_heads()
    if not heads:
        raise RuntimeError("no alembic head found")
    return tuple(sorted(heads))


def check_schema_head(engine: Any) -> tuple[bool, str]:
    """Compare the DB's ``alembic_version`` with the image's head(s) -> (ok, check string).

    Messages carry revision ids and exception type names only, never the exception text (a
    driver error can echo the connect string). The ``error:`` prefix on failure matches the
    other /readyz checks, which uptime-alert.yml uses to name the failing check.
    """
    try:
        heads = _image_heads()
    except Exception as exc:
        return False, f"error: cannot read image migration heads ({type(exc).__name__})"
    expected = ", ".join(heads)
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(text("SELECT version_num FROM alembic_version")).fetchall()
    except Exception as exc:  # missing table, permission, connection: all fail closed
        name = type(exc).__name__
        return False, f"error: db schema unreadable ({name}), image expects {expected}"
    current = tuple(sorted(str(r[0]) for r in rows))
    if current == heads:
        return True, f"ok ({expected})"
    return False, f"error: db at {', '.join(current) or 'none'}, image expects {expected}"


def _now() -> datetime:
    """Current UTC time (patched in tests)."""
    return datetime.now(timezone.utc)


@router.get("/healthz")
def liveness() -> dict:
    """Liveness probe — returns immediately; no dependency checks."""
    return {"status": "ok"}


@router.get("/readyz")
def readiness(response: Response) -> dict:
    """Readiness probe — verifies that heavy resources are loaded."""
    checks: dict[str, str] = {}
    ok = True

    try:
        r = deps.get_retriever()
        n = r.dense.index.ntotal if r.dense.index is not None else 0
        checks["retriever"] = f"ok ({n:,} vectors)"
    except Exception as exc:
        checks["retriever"] = f"error: {exc}"
        ok = False

    try:
        df = deps.get_catalogue_df()
        checks["catalogue"] = f"ok ({len(df):,} items)"
    except Exception as exc:
        checks["catalogue"] = f"error: {exc}"
        ok = False

    # Index build info (V3b). Informational: a legacy index or an old build NEVER fails
    # readiness (a stale index is /canary/status's job, not a reason to pull the instance).
    index_info: dict[str, Any] | None = None
    snap = deps.get_snapshot_or_none()
    if snap is not None:
        index_info = describe_index(snap, _now())
        reloader = deps.get_reloader()
        index_info["reload"] = (
            reloader.status()
            if reloader is not None
            else {"hot_reload": "disabled (INDEX_HOT_RELOAD not set)"}
        )
        if snap.manifest is None:
            checks["index"] = "legacy (no manifest)"
        else:
            checks["index"] = (
                f"ok (build_id={index_info['build_id']}, age_hours={index_info['age_hours']}, "
                f"stores={index_info['stores']}, items={index_info['items']})"
            )

    try:
        llm = deps.get_llm()
        probe = getattr(llm, "probe", None)  # only GroqClient has a chain to probe
        if probe is None:
            checks["llm"] = "ok"
        else:
            res = _llm_probe_cache.get(probe)
            served = f"{res['tier']}:{res['model']}" if res.get("model") else res["tier"]
            fallback = res["tier"] != "groq" or res["model"] != getattr(llm, "model", None)
            checks["llm"] = f"ok ({served}, {res['latency_ms']} ms{', FALLBACK' if fallback else ''})"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"
        ok = False

    db_engine = deps.get_db_engine()
    if db_engine is not None:
        try:
            from sqlalchemy import text

            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
            schema_ok, checks["schema"] = check_schema_head(db_engine)
            ok = ok and schema_ok
        except Exception as exc:
            # Driver errors echo the connect string; mask before it reaches the response body.
            checks["database"] = f"error: {Redactor.from_env().redact(str(exc))}"
            checks["schema"] = "skipped (database unreachable)"
            ok = False
    else:
        checks["database"] = "not configured"
        checks["schema"] = "not configured"  # no engine: unchanged behaviour, nothing to gate

    body: dict[str, Any] = {"status": "ready", "checks": checks}
    if index_info is not None:
        body["index"] = index_info
    body["auth_enabled"] = not _is_verification_disabled()
    if not ok:
        response.status_code = 503
        body["status"] = "not ready"
    return body
