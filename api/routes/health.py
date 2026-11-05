"""Health and readiness endpoints."""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Response

import api.deps as deps
from api.auth import _is_verification_disabled
from src.llm.resilience import ProbeCache
from src.utils.redact import Redactor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

# W4: /readyz makes a real minimal LLM call (max_tokens=1) through the whole chain.
# Cached so a burst of probes costs one call: the uptime cron (every 15 min) and the
# keep-warm cron (every 10 min) each trigger roughly one real call, ~240 calls/day
# at most, against Groq's free 1K requests/day per model. Failures are cached too.
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
        except Exception as exc:
            # Driver errors echo the connect string; mask before it reaches the response body.
            checks["database"] = f"error: {Redactor.from_env().redact(str(exc))}"
            ok = False
    else:
        checks["database"] = "not configured"

    if not ok:
        response.status_code = 503
        return {"status": "not ready", "checks": checks, "auth_enabled": not _is_verification_disabled()}

    return {"status": "ready", "checks": checks, "auth_enabled": not _is_verification_disabled()}
