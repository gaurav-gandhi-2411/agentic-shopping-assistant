"""Health and readiness endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Response

import api.deps as deps
from api.auth import _is_verification_disabled

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


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
        deps.get_llm()
        checks["llm"] = "ok"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"
        ok = False

    if not ok:
        response.status_code = 503
        return {"status": "not ready", "checks": checks, "auth_enabled": not _is_verification_disabled()}

    return {"status": "ready", "checks": checks, "auth_enabled": not _is_verification_disabled()}
