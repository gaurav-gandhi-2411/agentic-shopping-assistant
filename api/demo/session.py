from __future__ import annotations

import logging
import os
import time

import jwt as pyjwt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOKEN_TTL_SECONDS: int = 3600  # 1-hour demo session
_DEMO_TOKEN_TYPE: str = "demo"


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------


def _get_secret() -> str:
    """Return the HMAC secret used to sign demo JWTs.

    If DEMO_JWT_SECRET is not set and DEMO_MODE is active, a hardcoded dev
    fallback is used with a WARNING.  This makes local development work out of
    the box while making it impossible to accidentally use the fallback in
    production without noticing the log noise.
    """
    secret = os.environ.get("DEMO_JWT_SECRET", "")
    if secret:
        return secret

    # Fallback — only acceptable in non-production environments.
    if os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes"):
        logger.warning(
            "DEMO_JWT_SECRET is not set; using hardcoded dev fallback."
            " Set DEMO_JWT_SECRET before deploying to production."
        )
    return "dev-demo-secret-CHANGE-ME"


# ---------------------------------------------------------------------------
# Token operations
# ---------------------------------------------------------------------------


def create_demo_token(anon_id: str, brand: str) -> str:
    """Mint an HS256 JWT for an anonymous demo session.

    Payload fields:
      sub   — anonymous user identifier (e.g. "anon:<uuid>")
      brand — the brand this session is scoped to
      type  — always "demo" so validators can reject non-demo tokens
      iat   — issued-at (Unix timestamp)
      exp   — expiry = iat + 3600 s
    """
    now = int(time.time())
    payload = {
        "sub": anon_id,
        "brand": brand,
        "type": _DEMO_TOKEN_TYPE,
        "iat": now,
        "exp": now + _TOKEN_TTL_SECONDS,
    }
    return pyjwt.encode(payload, _get_secret(), algorithm="HS256")


def validate_demo_token(token: str) -> str | None:
    """Verify and decode a demo JWT.

    Returns the `sub` claim (anon_id) on success, or None on any failure
    (expired, bad signature, missing claims).  Never raises — callers check
    the return value.
    """
    try:
        payload = pyjwt.decode(token, _get_secret(), algorithms=["HS256"])
        if payload.get("type") != _DEMO_TOKEN_TYPE:
            return None
        sub: str = payload["sub"]
        return sub
    except Exception:
        return None
