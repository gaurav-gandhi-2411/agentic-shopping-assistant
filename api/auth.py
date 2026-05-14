"""JWT verification for FastAPI routes using Supabase RS256 tokens.

JWKS endpoint (verified 2026-05 against Supabase docs):
  {SUPABASE_URL}/auth/v1/.well-known/jwks.json
  Source: https://supabase.com/docs/guides/auth/jwks

  Supabase issues RS256 JWTs signed with a project-scoped RSA private key.
  The JWKS endpoint exposes only the corresponding public keys, so the
  backend can verify tokens without ever holding the signing secret.
  PyJWT's PyJWKClient selects the correct key by matching the JWT header's
  `kid` field and caches the JWKS response for `lifespan` seconds (24 h
  here; Supabase edge-caches the endpoint for 10 min so refreshes are cheap).

Algorithm: RS256.
aud claim: "authenticated" (Supabase default for logged-in users).
iss claim: not validated — the RS256 signature is sufficient; iss would
  require knowing the project URL at decode time without adding security.

Env vars (read at call time so monkeypatching works in tests):
  SUPABASE_URL              — https://abc.supabase.co (required in prod)
  SUPABASE_JWT_AUD          — audience to verify; default "authenticated"
  JWT_VERIFICATION_DISABLED — "true" skips all verification (dev only)
  JWT_TEST_PUBLIC_KEY       — PEM RSA public key; bypasses JWKS for unit
                              tests so no network calls are made
"""
from __future__ import annotations

import logging
import os
from typing import Any

import jwt as pyjwt
from fastapi import Header, HTTPException
from jwt import ExpiredSignatureError, InvalidTokenError

logger = logging.getLogger(__name__)

# Lazily created module-level PyJWKClient.  Tests bypass it entirely via
# JWT_TEST_PUBLIC_KEY so it is never instantiated during the test suite.
_jwks_client: Any = None

# Pluggable allow-list checker.  Default allows all (no DB configured).
# Replaced at startup by api.main when DATABASE_URL is set.
# Monkeypatchable in tests: monkeypatch.setattr(api.auth, "_check_allowlist", ...)
_check_allowlist: Any = lambda email: True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_aud() -> str:
    return os.environ.get("SUPABASE_JWT_AUD", "authenticated")


def _is_verification_disabled() -> bool:
    return os.environ.get("JWT_VERIFICATION_DISABLED", "").lower() in ("1", "true", "yes")


def _get_jwks_client():
    """Return (and lazily create) the module-level PyJWKClient.

    Raises RuntimeError if SUPABASE_URL is unset — this should never happen
    in production since the lifespan startup would have warned already.
    """
    global _jwks_client
    if _jwks_client is None:
        supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        if not supabase_url:
            raise RuntimeError(
                "SUPABASE_URL must be set when JWT_VERIFICATION_DISABLED is not true. "
                "Set it to your Supabase project URL, e.g. https://abc.supabase.co"
            )
        jwks_uri = f"{supabase_url}/auth/v1/.well-known/jwks.json"
        from jwt import PyJWKClient
        # lifespan=86400: cache JWKS for 24 h; PyJWKClient refreshes on expiry.
        _jwks_client = PyJWKClient(jwks_uri, cache_jwk_set=True, lifespan=86400)
        logger.debug("JWKS client initialised: %s", jwks_uri)
    return _jwks_client


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def verify_jwt(token: str) -> dict:
    """Verify a Supabase JWT and return its decoded payload claims.

    When JWT_TEST_PUBLIC_KEY is set, the provided PEM key is used instead of
    the JWKS endpoint so unit tests never make network requests.

    Raises:
        HTTPException(401) — expired token, invalid signature, wrong audience,
                             or any other verification failure.
    """
    test_key_pem = os.environ.get("JWT_TEST_PUBLIC_KEY", "").strip()
    aud = _get_aud()

    try:
        if test_key_pem:
            payload = pyjwt.decode(
                token,
                test_key_pem,
                algorithms=["RS256"],
                audience=aud,
            )
        else:
            client = _get_jwks_client()
            signing_key = client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=aud,
            )
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError as exc:
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid token")
    except RuntimeError as exc:
        logger.error("JWT verification error (config/JWKS): %s", exc, exc_info=True)
        raise HTTPException(status_code=401, detail="Unauthorized")

    return payload


def _enforce_allowlist(payload: dict) -> None:
    """Raise 401 if the JWT email claim is not on the allow-list.

    Skipped when email is absent (service-role tokens, phone-only accounts).
    The check is delegated to _check_allowlist so production and test code
    can swap the implementation without touching this logic.
    """
    email = payload.get("email", "")
    if email and not _check_allowlist(email):
        raise HTTPException(status_code=401, detail="Email not on allow-list")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_current_user_id(authorization: str = Header(default="")) -> str:
    """FastAPI dependency — extract user_id from the Bearer JWT.

    Returns the JWT ``sub`` claim (Supabase user UUID).

    When JWT_VERIFICATION_DISABLED=true this returns DEV_USER_ID without
    checking the header.  That env var must never be set in production.

    Raises:
        HTTPException(401) — missing/invalid/expired token.
    """
    if _is_verification_disabled():
        from api.deps import DEV_USER_ID
        return DEV_USER_ID

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header (expected: Bearer <token>)",
        )
    token = authorization.split(" ", 1)[1].strip()
    payload = verify_jwt(token)
    sub = payload.get("sub")
    if not sub:
        logger.warning("JWT payload missing sub claim")
        raise HTTPException(status_code=401, detail="Unauthorized")
    _enforce_allowlist(payload)
    return sub


def get_current_user_id_ws(token: str) -> str:
    """Extract user_id from a token string for WebSocket connections.

    The WebSocket client passes the token as ``?token=<jwt>`` in the connect
    URL.  The WS handler should close with code 1008 (policy violation) when
    this raises.

    When JWT_VERIFICATION_DISABLED=true returns DEV_USER_ID without checking.

    Raises:
        HTTPException(401) — missing/invalid/expired token.
    """
    if _is_verification_disabled():
        from api.deps import DEV_USER_ID
        return DEV_USER_ID

    if not token:
        raise HTTPException(status_code=401, detail="Missing token query parameter")
    payload = verify_jwt(token)
    sub = payload.get("sub")
    if not sub:
        logger.warning("JWT payload missing sub claim (WS path)")
        raise HTTPException(status_code=401, detail="Unauthorized")
    _enforce_allowlist(payload)
    return sub
