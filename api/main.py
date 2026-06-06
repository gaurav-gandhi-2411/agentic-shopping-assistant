"""FastAPI application factory.

Startup sequence (lifespan):
  1. Configure structured JSON logging.
  2. Load config.yaml.
  3. Load FAISS + BM25 retrievers and catalogue DataFrame (~5-10s on first run).
  4. Initialise the LLM client (Groq in production).
  5. Enable LangSmith tracing if LANGSMITH_API_KEY is set.
  6. Store all singletons in api.deps so route handlers can access them.

The Streamlit process and the API process share no state — each runs independently.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import sentry_sdk
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import api.deps as deps
from api.logging_config import setup_logging
from api.routes.auth import router as auth_router
from api.routes.brand import router as brand_router
from api.routes.catalogue import router as catalogue_router
from api.routes.chat import router as chat_router
from api.routes.conversations import router as conversations_router
from api.routes.feedback import router as feedback_router
from api.routes.health import router as health_router

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent


def _build_session_store(llm: Any, config: dict):
    """Return (session_store, engine|None).

    engine is non-None when DATABASE_URL is set; callers use it to wire the
    allow-list checker and for migrations.
    """
    from api.session import InMemorySessionStore

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return InMemorySessionStore(), None

    for prefix in ("postgresql://", "postgres://"):
        if db_url.startswith(prefix):
            db_url = "postgresql+psycopg://" + db_url[len(prefix):]
            break

    from sqlalchemy import create_engine

    from src.storage.postgres_session_store import PostgresSessionStore

    engine = create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=2)
    return PostgresSessionStore(engine, llm, config), engine
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_CONFIG_PATH = str(_REPO_ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# Sentry scrubbing helpers
# ---------------------------------------------------------------------------

def _scrub_token_from_url(url: str) -> str:
    """Replace the value of a `token` query param with [Filtered], leaving all
    other params untouched."""
    parsed = urlparse(url)
    if not parsed.query:
        return url
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(k == "token" for k, _ in pairs):
        return url
    pairs = [(k, "[Filtered]" if k == "token" else v) for k, v in pairs]
    return urlunparse(parsed._replace(query=urlencode(pairs)))


def _sentry_before_breadcrumb(crumb: dict, hint: object) -> dict:
    """Scrub token= from URLs in http and navigation breadcrumb data."""
    data = crumb.get("data") or {}
    for key in ("url", "from", "to"):
        if isinstance(data.get(key), str):
            data[key] = _scrub_token_from_url(data[key])
    return crumb


def _sentry_before_send(event: dict, hint: object) -> dict:
    """Scrub token= from the captured request URL and query string."""
    req = event.get("request") or {}
    if isinstance(req.get("url"), str):
        req["url"] = _scrub_token_from_url(req["url"])
    qs = req.get("query_string")
    if isinstance(qs, str):
        req["query_string"] = _scrub_token_from_url(f"http://x?{qs}")[len("http://x?"):]
    elif isinstance(qs, list):
        req["query_string"] = [
            [k, "[Filtered]" if k == "token" else v] for k, v in qs
        ]
    return event


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    setup_logging()
    logger.info("Starting up Shopping Assistant API")

    # Sentry — initialise early so startup errors are captured.
    sentry_dsn = os.environ.get("SENTRY_DSN", "")
    sentry_sdk.init(
        dsn=sentry_dsn or None,
        traces_sample_rate=0.1,
        send_default_pii=False,
        # Never capture request bodies — chat messages must not appear in error reports.
        max_request_body_size="never",
        before_breadcrumb=_sentry_before_breadcrumb,
        before_send=_sentry_before_send,
    )
    if sentry_dsn:
        logger.info("Sentry initialised (traces_sample_rate=0.1)")
    else:
        logger.info("Sentry not configured (SENTRY_DSN unset)")

    from src.catalogue.loader import load_config
    from src.llm.client import get_llm_client
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    config = load_config(_CONFIG_PATH)
    # Allow env var to override LLM provider (e.g. groq on Fly, ollama locally).
    if os.environ.get("LLM_PROVIDER"):
        config["llm"]["provider"] = os.environ["LLM_PROVIDER"]

    logger.info("Loading retrieval indices from %s", _DATA_DIR)
    df = pd.read_parquet(_DATA_DIR / "catalogue.parquet")
    dense = DenseRetriever.load(config, _DATA_DIR)
    sparse = SparseRetriever.load(config, _DATA_DIR)
    retriever = HybridRetriever(dense, sparse, df, config)
    n_vectors = dense.index.ntotal if dense.index is not None else 0
    logger.info("Retrieval ready: %d items, %d dense vectors", len(df), n_vectors)

    logger.info("Initialising LLM client (provider=%s)", config["llm"]["provider"])
    llm = get_llm_client(config)
    logger.info("LLM client ready")

    # LangSmith tracing — enabled when LANGSMITH_API_KEY is present.
    langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")
    if langsmith_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault(
            "LANGSMITH_PROJECT",
            os.environ.get("LANGSMITH_PROJECT", "agentic-shopping-assistant"),
        )
    tracing = bool(langsmith_key and os.environ.get("LANGCHAIN_TRACING_V2") == "true")
    logger.info("LangSmith tracing %s", "enabled" if tracing else "disabled")

    session_store, db_engine = _build_session_store(llm, config)
    deps._init(retriever, df, llm, config, session_store=session_store)

    # Wire defense-in-depth allow-list checker when a DB is available.
    if db_engine is not None:
        from sqlalchemy import text

        import api.auth as _auth

        def _db_check_allowlist(email: str) -> bool:
            with db_engine.connect() as conn:
                row = conn.execute(
                    text("SELECT public.check_email_allowed(:e)"),
                    {"e": email.lower()},
                ).scalar()
            return bool(row)

        _auth._check_allowlist = _db_check_allowlist
        logger.info("Allow-list DB checker wired (check_email_allowed)")

    # Auth status — warn loudly if verification is disabled.
    from api.auth import _is_verification_disabled
    if _is_verification_disabled():
        logger.warning(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        logger.warning(
            "JWT_VERIFICATION_DISABLED=true — ALL REQUESTS ARE UNAUTHENTICATED. "
            "Every call is treated as DEV_USER_ID. "
            "NEVER run with this setting in production."
        )
        logger.warning(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

    logger.info(
        "Startup complete (session_store=%s, auth_enabled=%s)",
        type(session_store).__name__,
        not _is_verification_disabled(),
    )

    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Shopping Assistant API",
        version="1.0.0",
        lifespan=lifespan,
    )

    allowed_origins = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next: Any) -> Response:
        t0 = time.monotonic()
        response = await call_next(request)
        latency_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "%s %s %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={"latency_ms": latency_ms},
        )
        return response

    app.include_router(health_router)
    app.include_router(brand_router)
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.include_router(catalogue_router)
    app.include_router(feedback_router)

    @app.get("/sentry-debug")
    def sentry_debug():
        """Raise a test exception to verify Sentry ingestion.

        Only active when SENTRY_DEBUG_ENDPOINT=true.  Never expose in production
        without that flag — the endpoint deliberately triggers an unhandled error.
        """
        if os.environ.get("SENTRY_DEBUG_ENDPOINT", "").lower() not in ("1", "true"):
            raise HTTPException(status_code=404, detail="Not found")
        raise ZeroDivisionError("Sentry debug endpoint triggered intentionally")

    # Serve product images if the directory was baked into the container.
    images_dir = _DATA_DIR / "images"
    if images_dir.is_dir():
        app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    return app


app = create_app()
