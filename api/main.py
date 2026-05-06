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

import pandas as pd
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import api.deps as deps
from api.logging_config import setup_logging
from api.routes.catalogue import router as catalogue_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "processed"
_CONFIG_PATH = str(_REPO_ROOT / "config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    setup_logging()
    logger.info("Starting up Shopping Assistant API")

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

    deps._init(retriever, df, llm, config)
    logger.info("Startup complete")

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
        "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
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
    app.include_router(chat_router)
    app.include_router(catalogue_router)

    return app


app = create_app()
