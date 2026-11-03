"""Minimal ASGI app for tests/test_db_leak_subprocess.py.

Wires the REAL logging setup, DB-engine construction (api.main._build_session_store),
/readyz router and feedback engine, without the heavy retrieval startup, so a
subprocess uvicorn can exercise the DB-failure paths hermetically.
"""
from __future__ import annotations

from fastapi import FastAPI

import api.deps as deps
import api.routes.feedback as feedback
from api.logging_config import setup_logging
from api.main import _build_session_store
from api.routes.health import router as health_router

setup_logging()
app = FastAPI()
app.include_router(health_router)
_, deps._db_engine = _build_session_store(None, {})


@app.get("/boom")
def boom() -> None:
    """Uncaught DB connect failure -> uvicorn logs 'Exception in ASGI application'."""
    with feedback._get_engine().connect():
        pass
