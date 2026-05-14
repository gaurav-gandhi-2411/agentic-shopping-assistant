"""Contextvars shared between the API layer and LLM clients.

Set in route handlers so LLM call logs can include user_id without the clients
depending on the api package.
"""
from contextvars import ContextVar

llm_user_id_var: ContextVar[str] = ContextVar("llm_user_id", default="")
