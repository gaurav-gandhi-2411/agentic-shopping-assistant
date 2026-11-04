"""LLM resilience helpers shared by the client and the API layer."""
from __future__ import annotations

# Shown to the user when every tier of the LLM chain failed or returned nothing.
# Plain ASCII on purpose: it is streamed as a normal assistant token.
LLM_BUSY_MESSAGE = (
    "I'm getting a lot of requests right now and couldn't put together a reply. "
    "Please try again in a minute."
)
