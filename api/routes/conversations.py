"""Conversation management routes.

GET    /conversations       — list all conversation summaries for the current user
GET    /conversations/{id}  — full conversation: messages + last retrieved items
POST   /conversations       — create a new empty conversation
DELETE /conversations/{id}  — delete a conversation
PATCH  /conversations/{id}  — update conversation metadata (custom title)

All routes require a valid JWT (get_current_user_id).  The ownership boundary is
enforced by the session store: InMemorySessionStore accepts user_id but does not
scope by it (single-process, UUID collision is negligible); PostgresSessionStore
enforces user_id at the DB level.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import api.deps as deps
from api.auth import get_current_user_id
from api.schemas import ItemSummary

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ---------------------------------------------------------------------------
# Response / request models
# ---------------------------------------------------------------------------

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    last_message: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    is_public: bool = False


class ConversationDetail(ConversationSummary):
    messages: list[dict[str, str]] = Field(default_factory=list)
    retrieved_items: list[ItemSummary] = Field(default_factory=list)


class ConversationPatch(BaseModel):
    title: str | None = None
    is_public: bool | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TITLE_MAX = 60
_SNIPPET_MAX = 120


def _derive_title(session: dict) -> str:
    for msg in session.get("messages", []):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            return text[:_TITLE_MAX] + ("…" if len(text) > _TITLE_MAX else "")
    return "New conversation"


def _to_summary(conversation_id: str, session: dict) -> ConversationSummary:
    messages = session.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]

    last_message: str | None = None
    if assistant_messages:
        last = assistant_messages[-1].get("content", "")
        last_message = last[:_SNIPPET_MAX] + ("…" if len(last) > _SNIPPET_MAX else "")

    title = session.get("_title") or _derive_title(session)

    return ConversationSummary(
        conversation_id=conversation_id,
        title=title,
        message_count=len(user_messages),
        last_message=last_message,
        filters=session.get("filters", {}),
        is_public=session.get("_is_public", False),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    user_id: str = Depends(get_current_user_id),
) -> list[ConversationSummary]:
    """Return a summary for each conversation belonging to the current user."""
    store = deps.get_session_store()
    summaries: list[ConversationSummary] = []
    for cid in store.list_ids(user_id):
        session = store.get(cid, user_id)
        if session is None:
            continue
        summaries.append(_to_summary(cid, session))
    # Most-recent first: proxy by message count since we have no timestamps.
    summaries.sort(key=lambda s: s.message_count, reverse=True)
    return summaries


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
) -> ConversationDetail:
    """Return the full message history and last retrieved items for a conversation."""
    store = deps.get_session_store()
    session = store.get(conversation_id, user_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id!r} not found",
        )

    summary = _to_summary(conversation_id, session)

    # Expose only user/assistant messages with non-empty content.
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in session.get("messages", [])
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    retrieved_items = [
        ItemSummary.from_agent_item(it)
        for it in session.get("retrieved_items", [])
    ]

    return ConversationDetail(
        **summary.model_dump(),
        messages=messages,
        retrieved_items=retrieved_items,
    )


@router.post("", response_model=ConversationSummary, status_code=201)
def create_conversation(
    user_id: str = Depends(get_current_user_id),
) -> ConversationSummary:
    """Create a new empty conversation and return its ID.

    Useful when the client wants a stable conversation_id before sending the
    first message over the WebSocket.
    """
    from src.memory.conversation import ConversationMemory

    llm = deps.get_llm()
    config = deps.get_config()
    conversation_id = str(uuid.uuid4())

    session: dict = {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "_memory": ConversationMemory(llm, config),
        "_summary": None,
        "_summary_message_count": 0,
    }
    deps.get_session_store().set(conversation_id, session, user_id)
    return _to_summary(conversation_id, session)


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
) -> None:
    """Permanently delete a conversation."""
    store = deps.get_session_store()
    if store.get(conversation_id, user_id) is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id!r} not found",
        )
    store.delete(conversation_id, user_id)


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def patch_conversation(
    conversation_id: str,
    body: ConversationPatch,
    user_id: str = Depends(get_current_user_id),
) -> ConversationSummary:
    """Update conversation metadata.  Currently supports renaming (title)."""
    store = deps.get_session_store()
    session = store.get(conversation_id, user_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id!r} not found",
        )
    if body.title is not None:
        session["_title"] = body.title
    if body.is_public is not None:
        session["_is_public"] = body.is_public
    if body.title is not None or body.is_public is not None:
        store.set(conversation_id, session, user_id)
    return _to_summary(conversation_id, session)
