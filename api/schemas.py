"""Pydantic request/response models for the Shopping Assistant API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared item model
# ---------------------------------------------------------------------------

class ItemSummary(BaseModel):
    article_id: str
    prod_name: str
    display_name: str
    colour: str
    product_type: str
    department: str
    image_url: str | None = None
    detail_desc: str | None = None
    score: float | None = None

    @classmethod
    def from_agent_item(cls, item: dict) -> "ItemSummary":
        return cls(
            article_id=item.get("article_id", ""),
            prod_name=item.get("prod_name", ""),
            display_name=item.get("display_name", ""),
            colour=item.get("colour", ""),
            product_type=item.get("product_type", ""),
            department=item.get("department", ""),
            image_url=item.get("image_url") or None,
            detail_desc=item.get("detail_desc") or None,
            score=item.get("score"),
        )


# ---------------------------------------------------------------------------
# HTTP /chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None,
        description="Omit to start a new conversation; server will mint a UUID4.",
    )
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    items: list[ItemSummary] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    routing: dict[str, Any] = Field(default_factory=dict)
    out_of_catalogue: bool = False
    new_items_this_turn: bool = False


# ---------------------------------------------------------------------------
# WebSocket /chat/stream  (Task 3 — types defined here, used in routes/chat.py)
# ---------------------------------------------------------------------------

# Client → Server
class WSUserMessage(BaseModel):
    type: Literal["user_message"] = "user_message"
    conversation_id: str | None = None
    message: str = Field(..., min_length=1, max_length=2000)


class WSCancelMessage(BaseModel):
    type: Literal["cancel"] = "cancel"


# Server → Client
class WSSessionMessage(BaseModel):
    type: Literal["session"] = "session"
    conversation_id: str


class WSRoutingMessage(BaseModel):
    type: Literal["routing"] = "routing"
    decision: dict[str, Any]


class WSToolStartMessage(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    tool: str


class WSItemsMessage(BaseModel):
    type: Literal["items"] = "items"
    items: list[ItemSummary]


class WSTokenMessage(BaseModel):
    type: Literal["token"] = "token"
    text: str


class WSDoneMessage(BaseModel):
    type: Literal["done"] = "done"
    final_state: dict[str, Any]


class WSErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
    code: str


class WSCancelledMessage(BaseModel):
    type: Literal["cancelled"] = "cancelled"
