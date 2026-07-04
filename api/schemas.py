"""Pydantic request/response models for the Shopping Assistant API."""
# v: abeadb3
from __future__ import annotations

import math as _math
from typing import Any, Literal

from pydantic import BaseModel, Field


def _ns(val: object, default: str = "") -> str:
    """Null-safe string coercion: returns default for None, float NaN, or the string 'nan'."""
    if val is None:
        return default
    if isinstance(val, float) and _math.isnan(val):
        return default
    s = str(val)
    return default if s.lower() == "nan" else s

# ---------------------------------------------------------------------------
# Cart / buy-link models
# ---------------------------------------------------------------------------


class ItemLink(BaseModel):
    """Per-item buy link included in cart action responses."""

    article_id: str
    name: str
    buy_url: str


# ---------------------------------------------------------------------------
# Shared item model
# ---------------------------------------------------------------------------


class PriceMatch(BaseModel):
    """A cross-store listing of the same product at a (possibly) different price.

    Prices are catalogue SNAPSHOTS — not real-time.  Always display a snapshot
    disclaimer when rendering PriceMatch data to users.
    """

    store: str
    store_display: str
    price_inr: float | None = None
    pdp_url: str | None = None
    confidence: float
    is_snapshot_price: bool = True   # always True — prices are not real-time


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
    price_inr: float | None = None
    pdp_handle: str | None = None
    outfit_slot: str | None = None    # e.g. "bottom", "accessory", "footwear"
    slot_role: str | None = None      # "seed" or "complement"
    # "Owned anchor" feature: True when this item is the user's OWN garment (e.g.
    # the seed resolved from an uploaded photo) rather than a catalogue item for
    # sale. Owned items are still rendered ("Your item") but pdp_url is always
    # None, and build_cart_action excludes them from cart/link/total machinery.
    is_owned: bool = False
    # Cross-store fields (populated in unified mode; None for legacy per-brand responses)
    store: str | None = None          # store slug, e.g. "myntra", "snitch"
    store_display: str | None = None  # human-readable name, e.g. "Myntra", "Snitch"
    pdp_url: str | None = None        # server-built deep-link; use directly in the frontend
    # Phase-D price-match: cross-store same-product listings, lowest price first.
    # None/empty = item not found in other stores (current reality for ~all items).
    # Prices are SNAPSHOT — never real-time.  Frontend must display snapshot disclaimer.
    price_matches: list[PriceMatch] | None = None

    @classmethod
    def from_agent_item(cls, item: dict) -> "ItemSummary":
        from src.config.stores import build_pdp_url, get_store_display_name

        store = item.get("store") or None
        is_owned = bool(item.get("_owned", False))
        return cls(
            article_id=item.get("article_id") or "",
            prod_name=_ns(item.get("prod_name")),
            display_name=_ns(item.get("display_name") or item.get("prod_name")),
            colour=_ns(item.get("colour")),
            product_type=_ns(item.get("product_type")),
            department=_ns(item.get("department")),
            image_url=item.get("image_url") or None,
            detail_desc=item.get("detail_desc") or None,
            score=item.get("score"),
            price_inr=item.get("price_inr"),
            pdp_handle=item.get("pdp_handle") or None,
            outfit_slot=item.get("_slot") or None,
            slot_role=item.get("_role") or None,
            is_owned=is_owned,
            store=store,
            store_display=get_store_display_name(store),
            # Owned items are never for sale — never emit a buy link for them.
            pdp_url=None if is_owned else build_pdp_url(store, item),
        )


# ---------------------------------------------------------------------------
# Outfit variants
# ---------------------------------------------------------------------------

class OutfitVariant(BaseModel):
    """One switchable outfit variant (base, colour story, or formality lean)."""

    variant_id: str
    label: str                    # short chip label: "Base", "Colour story", "Dressier", "Lighter"
    rationale: str
    items: list[ItemSummary]
    occasion: str | None = None
    budget_total_inr: float | None = None
    # Cart action fields — populated for outfit responses; None for non-outfit turns.
    # cart_url is None for cross-store looks (items span multiple stores): the spec
    # requires per-item deep-links in that case — there is no single cross-store cart.
    # item_links carry each item's OWN store PDP URL so the frontend can open them
    # individually ("Buy at {store_display}") without any Myntra fallback for non-Myntra items.
    cart_url: str | None = None
    item_links: list[ItemLink] | None = None


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
    look_id: str | None = None
    occasion: str | None = None
    look_gender: str | None = None
    budget_total_inr: float | None = None
    outfit_rationale: str | None = None
    outfit_variants: list[OutfitVariant] | None = None
    # Cart action fields — populated for outfit responses; None otherwise.
    cart_url: str | None = None
    item_links: list[ItemLink] | None = None
    # Colour refinement chips — available colours from the current result set.
    # Front-end renders these as tappable chip buttons that retrigger search with
    # the chosen colour filter applied to the current context.
    suggestion_chips: list[str] | None = None


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
    message_id: str | None = None  # UUID of the persisted assistant message; None in memory-only mode


class WSErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
    code: str


class WSCancelledMessage(BaseModel):
    type: Literal["cancelled"] = "cancelled"


# ---------------------------------------------------------------------------
# Saved looks / share
# ---------------------------------------------------------------------------


class SaveLookRequest(BaseModel):
    """Body for POST /looks — persist the current outfit board for sharing."""

    session_id: str = Field(..., description="Anonymous demo session identifier.")
    brand: str = Field(..., description="Brand slug, e.g. 'hm' or 'myntra'.")
    look_id: str | None = Field(default=None, description="Ephemeral look id from the outfit engine.")
    occasion: str | None = Field(default=None, description="Occasion label, e.g. 'casual'.")
    look_gender: str | None = Field(default=None, description="Gender label: 'men', 'women', 'unisex'.")
    anchor_item_id: str | None = Field(default=None, description="Anchor catalogue item id.")
    look_total_inr: int | None = Field(default=None, description="Basket total in INR.")
    snapshot: dict = Field(
        ...,
        description=(
            "Self-contained board payload: items[] with article_id/name/colour/type/"
            "slot/role/image_url/price_inr/pdp_handle/buy_url, plus rationale, "
            "cart_url, item_links, variant label."
        ),
    )
    # user_id is always None for anonymous sessions; reserved for future auth wiring.
    user_id: str | None = Field(default=None, description="Authenticated user UUID, or None for anonymous.")


class SaveLookResponse(BaseModel):
    """Response for POST /looks."""

    id: str = Field(..., description="UUID of the saved look (also the share slug).")
    share_path: str = Field(..., description="Relative path for the read-only shared board, e.g. /look/{id}.")


class SharedLookResponse(BaseModel):
    """Response for GET /looks/{look_id} — public read-only shared board payload."""

    id: str
    brand: str
    occasion: str | None = None
    look_gender: str | None = None
    look_total_inr: int | None = None
    snapshot: dict
    created_at: str
