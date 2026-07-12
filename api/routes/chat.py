"""Chat routes: POST /chat (non-streaming) and WS /chat/stream (Task 3)."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import threading
import uuid
from typing import Any, AsyncIterator

import sentry_sdk
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

import api.deps as deps
from api.auth import get_current_user_id_or_demo, get_current_user_id_ws
from api.logging_config import conversation_id_var
from api.rate_limit import get_rate_limiter
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ItemLink,
    ItemSummary,
    OutfitVariant,
    PartnerLook,
    WSCancelledMessage,
    WSDoneMessage,
    WSErrorMessage,
    WSItemsMessage,
    WSRoutingMessage,
    WSSessionMessage,
    WSTokenMessage,
    WSToolStartMessage,
    WSUserMessage,
)
from api.session import SessionStore
from src.agents.outfit.cart_links import build_cart_action
from src.llm.client import STREAM_ERROR_SENTINEL
from src.llm.context import llm_user_id_var
from src.retrieval.index_store import UNIFIED_BRAND

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _resolve_brand() -> str:
    """Resolve the active brand slug using the same logic as api/main.py.

    In unified (cross-store) mode — BRAND unset, BRAND=unified, or UNIFIED=1 —
    returns UNIFIED_BRAND so cart-link helpers receive the correct brand context.
    In legacy per-brand mode returns the BRAND env var value.
    """
    _unified_flag = os.environ.get("UNIFIED", "").lower() in ("1", "true", "yes")
    _brand_env = os.environ.get("BRAND", "")
    if _unified_flag or _brand_env == UNIFIED_BRAND or not _brand_env:
        return UNIFIED_BRAND
    return _brand_env

# In-memory session store for anonymous demo users.
# Keyed on conversation_id; accumulates until container restart (fine for ephemeral demos).
# Avoids passing anon:uuid user IDs to PostgresSessionStore which expects real UUIDs.
_DEMO_SESSIONS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _fresh_session(llm: Any, config: dict) -> dict:
    """Initialise an empty session dict including a ConversationMemory instance."""
    from src.memory.conversation import ConversationMemory

    return {
        "messages": [],
        "retrieved_items": [],
        "filters": {},
        "excluded_colours": None,
        "_memory": ConversationMemory(llm, config),
        "_summary": None,
        "_summary_message_count": 0,
    }


def _build_invoke_state(session: dict, user_message: str) -> dict:
    """Construct an AgentState-compatible dict for agent.invoke()."""
    return {
        "messages": session["messages"] + [{"role": "user", "content": user_message}],
        "user_query": user_message,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": session["retrieved_items"],
        "filters": session["filters"],
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": session.get("excluded_colours"),
        "anchor_article_id": session.get("anchor_article_id"),
        # "Owned anchor" feature: carries forward whether anchor_article_id refers
        # to a user-owned item (set by image_style.py). Consulted by outfit_node.
        "anchor_is_owned": session.get("anchor_is_owned", False),
        # Outfit fields — reset each turn; populated by outfit_node only
        "outfit_rationale": None,
        "outfit_variants": None,
        # ConversationMemory for this conversation — accessed by LLMRouterBackend
        # via state["_memory"] so the compiled graph singleton is memory-agnostic.
        "_memory": session["_memory"],
    }


def _persist_result(session: dict, result: dict) -> None:
    """Write the agent result back into the session dict (mutates in place)."""
    # messages accumulate via operator.add in the LangGraph reducer;
    # result["messages"] is the full accumulated list after the turn.
    session["messages"] = result.get("messages", session["messages"])
    session["retrieved_items"] = result.get("retrieved_items", session["retrieved_items"])
    session["filters"] = result.get("filters", session["filters"])
    if result.get("excluded_colours") is not None:
        session["excluded_colours"] = result["excluded_colours"]
    # Sync summary state: only present in result when get_context (re)computed a summary.
    if "_summary" in result:
        session["_summary"] = result["_summary"]
        session["_summary_message_count"] = result.get("_summary_message_count", 0)


def _build_outfit_variants(result: dict) -> list[OutfitVariant] | None:
    """Convert raw outfit_variants list from agent state into OutfitVariant schema objects.

    Returns None when no outfit_variants are present in the result (non-outfit turns).
    Each variant look dict is expected to carry: look_id, variant_label, rationale,
    seed_item, complements, occasion, budget_total_inr.

    Cart fields (cart_url, item_links) are populated per-variant using
    build_cart_action with the active BRAND env var.
    """
    raw_variants: list[dict] | None = result.get("outfit_variants")
    if not raw_variants:
        return None

    brand = _resolve_brand()

    out: list[OutfitVariant] = []
    for variant in raw_variants:
        seed = variant.get("seed_item")
        complements = variant.get("complements") or []
        all_items = ([seed] if seed else []) + complements
        try:
            item_summaries = [ItemSummary.from_agent_item(it) for it in all_items]

            # Build cart action for this variant's item set
            cart_action = build_cart_action(all_items, brand)
            cart_url = cart_action.get("cart_url")
            raw_links = cart_action.get("item_links") or []
            item_links = [
                ItemLink(
                    article_id=lk["article_id"],
                    name=lk["name"],
                    buy_url=lk["buy_url"],
                )
                for lk in raw_links
            ] or None

            ov = OutfitVariant(
                variant_id=variant.get("look_id") or "",
                label=variant.get("variant_label") or "Base",
                rationale=variant.get("rationale") or variant.get("outfit_rationale") or "",
                items=item_summaries,
                occasion=variant.get("occasion"),
                budget_total_inr=variant.get("budget_total_inr"),
                cart_url=cart_url,
                item_links=item_links,
                suppressed_slots=variant.get("suppressed_slots") or None,
            )
            out.append(ov)
        except Exception as _e:
            logger.warning("outfit_variants: skipping malformed variant: %s", _e)

    return out or None


def _build_base_cart_action(result: dict, brand: str) -> tuple[str | None, list[ItemLink] | None]:
    """Build cart action for the base outfit look (non-variant path).

    Extracts seed + complements from the top-level result dict (populated by the
    outfit node for single-look responses). Returns (cart_url, item_links) tuple;
    both None when no outfit items are present.
    """
    seed = result.get("seed_item")
    complements = result.get("complements") or []
    all_items = ([seed] if seed else []) + complements
    if not all_items:
        # Fall back to retrieved_items when no seed/complements (non-outfit turns)
        return None, None

    cart_action = build_cart_action(all_items, brand)
    cart_url = cart_action.get("cart_url")
    raw_links = cart_action.get("item_links") or []
    item_links = [
        ItemLink(article_id=lk["article_id"], name=lk["name"], buy_url=lk["buy_url"])
        for lk in raw_links
    ] or None
    return cart_url, item_links


def _normalize_look_role(role: str | None) -> str | None:
    """Map an internal AgentState look_role value to the external API contract.

    The external contract (ChatResponse/WSDoneMessage.look_role, OutfitBoard's
    lookRole prop) is exactly "partner" | None — see api/schemas.py and
    frontend/components/chat/OutfitBoard.tsx's `isPartnerLook` gate. The
    pre-existing single-partner-turn flow already sets look_role="partner"
    directly, so it passes through unchanged. The newer P2 couple-from-scratch
    flow (src/agents/graph.py's _compose_couple_from_scratch) sets the PRIMARY
    board's own look_role to "couple_primary" — internal bookkeeping to
    distinguish it from an ordinary primary turn — but from the frontend's
    perspective it IS an ordinary (non-partner) board, so it's normalized to
    None here (no "Partner look" badge on the primary board of a couple turn).
    """
    return "partner" if role == "partner" else None


def _partner_look_from_result(result: dict, brand: str) -> PartnerLook | None:
    """Build the SECOND ("couple-from-scratch") board from a turn's agent result.

    Reads the `partner_*`-prefixed AgentState fields (parallel to the existing
    singleton `look_id`/`look_role`/... fields — see src/agents/state.py) that
    outfit_node sets ONLY when it composed both a primary AND a partner look in
    the SAME turn. Returns None when no such second look was composed this
    turn — in particular the PRE-EXISTING single-partner-turn flow (e.g. "what
    should my husband wear with this?") never populates `partner_retrieved_items`
    and so always returns None here, leaving that flow's singleton
    look_role/look_title/coordinated_with fields as the only look data (unchanged
    behaviour).

    look_role on the returned PartnerLook is always the literal string
    "partner" (regardless of the internal value — currently "couple_partner",
    see src/agents/graph.py) so the frontend's OutfitBoard renders its
    "Partner look" badge + look_title + coordinated_with text exactly like the
    pre-existing single-partner-turn board does — the presence of a PartnerLook
    payload at all already signals "this is the companion board".

    "always visual" hard rule applies here too: items without an image_url are
    dropped before serialising.
    """
    raw_items = [it for it in (result.get("partner_retrieved_items") or []) if it.get("image_url")]
    if not raw_items:
        return None

    item_summaries = [ItemSummary.from_agent_item(it) for it in raw_items]

    cart_url, item_links = _build_base_cart_action(
        {
            "seed_item": result.get("partner_seed_item"),
            "complements": result.get("partner_complements"),
        },
        brand,
    )

    return PartnerLook(
        items=item_summaries,
        look_id=result.get("partner_look_id"),
        occasion=result.get("partner_occasion") or result.get("occasion"),
        look_gender=result.get("partner_look_gender"),
        budget_total_inr=result.get("partner_budget_total_inr"),
        outfit_rationale=result.get("partner_outfit_rationale"),
        cart_url=cart_url,
        item_links=item_links,
        suppressed_slots=result.get("partner_suppressed_slots") or None,
        look_role="partner",
        look_title=result.get("partner_look_title"),
        coordinated_with=result.get("partner_coordinated_with"),
    )


def _extract_routing(tool_calls: list[dict]) -> dict:
    for tc in tool_calls:
        if "router_decision" in tc:
            return tc["router_decision"]
    return {}


def _items_from_result(result: dict) -> list[ItemSummary]:
    if not result.get("new_items_this_turn"):
        return []
    # "always visual" hard rule: drop items without an image before serialising
    raw = [it for it in result.get("retrieved_items", []) if it.get("image_url")]
    return [ItemSummary.from_agent_item(it) for it in raw]


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def post_chat(
    request: Request,
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id_or_demo),
) -> ChatResponse:
    """Non-streaming chat endpoint.  Full agent round-trip; returns when done."""
    # Rate limit: anonymous demo sessions use Postgres-backed per-IP + daily guards;
    # authenticated users use the existing in-memory sliding-window limiter.
    if user_id.startswith("anon:"):
        client_ip = request.client.host if request.client else "0.0.0.0"
        _brand = _resolve_brand()
        _engine = deps.get_db_engine()
        if _engine is not None:
            from api.demo.guards import (
                check_daily_cap,
                check_daily_cost,
                check_ip_rate_limit,
                record_request,
            )
            ip_allowed, ip_retry = check_ip_rate_limit(client_ip, _brand, _engine)
            if not ip_allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(ip_retry)},
                )
            if not check_daily_cap(_brand, _engine) or not check_daily_cost(_brand):
                raise HTTPException(
                    status_code=429,
                    detail="Demo limit reached for today — try again tomorrow.",
                )
            record_request(_brand, _engine)
    else:
        limiter = get_rate_limiter()
        allowed, retry_after = limiter.is_allowed(user_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

    conversation_id = body.conversation_id or str(uuid.uuid4())
    token = conversation_id_var.set(conversation_id)
    llm_user_id_var.set(user_id)

    # Set Sentry scope — IDs only, no message content.
    sentry_sdk.set_user({"id": user_id})
    sentry_sdk.set_tag("conversation_id", conversation_id)

    store: SessionStore = deps.get_session_store()
    llm = deps.get_llm()
    config = deps.get_config()

    try:
        _is_anon = user_id.startswith("anon:")
        if _is_anon:
            session = _DEMO_SESSIONS.get(conversation_id) or _fresh_session(llm, config)
        else:
            session = store.get(conversation_id, user_id) or _fresh_session(llm, config)

        memory = session["_memory"]
        factory = deps.get_agent_factory()
        agent = factory(memory, streaming=False)

        state = _build_invoke_state(session, body.message)

        try:
            result = agent.invoke(state)
        except Exception as exc:
            logger.error("agent.invoke failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error") from exc

        _persist_result(session, result)
        if _is_anon:
            _DEMO_SESSIONS[conversation_id] = session
        else:
            store.set(conversation_id, session, user_id)

        tool_calls: list[dict] = result.get("tool_calls", [])
        routing = _extract_routing(tool_calls)
        items = _items_from_result(result)
        response_text = result.get("final_answer") or ""
        _brand = _resolve_brand()
        _outfit_variants = _build_outfit_variants(result)

        # Populate base-look cart action (covers single-look responses that don't
        # go through the outfit_variants path, and gives a top-level cart_url).
        _base_cart_url, _base_item_links = _build_base_cart_action(result, _brand)

        logger.info(
            "chat turn complete",
            extra={
                "action": routing.get("action", ""),
                "n_items": len(items),
            },
        )

        _chips = result.get("suggestion_chips") or None
        _partner_look = _partner_look_from_result(result, _brand)

        return ChatResponse(
            conversation_id=conversation_id,
            response=response_text,
            items=items,
            filters=result.get("filters", {}),
            tool_calls=tool_calls,
            routing=routing,
            out_of_catalogue=bool(result.get("out_of_catalogue")),
            new_items_this_turn=bool(result.get("new_items_this_turn")),
            look_id=result.get("look_id"),
            occasion=result.get("occasion"),
            look_gender=result.get("look_gender"),
            budget_total_inr=result.get("budget_total_inr"),
            outfit_rationale=result.get("outfit_rationale"),
            outfit_variants=_outfit_variants,
            cart_url=_base_cart_url,
            item_links=_base_item_links,
            suggestion_chips=_chips,
            suppressed_slots=result.get("suppressed_slots") or None,
            look_role=_normalize_look_role(result.get("look_role")),
            look_title=result.get("look_title"),
            coordinated_with=result.get("coordinated_with"),
            partner_look=_partner_look,
        )

    finally:
        conversation_id_var.reset(token)


# ---------------------------------------------------------------------------
# Token streaming helper
# ---------------------------------------------------------------------------

async def _iter_tokens(llm: Any, prompt: str) -> AsyncIterator[str]:
    """Bridge llm.generate_stream() (blocking sync iterator) to async.

    Runs the generator in a daemon thread and funnels tokens via an asyncio
    Queue, so the WebSocket coroutine can await each token without blocking
    the event loop.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _produce() -> None:
        try:
            for tok in llm.generate_stream(prompt):
                loop.call_soon_threadsafe(queue.put_nowait, tok)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    ctx = contextvars.copy_context()
    t = threading.Thread(target=ctx.run, args=(_produce,), daemon=True)
    t.start()
    try:
        while True:
            tok = await queue.get()
            if tok is None:
                break
            yield tok
    finally:
        t.join(timeout=10)


# ---------------------------------------------------------------------------
# WS /chat/stream
# ---------------------------------------------------------------------------

@router.websocket("/chat/stream")
async def ws_chat(websocket: WebSocket) -> None:
    """Streaming chat over WebSocket.

    Client → Server frames:
      WSUserMessage   — starts a turn (sent first)
      WSCancelMessage — cancels the in-flight turn at any point

    Server → Client frames (in order):
      WSSessionMessage   — echoes conversation_id immediately after user_message
      WSRoutingMessage   — router decision
      WSToolStartMessage — one per tool call that ran
      WSItemsMessage     — retrieved items (only when new_items_this_turn)
      WSTokenMessage     — one per LLM output token (or full canned answer chunk)
      WSDoneMessage      — final state; conversation persisted
      WSCancelledMessage — turn was cancelled before completion
      WSErrorMessage     — unrecoverable error

    Streaming strategy: all routing+tool nodes run synchronously in a
    background thread (streaming_mode=True so respond_node is bypassed).
    After the thread completes we emit routing/tool/items events from the
    result, then stream the final LLM response tokens via llm.generate_stream
    bridged through an asyncio.Queue (_iter_tokens).  This avoids the quirks
    of astream_events with custom LangGraph node functions.
    """
    await websocket.accept()

    # Authenticate: preferred path uses ?ticket=<nonce> (minted by POST /auth/ws-ticket).
    # Legacy ?token=<jwt> is still accepted for backward compatibility (Streamlit Spaces).
    # Close immediately with 1008 (policy violation) on any auth failure.
    try:
        user_id: str = get_current_user_id_ws(websocket)
    except HTTPException:
        await websocket.close(code=1008, reason="Policy violation: invalid or missing token")
        return

    # Rate limit: anonymous demo sessions use Postgres-backed per-IP + daily guards;
    # authenticated users use the existing in-memory sliding-window limiter.
    if user_id.startswith("anon:"):
        client_ip = websocket.client.host if websocket.client else "0.0.0.0"
        _brand = _resolve_brand()
        _engine = deps.get_db_engine()
        if _engine is not None:
            from api.demo.guards import (
                check_daily_cap,
                check_daily_cost,
                check_ip_rate_limit,
                record_request,
            )
            ip_allowed, ip_retry = check_ip_rate_limit(client_ip, _brand, _engine)
            if not ip_allowed:
                await websocket.send_text(
                    WSErrorMessage(
                        message=f"Rate limit exceeded. Retry in {ip_retry}s",
                        code="rate_limited",
                    ).model_dump_json()
                )
                await websocket.close(code=1008, reason="Rate limit exceeded")
                return
            if not check_daily_cap(_brand, _engine) or not check_daily_cost(_brand):
                await websocket.send_text(
                    WSErrorMessage(
                        message="Demo limit reached for today — try again tomorrow.",
                        code="demo_limit",
                    ).model_dump_json()
                )
                await websocket.close(code=1008, reason="Demo limit reached")
                return
            record_request(_brand, _engine)
    else:
        limiter = get_rate_limiter()
        allowed, retry_after = limiter.is_allowed(user_id)
        if not allowed:
            await websocket.send_text(
                WSErrorMessage(
                    message=f"Rate limit exceeded. Retry in {retry_after}s",
                    code="rate_limited",
                ).model_dump_json()
            )
            await websocket.close(code=1008, reason="Rate limit exceeded")
            return

    llm_user_id_var.set(user_id)
    sentry_sdk.set_user({"id": user_id})

    cid_token = None
    try:
        # ------------------------------------------------------------------
        # 1. Receive first frame — must be WSUserMessage
        # ------------------------------------------------------------------
        raw = await websocket.receive_text()
        try:
            user_msg = WSUserMessage.model_validate_json(raw)
        except Exception as exc:
            await websocket.send_text(
                WSErrorMessage(message=f"Invalid message: {exc}", code="bad_request").model_dump_json()
            )
            return

        # ------------------------------------------------------------------
        # 2. Establish conversation_id; set contextvar immediately so all
        #    downstream logging for this turn carries the conversation_id.
        # ------------------------------------------------------------------
        conversation_id = user_msg.conversation_id or str(uuid.uuid4())
        cid_token = conversation_id_var.set(conversation_id)
        sentry_sdk.set_tag("conversation_id", conversation_id)

        # ------------------------------------------------------------------
        # 3. Acknowledge session
        # ------------------------------------------------------------------
        await websocket.send_text(
            WSSessionMessage(conversation_id=conversation_id).model_dump_json()
        )

        # ------------------------------------------------------------------
        # 4. Load / create session and build agent
        # ------------------------------------------------------------------
        store: SessionStore = deps.get_session_store()
        llm = deps.get_llm()
        config = deps.get_config()
        _is_anon_ws = user_id.startswith("anon:")
        if _is_anon_ws:
            session = _DEMO_SESSIONS.get(conversation_id) or _fresh_session(llm, config)
        else:
            session = store.get(conversation_id, user_id) or _fresh_session(llm, config)

        factory = deps.get_agent_factory()
        agent = factory(session["_memory"], streaming=True)
        state = _build_invoke_state(session, user_msg.message)

        # ------------------------------------------------------------------
        # 5. Run agent in thread; concurrently watch for WSCancelMessage
        # ------------------------------------------------------------------
        cancel_event = asyncio.Event()

        async def _watch_cancel() -> None:
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        frame = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if frame.get("type") == "cancel":
                        cancel_event.set()
                        return
            except (WebSocketDisconnect, Exception):
                cancel_event.set()

        agent_task = asyncio.create_task(asyncio.to_thread(agent.invoke, state))
        cancel_task = asyncio.create_task(_watch_cancel())

        # Wave 7 hang fix (turn 2 of the body-type-then-occasion sequence): bound
        # total agent-turn wall-clock time so a downstream dependency stall (an
        # LLM provider's rate-limit retry loop looping for minutes with zero
        # feedback — see GroqClient.chat's TPD branch in src/llm/client.py,
        # which has no ceiling on its own total retry time — or any other slow
        # dependency) can never leave the WS connection open indefinitely with
        # no response and no error, which is exactly the live symptom ("Stop"
        # forever, 180s+, nothing surfaced). Direct agent.invoke() reproduction
        # of this exact 2-turn sequence with a mocked LLM completes in under 5s
        # with a correct result (see tests/test_body_type_bare_statement.py),
        # ruling out a graph-logic hang — the deadline below is the actual fix,
        # not a longer proof timeout, because it converts an unbounded silent
        # stall into a bounded, honest error response for the live user.
        # 90s mirrors the app's existing most-generous legitimate turn-latency
        # convention (scripts/browser_proof.py's IMAGE_WAIT_TIMEOUT_S=90).
        # agent_task itself cannot be forcibly killed — it wraps a blocking
        # sync call in a worker thread — so on timeout we only stop WAITING
        # for it; the thread runs to completion in the background and its
        # result is discarded.
        _turn_deadline_s = float(os.environ.get("WS_TURN_DEADLINE_SECONDS", "90"))
        done, _ = await asyncio.wait(
            {agent_task, cancel_task},
            timeout=_turn_deadline_s,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_event.is_set():
            agent_task.cancel()
            await websocket.send_text(WSCancelledMessage().model_dump_json())
            return

        if not done:
            # Deadline elapsed before the agent (or a cancel) finished.
            cancel_task.cancel()
            agent_task.cancel()
            logger.error(
                "[ws_chat] turn exceeded %.0fs deadline with no response — "
                "surfacing timeout error (agent thread abandoned, not killed)",
                _turn_deadline_s,
            )
            await websocket.send_text(
                WSErrorMessage(
                    message="This is taking longer than expected — please try again.",
                    code="turn_timeout",
                ).model_dump_json()
            )
            return

        # Agent finished first — stop the cancel watcher.
        cancel_task.cancel()
        result: dict = agent_task.result()

        # ------------------------------------------------------------------
        # 6. Emit intermediate events from the completed result
        # ------------------------------------------------------------------
        tool_calls: list[dict] = result.get("tool_calls", [])

        routing = _extract_routing(tool_calls)
        if routing:
            await websocket.send_text(
                WSRoutingMessage(decision=routing).model_dump_json()
            )

        for tc in tool_calls:
            tool_name = next((k for k in tc if k != "router_decision"), None)
            if tool_name:
                await websocket.send_text(
                    WSToolStartMessage(tool=tool_name).model_dump_json()
                )

        if result.get("new_items_this_turn") and result.get("retrieved_items"):
            # Full ItemSummary inline (not just article_ids) so the frontend can render
            # product cards without N+1 catalogue fetches per item.
            # "always visual" hard rule: drop items without an image before serialising.
            _ws_raw = [it for it in result["retrieved_items"] if it.get("image_url")]
            items = [ItemSummary.from_agent_item(it) for it in _ws_raw]
            await websocket.send_text(WSItemsMessage(items=items).model_dump_json())

        # ------------------------------------------------------------------
        # 7. Stream response tokens
        #    pending_respond → call llm.generate_stream and stream real tokens
        #    pending_answer  → canned text (OOC / outfit / zero-stock)
        #    final_answer    → already set (streaming_mode=False fallback)
        # ------------------------------------------------------------------
        plan: dict = {}
        try:
            plan = json.loads(result.get("current_plan") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        full_response = ""

        # Bug fix (WS session-history overwrite): the streaming-mode graph nodes
        # (respond_node/outfit_node/clarify_node) return "messages": [] when they
        # hand off a pending_respond/pending_answer plan — the assistant reply text
        # isn't known until it's streamed here. `result["messages"]` at this point is
        # therefore still the FULL accumulated history (prior session messages + this
        # turn's user message), NOT a fresh single-element list. Capture it before the
        # branches below and APPEND the assistant reply to it — previously this was
        # overwritten with a single-element list, truncating multi-turn history on
        # every streamed turn (the primary frontend path).
        _accumulated_messages: list[dict] = result.get("messages", session["messages"])

        if plan.get("action") == "pending_respond":
            chunks: list[str] = []
            async for tok in _iter_tokens(llm, plan["prompt"]):
                if tok == STREAM_ERROR_SENTINEL:
                    await websocket.send_text(
                        WSErrorMessage(message="Stream generation failed", code="stream_error").model_dump_json()
                    )
                    return
                if cancel_event.is_set():
                    await websocket.send_text(WSCancelledMessage().model_dump_json())
                    return
                await websocket.send_text(WSTokenMessage(text=tok).model_dump_json())
                chunks.append(tok)
            full_response = "".join(chunks)
            result = {
                **result,
                "final_answer": full_response,
                "messages": _accumulated_messages + [{"role": "assistant", "content": full_response}],
            }

        elif plan.get("action") == "pending_answer":
            full_response = plan.get("text", "")
            await websocket.send_text(WSTokenMessage(text=full_response).model_dump_json())
            result = {
                **result,
                "final_answer": full_response,
                "messages": _accumulated_messages + [{"role": "assistant", "content": full_response}],
            }

        else:
            # final_answer already set (clarify node or non-streaming fallback)
            full_response = result.get("final_answer") or ""
            if full_response:
                await websocket.send_text(WSTokenMessage(text=full_response).model_dump_json())

        # ------------------------------------------------------------------
        # 8. Persist session and send done
        # ------------------------------------------------------------------
        _persist_result(session, result)
        if _is_anon_ws:
            _DEMO_SESSIONS[conversation_id] = session
        else:
            store.set(conversation_id, session, user_id)

        # Fetch the persisted assistant message UUID so the frontend can
        # submit feedback.  Only available when PostgresSessionStore is in use;
        # returns None for InMemorySessionStore (no-op feedback buttons).
        last_message_id: str | None = None
        if not _is_anon_ws and hasattr(store, "get_last_assistant_message_id"):
            try:
                last_message_id = store.get_last_assistant_message_id(conversation_id)
            except Exception as _mid_exc:
                logger.warning("could not fetch last message id: %s", _mid_exc)

        logger.info(
            "ws turn complete",
            extra={
                "action": routing.get("action", ""),
                "n_items": len(result.get("retrieved_items", [])),
            },
        )

        _ws_brand = _resolve_brand()
        _ws_variants = _build_outfit_variants(result)
        _ws_base_cart_url, _ws_base_item_links = _build_base_cart_action(result, _ws_brand)
        _ws_partner_look = _partner_look_from_result(result, _ws_brand)
        await websocket.send_text(
            WSDoneMessage(
                final_state={
                    "filters": result.get("filters", {}),
                    "out_of_catalogue": bool(result.get("out_of_catalogue")),
                    "new_items_this_turn": bool(result.get("new_items_this_turn")),
                    "response": full_response,
                    "look_id": result.get("look_id"),
                    "occasion": result.get("occasion"),
                    "look_gender": result.get("look_gender"),
                    "budget_total_inr": result.get("budget_total_inr"),
                    "outfit_rationale": result.get("outfit_rationale"),
                    "outfit_variants": (
                        [v.model_dump() for v in _ws_variants]
                        if _ws_variants else None
                    ),
                    "cart_url": _ws_base_cart_url,
                    "item_links": (
                        [lk.model_dump() for lk in _ws_base_item_links]
                        if _ws_base_item_links else None
                    ),
                    "suggestion_chips": result.get("suggestion_chips") or None,
                    "suppressed_slots": result.get("suppressed_slots") or None,
                    "look_role": _normalize_look_role(result.get("look_role")),
                    "look_title": result.get("look_title"),
                    "coordinated_with": result.get("coordinated_with"),
                    # Wave 7 — "couple-from-scratch" turn: a SECOND coordinated board
                    # alongside the primary one above; None on every other turn.
                    "partner_look": (
                        _ws_partner_look.model_dump() if _ws_partner_look else None
                    ),
                },
                message_id=last_message_id,
            ).model_dump_json()
        )

        # Explicitly close the connection after the terminal frame so the
        # client sees a clean handshake (code 1000) rather than relying on
        # the ASGI server to close it implicitly when the handler returns.
        # Without this, some clients intermittently observe an abnormal
        # close ("no close frame received or sent") and silently retry,
        # even though items/done were already sent successfully.
        try:
            await websocket.close(code=1000)
        except Exception:
            # Client may have already disconnected — nothing to do.
            pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_text(
                WSErrorMessage(message="Internal server error", code="internal_error").model_dump_json()
            )
        except Exception:
            pass
    finally:
        if cid_token is not None:
            conversation_id_var.reset(cid_token)
