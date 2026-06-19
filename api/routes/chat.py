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

        done, _ = await asyncio.wait(
            {agent_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if cancel_event.is_set():
            agent_task.cancel()
            await websocket.send_text(WSCancelledMessage().model_dump_json())
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
                "messages": [{"role": "assistant", "content": full_response}],
            }

        elif plan.get("action") == "pending_answer":
            full_response = plan.get("text", "")
            await websocket.send_text(WSTokenMessage(text=full_response).model_dump_json())
            result = {
                **result,
                "final_answer": full_response,
                "messages": [{"role": "assistant", "content": full_response}],
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
                },
                message_id=last_message_id,
            ).model_dump_json()
        )

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
