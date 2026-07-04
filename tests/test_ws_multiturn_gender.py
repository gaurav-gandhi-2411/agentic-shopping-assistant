"""Full-stack WS /chat/stream reproduction of the mixed-gender colour-refinement bug.

The graph-level repro in tests/test_agent.py (real agent.invoke(), sync graph) stayed
green even before the index_group_name sync fix (commit f7f74f1), because that fixture's
retrieval window always had enough men's-shirt inventory to avoid search_node's
zero-result fallback ladder. The live browser bug reproduces on the STREAMING graph
driven through the real WS route (api/routes/chat.py::ws_chat), with the demo-anon
session path (_DEMO_SESSIONS) and a per-turn ticket reconnect — exactly like the
frontend. This test drives that real path end-to-end, turn by turn, with a fresh
WS connection (and fresh single-use ticket) per turn but the SAME conversation_id,
mirroring the browser exactly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from api.session import InMemorySessionStore
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

UNIFIED_DIR = Path("data/processed/unified")

_MINIMAL_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
    "retrieval": {
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dense_dim": 384,
        "rrf_k": 60,
        "top_k": 50,
        "final_k": 10,
        "store_diversity": 0.2,
    },
}


class _MockLLM:
    """Cycles through canned responses; repeats the last one when exhausted.

    IntentParser handles routing deterministically for plain product queries (no
    LLM call), so these responses only need to satisfy respond_node's streamed
    token generation (generate_stream) — the actual text content is irrelevant to
    this test, which asserts on retrieved items / filters, not response prose.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs) -> Iterator[str]:
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        yield self._next()


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


@pytest.fixture
def demo_client(monkeypatch: pytest.MonkeyPatch, _unified_index) -> TestClient:
    """Wires the real compiled agent graphs (unified index) into api.deps and
    enables the demo-anon auth path, exactly like the deployed service.
    """
    retriever, catalogue_df = _unified_index
    llm = _MockLLM(["Here you go."] * 40)

    deps._init(
        retriever=retriever,
        catalogue_df=catalogue_df,
        llm=llm,
        config=_MINIMAL_CONFIG,
        session_store=InMemorySessionStore(),
    )
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_JWT_SECRET", "ws-multiturn-test-secret")
    monkeypatch.delenv("JWT_VERIFICATION_DISABLED", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")
    monkeypatch.setattr(deps, "_db_engine", None)  # skip daily-cap DB checks

    # api.routes.chat._DEMO_SESSIONS is a module-level dict keyed by conversation_id;
    # reset it so tests never see state left behind by another test/run.
    import api.routes.chat as chat_module

    monkeypatch.setattr(chat_module, "_DEMO_SESSIONS", {})

    return TestClient(app, raise_server_exceptions=True)


def _mint_ticket(client: TestClient, session_token: str) -> str:
    """POST /auth/ws-ticket with the demo session token — mints a fresh, single-use
    60s ticket, exactly as the frontend does before opening each new WS connection.
    """
    resp = client.post(
        "/auth/ws-ticket", headers={"Authorization": f"Bearer {session_token}"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["ticket"]


def _run_ws_turn(
    client: TestClient, ticket: str, message: str, conversation_id: str | None
) -> tuple[str, list[dict]]:
    """Open one fresh WS connection (per-turn reconnect, matching the browser),
    send one user message, and collect (conversation_id, items) from the response
    frames. Blocks until the "done" frame or an "error" frame.
    """
    with client.websocket_connect(f"/chat/stream?ticket={ticket}") as ws:
        payload: dict = {"type": "user_message", "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        ws.send_json(payload)

        cid: str | None = conversation_id
        items: list[dict] = []
        for _ in range(200):
            frame = ws.receive_json()
            ftype = frame.get("type")
            if ftype == "session":
                cid = frame["conversation_id"]
            elif ftype == "items":
                items = frame.get("items", [])
            elif ftype == "error":
                raise AssertionError(f"WS error frame: {frame}")
            elif ftype in ("done", "cancelled"):
                break
        assert cid is not None
        return cid, items


@pytest.mark.requires_index
def test_ws_colour_refinement_inherits_most_recent_turn_gender(
    demo_client: TestClient,
) -> None:
    """Reproduces the live browser bug through the REAL WS route, demo-anon session
    path, and per-turn reconnect (fresh ticket + fresh WS connection per turn, same
    conversation_id) — exactly like the frontend drives /chat/stream.

    4-turn conversation: saree -> black dress for women -> white shirt men (correct)
    -> "in blue now". Turn 4 must stay men's shirts; must NOT fall back to a generic
    women's search.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"

    resp = demo_client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    session_token = body["session_token"]
    first_ticket = body["ws_ticket"]

    import api.routes.chat as chat_module

    conversation_id: str | None = None
    conversation_id, _ = _run_ws_turn(demo_client, first_ticket, "saree", conversation_id)

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, _ = _run_ws_turn(
        demo_client, ticket, "black dress for women", conversation_id
    )

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, turn3_items = _run_ws_turn(
        demo_client, ticket, "white shirt men", conversation_id
    )

    # Precondition: turn 3 must actually land on men's shirts, matching the live report.
    turn3_types = [it.get("product_type", "").lower() for it in turn3_items]
    assert turn3_types and all("shirt" in t for t in turn3_types), (
        f"Precondition failed: turn 3 should be all shirts, got {turn3_types}"
    )
    session_after_3 = chat_module._DEMO_SESSIONS[conversation_id]
    turn3_genders = [
        it.get("gender", "").lower() for it in session_after_3["retrieved_items"]
    ]
    assert turn3_genders and all(g == "men" for g in turn3_genders), (
        f"Precondition failed: turn 3 should be all men's, got {turn3_genders}"
    )

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, turn4_items = _run_ws_turn(
        demo_client, ticket, "in blue now", conversation_id
    )

    session_after_4 = chat_module._DEMO_SESSIONS[conversation_id]
    filters = session_after_4.get("filters", {})
    turn4_types = [it.get("product_type", "").lower() for it in turn4_items]
    turn4_genders = [
        it.get("gender", "").lower() for it in session_after_4["retrieved_items"]
    ]

    assert filters.get("product_type_name", "").lower() == "shirt", (
        f"Expected garment_type carried forward as 'shirt', got filters={filters}"
    )
    assert filters.get("index_group_name") == "menswear", (
        f"Expected gender carried forward as 'men' (index_group_name=menswear), "
        f"got filters={filters}"
    )
    assert turn4_types and all("shirt" in t for t in turn4_types), (
        f"Expected turn 4 items to stay shirts, got {turn4_types} "
        f"(display_names={[it.get('display_name') for it in turn4_items]})"
    )
    assert turn4_genders and all(g == "men" for g in turn4_genders), (
        f"Expected turn 4 items to stay men's, got {turn4_genders} "
        f"(display_names={[it.get('display_name') for it in turn4_items]})"
    )


@pytest.mark.requires_index
def test_ws_colour_refinement_forced_fallback_preserves_gender_and_type(
    demo_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same real WS path as above, but forces search_node's progressive fallback
    ladder so it must fall through to the {product_type_name, index_group_name}
    candidate (the one that reconstructs gender purely from index_group_name, with
    no "gender" key) — the exact seam where a stale index_group_name corrupts the
    result. The turn-3 -> turn-4 conversation above shows the corrupted *state*
    survives even through the real WS route with ample real inventory (state
    assertions there still fail pre-fix); this test additionally forces the
    fallback branch so the corrupted state also corrupts the returned *items*,
    matching the live browser report where turn 4 rendered generic women's items
    (denim jacket / trousers / off-shoulder top / blazer) instead of men's shirts.
    """
    os.environ["AGENT_LOOP_FAST_PATH"] = "true"
    import api.routes.chat as chat_module
    import src.agents.graph as graph_module
    from src.agents.tools import search_catalogue as real_search_catalogue

    def _fake_search_catalogue(query, filters, retriever, top_k):
        if filters and filters.get("gender"):
            return {"items": [], "query": query, "n_results": 0}
        return real_search_catalogue(query, filters, retriever, top_k)

    monkeypatch.setattr(graph_module, "search_catalogue", _fake_search_catalogue)

    resp = demo_client.post("/demo/session")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    session_token = body["session_token"]
    first_ticket = body["ws_ticket"]

    conversation_id: str | None = None
    conversation_id, _ = _run_ws_turn(demo_client, first_ticket, "saree", conversation_id)

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, _ = _run_ws_turn(
        demo_client, ticket, "black dress for women", conversation_id
    )

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, turn3_items = _run_ws_turn(
        demo_client, ticket, "white shirt men", conversation_id
    )
    turn3_types = [it.get("product_type", "").lower() for it in turn3_items]
    assert turn3_types and all("shirt" in t for t in turn3_types), (
        f"Precondition failed: turn 3 should be all shirts, got {turn3_types}"
    )
    # The forced fallback also fires on turn 3 itself (its own full-filter search
    # is forced to zero too), so the gender precondition must be checked here via
    # the internal session state (item["gender"], not exposed on the WS wire) —
    # otherwise a turn 3 that already silently landed on the wrong gender would go
    # unnoticed and this test would validate nothing.
    turn3_genders = [
        it.get("gender", "").lower()
        for it in chat_module._DEMO_SESSIONS[conversation_id]["retrieved_items"]
    ]
    assert turn3_genders and all(g == "men" for g in turn3_genders), (
        f"Precondition failed: turn 3 should be all men's, got {turn3_genders}"
    )

    ticket = _mint_ticket(demo_client, session_token)
    conversation_id, turn4_items = _run_ws_turn(
        demo_client, ticket, "in blue now", conversation_id
    )

    turn4_types = [it.get("product_type", "").lower() for it in turn4_items]
    turn4_genders = [
        it.get("gender", "").lower()
        for it in chat_module._DEMO_SESSIONS[conversation_id]["retrieved_items"]
    ]
    assert turn4_types and all("shirt" in t for t in turn4_types), (
        f"Expected turn 4 items to stay shirts (men's shirts, colour constraint may be "
        f"lost in the fallback), got {turn4_types} "
        f"(display_names={[it.get('display_name') for it in turn4_items]})"
    )
    assert turn4_genders and all(g == "men" for g in turn4_genders), (
        f"Expected turn 4 items to stay men's, got {turn4_genders} "
        f"(display_names={[it.get('display_name') for it in turn4_items]})"
    )
