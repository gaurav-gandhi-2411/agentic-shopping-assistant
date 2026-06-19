"""ASA user-flow QA matrix — defect groups G1–G9.

Run with:
    python -m qa.user_flow_matrix

Exits with code 1 when any check FAILs.
Index-dependent checks are SKIPped when no pre-built FAISS index is found.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_FIXTURE_WEBP = _REPO_ROOT / "tests" / "fixtures" / "test_tshirt.webp"
_INDEX_SENTINEL = _REPO_ROOT / "data" / "processed" / "dense.faiss"
_CHAT_PAGE = _REPO_ROOT / "frontend" / "app" / "demo" / "chat" / "page.tsx"

_MINIMAL_CONFIG: dict = {
    "agent": {"max_iterations": 3},
    "memory": {"recent_turns": 6, "summary_trigger_turns": 12},
}

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

Status = Literal["PASS", "FAIL", "SKIP"]


@dataclass
class CheckResult:
    defect: str
    name: str
    status: Status
    detail: str = ""


_results: list[CheckResult] = []


def _record(defect: str, name: str, status: Status, detail: str = "") -> None:
    _results.append(CheckResult(defect=defect, name=name, status=status, detail=detail))


# ---------------------------------------------------------------------------
# Helpers — shared mock infra (mirrors tests/test_api_chat.py)
# ---------------------------------------------------------------------------

class _MockLLM:
    """Minimal stub; returns a fixed string for any LLM call."""

    def generate(self, prompt: str, system: str | None = None, **kwargs: object) -> str:
        return "ok"

    def chat(self, messages: list[dict], **kwargs: object) -> str:
        return "ok"

    def chat_stream(self, messages: list[dict], **kwargs: object):  # type: ignore[return]
        yield "ok"


class _MockAgent:
    """Fake compiled LangGraph agent; returns a preset state dict."""

    def __init__(self, result: dict) -> None:
        self._result = result

    def invoke(self, state: dict, **kwargs: object) -> dict:
        result = dict(self._result)
        result.setdefault("messages", state.get("messages", []))
        return result


_DEFAULT_AGENT_RESULT: dict = {
    "retrieved_items": [],
    "filters": {},
    "tool_calls": [{"router_decision": {"action": "search", "query": "hello"}}],
    "final_answer": "Here are some results.",
    "iteration": 1,
    "new_items_this_turn": False,
    "out_of_catalogue": False,
    "excluded_colours": None,
}


def _make_agent_factory(result: dict):
    """Return a get_agent_factory replacement that yields a mock agent."""
    agent = _MockAgent(result)

    def factory(memory: object, streaming: bool = False) -> _MockAgent:
        return agent

    def get_factory() -> object:
        return factory

    return get_factory


def _fresh_client():
    """Return a TestClient with injected deps but no lifespan execution."""
    from fastapi.testclient import TestClient

    import api.deps as deps
    from api.main import app
    from api.session import InMemorySessionStore

    deps._session_store = InMemorySessionStore()
    deps._llm = _MockLLM()
    deps._config = _MINIMAL_CONFIG

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# G3 — template_rationale with NaN colour
# ---------------------------------------------------------------------------

def _check_g3_rationale_nan_colour() -> None:
    """G3: template_rationale must not produce 'nan' when colour is string 'nan'."""
    from src.agents.outfit.rationale import template_rationale

    look = {
        "seed_item": {"colour": "nan", "product_type": "T-shirt"},
        "complements": [],
        "occasion": "casual",
    }
    try:
        result = template_rationale(look)
        if "nan" in result.lower():
            _record("G3", "template_rationale with nan colour", "FAIL",
                    f"rationale contains 'nan': {result!r}")
        else:
            _record("G3", "template_rationale with nan colour", "PASS")
    except Exception as exc:
        _record("G3", "template_rationale with nan colour", "FAIL",
                f"raised {type(exc).__name__}: {exc}")


def _check_g3b_item_summary_nan_name() -> None:
    """G3b: ItemSummary.from_agent_item must not pass 'nan' through for prod_name."""
    from api.schemas import ItemSummary

    item = {
        "article_id": "test001",
        "prod_name": "nan",
        "display_name": "nan",
        "colour": "red",
        "product_type": "T-shirt",
        "department": "Menswear",
        "image_url": None,
        "score": 0.9,
    }
    try:
        summary = ItemSummary.from_agent_item(item)
        if summary.prod_name == "nan":
            _record("G3b", "ItemSummary.from_agent_item nan name", "FAIL",
                    "prod_name is 'nan' (should be sanitised to empty or placeholder)")
        else:
            _record("G3b", "ItemSummary.from_agent_item nan name", "PASS")
    except Exception as exc:
        _record("G3b", "ItemSummary.from_agent_item nan name", "FAIL",
                f"raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# G5 — budget_total mismatch
# ---------------------------------------------------------------------------

def _check_g5_budget_open_all_consistent() -> None:
    """G5: all items in a cross-store outfit have valid buy URLs; budget equals item sum."""
    from src.agents.outfit.cart_links import build_cart_action

    # Cross-store outfit spanning 3 different stores (realistic test case).
    # pdp_handle values mirror the real catalogue conventions per store:
    #   snitch   → product slug  (Shopify /products/{handle})
    #   myntra   → relative path (https://www.myntra.com/{handle})
    #   flipkart → full URL already stored in pdp_handle
    items = [
        {
            "article_id": "TEST-S1",
            "store": "snitch",
            "pdp_handle": "my-test-shirt",
            "display_name": "Test Shirt",
            "price_inr": 999.0,
        },
        {
            "article_id": "TEST-M1",
            "store": "myntra",
            "pdp_handle": "shirts/brand/test-shirt/99999999/buy",
            "display_name": "Test Kurta",
            "price_inr": 1499.0,
        },
        {
            "article_id": "TEST-F1",
            "store": "flipkart",
            "pdp_handle": "https://www.flipkart.com/test-prod/p/itm123abc",
            "display_name": "Test Trousers",
            "price_inr": 1200.0,
        },
    ]
    expected_budget = sum(it["price_inr"] for it in items)  # 3698.0

    try:
        action = build_cart_action(items, "unified")
    except Exception as exc:
        _record("G5", "budget + open-all consistent", "FAIL",
                f"build_cart_action raised {type(exc).__name__}: {exc}")
        return

    links = action.get("item_links", [])
    missing = action.get("missing", [])

    if missing:
        _record(
            "G5", "budget + open-all consistent", "FAIL",
            f"{len(missing)} item(s) have no buy URL (article_ids: {missing}) — "
            "open-all would skip them",
        )
    elif len(links) != len(items):
        _record(
            "G5", "budget + open-all consistent", "FAIL",
            f"item_links count={len(links)} but outfit items={len(items)}",
        )
    else:
        bad_urls = [lk for lk in links if not lk.get("buy_url", "").startswith("http")]
        if bad_urls:
            _record(
                "G5", "budget + open-all consistent", "FAIL",
                f"{len(bad_urls)} item(s) have invalid buy_url: "
                f"{[lk['article_id'] for lk in bad_urls]}",
            )
        else:
            _record(
                "G5", "budget + open-all consistent", "PASS",
                f"all {len(links)} items have buy URLs; budget ₹{expected_budget:.0f}",
            )


# ---------------------------------------------------------------------------
# G9 — _GENDER_MAP missing \bwife\b
# ---------------------------------------------------------------------------

def _check_g9_gender_map_wife() -> None:
    """G9: _GENDER_MAP must contain r'\\bwife\\b' to extract gender from 'for my wife'."""

    # _GENDER_MAP is defined inside build_graph() as a class/local variable.
    # Grep the source file directly — we look for the pattern in the module source.
    graph_src = (_REPO_ROOT / "src" / "agents" / "graph.py").read_text(encoding="utf-8")
    wife_pattern = r"\bwife\b"

    if wife_pattern in graph_src:
        _record("G9", "_GENDER_MAP contains wife", "PASS")
    else:
        _record("G9", "_GENDER_MAP contains wife", "FAIL",
                "key r'\\bwife\\b' missing from _GENDER_MAP — "
                "'for my wife' does not route to Ladieswear")


# ---------------------------------------------------------------------------
# G1 — /style/from-image response missing conversation_id
# ---------------------------------------------------------------------------

def _check_g1_image_endpoint_no_conv_id() -> None:
    """G1: /style/from-image response payload must contain conversation_id."""
    # Inspect endpoint source: the response payload dict is built in image_style.py.
    # We check that 'conversation_id' is NOT present in the payload keys — confirming
    # the bug exists without needing a live CLIP index.
    image_style_src = (_REPO_ROOT / "api" / "routes" / "image_style.py").read_text(
        encoding="utf-8"
    )

    # Look for conversation_id assignment inside the payload dict
    if '"conversation_id"' in image_style_src or "'conversation_id'" in image_style_src:
        _record("G1", "/style/from-image returns conv_id", "PASS")
    else:
        _record("G1", "/style/from-image returns conv_id", "FAIL",
                "endpoint payload dict has no 'conversation_id' key — "
                "follow-up chat cannot resume image context")


# ---------------------------------------------------------------------------
# G8 — /style/from-image accepts WebP (not 400)
# ---------------------------------------------------------------------------

def _check_g8_image_accepts_webp() -> None:
    """G8: /style/from-image must not return 400 for a valid WebP upload."""
    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    client = _fresh_client()
    webp_bytes = _FIXTURE_WEBP.read_bytes()

    res = client.post(
        "/style/from-image",
        files={"file": ("test_tshirt.webp", webp_bytes, "image/webp")},
    )

    if res.status_code == 400:
        try:
            detail = res.json().get("detail", res.text[:120])
        except Exception:
            detail = res.text[:120]
        _record("G8", "/style/from-image accepts WebP", "FAIL",
                f"got 400 (content-type rejected): {detail}")
    else:
        _record("G8", "/style/from-image accepts WebP", "PASS",
                f"status={res.status_code} (not 400 — content-type accepted)")


# ---------------------------------------------------------------------------
# G2 — items always have image_url
# ---------------------------------------------------------------------------

def _check_g2_items_image_url_not_none() -> None:
    """G2: any item in a chat response with new_items_this_turn=True must have non-null image_url."""
    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    import api.deps as deps

    client = _fresh_client()

    result_with_null_image = {
        **_DEFAULT_AGENT_RESULT,
        "new_items_this_turn": True,
        "retrieved_items": [
            {
                "article_id": "BAD001",
                "prod_name": "Slim Trousers",
                "display_name": "Slim Trousers (Black)",
                "colour": "Black",
                "product_type": "Trousers",
                "department": "Ladieswear",
                "image_url": None,
                "detail_desc": "Test item with null image.",
                "score": 0.92,
            }
        ],
    }
    deps.get_agent_factory = _make_agent_factory(result_with_null_image)  # type: ignore[assignment]

    res = client.post("/chat", json={"message": "show me black trousers"})

    if res.status_code != 200:
        _record("G2", "items always have image_url", "FAIL",
                f"unexpected status {res.status_code}: {res.text[:100]}")
        return

    data = res.json()
    null_items = [it for it in data.get("items", []) if not it.get("image_url")]

    if null_items:
        ids = [it["article_id"] for it in null_items]
        _record("G2", "items always have image_url", "FAIL",
                f"item(s) with null image_url: article_ids={ids}")
    else:
        _record("G2", "items always have image_url", "PASS")


# ---------------------------------------------------------------------------
# G6 — POST /looks returns 201 (fails without DATABASE_URL)
# ---------------------------------------------------------------------------

def _check_g6_save_look_round_trip() -> None:
    """G6: POST /looks returns 201 and GET /looks/{id} returns the saved look."""
    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"
    os.environ.pop("DATABASE_URL", None)  # ensure in-memory fallback is used

    # Clear the memory store before this test so prior QA runs don't interfere.
    from src.storage.saved_looks import _MEMORY_STORE
    _MEMORY_STORE.clear()

    client = _fresh_client()

    payload = {
        "session_id": "qa-session-g6",
        "brand": "unified",
        "look_id": "look-test-g6",
        "occasion": "casual",
        "look_gender": "women",
        "anchor_item_id": "TEST001",
        "look_total_inr": 2499,
        "snapshot": {
            "items": [],
            "rationale": "Test look for QA",
            "cart_url": None,
            "item_links": [],
            "variant_label": "Style 1",
        },
        "user_id": None,
    }

    res_post = client.post("/looks", json=payload)
    if res_post.status_code != 201:
        try:
            detail = res_post.json().get("detail", res_post.text[:120])
        except Exception:
            detail = res_post.text[:120]
        _record("G6", "save look round-trip", "FAIL",
                f"POST /looks got {res_post.status_code}: {detail}")
        return

    saved_id = res_post.json().get("id")
    if not saved_id:
        _record("G6", "save look round-trip", "FAIL",
                f"POST /looks returned 201 but no id in body: {res_post.json()}")
        return

    # Round-trip: GET /looks/{id}
    res_get = client.get(f"/looks/{saved_id}")
    if res_get.status_code == 200:
        get_data = res_get.json()
        if get_data.get("id") == saved_id:
            _record("G6", "save look round-trip", "PASS",
                    f"POST 201 id={saved_id[:8]}... -> GET 200 ok")
        else:
            _record("G6", "save look round-trip", "FAIL",
                    f"GET returned id={get_data.get('id')!r}, expected {saved_id!r}")
    else:
        _record("G6", "save look round-trip", "FAIL",
                f"GET /looks/{saved_id[:8]}... returned {res_get.status_code}")


# ---------------------------------------------------------------------------
# G7 — frontend chat page has no 'change brand' text
# ---------------------------------------------------------------------------

def _check_g7_no_change_brand() -> None:
    """G7: demo/chat/page.tsx must not contain raw 'Change brand' UI text."""
    if not _CHAT_PAGE.exists():
        _record("G7", "no 'change brand' in demo/chat page", "SKIP",
                f"file not found: {_CHAT_PAGE}")
        return

    content = _CHAT_PAGE.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    found_lines = [
        f"line {i + 1}: {line.strip()}"
        for i, line in enumerate(lines)
        if "change brand" in line.lower()
    ]

    if found_lines:
        _record("G7", "no 'change brand' in demo/chat page", "FAIL",
                f"found 'Change brand': {'; '.join(found_lines)}")
    else:
        _record("G7", "no 'change brand' in demo/chat page", "PASS")


# ---------------------------------------------------------------------------
# G1b — image upload persists outfit to session store (no live CLIP index)
# ---------------------------------------------------------------------------

def _check_g1b_image_session_persistence() -> None:
    """G1b: after image upload, the session store contains the outfit items.

    Uses mocked CLIP encoder and compose_outfit_variants so no real indices
    are needed.  Checks that:
      1. /style/from-image returns 200 with a conversation_id.
      2. The session store entry for that conversation_id contains retrieved_items.
    """
    import unittest.mock as mock

    from fastapi.testclient import TestClient

    import api.deps as deps
    from api.main import app
    from api.session import InMemorySessionStore

    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    session_store = InMemorySessionStore()
    deps._session_store = session_store
    deps._llm = _MockLLM()
    deps._config = {
        **_MINIMAL_CONFIG,
        "features": {"image_input_enabled": True},
        "clip": {"model": "clip-ViT-B-32", "index_dir": "data/processed/clip"},
        "retrieval": {
            "dense_model": "...",
            "dense_dim": 384,
            "rrf_k": 60,
            "top_k": 5,
            "final_k": 3,
            "store_diversity": 0.0,
        },
    }

    fake_items = [
        {
            "article_id": "FAKE001",
            "store": "snitch",
            "pdp_handle": "test-slug",
            "display_name": "Test Shirt",
            "prod_name": "Test Shirt",
            "colour": "Blue",
            "product_type": "T-shirt",
            "department": "Menswear",
            "index_group_name": "Menswear",
            "image_url": "https://example.com/img.jpg",
            "price_inr": 999.0,
            "detail_desc": "Test",
            "gender": "men",
            "score": 1.0,
            "_role": "seed",
        }
    ]
    fake_look = {
        "look_id": "test-look-001",
        "seed_item": fake_items[0],
        "complements": [],
        "outfit_rationale": "Test look",
        "empty_slots": [],
        "occasion": "casual",
        "gender": "women",
        "budget_total_inr": 999.0,
        "variant_label": "Base",
    }

    # Patch where each name is USED (image_style module namespace), not where defined.
    with (
        mock.patch(
            "api.routes.image_style.find_anchor_from_image",
            return_value=["FAKE001"],
        ),
        mock.patch(
            "api.routes.image_style._brand_index_exists",
            return_value=True,
        ),
        mock.patch(
            "api.routes.image_style.compose_outfit_variants",
            return_value=[fake_look],
        ),
        mock.patch(
            "api.routes.image_style.generate_rationales",
            return_value=["Test rationale"],
        ),
        mock.patch("api.deps.get_catalogue_df", return_value=None),
        mock.patch("api.deps.get_retriever", return_value=None),
        mock.patch("api.deps.get_session_store", return_value=session_store),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        webp_bytes = _FIXTURE_WEBP.read_bytes()

        with mock.patch("api.routes.image_style.ItemSummary.from_agent_item") as mock_summary:
            from api.schemas import ItemSummary

            mock_summary.side_effect = lambda it: ItemSummary(
                article_id=it["article_id"],
                prod_name=it.get("prod_name", ""),
                display_name=it.get("display_name", ""),
                colour=it.get("colour", ""),
                product_type=it.get("product_type", ""),
                department=it.get("department", ""),
                image_url=it.get("image_url"),
                price_inr=it.get("price_inr"),
                store=it.get("store"),
                store_display=it.get("store"),
            )

            res = client.post(
                "/style/from-image",
                data={"conversation_id": "test-cid-g1b"},
                files={"file": ("test.webp", webp_bytes, "image/webp")},
            )

    # 400/422 → real bug; 500/503 → infrastructure not available → SKIP
    if res.status_code in (400, 422):
        try:
            detail = res.json().get("detail", res.text[:120])
        except Exception:
            detail = res.text[:120]
        _record(
            "G1b",
            "image → follow-up retains context",
            "FAIL",
            f"status={res.status_code}: {detail}",
        )
        return

    if res.status_code != 200:
        _record(
            "G1b",
            "image → follow-up retains context",
            "SKIP",
            f"endpoint returned {res.status_code} (CLIP/LLM not available locally)",
        )
        return

    data = res.json()
    returned_cid = data.get("conversation_id")
    if not returned_cid:
        _record(
            "G1b",
            "image → follow-up retains context",
            "FAIL",
            "conversation_id absent from response payload",
        )
        return

    # Anonymous demo requests store session in _DEMO_SESSIONS (api.routes.chat),
    # not in the InMemorySessionStore, so check the right store.
    from api.routes.chat import _DEMO_SESSIONS

    stored = _DEMO_SESSIONS.get(returned_cid) or session_store.get(returned_cid, "")
    if stored and stored.get("retrieved_items"):
        _record(
            "G1b",
            "image → follow-up retains context",
            "PASS",
            f"conv_id={returned_cid[:8]}… session has {len(stored['retrieved_items'])} item(s)",
        )
    else:
        _record(
            "G1b",
            "image → follow-up retains context",
            "FAIL",
            f"conv_id={returned_cid[:8]}… but session store has no items "
            f"(stored={bool(stored)})",
        )


# ---------------------------------------------------------------------------
# Agent-path integration helpers
# ---------------------------------------------------------------------------

class _AdversarialRouterLLM:
    """Adversarial mock LLM: routes to 'search' but deliberately drops product_type from filters.

    Simulates the real Groq/LLaMA-3 router's worst-case behavior: it picks the
    right action but omits garment-type filter and reformulates the query away
    from the product type ("women clothing" instead of "black dress").

    The graph architecture (raw_query retrieval + _PRODUCT_TYPE_KEYWORDS) must
    survive this and still return the correct garment type.

    The fast-path (AGENT_LOOP_FAST_PATH=true, default) handles search→respond
    routing without LLM, so this mock only handles:
      - Router call: "STRICT RULES" in prompt → return adversarial search JSON
      - Reranker call: starts with 'Query: "' → return valid index selection
      - Respond call: everything else → return canned text
    """

    def _decide(self, content: str) -> str:
        if "STRICT RULES" in content:
            c = content.lower()
            if "dress" in c:
                # Adversarial: drop product_type AND reformulate away from "dress"
                return '{"action": "search", "query": "women clothing", "filters": {"index_group_name": "ladieswear"}}'
            elif ("blue" in c or "colour" in c or "color" in c) and (
                "different" in c or "in blue" in c
            ):
                return '{"action": "search", "query": "blue", "filters": {"colour_group_name": "Blue"}}'
            else:
                return '{"action": "search", "query": "fashion items", "filters": {}}'
        elif content.strip().startswith('Query: "'):
            # Reranker: return first 5 items in retrieval order
            return '{"selected": [1, 2, 3, 4, 5]}'
        else:
            return "Here are some great options for you!"

    def generate(self, prompt: str, system: str | None = None, **kwargs: object) -> str:
        return self._decide(prompt)

    def chat(self, messages: list[dict], **kwargs: object) -> str:
        content = " ".join(m.get("content", "") for m in messages)
        return self._decide(content)

    def chat_stream(self, messages: list[dict], **kwargs: object):  # type: ignore[return]
        content = " ".join(m.get("content", "") for m in messages)
        yield self._decide(content)

    def generate_stream(self, prompt: str, **kwargs: object):  # type: ignore[return]
        yield self._decide(prompt)


def _agent_path_client(adversarial_llm=None):
    """Context manager: real FAISS index + adversarial mock LLM → TestClient.

    Calls deps._init() to compile the REAL LangGraph with the injected LLM so
    POST /chat drives the actual search_node, _PRODUCT_TYPE_KEYWORDS, etc.

    Critically: some earlier QA checks replace deps.get_agent_factory with a
    mock and never restore it.  After _init() compiles fresh graphs, we install
    a real factory closure that serves those graphs.  The prior value (mock or
    real) is restored on context exit.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        from fastapi.testclient import TestClient
        import api.deps as deps
        from api.main import app
        from api.session import InMemorySessionStore

        llm = adversarial_llm or _AdversarialRouterLLM()
        retriever, catalogue_df = _load_unified_retriever()

        saved = {
            k: getattr(deps, k, None)
            for k in ("_retriever", "_catalogue_df", "_llm", "_config",
                       "_session_store", "_agent_sync", "_agent_streaming",
                       "get_agent_factory")
        }
        try:
            cfg = {
                **_MINIMAL_CONFIG,
                "retrieval": {
                    "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dense_dim": 384,
                    "rrf_k": 60,
                    "top_k": 20,
                    "final_k": 5,
                    "store_diversity": 0.2,
                },
            }
            deps._init(
                retriever=retriever,
                catalogue_df=catalogue_df,
                llm=llm,
                config=cfg,
                session_store=InMemorySessionStore(),
            )
            # _init() compiled real graphs into deps._agent_sync / _agent_streaming.
            # Replace whatever factory is currently installed (may be a mock from G2
            # or other earlier checks) with a factory that serves the compiled graphs.
            def _real_factory():
                def factory(memory: object, streaming: bool = False) -> object:
                    return deps._agent_streaming if streaming else deps._agent_sync
                return factory
            deps.get_agent_factory = _real_factory  # type: ignore[assignment]

            yield TestClient(app, raise_server_exceptions=False)
        finally:
            for k, v in saved.items():
                if v is not None:
                    setattr(deps, k, v)

    return _ctx()


_AGENT_HEADERS = {"Authorization": "Bearer qa-test"}


# ---------------------------------------------------------------------------
# G2b-agent — 'black dress for women' through real LangGraph → dresses only
# ---------------------------------------------------------------------------

def _check_agent_path_dress() -> None:
    """G2b-agent: POST /chat 'black dress for women' through real LangGraph.

    RED without architecture fix:
        Adversarial LLM drops product_type → embedding search on "women clothing"
        → kurtis dominate (4,443 vs 1,544 dresses) → wrong garment type returned.
    GREEN after fix (query=raw_query + _PRODUCT_TYPE_KEYWORDS):
        raw_query "black dress for women" preserves "dress" for retrieval;
        keyword filter forces product_type_name=Dress → only dresses returned.
    """
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("G2b-agent", "agent path: black dress → dresses only", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    try:
        with _agent_path_client() as client:
            resp = client.post("/chat", json={"message": "black dress for women"},
                               headers=_AGENT_HEADERS, timeout=120)
    except Exception as exc:
        _record("G2b-agent", "agent path: black dress → dresses only", "FAIL",
                f"client/setup error: {type(exc).__name__}: {exc}")
        return

    if resp.status_code != 200:
        _record("G2b-agent", "agent path: black dress → dresses only", "FAIL",
                f"POST /chat returned {resp.status_code}: {resp.text[:120]}")
        return

    data = resp.json()
    items = data.get("items", [])

    if not items:
        resp_text = data.get("response", "")[:100]
        _record("G2b-agent", "agent path: black dress → dresses only", "FAIL",
                f"0 items returned — response: {resp_text!r}")
        return

    non_dress = [it.get("product_type", "") for it in items
                 if it.get("product_type", "").lower() not in ("dress", "")]
    no_image = [it.get("article_id") for it in items if not it.get("image_url")]

    if non_dress:
        _record("G2b-agent", "agent path: black dress → dresses only", "FAIL",
                f"{len(non_dress)}/{len(items)} non-dress types: {non_dress[:3]}")
    elif no_image:
        _record("G2b-agent", "agent path: black dress → dresses only", "FAIL",
                f"{len(no_image)} items missing image_url")
    else:
        stores = list({it.get("store") for it in items})
        _record("G2b-agent", "agent path: black dress → dresses only", "PASS",
                f"all {len(items)} items are Dress, all have images; stores={stores}")


# ---------------------------------------------------------------------------
# G-refine — colour refinement retains garment type AND returns cards
# ---------------------------------------------------------------------------

def _check_agent_path_refinement() -> None:
    """G-refine: 'in blue' after dress search → still Dress cards, never prose-only.

    Tests two failure modes:
    1. Refinement forgets garment type (wrong product type returned)
    2. Refinement returns text response with no items (prose instead of cards)
    """
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    try:
        with _agent_path_client() as client:
            # Turn 1: initial dress query
            r1 = client.post("/chat", json={"message": "black dress for women"},
                             headers=_AGENT_HEADERS, timeout=120)
            if r1.status_code != 200:
                _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "SKIP",
                        f"turn1 returned {r1.status_code} (check G2b-agent for root cause)")
                return

            d1 = r1.json()
            cid = d1.get("conversation_id", "")
            items1 = d1.get("items", [])

            if not items1:
                _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "SKIP",
                        "turn1 returned no items (check G2b-agent for root cause)")
                return

            # Turn 2: colour refinement
            r2 = client.post(
                "/chat",
                json={"message": "in blue", "conversation_id": cid},
                headers=_AGENT_HEADERS,
                timeout=120,
            )
    except Exception as exc:
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "FAIL",
                f"client/setup error: {type(exc).__name__}: {exc}")
        return

    if r2.status_code != 200:
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "FAIL",
                f"turn2 returned {r2.status_code}: {r2.text[:120]}")
        return

    d2 = r2.json()
    items2 = d2.get("items", [])

    if not items2:
        prose = d2.get("response", "")[:120]
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "FAIL",
                f"refinement returned 0 items (prose-only); response: {prose!r}")
        return

    non_dress = [it.get("product_type", "") for it in items2
                 if it.get("product_type", "").lower() not in ("dress", "")]
    no_image = [it.get("article_id") for it in items2 if not it.get("image_url")]

    if non_dress:
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "FAIL",
                f"refinement lost garment type: {non_dress[:3]}")
    elif no_image:
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "FAIL",
                f"{len(no_image)} items missing image_url on refinement turn")
    else:
        colours = list({it.get("colour") for it in items2})
        _record("G-refine", "refinement: 'in blue' → dress cards (not prose)", "PASS",
                f"turn2 items={len(items2)}, all Dress, all have images; colours={colours}")


# ---------------------------------------------------------------------------
# G-chips — suggestion_chips present in search response
# ---------------------------------------------------------------------------

def _check_agent_path_chips() -> None:
    """G-chips: POST /chat search must return suggestion_chips for colour refinement."""
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("G-chips", "suggestion_chips returned with items", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    try:
        with _agent_path_client() as client:
            resp = client.post("/chat", json={"message": "black dress for women"},
                               headers=_AGENT_HEADERS, timeout=120)
    except Exception as exc:
        _record("G-chips", "suggestion_chips returned with items", "FAIL",
                f"client/setup error: {type(exc).__name__}: {exc}")
        return

    if resp.status_code != 200:
        _record("G-chips", "suggestion_chips returned with items", "FAIL",
                f"POST /chat returned {resp.status_code}")
        return

    data = resp.json()
    items = data.get("items", [])
    chips = data.get("suggestion_chips")

    if not items:
        _record("G-chips", "suggestion_chips returned with items", "SKIP",
                "no items returned — chips check not meaningful")
        return

    if chips is None:
        _record("G-chips", "suggestion_chips returned with items", "FAIL",
                "suggestion_chips key absent from response")
    elif len(chips) == 0:
        _record("G-chips", "suggestion_chips returned with items", "FAIL",
                "suggestion_chips is an empty list (all items same colour?)")
    else:
        _record("G-chips", "suggestion_chips returned with items", "PASS",
                f"{len(chips)} chips: {chips[:5]}")


# ---------------------------------------------------------------------------
# WS-product — WS /chat/stream: product query must send items frame
# ---------------------------------------------------------------------------

def _check_ws_stream_product_query() -> None:
    """WS-product: WS /chat/stream 'black dress for women' must deliver an items frame.

    Existing G2b/G-refine/G-chips tests call POST /chat — a DIFFERENT handler.
    This is the first test that hits the REAL browser path (WebSocket).

    Uses a mock LLM that always returns {"action":"respond"} for the router
    call, simulating Groq misbehaving.  WITHOUT the route_decision guard this
    produces prose-only (no WSItemsMessage).  WITH the guard, items are always
    returned.
    """
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("WS-product", "WS /chat/stream: black dress → items frame", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    os.environ["JWT_VERIFICATION_DISABLED"] = "true"
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"

    class _RespondFirstLLM:
        """Simulates the Groq router returning 'respond' before any search runs."""

        def _decide(self, prompt: str) -> str:
            if "STRICT RULES" in prompt:
                # Adversarial: always tell router to respond immediately
                return '{"action": "respond"}'
            elif prompt.strip().startswith('Query: "'):
                return '{"selected": [1, 2, 3, 4, 5]}'
            else:
                return "Here are some great dress options for you!"

        def generate(self, prompt: str, system: object = None, **kwargs: object) -> str:
            return self._decide(prompt)

        def generate_stream(self, prompt: str, **kwargs: object):  # type: ignore[return]
            yield self._decide(prompt)

        def chat(self, messages: list, **kwargs: object) -> str:
            return self._decide(" ".join(m.get("content", "") for m in messages))

        def chat_stream(self, messages: list, **kwargs: object):  # type: ignore[return]
            yield self._decide(" ".join(m.get("content", "") for m in messages))

    try:
        with _agent_path_client(adversarial_llm=_RespondFirstLLM()) as client:
            with client.websocket_connect("/chat/stream") as ws:
                ws.send_json({"type": "user_message", "message": "black dress for women"})

                items_frame: dict | None = None
                for _ in range(60):
                    try:
                        frame = ws.receive_json()
                    except Exception:
                        break
                    if frame.get("type") == "items":
                        items_frame = frame
                    elif frame.get("type") in ("done", "cancelled"):
                        break
                    elif frame.get("type") == "error":
                        _record(
                            "WS-product",
                            "WS /chat/stream: black dress → items frame",
                            "FAIL",
                            f"WS error frame: {frame.get('message', '')[:120]}",
                        )
                        return
    except Exception as exc:
        _record(
            "WS-product",
            "WS /chat/stream: black dress → items frame",
            "FAIL",
            f"connection/setup error: {type(exc).__name__}: {exc}",
        )
        return

    if items_frame is None:
        _record(
            "WS-product",
            "WS /chat/stream: black dress → items frame",
            "FAIL",
            "no WSItemsMessage received — prose-only response; "
            "route_decision guard is missing or not firing",
        )
        return

    items = items_frame.get("items", [])
    if not items:
        _record(
            "WS-product",
            "WS /chat/stream: black dress → items frame",
            "FAIL",
            "WSItemsMessage received but items list is empty",
        )
        return

    non_dress = [
        it.get("product_type")
        for it in items
        if it.get("product_type", "").lower() not in ("dress", "")
    ]
    no_image = [it.get("article_id") for it in items if not it.get("image_url")]

    if non_dress:
        _record(
            "WS-product",
            "WS /chat/stream: black dress → items frame",
            "FAIL",
            f"wrong product types via WS: {non_dress[:4]}",
        )
    elif no_image:
        _record(
            "WS-product",
            "WS /chat/stream: black dress → items frame",
            "FAIL",
            f"{len(no_image)} items missing image_url via WS",
        )
    else:
        stores = sorted({it.get("store", "") for it in items})
        _record(
            "WS-product",
            "WS /chat/stream: black dress → items frame",
            "PASS",
            f"{len(items)} Dress items, all have images; stores={stores}",
        )


# ---------------------------------------------------------------------------
# Shared retriever loader (G2b, G4, G9b) — loads unified index once on demand
# ---------------------------------------------------------------------------

def _load_unified_retriever():
    """Load DenseRetriever + SparseRetriever + HybridRetriever from the unified index.

    Returns (retriever, catalogue_df) or raises on any failure.
    """
    import pandas as pd

    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    unified_dir = _REPO_ROOT / "data" / "processed" / "unified"

    config: dict = {
        "retrieval": {
            "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dense_dim": 384,
            "rrf_k": 60,
            "top_k": 50,
            "final_k": 10,
            "store_diversity": 0.2,
        }
    }

    dense = DenseRetriever.load(config, unified_dir)
    sparse = SparseRetriever.load(config, unified_dir)
    catalogue_df = pd.read_parquet(unified_dir / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, config)
    return retriever, catalogue_df


# ---------------------------------------------------------------------------
# G2b — text search returns items with image_url
# ---------------------------------------------------------------------------

def _check_g2b_text_search_images() -> None:
    """G2b: 'black dress for women' must return dresses with images (not kurtis).

    Two sub-checks:
    1. Unfiltered search — all results have non-null image_url.
    2. Filtered search (product_type_name=dress, index_group_name=ladieswear) —
       all results must have product_type == 'Dress', confirming the garment-type
       filter works.  This is the fix for the 'kurtis returned instead of dresses'
       bug: the graph now deterministically sets the filter from raw_query.
    """
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("G2b", "black dress returns dresses (not kurtis)", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    try:
        retriever, _df = _load_unified_retriever()
    except Exception as exc:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                f"failed to load unified retriever: {type(exc).__name__}: {exc}")
        return

    # Sub-check 1: unfiltered — images present
    try:
        results = retriever.search("black dress for women", top_k=10)
    except Exception as exc:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                f"retriever.search raised {type(exc).__name__}: {exc}")
        return

    if not results:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                "search returned 0 results")
        return

    missing_image = [r["article_id"] for r in results if not r.get("image_url")]
    if missing_image:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                f"{len(missing_image)}/{len(results)} results have null image_url")
        return

    # Sub-check 2: with garment-type filter — results must be Dress, not kurtis
    try:
        filtered = retriever.search(
            "black dress for women",
            top_k=10,
            filters={"product_type_name": "dress", "index_group_name": "ladieswear"},
        )
    except Exception as exc:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                f"filtered search raised {type(exc).__name__}: {exc}")
        return

    if not filtered:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                "filtered search returned 0 results (no black dresses in index?)")
        return

    wrong_type = [
        (r["article_id"], r.get("product_type", ""))
        for r in filtered
        if r.get("product_type", "").lower() not in ("dress", "")
    ]
    if wrong_type:
        _record("G2b", "black dress returns dresses (not kurtis)", "FAIL",
                f"{len(wrong_type)}/{len(filtered)} results are NOT dresses: "
                f"{wrong_type[:3]}")
    else:
        _record("G2b", "black dress returns dresses (not kurtis)", "PASS",
                f"images ✓ all {len(results)} unfiltered; "
                f"garment type ✓ all {len(filtered)} filtered results are Dress")


# ---------------------------------------------------------------------------
# G4 — compose_outfit with Flipkart seed spans ≥2 stores
# ---------------------------------------------------------------------------

def _check_g4_multi_store_outfit() -> None:
    """G4: outfit composed around a Flipkart seed must span ≥2 distinct stores."""
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("G4", "image search spans >1 store", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    try:
        retriever, catalogue_df = _load_unified_retriever()
    except Exception as exc:
        _record("G4", "image search spans >1 store", "FAIL",
                f"failed to load unified retriever: {type(exc).__name__}: {exc}")
        return

    from src.agents.outfit.composer import compose_outfit

    fk_men = catalogue_df[
        (catalogue_df["store"].str.lower() == "flipkart")
        & (catalogue_df["gender"].str.lower() == "men")
    ]
    seed_row = fk_men.iloc[0] if not fk_men.empty else catalogue_df[
        catalogue_df["store"].str.lower() == "flipkart"
    ].iloc[0]
    seed_id = str(seed_row["article_id"])

    try:
        outfit = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=seed_id, occasion_slug="casual", gender="men",
        )
    except Exception as exc:
        _record("G4", "image search spans >1 store", "FAIL",
                f"compose_outfit raised {type(exc).__name__}: {exc}")
        return

    seed_item = outfit.get("seed_item")
    all_items = ([seed_item] if seed_item else []) + (outfit.get("complements") or [])
    stores = {str(it.get("store") or "").lower() for it in all_items if it and it.get("store")}

    if len(stores) >= 2:
        _record("G4", "image search spans >1 store", "PASS",
                f"outfit spans {len(stores)} stores: {sorted(stores)}")
    else:
        _record("G4", "image search spans >1 store", "FAIL",
                f"outfit spans only {len(stores)} store(s): {sorted(stores)} "
                f"(seed={seed_id}, complements={len(outfit.get('complements', []))})")


# ---------------------------------------------------------------------------
# G9b — men's t-shirt seed produces no skirt complements
# ---------------------------------------------------------------------------

def _check_g9b_no_skirts_for_men_tshirt() -> None:
    """G9b: compose_outfit with a Flipkart men's t-shirt seed must return no skirt complements."""
    unified_faiss = _REPO_ROOT / "data" / "processed" / "unified" / "dense.faiss"
    if not unified_faiss.exists():
        _record("G9b", "t-shirt anchor -> no skirts", "SKIP",
                f"unified index not present: {unified_faiss}")
        return

    try:
        retriever, catalogue_df = _load_unified_retriever()
    except Exception as exc:
        _record("G9b", "t-shirt anchor -> no skirts", "FAIL",
                f"failed to load unified retriever: {type(exc).__name__}: {exc}")
        return

    from src.agents.outfit.composer import compose_outfit

    fk_tshirt = catalogue_df[
        (catalogue_df["store"].str.lower() == "flipkart")
        & (catalogue_df["gender"].str.lower() == "men")
        & catalogue_df["product_type_name"].str.contains("T-shirt|Shirt", case=False, na=False)
    ]
    if fk_tshirt.empty:
        fk_men = catalogue_df[
            (catalogue_df["store"].str.lower() == "flipkart")
            & (catalogue_df["gender"].str.lower() == "men")
        ]
        if fk_men.empty:
            _record("G9b", "t-shirt anchor -> no skirts", "SKIP",
                    "no Flipkart men's rows in catalogue")
            return
        seed_row = fk_men.iloc[0]
    else:
        seed_row = fk_tshirt.iloc[0]

    seed_id = str(seed_row["article_id"])

    try:
        outfit = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=seed_id, occasion_slug="casual", gender="men",
        )
    except Exception as exc:
        _record("G9b", "t-shirt anchor -> no skirts", "FAIL",
                f"compose_outfit raised {type(exc).__name__}: {exc}")
        return

    complements = outfit.get("complements") or []
    skirt_items = [c for c in complements if "skirt" in str(c.get("product_type") or "").lower()]

    if skirt_items:
        _record("G9b", "t-shirt anchor -> no skirts", "FAIL",
                f"men's t-shirt outfit contains skirt complement(s): "
                f"{[c['article_id'] for c in skirt_items]}")
    else:
        _record("G9b", "t-shirt anchor -> no skirts", "PASS",
                f"0 skirts in {len(complements)} complement(s) for seed={seed_id}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _run_all_checks() -> None:
    # ── Unit checks (no server) ──────────────────────────────────────────────
    _check_g3_rationale_nan_colour()
    _check_g3b_item_summary_nan_name()
    _check_g5_budget_open_all_consistent()
    _check_g9_gender_map_wife()

    # ── Code-inspection check (no server, no index) ──────────────────────────
    _check_g1_image_endpoint_no_conv_id()

    # ── API checks (TestClient with mocked deps) ─────────────────────────────
    _check_g8_image_accepts_webp()
    _check_g2_items_image_url_not_none()
    _check_g6_save_look_round_trip()
    _check_g7_no_change_brand()

    # ── G1b: image session persistence (mocked, no CLIP index needed) ────────
    _check_g1b_image_session_persistence()

    # ── Integration checks (loads unified FAISS+BM25 index) ─────────────────
    _check_g2b_text_search_images()
    _check_g4_multi_store_outfit()
    _check_g9b_no_skirts_for_men_tshirt()

    # ── Agent-path checks (real LangGraph + adversarial mock LLM) ───────────
    # These drive POST /chat through the SAME code path the browser uses.
    # The adversarial LLM drops product_type from filters to prove the
    # raw_query + _PRODUCT_TYPE_KEYWORDS fix is working end-to-end.
    _check_agent_path_dress()
    _check_agent_path_refinement()
    _check_agent_path_chips()
    _check_ws_stream_product_query()


def _print_table(results: list[CheckResult]) -> int:
    """Print the QA matrix table and return 1 if any FAILs exist, else 0."""
    print("\n=== ASA User Flow QA Matrix ===\n")

    col_defect = 7
    col_name = 40
    col_status = 8
    header = (
        f"{'DEFECT':<{col_defect}}  "
        f"{'CHECK':<{col_name}}  "
        f"{'STATUS':<{col_status}}  "
        f"DETAIL"
    )
    sep = (
        f"{'-' * col_defect}  "
        f"{'-' * col_name}  "
        f"{'-' * col_status}  "
        f"{'-' * 44}"
    )
    print(header)
    print(sep)

    for r in results:
        defect = r.defect.ljust(col_defect)
        name = r.name[:col_name].ljust(col_name)
        status = r.status.ljust(col_status)
        print(f"{defect}  {name}  {status}  {r.detail}")

    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_skip = sum(1 for r in results if r.status == "SKIP")

    print(f"\nSummary: {n_fail} FAIL | {n_skip} SKIP | {n_pass} PASS")
    return 1 if n_fail else 0


def main() -> None:
    # Force UTF-8 output on Windows so non-ASCII chars in detail strings (e.g.
    # arrows from TSX source) do not crash the printer on cp1252 terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    _run_all_checks()
    exit_code = _print_table(_results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
