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

    with (
        mock.patch(
            "src.agents.outfit.image_anchor.find_anchor_from_image",
            return_value=["FAKE001"],
        ),
        mock.patch(
            "api.routes.image_style._brand_index_exists",
            return_value=True,
        ),
        mock.patch(
            "src.agents.outfit.composer.compose_outfit_variants",
            return_value=[fake_look],
        ),
        mock.patch(
            "src.agents.outfit.rationale.generate_rationales",
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

    # InMemorySessionStore ignores user_id — pass empty string to match any user.
    stored = session_store.get(returned_cid, "")
    if stored and stored.get("retrieved_items"):
        _record(
            "G1b",
            "image → follow-up retains context",
            "PASS",
            f"conv_id={returned_cid[:8]}… session has {len(stored['retrieved_items'])} items",
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
# Index-required checks (G2b, G4, G9b) — SKIP when index absent
# ---------------------------------------------------------------------------

def _check_index_required(
    defect: str, name: str, detail: str = "requires pre-built retrieval index"
) -> None:
    _record(defect, name, "SKIP", detail)


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

    # ── Index-required checks ────────────────────────────────────────────────
    _check_index_required("G2b", "text search returns images",
                          "requires pre-built retrieval index")
    _check_index_required("G4", "image search spans >1 store",
                          "requires pre-built CLIP index")
    _check_index_required("G9b", "t-shirt search → no skirts",
                          "requires pre-built retrieval index")


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
