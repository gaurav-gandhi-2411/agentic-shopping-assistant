"""Wave 9 — gym/activewear occasion model expansion.

Adds "gym" as a first-class occasion slug (formality=1, EITHER-lean),
following the exact haldi/sangeet/office (wave 7) and diwali/navratri/
karva_chauth/raksha_bandhan/eid (wave 8, commit a7e5b09) pattern across the 6
standard touchpoints: OCCASIONS registry (occasions.py), keyword routing
(intent_parser.py's _OCCASION_MAP), retrieval register tokens + bottom/
footwear query overrides (slots.py), anchor query text (composer.py), LLM
rationale hint (rationale.py), and graph.py's deterministic pre-LLM fast-path
routing regexes.

Two requirements make gym structurally different from every prior occasion,
and are the actual point of this module (see the dedicated test classes
below):

  1. No ethnic/festive bleed — coherence.py gate 5 (a NEW
     _ATHLETIC_REGISTER_OCCASIONS gate, alongside the existing
     _WESTERN_REGISTER_OCCASIONS/office gate 4, not folded into it — gym's
     footwear-specific rule below has no office equivalent) rejects ethnic
     items and festive/quirky markers, identical in shape to gate 4.

  2. No loungewear/sleepwear leak — graph.py's _apply_loungewear_gate trigger
     set is extended from _FORMAL_ETHNIC_OCCASIONS to
     _LOUNGEWEAR_GATE_OCCASIONS (= _FORMAL_ETHNIC_OCCASIONS | {"gym"}). gym is
     deliberately NOT added to _FORMAL_ETHNIC_OCCASIONS itself (that set also
     drives "footwear required", and gym's footwear stays OPTIONAL). Verified
     against the REAL unified catalogue: "comfortable loose clothes for gym"
     surfaces 3 literal "Nightdress" rows in the raw top-40 retrieval pool
     before the gate runs (see TestLoungewearGateCoversGymRealIndex) — this is
     not a hypothetical risk.

  3. Honest footwear suppression, not substitution — coherence.py gate 5 also
     rejects any non-athletic-typed footwear from a gym look's footwear slot
     (via slots.is_athletic_footwear_item). Catalogue audit against the real
     unified catalogue: 0 women's, ~20 men's (all store=flipkart) rows match
     the athletic-footwear pattern. gym is NOT added to
     _FORMAL_ETHNIC_OCCASIONS, so footwear stays optional — when no genuine
     athletic shoe survives the gates, the slot goes through the EXISTING
     honest-suppression mechanism (composer._suppression_reason /
     suppressed_slots) instead of falling back to a jutti/mojari/oxford/heel.

No network/LLM calls in the deterministic sections — pure unit tests against
occasions.py/intent_parser.py/composer.py/slots.py/coherence.py/rationale.py/
graph.py, mirroring tests/test_festival_occasions.py's structure. The two
real-index classes are @pytest.mark.requires_index and mirror
tests/test_occasion_merchandise_leak.py's / tests/test_price_outlier_guard.py's
real-WS/real-retriever harness patterns.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.agents.graph import (
    _LOUNGEWEAR_GATE_OCCASIONS,
    _OCCASION_LOOK_RE,
    _OUTFIT_INTENT_RE,
    _OUTFIT_OCCASION_RE,
    _apply_athletic_footwear_gate,
)
from src.agents.intent_parser import parse_intent
from src.agents.outfit.coherence import (
    colour_score,
    is_athletic_register_occasion,
    is_coherent_candidate,
)
from src.agents.outfit.composer import _anchor_query_for_occasion
from src.agents.outfit.occasions import EITHER, OCCASIONS, get_occasion
from src.agents.outfit.rationale import _OCCASION_REGISTER_HINTS
from src.agents.outfit.slots import (
    _FORMAL_ETHNIC_OCCASIONS,
    _default_bottom_query,
    _occasion_register_tokens,
    is_athletic_footwear_item,
)

# ── (a) OCCASIONS contains "gym" with the expected formality/ethnic_lean ────


class TestGymOccasionPresent:
    def test_gym_formality_and_ethnic_lean(self) -> None:
        occ = OCCASIONS["gym"]
        assert occ.formality == 1
        assert occ.ethnic_lean == EITHER

    def test_get_occasion_resolves_gym(self) -> None:
        assert get_occasion("gym").slug == "gym"


# ── (b) intent parsing for gym/workout free text ─────────────────────────────


class TestIntentParsingGym:
    def test_gym(self) -> None:
        assert parse_intent("gym outfit for women").occasion == "gym"

    def test_workout_single_word(self) -> None:
        assert parse_intent("workout clothes for women").occasion == "gym"

    def test_work_out_two_words(self) -> None:
        assert parse_intent("what to wear to work out").occasion == "gym"

    def test_athleisure(self) -> None:
        assert parse_intent("athleisure outfit").occasion == "gym"

    def test_athletic_wear(self) -> None:
        assert parse_intent("athletic wear for men").occasion == "gym"

    def test_yoga_resolves_to_gym_slug(self) -> None:
        """yoga deliberately shares the "gym" slug rather than getting its
        own — see occasions.py's "gym" entry docstring."""
        assert parse_intent("yoga outfit for women").occasion == "gym"

    def test_work_alone_still_resolves_to_office(self) -> None:
        """"work" (existing office keyword) must not be shadowed by the new
        "work out"/"workout" entries — regression guard."""
        assert parse_intent("outfit for work").occasion == "office"

    def test_workout_not_falsely_matched_as_work(self) -> None:
        """Word-boundary regression: "workout" is ONE word, must resolve to
        gym, not accidentally short-circuit on the "work" substring."""
        assert parse_intent("workout gear").occasion == "gym"


# ── (c) anchor query non-empty + contains signature tokens ──────────────────


class TestAnchorQueryForGym:
    def test_gym_query_women(self) -> None:
        query = _anchor_query_for_occasion("gym", "women")
        assert query
        assert "sports bra" in query and "leggings" in query

    def test_gym_query_men(self) -> None:
        query = _anchor_query_for_occasion("gym", "men")
        assert query
        assert "joggers" in query and "athletic" in query


# ── (d) footwear stays OPTIONAL for gym (never added to _FORMAL_ETHNIC_OCCASIONS) ──


def test_gym_not_in_formal_ethnic_occasions() -> None:
    """Footwear must stay optional (matches casual/office precedent) — the
    honest-suppression requirement (point 3) depends on this: required=False
    lets compose_outfit leave the slot silently absent+suppressed instead of
    an empty_slots hard failure."""
    assert "gym" not in _FORMAL_ETHNIC_OCCASIONS


# ── (e) colour_score falls through to the generic EITHER branch (no override) ──


class TestColourScoreNoOverride:
    def test_gym_falls_through_to_generic_either_branch(self) -> None:
        """No dedicated palette override — same light-touch treatment as
        raksha_bandhan."""
        neutral_score = colour_score("white", "red", "gym")
        assert neutral_score == 1.0


# ── (f) occasion register tokens + bottom-query override ────────────────────


class TestGymRegisterAndQueries:
    def test_register_tokens(self) -> None:
        assert _occasion_register_tokens("gym") == "activewear athletic gym sport"

    def test_default_bottom_query_biased_to_activewear(self) -> None:
        query = _default_bottom_query("gym")
        assert "leggings" in query
        assert "joggers" in query
        # Must NOT retain the generic "trousers jeans skirt" fallback text —
        # that would actively steer retrieval toward the wrong register.
        assert "skirt" not in query


# ── (g) rationale register hint present ──────────────────────────────────────


def test_rationale_register_hint_present_for_gym() -> None:
    assert "gym" in _OCCASION_REGISTER_HINTS
    assert _OCCASION_REGISTER_HINTS["gym"]


# ── (h) coherence gate 5: athletic-register rejects ethnic + festive markers ─


class TestAthleticRegisterCoherenceGate:
    def test_is_athletic_register_occasion_helper(self) -> None:
        assert is_athletic_register_occasion("gym") is True
        assert is_athletic_register_occasion("office") is False
        assert is_athletic_register_occasion("casual") is False

    def test_ethnic_kurta_rejected_from_gym_top(self) -> None:
        item = {"product_type": "kurta", "prod_name": "Blue Cotton Kurta", "gender": "women"}
        assert not is_coherent_candidate(item, "gym", "women", "top")

    def test_ethnic_lehenga_rejected_from_gym_bottom(self) -> None:
        item = {"product_type": "lehenga", "prod_name": "Red Silk Lehenga", "gender": "women"}
        assert not is_coherent_candidate(item, "gym", "women", "bottom")

    def test_festive_marker_rejected(self) -> None:
        item = {"product_type": "top", "prod_name": "Quirky Printed Gym Tank Top",
                "gender": "women"}
        assert not is_coherent_candidate(item, "gym", "women", "top")

    def test_western_activewear_top_passes(self) -> None:
        item = {"product_type": "sports_bra", "prod_name": "Ultimate Printed Comfort Sports Bra",
                "gender": "women"}
        assert is_coherent_candidate(item, "gym", "women", "top")

    def test_western_activewear_bottom_passes(self) -> None:
        item = {"product_type": "leggings", "prod_name": "Ultimate Printed Leggings",
                "gender": "women"}
        assert is_coherent_candidate(item, "gym", "women", "bottom")


# ── (i) coherence gate 5: footwear-specific athletic-only rule ──────────────


class TestAthleticFootwearGate:
    def test_jutti_rejected_from_gym_footwear(self) -> None:
        item = {"product_type": "footwear", "prod_name": "Golden Embroidered Juttis",
                "gender": "women"}
        assert not is_coherent_candidate(item, "gym", "women", "footwear")

    def test_mojari_rejected_from_gym_footwear(self) -> None:
        item = {"product_type": "footwear", "prod_name": "Tan Leather Mojaris",
                "gender": "men"}
        assert not is_coherent_candidate(item, "gym", "men", "footwear")

    def test_formal_oxford_rejected_from_gym_footwear(self) -> None:
        item = {"product_type": "footwear", "prod_name": "Black Formal Oxford Shoes",
                "gender": "men"}
        assert not is_coherent_candidate(item, "gym", "men", "footwear")

    def test_wedding_heel_rejected_from_gym_footwear(self) -> None:
        item = {"product_type": "footwear", "prod_name": "Red Stiletto Heels",
                "gender": "women"}
        assert not is_coherent_candidate(item, "gym", "women", "footwear")

    def test_running_shoe_accepted(self) -> None:
        item = {"product_type": "footwear", "prod_name": "Running Shoes For Men (Black)",
                "gender": "men"}
        assert is_coherent_candidate(item, "gym", "men", "footwear")

    def test_sneaker_accepted(self) -> None:
        item = {"product_type": "footwear", "prod_name": "Sneakers For Men (Black)",
                "gender": "men"}
        assert is_coherent_candidate(item, "gym", "men", "footwear")

    def test_non_footwear_slot_unaffected_by_footwear_rule(self) -> None:
        """The athletic-footwear-only rule is scoped to slot_name=="footwear"
        only — a genuine western top must not be rejected by it."""
        item = {"product_type": "sports_bra", "prod_name": "Ultimate Printed Comfort Sports Bra",
                "gender": "women"}
        assert is_coherent_candidate(item, "gym", "women", "top")


class TestIsAthleticFootwearItem:
    def test_sneaker_true(self) -> None:
        assert is_athletic_footwear_item("Sneakers For Men (Black)") is True

    def test_running_shoes_true(self) -> None:
        assert is_athletic_footwear_item("RS-5006 Running Shoes For Men (Black, Yellow)") is True

    def test_training_gym_shoes_true(self) -> None:
        assert is_athletic_footwear_item("Training & Gym Shoes For Men (White)") is True

    def test_jutti_false(self) -> None:
        assert is_athletic_footwear_item("Golden Embroidered Juttis") is False

    def test_mojari_false(self) -> None:
        assert is_athletic_footwear_item("Tan Leather Mojaris") is False

    def test_oxford_false(self) -> None:
        assert is_athletic_footwear_item("Black Formal Oxford Shoes") is False

    def test_heel_false(self) -> None:
        assert is_athletic_footwear_item("Red Stiletto Heels") is False

    def test_none_and_empty_safe(self) -> None:
        assert is_athletic_footwear_item(None) is False
        assert is_athletic_footwear_item("") is False


# ── (j) loungewear gate trigger set includes gym ─────────────────────────────


def test_loungewear_gate_occasions_includes_gym() -> None:
    assert "gym" in _LOUNGEWEAR_GATE_OCCASIONS


def test_loungewear_gate_occasions_still_includes_formal_ethnic_set() -> None:
    """The gym addition is additive — every pre-existing formal-ethnic
    occasion is untouched."""
    assert _FORMAL_ETHNIC_OCCASIONS.issubset(_LOUNGEWEAR_GATE_OCCASIONS)


# ── (k) deterministic pre-LLM outfit-routing fast path covers gym/workout ───


def _routes_to_outfit(query: str) -> bool:
    return bool(
        _OUTFIT_OCCASION_RE.search(query)
        and (_OUTFIT_INTENT_RE.search(query) or _OCCASION_LOOK_RE.search(query))
    )


class TestGymFastPathRouting:
    def test_gym_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("gym outfit") is True

    def test_gym_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("gym look for women") is True

    def test_workout_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("workout outfit") is True

    def test_work_out_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("work out look") is True

    def test_athleisure_outfit_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("athleisure outfit") is True

    def test_yoga_look_routes_to_outfit(self) -> None:
        assert _routes_to_outfit("yoga look") is True


# ── (l) POINT 2: real-index loungewear-leak regression (requires_index) ─────


class TestLoungewearGateCoversGymRealIndex:
    """Reproduces the exact risk class this project was live-bitten by before
    ("minimalist wedding guest dress" surfacing a literal nightgown, see
    graph.py's _apply_loungewear_gate docstring) for the gym occasion.

    "comfortable loose clothes for gym" is verified (offline, against the
    real unified catalogue's raw retrieval pool, before any gate runs) to
    surface 3 literal "Nightdress"-typed rows in the top-40 window — this is
    not a hypothetical risk being guarded against defensively; it is a real,
    reproducible retrieval-pool collision between activewear's soft/
    comfortable vocabulary and loungewear's own soft/comfortable vocabulary.
    """

    @staticmethod
    def _run_search(query: str):
        from src.agents.graph import build_graph
        from src.memory.conversation import ConversationMemory
        from src.retrieval.dense_search import DenseRetriever
        from src.retrieval.hybrid_search import HybridRetriever
        from src.retrieval.sparse_search import SparseRetriever

        unified_dir = "data/processed/unified"
        config: dict = {
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
        dense = DenseRetriever.load(config, unified_dir)
        sparse = SparseRetriever.load(config, unified_dir)
        catalogue_df = pd.read_parquet(f"{unified_dir}/catalogue.parquet")
        retriever = HybridRetriever(dense, sparse, catalogue_df, config)

        llm = _RealIndexMockLLM(["Here you go."] * 5)
        memory = ConversationMemory(llm, config)
        agent = build_graph(retriever, catalogue_df, llm, config, streaming_mode=True)

        state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
            "current_plan": None,
            "tool_calls": [],
            "retrieved_items": [],
            "filters": {},
            "final_answer": None,
            "iteration": 0,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
            "anchor_article_id": None,
            "outfit_rationale": None,
            "outfit_variants": None,
            "_memory": memory,
        }
        return agent.invoke(state)

    def _assert_no_loungewear(self, query: str) -> None:
        from src.catalogue.cleaning import is_loungewear_text

        result = self._run_search(query)
        items = result.get("retrieved_items", [])
        assert items, f"expected items for {query!r}"
        leaked = [
            it for it in items
            if is_loungewear_text(it.get("prod_name") or it.get("display_name") or "")
        ]
        assert not leaked, (
            f"loungewear item(s) leaked into {query!r} results: "
            f"{[it.get('prod_name') for it in leaked]}"
        )

    @pytest.mark.requires_index
    def test_comfortable_loose_clothes_for_gym_no_loungewear_leak(self) -> None:
        """The exact query shape verified (offline) to surface nightdresses in
        the raw pre-gate retrieval pool."""
        self._assert_no_loungewear("comfortable loose clothes for gym")

    @pytest.mark.requires_index
    def test_gym_wear_for_women_no_loungewear_leak(self) -> None:
        self._assert_no_loungewear("gym wear for women")

    @pytest.mark.requires_index
    def test_workout_clothes_for_women_no_loungewear_leak(self) -> None:
        self._assert_no_loungewear("workout clothes for women")


# ── (m) POINT 3: real-index footwear honest-suppression (requires_index) ────


class TestGymFootwearHonestSuppressionRealIndex:
    """Composes a real gym look for women against the real unified catalogue
    and asserts EITHER the footwear slot is absent with a suppression reason
    recorded, OR (if present) the item is a genuine athletic-typed shoe —
    never a jutti/mojari/oxford/heel. Mirrors
    tests/test_price_outlier_guard.py's compose_outfit real-index harness.
    """

    UNIFIED_DIR = "data/processed/unified"
    _CONFIG: dict = {
        "retrieval": {
            "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dense_dim": 384,
            "rrf_k": 60,
            "top_k": 50,
            "final_k": 10,
            "store_diversity": 0.2,
        },
    }
    # Real women's activewear anchors from the wave-9 catalogue merge
    # (data/processed/unified/catalogue.parquet, store=blissclub).
    _SPORTS_BRA_ANCHOR = "9327951347968"  # "Ultimate Printed Comfort Sports Bra"
    _LEGGINGS_ANCHOR = "9327940075776"  # "Ultimate Printed Leggings"

    @pytest.fixture(scope="class")
    def _unified_index(self):
        from src.retrieval.dense_search import DenseRetriever
        from src.retrieval.hybrid_search import HybridRetriever
        from src.retrieval.sparse_search import SparseRetriever

        dense = DenseRetriever.load(self._CONFIG, self.UNIFIED_DIR)
        sparse = SparseRetriever.load(self._CONFIG, self.UNIFIED_DIR)
        catalogue_df = pd.read_parquet(f"{self.UNIFIED_DIR}/catalogue.parquet")
        retriever = HybridRetriever(dense, sparse, catalogue_df, self._CONFIG)
        return retriever, catalogue_df

    def _assert_honest_footwear(self, look: dict) -> None:
        footwear = [c for c in look["complements"] if c.get("_slot") == "footwear"]
        if not footwear:
            reasons = {s["slot"]: s["reason"] for s in look["suppressed_slots"]}
            assert "footwear" in reasons, (
                "footwear slot is absent but carries no suppression reason — "
                "silent drop, not honest suppression"
            )
            assert reasons["footwear"]
            return
        assert len(footwear) == 1
        assert is_athletic_footwear_item(footwear[0].get("prod_name") or ""), (
            f"gym look's footwear slot filled with a non-athletic item: "
            f"{footwear[0].get('prod_name')!r}"
        )

    @pytest.mark.requires_index
    def test_unbudgeted_womens_gym_look_footwear_honest(self, _unified_index) -> None:
        from src.agents.outfit.composer import compose_outfit

        retriever, catalogue_df = _unified_index
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=self._SPORTS_BRA_ANCHOR,
            occasion_slug="gym", gender="women", budget_inr=None,
        )
        self._assert_honest_footwear(look)

    @pytest.mark.requires_index
    def test_tightly_budgeted_womens_gym_look_footwear_honest(self, _unified_index) -> None:
        """A tight budget (₹800) is far more likely to genuinely exhaust the
        already-thin women's athletic-footwear inventory — exercises the
        suppression path, not just the "happy path found a sneaker" path."""
        from src.agents.outfit.composer import compose_outfit

        retriever, catalogue_df = _unified_index
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=self._LEGGINGS_ANCHOR,
            occasion_slug="gym", gender="women", budget_inr=800.0,
        )
        self._assert_honest_footwear(look)

    @pytest.mark.requires_index
    def test_mens_gym_look_footwear_honest(self, _unified_index) -> None:
        """Men's inventory has real athletic-footwear depth (~20 rows,
        store=flipkart) — this exercises the "genuine athletic shoe found"
        branch of the same honest assertion."""
        from src.agents.outfit.composer import compose_outfit

        retriever, catalogue_df = _unified_index
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=None,
            occasion_slug="gym", gender="men", budget_inr=None,
        )
        self._assert_honest_footwear(look)


# ── (n) _apply_athletic_footwear_gate — plain-search path fix ───────────────
#
# Live-proven bug (2026-07-24, revision asa-stylist-api-00086-5qh): "gym shoes
# for women under 1500" (garment_type="footwear" + occasion="gym", no "look"/
# "outfit" framing) routes through search_node's PLAIN SEARCH path, not
# compose_outfit — and that path's own coherence check (_occ_gated in
# graph.py's search_node) always calls is_coherent_candidate with
# slot_name="top", so gate 5's footwear-specific athletic-only rule never
# fired. It returned literal formal heels ("Black Sierra Heels", "Sarah
# Tie-up Heels", store=houseofvian) and the LLM's own reply called them "a
# bit of a stretch for a gym shoe search" while still showing them.
# _apply_athletic_footwear_gate (graph.py) closes this gap by reusing
# is_athletic_footwear_item directly, keyed on garment_type=="footwear" +
# occasion=="gym" (via is_athletic_register_occasion).


_HEEL_ITEM = {"product_type": "footwear", "prod_name": "Black Sierra Heels", "gender": "women"}
_TIEUP_HEEL_ITEM = {"product_type": "footwear", "prod_name": "Sarah Tie-up Heels", "gender": "women"}
_JUTTI_ITEM = {"product_type": "footwear", "prod_name": "Golden Embroidered Juttis", "gender": "women"}
_SNEAKER_ITEM = {"product_type": "footwear", "prod_name": "Sneakers For Men (Black)", "gender": "men"}
_RUNNING_SHOE_ITEM = {
    "product_type": "footwear", "prod_name": "RS-5006 Running Shoes For Men (Black, Yellow)",
    "gender": "men",
}


class TestApplyAthleticFootwearGate:
    def test_gym_footwear_query_strips_non_athletic(self) -> None:
        """The exact live-reproduced item shapes: formal heels stripped,
        genuine athletic shoes survive."""
        items = [_HEEL_ITEM, _TIEUP_HEEL_ITEM, _JUTTI_ITEM, _SNEAKER_ITEM, _RUNNING_SHOE_ITEM]
        out = _apply_athletic_footwear_gate(items, "gym", "footwear")
        assert _HEEL_ITEM not in out
        assert _TIEUP_HEEL_ITEM not in out
        assert _JUTTI_ITEM not in out
        assert _SNEAKER_ITEM in out
        assert _RUNNING_SHOE_ITEM in out

    def test_non_athletic_register_occasion_is_noop(self) -> None:
        items = [_HEEL_ITEM]
        out = _apply_athletic_footwear_gate(items, "office", "footwear")
        assert out == items

    def test_none_occasion_is_noop(self) -> None:
        items = [_HEEL_ITEM]
        out = _apply_athletic_footwear_gate(items, None, "footwear")
        assert out == items

    def test_gym_non_footwear_garment_type_is_noop(self) -> None:
        """A gym query for a non-footwear garment (e.g. "gym top") must never
        be touched by this gate — it is footwear-slot-specific only."""
        items = [{"product_type": "top", "prod_name": "Quirky Printed Gym Tank Top"}]
        out = _apply_athletic_footwear_gate(items, "gym", "top")
        assert out == items

    def test_gym_no_garment_type_is_noop(self) -> None:
        """garment_type=None (no explicit footwear noun, e.g. "gym look for
        women") is the compose_outfit path's territory, already covered by
        coherence.py's gate 5 — this gate deliberately no-ops so it never
        double-applies or interferes with that path."""
        items = [_HEEL_ITEM]
        out = _apply_athletic_footwear_gate(items, "gym", None)
        assert out == items

    def test_pool_not_underflow_protected(self) -> None:
        """Mirrors _apply_loungewear_gate / _apply_occasion_merchandise_gate's
        discipline: a non-athletic shoe is never an acceptable gym-shoe
        result even as a last resort — this MAY legitimately empty the
        result list."""
        items = [_HEEL_ITEM, _JUTTI_ITEM]
        out = _apply_athletic_footwear_gate(items, "gym", "footwear")
        assert out == []


# ── (o) POINT 1 (fix 1): real-index end-to-end plain-search regression ──────


class TestAthleticFootwearGateRealIndex:
    """Reproduces the live bug end-to-end against the real unified catalogue,
    mirroring tests/test_occasion_merchandise_leak.py's / this file's own
    TestLoungewearGateCoversGymRealIndex's real-WS harness pattern.
    """

    @staticmethod
    def _run_search(query: str):
        from src.agents.graph import build_graph
        from src.memory.conversation import ConversationMemory
        from src.retrieval.dense_search import DenseRetriever
        from src.retrieval.hybrid_search import HybridRetriever
        from src.retrieval.sparse_search import SparseRetriever

        unified_dir = "data/processed/unified"
        config: dict = {
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
        dense = DenseRetriever.load(config, unified_dir)
        sparse = SparseRetriever.load(config, unified_dir)
        catalogue_df = pd.read_parquet(f"{unified_dir}/catalogue.parquet")
        retriever = HybridRetriever(dense, sparse, catalogue_df, config)

        llm = _RealIndexMockLLM(["Here you go."] * 5)
        memory = ConversationMemory(llm, config)
        agent = build_graph(retriever, catalogue_df, llm, config, streaming_mode=True)

        state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
            "current_plan": None,
            "tool_calls": [],
            "retrieved_items": [],
            "filters": {},
            "final_answer": None,
            "iteration": 0,
            "new_items_this_turn": False,
            "out_of_catalogue": False,
            "excluded_colours": None,
            "anchor_article_id": None,
            "outfit_rationale": None,
            "outfit_variants": None,
            "_memory": memory,
        }
        return agent.invoke(state)

    @pytest.mark.requires_index
    def test_gym_shoes_for_women_under_1500_no_non_athletic_footwear(self) -> None:
        """The exact live-reproduced query. Verified offline (see this
        module's docstring/commit) that the catalogue carries ~0 genuine
        women's athletic-footwear rows, so the honest outcome is zero items
        — never a non-athletic substitute like the live-proven heels."""
        result = self._run_search("gym shoes for women under 1500")
        items = result.get("retrieved_items", [])
        non_athletic = [
            it for it in items
            if not is_athletic_footwear_item(it.get("prod_name") or it.get("display_name") or "")
        ]
        assert not non_athletic, (
            f"non-athletic footwear leaked into 'gym shoes for women under "
            f"1500': {[it.get('prod_name') for it in non_athletic]}"
        )

    @pytest.mark.requires_index
    def test_gym_shoes_unbudgeted_men_returns_genuine_athletic_inventory(self) -> None:
        """The contrast case this fix must not regress: men's side has real
        athletic-footwear depth (~20 rows, store=flipkart) — the gate must be
        a genuine-athletic-only filter, never a blanket suppressor."""
        result = self._run_search("gym shoes for men")
        items = result.get("retrieved_items", [])
        assert items, "expected genuine athletic footwear results for 'gym shoes for men'"
        for it in items:
            assert is_athletic_footwear_item(it.get("prod_name") or it.get("display_name") or ""), (
                f"non-athletic item surfaced for men's gym shoes: {it.get('prod_name')!r}"
            )


class _RealIndexMockLLM:
    """Minimal fixed-response LLM stub — mirrors
    tests/test_occasion_merchandise_leak.py's identical helper."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def _next(self) -> str:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r

    def generate(self, prompt: str, system: str = None, **kwargs) -> str:
        return self._next()

    def generate_stream(self, prompt: str, system: str = None, **kwargs):
        yield self._next()

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self._next()

    def chat_stream(self, messages: list[dict], **kwargs):
        yield self._next()
