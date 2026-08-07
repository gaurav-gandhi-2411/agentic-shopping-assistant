"""Regression tests for the 2026-07-13 live-sweep Batch 2 trust-killer fixes.

Each test class targets one confirmed live bug from this batch:
  Part A — honest disclosure: "footwear for lehenga" / "jacket style lehenga" /
    "what's trending for wedding season 2026" got a confident, fabricated pitch
    instead of an honest hedge; a gibberish query on turn 2+ skipped the
    gibberish guard entirely (scoped to is_first_search only).
  Part B — "I don't have pricing information" claimed despite price_inr being
    present on every retrieved item.
  Part C — formality_softener ("something comfortable for sangeet dancing",
    "not too flashy" for wedding_guest) had no ranking effect on plain search
    or the outfit composer.
  Part D — price_qualifier ("cheap lehenga") had no ranking/filtering effect:
    cheapest item ranked 3rd of 5, an 11x-median-price outlier was included.
  Part E — "minimalist wedding guest dress" surfaced a literal "Night Dress"
    (genuine sleepwear) with no exclusion mechanism.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

from src.agents.graph import (
    _LOW_CONFIDENCE_MIN_ITEMS,
    _LOW_CONFIDENCE_SCORE_MULT,
    _apply_loungewear_gate,
    _apply_price_qualifier,
    _format_items_for_response,
    _gibberish_check_applies,
    _is_low_confidence_result,
    _query_names_unsupported_attribute,
    build_graph,
)
from src.agents.grounding import validate_response
from src.agents.outfit.composer import _score_candidates
from src.agents.outfit.slots import SANGEET_EMBELLISHMENT_KEYWORDS
from src.memory.conversation import ConversationMemory
from src.retrieval.dense_search import DenseRetriever
from src.retrieval.hybrid_search import _RELEVANCE_FLOOR, HybridRetriever
from src.retrieval.sparse_search import SparseRetriever

# ── Part A: gibberish guard beyond turn 1 ───────────────────────────────────


class TestGibberishGuardAppliesBeyondTurnOne:
    def test_turn_one_always_applies(self) -> None:
        assert _gibberish_check_applies(True, "asdkfjhqwoiuerlkj") is True
        assert _gibberish_check_applies(True, "red dress") is True

    def test_turn_two_plus_true_gibberish_still_applies(self) -> None:
        # Live gap: a keyboard-mash fragment injected mid-conversation
        # previously skipped the gibberish guard entirely because is_first_
        # search was False on turn 2+.
        assert _gibberish_check_applies(False, "asdkfjhqwoiuerlkj qwxyz") is True

    def test_turn_two_plus_colour_refinement_does_not_apply(self) -> None:
        # "in blue" is short but parse_intent extracts colour="blue" — must
        # never be routed into the gibberish check on a later turn.
        assert _gibberish_check_applies(False, "in blue") is False

    def test_turn_two_plus_occasion_refinement_does_not_apply(self) -> None:
        assert _gibberish_check_applies(False, "for a wedding") is False

    def test_turn_two_plus_budget_refinement_does_not_apply(self) -> None:
        assert _gibberish_check_applies(False, "under 2000") is False

    def test_turn_two_plus_zero_signal_nonsense_applies(self) -> None:
        assert _gibberish_check_applies(False, "zzxxccvvbb qwopiuytrewq") is True


# ── Part A: low-confidence result-set signal ────────────────────────────────


class TestLowConfidenceResultSignal:
    def test_empty_items_never_low_confidence(self) -> None:
        # The separate, stronger "zero_confidence" case handles this — see
        # search_node's search_meta wiring, not this function.
        assert _is_low_confidence_result([]) is False

    def test_high_score_and_enough_items_is_confident(self) -> None:
        items = [
            {"score": _RELEVANCE_FLOOR * 6},
            {"score": _RELEVANCE_FLOOR * 5},
            {"score": _RELEVANCE_FLOOR * 5},
        ]
        assert _is_low_confidence_result(items) is False

    def test_score_barely_above_floor_is_low_confidence(self) -> None:
        items = [
            {"score": _RELEVANCE_FLOOR * (_LOW_CONFIDENCE_SCORE_MULT - 0.5)},
            {"score": _RELEVANCE_FLOOR * 3},
            {"score": _RELEVANCE_FLOOR * 3},
        ]
        assert _is_low_confidence_result(items) is True

    def test_thin_item_count_is_low_confidence_even_with_high_score(self) -> None:
        items = [{"score": _RELEVANCE_FLOOR * 10}] * (_LOW_CONFIDENCE_MIN_ITEMS - 1)
        assert _is_low_confidence_result(items) is True

    def test_missing_score_key_treated_as_zero(self) -> None:
        items = [{}, {}, {}]
        assert _is_low_confidence_result(items) is True


class TestQueryNamesUnsupportedAttribute:
    """Fix #8: "jacket style lehenga" retrieves lehengas strongly enough to
    clear _is_low_confidence_result's score threshold (see that function's
    own docstring — a documented residual gap), so this is a SEPARATE,
    independent query-attribute-presence signal feeding the same hedge path,
    not a change to the score-based function."""

    def test_named_attribute_absent_from_all_items_is_unsupported(self) -> None:
        items = [
            {"detail_desc": "A flowy embroidered lehenga skirt", "display_name": "Lehenga"},
            {"detail_desc": "Silk lehenga with dupatta", "display_name": "Lehenga Set"},
        ]
        assert _query_names_unsupported_attribute("jacket style lehenga", items) is True

    def test_named_attribute_present_in_backing_text_is_supported(self) -> None:
        items = [
            {"detail_desc": "Jacket style lehenga with embroidered blazer", "display_name": ""},
        ]
        assert _query_names_unsupported_attribute("jacket style lehenga", items) is False

    def test_no_structural_attribute_named_never_flagged(self) -> None:
        items = [{"detail_desc": "Silk lehenga", "display_name": "Lehenga"}]
        assert _query_names_unsupported_attribute("embellished lehenga", items) is False

    def test_empty_items_never_flagged(self) -> None:
        assert _query_names_unsupported_attribute("jacket style lehenga", []) is False


# ── Part B: price_inr shown to the LLM + grounding exemption ───────────────


class TestPriceShownToLLM:
    def test_price_included_in_formatted_item_block(self) -> None:
        items = [{"display_name": "Red Saree", "price_inr": 2999.0}]
        formatted = _format_items_for_response(items)
        assert "₹2999" in formatted

    def test_missing_price_renders_blank_not_none(self) -> None:
        items = [{"display_name": "Red Saree", "price_inr": None}]
        formatted = _format_items_for_response(items)
        assert "None" not in formatted


class TestPriceGroundingExemption:
    """validate_response previously scrubbed a grounded "the price is ₹2999"
    sentence to a false "I don't have pricing information" fallback because
    the literal word "price" never appeared in an item's own field values."""

    _ITEMS = [{"display_name": "Red Saree", "price_inr": 2999.0, "colour": "red"}]
    _ITEMS_KURTA = [
        {"display_name": "Men Solid Cotton Kurta", "price_inr": 449.0, "colour": "white"}
    ]

    def test_price_word_scrubbed_without_exemption(self) -> None:
        resp = "The Red Saree is a great pick. The price is around 2999 rupees."
        cleaned, flags = validate_response(resp, self._ITEMS)
        assert "price:\\bprice\\b" in flags
        assert "I don't have pricing information" in cleaned

    def test_price_word_survives_with_exemption(self) -> None:
        resp = "The Red Saree is a great pick. The price is around 2999 rupees."
        cleaned, flags = validate_response(resp, self._ITEMS, allow_price_mentions=True)
        assert flags == []
        assert "2999" in cleaned

    def test_subjective_price_words_still_scrubbed_with_exemption(self) -> None:
        # A raw price number does not ground a claim that it's "on sale" or
        # "affordable" — those stay scrubbed even with allow_price_mentions.
        resp = "The Red Saree is on sale and very affordable."
        cleaned, flags = validate_response(resp, self._ITEMS, allow_price_mentions=True)
        assert flags
        assert "I don't have pricing information" in cleaned

    # 2026-07-19 fix (live bug: "cheap kurta for men" response falsely claimed
    # "I don't have pricing information" despite price_inr populated on every
    # returned item). Mechanism: "This affordable kurta at ₹449 is a great
    # pick" was scrubbed wholesale because "affordable" doesn't appear verbatim
    # in any item's field values — even though the SAME sentence cites a real
    # price (₹449) genuinely belonging to a returned item. A sentence whose own
    # rupee figure matches a real item price is now price-grounded outright,
    # regardless of which price vocabulary ("affordable"/"budget"/"cheaper"/
    # "on sale"/"discount") it's phrased with.
    def test_real_price_citation_survives_alongside_subjective_word(self) -> None:
        resp = "This affordable kurta at ₹449 is a great budget pick for everyday wear."
        cleaned, flags = validate_response(resp, self._ITEMS_KURTA, allow_price_mentions=True)
        assert flags == []
        assert "₹449" in cleaned
        assert "I don't have pricing information" not in cleaned

    def test_real_price_citation_still_scrubbed_without_price_mentions_flag(self) -> None:
        # The exemption is gated behind allow_price_mentions (respond_node's
        # existing opt-in) so validate_rationale's separate contract — cost/
        # cheaper/expensive/sale/discount stay scrubbed unconditionally there
        # (see its docstring) — is never silently loosened by this fix.
        resp = "This affordable kurta at ₹449 is a great budget pick for everyday wear."
        cleaned, flags = validate_response(resp, self._ITEMS_KURTA)
        assert flags
        assert "I don't have pricing information" in cleaned

    def test_fabricated_price_not_matching_any_item_still_scrubbed(self) -> None:
        # Guards against a hallucinated number: citing a rupee figure that does
        # NOT match any real item price must not be treated as grounded.
        resp = "This affordable kurta is only ₹99, a steal."
        cleaned, flags = validate_response(resp, self._ITEMS_KURTA, allow_price_mentions=True)
        assert flags
        assert "I don't have pricing information" in cleaned


# ── Part D: price_qualifier ranking ─────────────────────────────────────────


class TestPriceQualifierRanking:
    _POOL = [
        {"article_id": "A", "price_inr": 4150.0},
        {"article_id": "B", "price_inr": 4499.0},
        {"article_id": "C", "price_inr": 6400.0},
        {"article_id": "D", "price_inr": 6799.0},
        {"article_id": "E", "price_inr": 28900.0},  # outlier: ~4.9x median (5849)
    ]

    def test_cheap_sorts_ascending(self) -> None:
        out = _apply_price_qualifier(self._POOL, "cheap")
        prices = [it["price_inr"] for it in out]
        assert prices == sorted(prices)

    def test_cheap_excludes_genuine_outlier(self) -> None:
        # Live-proven repro shape ("cheap lehenga"): a ~4-11x-median outlier
        # must not appear in the final ranked list.
        out = _apply_price_qualifier(self._POOL, "cheap")
        assert "E" not in {it["article_id"] for it in out}
        assert out[0]["article_id"] == "A"  # cheapest ranks first

    def test_cheap_never_filters_to_fewer_than_two_pool_underflow_protected(self) -> None:
        pool = [
            {"article_id": "A", "price_inr": 500.0},
            {"article_id": "B", "price_inr": 50000.0},
        ]
        out = _apply_price_qualifier(pool, "cheap")
        assert len(out) == 2  # would drop to 1 without the >=2 floor

    def test_expensive_sorts_descending_no_exclusion(self) -> None:
        out = _apply_price_qualifier(self._POOL, "expensive")
        prices = [it["price_inr"] for it in out]
        assert prices == sorted(prices, reverse=True)
        assert out[0]["article_id"] == "E"  # outlier is NOT excluded for expensive

    def test_no_qualifier_is_noop(self) -> None:
        assert _apply_price_qualifier(self._POOL, None) == self._POOL

    def test_empty_items_is_noop(self) -> None:
        assert _apply_price_qualifier([], "cheap") == []

    def test_items_missing_price_sort_last_for_cheap(self) -> None:
        pool = [
            {"article_id": "A", "price_inr": None},
            {"article_id": "B", "price_inr": 500.0},
        ]
        out = _apply_price_qualifier(pool, "cheap")
        assert out[0]["article_id"] == "B"


# ── Part E: loungewear occasion gate ────────────────────────────────────────


class TestLoungewearOccasionGate:
    _NIGHT_DRESS = {"prod_name": "Green Geometric Printed Cotton Kaftan Night Dress"}
    _REAL_DRESS = {"prod_name": "Women's Printed V-Neck Black Dress"}

    def test_excluded_for_formal_wedding_tier_occasion(self) -> None:
        out = _apply_loungewear_gate([self._NIGHT_DRESS, self._REAL_DRESS], "wedding_guest")
        assert out == [self._REAL_DRESS]

    def test_untouched_for_casual_occasion(self) -> None:
        # A bare "night dress" query with no formal-occasion signal has a
        # legitimate reason to want these items — see is_loungewear_text.
        out = _apply_loungewear_gate([self._NIGHT_DRESS, self._REAL_DRESS], "casual")
        assert out == [self._NIGHT_DRESS, self._REAL_DRESS]

    def test_applies_across_every_formal_wedding_tier_occasion(self) -> None:
        for occasion in (
            "sangeet", "haldi", "mehendi", "reception", "engagement",
            "festive_puja", "traditional_ethnic", "wedding_guest",
        ):
            assert _apply_loungewear_gate([self._NIGHT_DRESS], occasion) == []

    def test_never_pool_underflow_protected_can_return_empty(self) -> None:
        # Deliberately NOT protected like every other search_node gate — a
        # sleepwear item is never an acceptable formal-occasion result, even
        # as the sole survivor. Fixes the exact "minimalist wedding guest
        # dress" repro where the nightgown was the only candidate.
        assert _apply_loungewear_gate([self._NIGHT_DRESS], "wedding_guest") == []


# ── Part C: formality_softener wiring through composer._score_candidates ───


def _kurta_candidate(article_id: str, prod_name: str) -> dict:
    return {
        "article_id": article_id,
        "product_type": "Kurta",
        "prod_name": prod_name,
        "display_name": prod_name,
        "detail_desc": "",
        "colour": "black",
        "gender": "women",
        "score": 0.5,
        "price_inr": 800.0,
        "store": "storea",
    }


class TestFormalityOverrideWiringThroughScoreCandidates:
    """composer._score_candidates must thread formality_override straight to
    fabric_score_delta (fabric_score_delta itself is already fully tested in
    tests/test_outfit_package.py — this only proves the WIRING gap this batch
    closes, mirroring test_body_type.py's body_type wiring test shape)."""

    _common_kwargs = {
        "query": "ethnic top",
        "slot_name": "top",
        "occasion_slug": "wedding_guest",  # zero base signal without override
        "gender": "women",
        "anchor_colour": "black",
        "seen_ids": set(),
        "seen_prod_colour": set(),
        "budget_remaining": None,
        "pairing_stats": None,
        "anchor_class": "ethnic_bottom",
        "seen_stores": None,
        "neutral_fallback_ids": set(),
    }

    def test_no_override_is_backward_compatible_zero_delta_for_wedding_guest(self) -> None:
        embellished = _kurta_candidate("E1", "Heavy Embroidered Sequin Bridal Kurta")
        scored = _score_candidates([embellished], **self._common_kwargs)
        assert scored  # survives every hard gate
        # fab_delta contributes 0.0 for wedding_guest with no override — score
        # is driven purely by base_score/colour_score/flywheel/body_type terms.
        base_score, item = scored[0]
        assert item["article_id"] == "E1"

    def test_comfortable_override_penalizes_embellished_favours_plain(self) -> None:
        embellished = _kurta_candidate("E1", "Heavy Embroidered Sequin Bridal Kurta")
        plain = _kurta_candidate("P1", "Cotton Floral Printed Kurta")
        scored = _score_candidates(
            [embellished, plain], **self._common_kwargs, formality_override="comfortable"
        )
        scored.sort(key=lambda t: t[0], reverse=True)
        assert scored[0][1]["article_id"] == "P1"  # plain outranks embellished

    def test_same_candidate_set_with_and_without_override(self) -> None:
        # Never a filter — only a score nudge (mirrors body_type's own
        # NEVER-FILTER invariant test in test_body_type.py).
        embellished = _kurta_candidate("E1", "Heavy Embroidered Sequin Bridal Kurta")
        plain = _kurta_candidate("P1", "Cotton Floral Printed Kurta")
        scored_without = _score_candidates([embellished, plain], **self._common_kwargs)
        scored_with = _score_candidates(
            [embellished, plain], **self._common_kwargs, formality_override="comfortable"
        )
        ids_without = {item["article_id"] for _, item in scored_without}
        ids_with = {item["article_id"] for _, item in scored_with}
        assert ids_without == ids_with == {"E1", "P1"}


# ── Full-pipeline integration tests (real unified index) ───────────────────
# Mirrors tests/test_cheaper_refinement.py and tests/test_respond_history_and_
# stylist_reply.py's fixture shape (each test file duplicates its own
# _unified_index fixture — established convention, not a shared conftest).

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


class _CapturingLLM:
    """Records every prompt passed to generate(); returns a fixed canned reply.

    Mirrors tests/test_respond_history_and_stylist_reply.py's stub exactly.
    """

    def __init__(self, reply: str = "Great pick! Here's why it works for you.") -> None:
        self.prompts: list[str] = []
        self._reply = reply

    def generate(self, prompt: str, system: str = None, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return self._reply

    def generate_stream(self, prompt: str, system: str = None, **kwargs: Any) -> Iterator[str]:
        self.prompts.append(prompt)
        yield self._reply

    def chat(self, messages: list[dict], **kwargs: Any) -> str:
        return self._reply

    def chat_stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        yield self._reply


@pytest.fixture(scope="module")
def _unified_index() -> tuple[HybridRetriever, pd.DataFrame]:
    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _blank_state(query: str, memory: Any) -> dict:
    return {
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


def _next_turn_state(prior_result: dict, query: str, memory: Any) -> dict:
    return {
        "messages": prior_result.get("messages", []) + [{"role": "user", "content": query}],
        "user_query": query,
        "current_plan": None,
        "tool_calls": [],
        "retrieved_items": prior_result.get("retrieved_items", []),
        "filters": prior_result.get("filters", {}),
        "final_answer": None,
        "iteration": 0,
        "new_items_this_turn": False,
        "out_of_catalogue": False,
        "excluded_colours": prior_result.get("excluded_colours"),
        "anchor_article_id": prior_result.get("anchor_article_id"),
        "outfit_rationale": None,
        "outfit_variants": None,
        "_memory": memory,
    }


@pytest.mark.requires_index
class TestLiveRepros:
    """End-to-end repros of the confirmed live bug queries against the real
    catalogue, via build_graph + a stub LLM (no live LLM API call — see PR
    notes for what this does and does not verify)."""

    def test_low_confidence_query_gets_weak_match_hedge(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        # This test's original repro query ("footwear for lehenga", 0 real
        # footwear as of 2026-07-13, max score 3.76x the relevance floor) went
        # stale: the 2026-08-06 retrieval filter-pushdown fix (pushing gender/
        # colour/price into fetch instead of post-filtering) genuinely
        # improved recall for exactly this kind of thin-inventory query, and
        # it now surfaces real festive footwear (juttis/heels marketed for
        # lehenga pairing) at 4.19x the floor — ABOVE the hedge threshold. A
        # good match now existing is the fix working as intended, not a bug;
        # confirmed on 2026-08-07 that plain master (pre-pushdown-fix) still
        # scores the old query 3.76x/hedges, and the fixed code no longer
        # does for that specific query.
        #
        # Swapped to "raincoat for toddlers" (verified 2026-08-07 against the
        # current index+pushdown fix together: max score 2.73x the floor,
        # still hedges) so this test keeps covering the actual thing it
        # exists for — that respond_node wires _is_low_confidence_result's
        # signal into the LLM prompt — rather than a specific query that
        # happened to be thin on one date. _is_low_confidence_result's own
        # threshold logic has separate, query-independent unit coverage above
        # (see TestLowConfidenceResultSignal), so this integration test's
        # only job is proving the wiring, not re-deriving the threshold.
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("raincoat for toddlers", memory))

        assert result.get("retrieved_items"), "precondition: search returns items"
        respond_prompt = llm.prompts[-1]
        assert "weak match" in respond_prompt

    def test_minimalist_wedding_guest_dress_gets_honest_canned_message(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        # Live-proven: the only "dress"-bucket item coherent for wedding_guest
        # was a literal "Kaftan Night Dress" (genuine sleepwear) — the
        # loungewear gate strips it, leaving zero items, which must produce
        # the honest canned message rather than any LLM improvisation.
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("minimalist wedding guest dress", memory))

        assert result.get("retrieved_items") == []
        assert "couldn't find a good match" in result.get("final_answer", "")

    def test_cheap_lehenga_sorts_ascending_and_excludes_outlier(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("cheap lehenga", memory))
        items = result.get("retrieved_items", [])
        prices = [it["price_inr"] for it in items if it.get("price_inr")]

        assert prices, "precondition: cheap lehenga returns priced items"
        assert prices == sorted(prices), "expected ascending price order"

    def test_gibberish_on_turn_two_still_gets_clarify(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Mid-conversation gap fix: a gibberish query on turn 2+ that still
        carries buy-signal intent ("show me...", reaching search_node via
        router_node's is_product_query gate — see IntentParser._BUY_SIGNAL_RE)
        but extracts ZERO structured signal (no garment/occasion/colour/
        budget/gender) must still be caught by search_node's own gibberish
        check, not silently searched with a confident pitch.

        A query with NEITHER buy-signal intent NOR any structured signal at
        all (e.g. bare "asdkfjhqwoiuerlkj zzxxccvv") previously never reached
        search_node at all on turn 2+ — router_node's own upstream
        is_product_query gate routed it straight to conversational "respond"
        with the PRIOR turn's stale retrieved_items, before search_node's own
        gibberish check could ever run. Fixed 2026-07-16: router_node's
        is_product_query=False branch now itself checks _is_unrecognized_query
        before falling to "respond" — see
        test_pure_gibberish_no_buy_signal_on_turn_two_gets_clarify below for
        that specific case.
        """
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        turn1_result = agent.invoke(_blank_state("red saree for wedding", memory))
        assert turn1_result.get("retrieved_items"), "precondition: turn 1 returns items"

        turn2_state = _next_turn_state(turn1_result, "show me qwxyzhjklp", memory)
        turn2_result = agent.invoke(turn2_state)

        assert turn2_result.get("out_of_catalogue") is True
        assert "didn't quite catch that" in turn2_result.get("final_answer", "")

    def test_pure_gibberish_no_buy_signal_on_turn_two_gets_clarify(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Batch 2 residual gap, fixed 2026-07-16: a query with NEITHER
        buy-signal intent NOR any structured signal at all
        ("asdkfjhqwoiuerlkj zzxxccvv") is not is_product_query, so it never
        reached search_node's own gibberish guard on turn 2+ — router_node
        routed it straight to "respond" over the PRIOR turn's stale
        retrieved_items, producing a confident LLM pitch for an unrelated
        item set. router_node's is_product_query=False branch now runs the
        same _is_unrecognized_query check before falling through to respond,
        routing true gibberish through the SAME deterministic
        search -> out_of_catalogue -> honest-clarify path search_node's own
        guard already used.
        """
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        turn1_result = agent.invoke(_blank_state("red saree for wedding", memory))
        assert turn1_result.get("retrieved_items"), "precondition: turn 1 returns items"

        turn2_state = _next_turn_state(turn1_result, "asdkfjhqwoiuerlkj zzxxccvv", memory)
        turn2_result = agent.invoke(turn2_state)

        assert turn2_result.get("out_of_catalogue") is True
        assert "didn't quite catch that" in turn2_result.get("final_answer", "")

    def test_conversational_reply_on_turn_two_not_treated_as_gibberish(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Explicit non-regression for the router_node-level gibberish check
        added alongside test_pure_gibberish_no_buy_signal_on_turn_two_gets_
        clarify above: genuine conversational replies that merely lack
        product signal (real English words, not keyboard-mash) must still
        reach the normal conversational "respond" path, not the clarify
        template."""
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        turn1_result = agent.invoke(_blank_state("red saree for wedding", memory))
        assert turn1_result.get("retrieved_items"), "precondition: turn 1 returns items"

        turn2_state = _next_turn_state(turn1_result, "thank you", memory)
        turn2_result = agent.invoke(turn2_state)

        assert turn2_result.get("out_of_catalogue") is not True

    def test_in_blue_refinement_on_turn_two_not_treated_as_gibberish(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Explicit non-regression: the mid-conversation gibberish gate must
        never false-positive on a legitimate short colour refinement."""
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        turn1_result = agent.invoke(_blank_state("red saree for wedding", memory))
        assert turn1_result.get("retrieved_items"), "precondition: turn 1 returns items"

        turn2_state = _next_turn_state(turn1_result, "in blue", memory)
        turn2_result = agent.invoke(turn2_state)

        assert turn2_result.get("out_of_catalogue") is not True
        assert turn2_result.get("retrieved_items")

    # ── Formality-aware occasion retrieval (2026-07-13 root-cause fix) ─────

    def test_comfortable_sangeet_dancing_surfaces_non_embellished_items(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Live-proven 2026-07-13: "something comfortable for sangeet dancing"
        correctly set occasion=sangeet + formality_softener=comfortable and
        fabric_score_delta correctly scored every embellished item -0.1, but
        the bug was upstream of that sort entirely — _OCCASION_QUERY_TERMS
        unconditionally appended the literal word "embellished" to the
        retrieval query whenever no garment type was present, so the
        candidate POOL itself was already 100% embellishment-biased (10/10)
        before the sort ever ran. Fixed by making the occasion query-
        expansion terms formality-aware: a comfortable/minimalist variant
        with no embellishment word is substituted when formality_softener
        requests it.
        """
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(
            _blank_state("something comfortable for sangeet dancing", memory)
        )
        items = result.get("retrieved_items", [])
        assert items, "precondition: query returns items"

        def _is_embellished(item: dict) -> bool:
            text = (
                (item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")
            ).lower()
            return any(kw in text for kw in SANGEET_EMBELLISHMENT_KEYWORDS)

        embellished_flags = [_is_embellished(it) for it in items]
        assert not all(embellished_flags), (
            "expected at least one non-embellished sangeet item to surface; "
            f"got {sum(embellished_flags)}/{len(items)} embellished"
        )
        # The non-embellished items must actually rank above the heavy ones,
        # not just be present somewhere in the pool — this is what the stable
        # sort couldn't do on its own when the pool was 100% biased.
        assert embellished_flags[0] is False

    def test_plain_sangeet_occasion_terms_unaffected(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Non-regression: a sangeet query with NO formality softener must
        keep the original (embellishment-favouring) occasion-term expansion —
        the formality-aware branch only fires when the user actually asked
        for something comfortable/minimalist."""
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("something nice for sangeet", memory))
        items = result.get("retrieved_items", [])
        assert items, "precondition: query returns items"

        def _is_embellished(item: dict) -> bool:
            text = (
                (item.get("prod_name") or "") + " " + (item.get("detail_desc") or "")
            ).lower()
            return any(kw in text for kw in SANGEET_EMBELLISHMENT_KEYWORDS)

        embellished_flags = [_is_embellished(it) for it in items]
        # Base sangeet behaviour still favours embellishment — most results
        # embellished, matching the pre-fix distribution (verified 9/10 live).
        assert sum(embellished_flags) >= len(items) - 2

    # ── Relational-noun gender guard, second map (2026-07-13 root-cause fix) ─

    def test_grooms_sister_outfit_ideas_does_not_force_menswear(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Live-proven 2026-07-13: "groom's sister outfit ideas" returned
        generic men's T-shirts/sweatshirts. Root cause was search_node's OWN
        local _GENDER_MAP (separate from intent_parser.py's and from
        outfit/partner.py's already-fixed _GROOM_RE/_BRIDE_RE) matching
        \\bgroom\\b inside "groom's" with no relational-noun guard, hard-
        setting index_group_name="menswear" via the raw-query regex fallback.
        Fixed by applying the same _RELATIONAL_NOUN_ALT negative-lookahead
        (imported from outfit.partner, the source of truth) to this second,
        independent map. No gender signal is asserted here by design — an
        unfiltered fallback is honest/acceptable for a deterministic parser
        with no real signal (see module docstring — inferring "sister"
        implies gender="women" is explicitly out of scope for this fix).
        """
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("groom's sister outfit ideas", memory))

        assert result.get("filters", {}).get("index_group_name") != "menswear"

    def test_sherwani_for_groom_still_filters_menswear(
        self, _unified_index: tuple[HybridRetriever, pd.DataFrame]
    ) -> None:
        """Explicit non-regression: a query that SHOULD infer men's gender
        from "groom" alone in a non-relational context must still correctly
        filter to menswear — mirrors outfit/partner.py's own "sherwani for
        groom" non-regression coverage for the sibling _GROOM_RE fix."""
        retriever, catalogue_df = _unified_index
        llm = _CapturingLLM()
        memory = ConversationMemory(llm, _MINIMAL_CONFIG)
        agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=False)

        result = agent.invoke(_blank_state("sherwani for groom", memory))

        assert result.get("filters", {}).get("index_group_name") == "menswear"
