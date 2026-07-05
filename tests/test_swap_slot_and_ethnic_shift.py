"""Tests for the "swap the {slot}" and "make this look more ethnic" refinement fixes,
plus the owned-anchor budget-accounting fix.

Covers:
- composer.swap_slot_in_look: replaces ONLY the named slot, keeps seed + other
  complements fixed, excludes the currently-shown item and any explicit exclusion
  ids, and returns None gracefully when the slot isn't in the current look or
  isn't valid for the anchor.
- composer._BiasedRetriever bias_mode="ethnic_shift": positive bias for ethnic
  garment types/keywords, negative for western, and (at the wrapper level) can
  promote an ethnic candidate back into a truncated top_k candidate set — without
  ever bypassing the gender/coherence gates in _find_best_candidate.
- composer.compose_outfit(owned_anchor=True): running_total/budget_total_inr
  excludes the seed's own price — the budget buys complements only.
- graph.py router_node + outfit_node integration (real unified index):
  "swap the {slot} in this look" changes only that slot; "make this look more
  ethnic" routes through the ethnic_shift bias without crashing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from src.agents.outfit.composer import (
    _BiasedRetriever,
    compose_biased_look,
    compose_outfit,
    swap_slot_in_look,
)

# ---------------------------------------------------------------------------
# Shared fixtures (mirrors tests/test_outfit_package.py / test_owned_anchor.py)
# ---------------------------------------------------------------------------


def _make_seed_catalogue_row(article_id: str, price_inr: float = 799.0) -> pd.DataFrame:
    """Single-row catalogue DataFrame for a western-top seed item."""
    return pd.DataFrame(
        [
            {
                "article_id": article_id,
                "prod_name": "White Shirt",
                "display_name": "White Shirt",
                "colour_group_name": "white",
                "product_type_name": "Shirt",
                "department_name": "Women",
                "index_group_name": "Ladieswear",
                "detail_desc": "",
                "image_url": None,
                "price_inr": price_inr,
                "pdp_handle": "white-shirt",
                "store": "myntra",
                "gender": "women",
                "facets": {
                    "colour_group_name": "white",
                    "product_type_name": "Shirt",
                    "department_name": "Women",
                },
            }
        ]
    )


class _FillSlotFakeRetriever:
    """Returns bottom + footwear complement candidates regardless of query text."""

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        if "trousers" in query or "jeans" in query or "skirt" in query:
            return [
                {
                    "article_id": "C1",
                    "prod_name": "Black Trousers",
                    "display_name": "Black Trousers",
                    "store": "myntra",
                    "colour": "black",
                    "product_type": "Trousers",
                    "detail_desc": "",
                    "score": 0.9,
                    "price_inr": 999.0,
                    "gender": "women",
                }
            ]
        if "sneakers" in query or "flats" in query or "heels" in query:
            return [
                {
                    "article_id": "C2",
                    "prod_name": "White Sneakers",
                    "display_name": "White Sneakers",
                    "store": "myntra",
                    "colour": "white",
                    "product_type": "Sneakers",
                    "detail_desc": "",
                    "score": 0.85,
                    "price_inr": 1200.0,
                    "gender": "women",
                }
            ]
        return []


def _compose_base_look(article_id: str = "SEED1", price_inr: float = 799.0) -> dict:
    catalogue_df = _make_seed_catalogue_row(article_id, price_inr=price_inr)
    retriever = _FillSlotFakeRetriever()
    return compose_outfit(
        catalogue_df,
        retriever,
        seed_article_id=article_id,
        occasion_slug="casual",
        gender="women",
    )


# ---------------------------------------------------------------------------
# swap_slot_in_look
# ---------------------------------------------------------------------------


class _SwapBottomFakeRetriever:
    """Bottom-slot query returns the currently-shown item plus one alternative."""

    def __init__(self, alt_price: float = 899.0) -> None:
        self._alt_price = alt_price

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        if "trousers" in query or "jeans" in query or "skirt" in query:
            return [
                {
                    "article_id": "C1",  # currently-shown bottom — must be excluded
                    "prod_name": "Black Trousers",
                    "display_name": "Black Trousers",
                    "store": "myntra",
                    "colour": "black",
                    "product_type": "Trousers",
                    "detail_desc": "",
                    "score": 0.95,
                    "price_inr": 999.0,
                    "gender": "women",
                },
                {
                    "article_id": "C3",
                    "prod_name": "Blue Jeans",
                    "display_name": "Blue Jeans",
                    "store": "myntra",
                    "colour": "blue",
                    "product_type": "Jeans",
                    "detail_desc": "",
                    "score": 0.9,
                    "price_inr": self._alt_price,
                    "gender": "women",
                },
                {
                    "article_id": "C4",
                    "prod_name": "Grey Skirt",
                    "display_name": "Grey Skirt",
                    "store": "myntra",
                    "colour": "grey",
                    "product_type": "Skirt",
                    "detail_desc": "",
                    "score": 0.6,
                    "price_inr": 500.0,
                    "gender": "women",
                },
            ]
        return []


class TestSwapSlotInLook:
    def test_swaps_only_named_slot_keeps_others_fixed(self) -> None:
        base = _compose_base_look()
        seed = base["seed_item"]
        complements = base["complements"]
        assert any(c["_slot"] == "bottom" for c in complements), "precondition: bottom slot filled"
        assert any(c["_slot"] == "footwear" for c in complements), "precondition: footwear filled"

        footwear_before = next(c for c in complements if c["_slot"] == "footwear")

        new_look = swap_slot_in_look(
            _SwapBottomFakeRetriever(),
            seed_item=seed,
            complements=complements,
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
        )

        assert new_look is not None
        assert new_look["seed_item"]["article_id"] == seed["article_id"]
        new_bottom = next(c for c in new_look["complements"] if c["_slot"] == "bottom")
        new_footwear = next(c for c in new_look["complements"] if c["_slot"] == "footwear")
        assert new_bottom["article_id"] != "C1", "bottom must change"
        assert new_footwear["article_id"] == footwear_before["article_id"], (
            "footwear (a slot not being swapped) must be unchanged"
        )

    def test_currently_shown_item_never_re_picked(self) -> None:
        base = _compose_base_look()
        new_look = swap_slot_in_look(
            _SwapBottomFakeRetriever(),
            seed_item=base["seed_item"],
            complements=base["complements"],
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
        )
        assert new_look is not None
        new_bottom = next(c for c in new_look["complements"] if c["_slot"] == "bottom")
        assert new_bottom["article_id"] == "C3", "highest-scored non-excluded candidate must win"

    def test_explicit_exclude_ids_are_honoured(self) -> None:
        base = _compose_base_look()
        new_look = swap_slot_in_look(
            _SwapBottomFakeRetriever(),
            seed_item=base["seed_item"],
            complements=base["complements"],
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
            exclude_article_ids={"C3"},
        )
        assert new_look is not None
        new_bottom = next(c for c in new_look["complements"] if c["_slot"] == "bottom")
        assert new_bottom["article_id"] == "C4", "explicitly excluded id must never be re-picked"

    def test_returns_none_when_slot_not_in_current_look(self) -> None:
        base = _compose_base_look()
        result = swap_slot_in_look(
            _SwapBottomFakeRetriever(),
            seed_item=base["seed_item"],
            complements=base["complements"],
            slot_name="accessory",  # not filled by _FillSlotFakeRetriever's look
            occasion_slug="casual",
            gender="women",
        )
        assert result is None

    def test_returns_none_when_no_alternative_candidate_found(self) -> None:
        class _EmptyRetriever:
            def search(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[dict]:  # noqa: ARG002
                return []

        base = _compose_base_look()
        result = swap_slot_in_look(
            _EmptyRetriever(),
            seed_item=base["seed_item"],
            complements=base["complements"],
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
        )
        assert result is None

    def test_owned_seed_excluded_from_budget_calculation(self) -> None:
        """An owned seed's price must not eat into the budget available for the
        swapped-in slot item (mirrors compose_outfit's owned_anchor behaviour)."""
        base = _compose_base_look(price_inr=5000.0)  # would blow any reasonable budget if counted
        seed = dict(base["seed_item"])
        seed["_owned"] = True

        new_look = swap_slot_in_look(
            _SwapBottomFakeRetriever(alt_price=899.0),
            seed_item=seed,
            complements=base["complements"],
            slot_name="bottom",
            occasion_slug="casual",
            gender="women",
            # footwear (1200) leaves 1000 remaining for the swapped bottom — enough
            # for C3 (899) but nowhere near enough if the owned seed's 5000 counted.
            budget_inr=2200.0,
        )
        assert new_look is not None, "owned seed's price must not count against the budget"
        new_bottom = next(c for c in new_look["complements"] if c["_slot"] == "bottom")
        assert new_bottom["article_id"] == "C3"


# ---------------------------------------------------------------------------
# owned-anchor budget fix (compose_outfit)
# ---------------------------------------------------------------------------


class TestOwnedAnchorBudgetExcludesSeedPrice:
    def test_owned_anchor_budget_total_excludes_seed_price(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED_OWNED_BUDGET", price_inr=5000.0)
        retriever = _FillSlotFakeRetriever()

        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_OWNED_BUDGET",
            occasion_slug="casual",
            gender="women",
            budget_inr=2500.0,  # would be blown instantly (5000 > 2500) if seed counted
            owned_anchor=True,
        )

        assert look["complements"], (
            "expected complements within budget — the owned seed's ₹5000 price must "
            "not count against the ₹2500 budget"
        )
        expected_total = sum(c.get("price_inr") or 0.0 for c in look["complements"])
        assert look["budget_total_inr"] == pytest.approx(expected_total), (
            "budget_total_inr must be complements-only when the seed is owned"
        )

    def test_non_owned_anchor_budget_total_includes_seed_price(self) -> None:
        """Regression guard: non-owned seeds still count toward the budget total,
        unchanged from prior behaviour."""
        catalogue_df = _make_seed_catalogue_row("SEED_NOT_OWNED_BUDGET", price_inr=799.0)
        retriever = _FillSlotFakeRetriever()

        look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_NOT_OWNED_BUDGET",
            occasion_slug="casual",
            gender="women",
        )
        expected_total = 799.0 + sum(c.get("price_inr") or 0.0 for c in look["complements"])
        assert look["budget_total_inr"] == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# _BiasedRetriever bias_mode="ethnic_shift"
# ---------------------------------------------------------------------------


class TestBiasedRetrieverEthnicShiftScoring:
    def _retriever(self) -> _BiasedRetriever:
        return _BiasedRetriever(
            retriever=None,  # not used by _bias_score directly
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
        )

    def test_ethnic_item_gets_positive_bias(self) -> None:
        item = {"product_type": "Palazzo", "prod_name": "Cotton Palazzo", "detail_desc": ""}
        assert self._retriever()._bias_score(item) == pytest.approx(0.15)

    def test_western_item_gets_negative_bias(self) -> None:
        item = {"product_type": "Jeans", "prod_name": "Blue Jeans", "detail_desc": ""}
        assert self._retriever()._bias_score(item) == pytest.approx(-0.1)

    def test_unclassified_item_gets_zero_bias(self) -> None:
        item = {"product_type": "Widget", "prod_name": "Mystery Item", "detail_desc": ""}
        assert self._retriever()._bias_score(item) == pytest.approx(0.0)


class _QueryRecordingFakeRetriever:
    """Records every query string handed to `.search()`; returns no candidates.

    Used to assert what text _BiasedRetriever actually sends downstream, without
    needing any candidate-selection logic.
    """

    def __init__(self) -> None:
        self.queries_seen: list[str] = []

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        self.queries_seen.append(query)
        return []


class TestBiasedRetrieverEthnicShiftQueryAugmentation:
    """Live-proven bug: get_fill_slots() (slots.py) derives search_query purely from
    the seed's anchor_class, never from bias mode — so a western-anchored look's
    slot pools were western-only regardless of "make this look more ethnic". The
    fix augments the query text itself for bias_mode="ethnic_shift" so ethnic
    candidates (kurta/palazzo/dupatta/jutti/...) actually exist in-pool.
    """

    def test_ethnic_shift_augments_bottom_query_with_ethnic_terms(self) -> None:
        fake = _QueryRecordingFakeRetriever()
        retriever = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
            gender="women",
        )
        retriever.search("trousers jeans skirt", top_k=20)
        assert fake.queries_seen, "underlying retriever must have been called"
        sent = fake.queries_seen[0]
        assert "palazzo" in sent and "churidar" in sent and "salwar" in sent

    def test_ethnic_shift_augments_outerwear_query_with_ethnic_terms(self) -> None:
        fake = _QueryRecordingFakeRetriever()
        retriever = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
            gender="women",
        )
        retriever.search("jacket blazer coat cardigan", top_k=20)
        assert "nehru jacket" in fake.queries_seen[0]

    def test_ethnic_shift_augments_accessory_query_with_ethnic_terms(self) -> None:
        fake = _QueryRecordingFakeRetriever()
        retriever = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
            gender="women",
        )
        retriever.search("handbag sling bag earrings women accessory", top_k=20)
        assert "dupatta" in fake.queries_seen[0]

    def test_ethnic_shift_augments_footwear_query_with_ethnic_terms(self) -> None:
        fake = _QueryRecordingFakeRetriever()
        retriever = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
            gender="women",
        )
        retriever.search("sneakers flats heels casual shoes women", top_k=20)
        assert "juttis" in fake.queries_seen[0]

    def test_ethnic_shift_query_augmentation_is_gendered_for_men(self) -> None:
        fake = _QueryRecordingFakeRetriever()
        retriever = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
            gender="men",
        )
        retriever.search("jacket blazer coat cardigan", top_k=20)
        sent = fake.queries_seen[0]
        assert "waistcoat" in sent
        assert "shrug" not in sent, "shrug is the women's-branch term, must not leak into men's"

    def test_non_ethnic_shift_bias_modes_do_not_augment_the_query(self) -> None:
        for bias_mode in ("formality_shift", "alternate_colour"):
            fake = _QueryRecordingFakeRetriever()
            retriever = _BiasedRetriever(
                retriever=fake,
                bias_mode=bias_mode,
                base_complement_colours=set(),
                occasion_slug="casual",
                gender="women",
            )
            retriever.search("trousers jeans skirt", top_k=20)
            assert fake.queries_seen[0] == "trousers jeans skirt", (
                f"bias_mode={bias_mode} must not touch the query text"
            )

    def test_already_ethnic_query_is_returned_unchanged(self) -> None:
        """A query that already carries no western hint words (e.g. slots.py's own
        ethnic branch query "kurta ethnic top") must pass through unmodified."""
        fake = _QueryRecordingFakeRetriever()
        retriever = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
            gender="women",
        )
        retriever.search("kurta ethnic top", top_k=20)
        assert fake.queries_seen[0] == "kurta ethnic top"


class _EthnicVsWesternPoolFakeRetriever:
    """Models a real hybrid retriever's text-similarity behaviour: candidates for a
    given slot are only returned when the QUERY TEXT actually contains matching
    vocabulary. A purely western query ("trousers jeans skirt") never surfaces the
    ethnic candidate — proving the live-proven bug (western-only slot queries mean
    no ethnic candidate is ever in-pool for the bias to promote). Once the query is
    augmented with ethnic terms, the ethnic candidate is returned and — modelling a
    strong text match — scores higher than the western competitor.
    """

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        q = query.lower()
        results: list[dict] = []
        if any(w in q for w in ("trousers", "jeans", "skirt")):
            results.append(
                {
                    "article_id": "WEST_BOTTOM",
                    "prod_name": "Blue Jeans",
                    "display_name": "Blue Jeans",
                    "colour": "blue",
                    "product_type": "Jeans",
                    "gender": "women",
                    "score": 0.5,
                    "price_inr": 999.0,
                    "store": "myntra",
                    "detail_desc": "",
                }
            )
        if "palazzo" in q:
            results.append(
                {
                    "article_id": "ETH_BOTTOM",
                    "prod_name": "Cotton Palazzo",
                    "display_name": "Cotton Palazzo",
                    "colour": "blue",
                    "product_type": "Palazzo",
                    "gender": "women",
                    "score": 0.9,
                    "price_inr": 899.0,
                    "store": "myntra",
                    "detail_desc": "",
                }
            )
        if any(w in q for w in ("bag", "handbag", "sling", "earrings")):
            results.append(
                {
                    "article_id": "WEST_ACCESSORY",
                    "prod_name": "Sling Bag",
                    "display_name": "Sling Bag",
                    "colour": "black",
                    "product_type": "Bag",
                    "gender": "women",
                    "score": 0.5,
                    "price_inr": 599.0,
                    "store": "myntra",
                    "detail_desc": "",
                }
            )
        if "dupatta" in q:
            results.append(
                {
                    "article_id": "ETH_ACCESSORY",
                    "prod_name": "Silk Dupatta",
                    "display_name": "Silk Dupatta",
                    "colour": "red",
                    "product_type": "Dupatta",
                    "gender": "women",
                    "score": 0.9,
                    "price_inr": 499.0,
                    "store": "myntra",
                    "detail_desc": "",
                }
            )
        return results


class TestEthnicShiftRecomposePullsEthnicCandidates:
    """End-to-end: compose_biased_look(bias_mode="ethnic_shift") must actually
    select ethnic-typed complements when the fake pool contains both western and
    ethnic candidates, keyed by whether ethnic vocabulary appears in the query —
    directly reproducing the live "make this look more ethnic" recompose."""

    def test_ethnic_shift_recompose_pulls_ethnic_candidates(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED_ETHNIC_SHIFT")
        retriever = _EthnicVsWesternPoolFakeRetriever()
        base_look = compose_outfit(
            catalogue_df,
            retriever,
            seed_article_id="SEED_ETHNIC_SHIFT",
            occasion_slug="casual",
            gender="women",
        )
        # Precondition: without the ethnic-shift bias, the base look is all-western
        # (mirrors the live bug's blouse/blazer/bag result).
        base_types = {c["product_type"] for c in base_look["complements"]}
        assert "Palazzo" not in base_types and "Dupatta" not in base_types

        ethnic_look = compose_biased_look(
            catalogue_df=catalogue_df,
            retriever=retriever,
            base_look=base_look,
            seed_article_id="SEED_ETHNIC_SHIFT",
            occasion_slug="casual",
            gender="women",
            budget_inr=None,
            pairing_stats=None,
            brand_gender_default="women",
            bias_mode="ethnic_shift",
        )

        assert ethnic_look is not None
        ethnic_names = {
            (c.get("prod_name") or "").lower() for c in ethnic_look["complements"]
        }
        assert any(
            "palazzo" in n or "dupatta" in n or "kurta" in n for n in ethnic_names
        ), f"expected an ethnic-typed complement, got: {ethnic_names}"


class _LargeWesternPoolFakeRetriever:
    """25 western candidates (score 0.5) rank ahead of 1 ethnic candidate (score 0.5)
    purely by list order — models a real retriever whose text-similarity ranking for
    a western-slanted query ("trousers jeans skirt") naturally favours western hits.
    """

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        western = [
            {
                "article_id": f"WEST{i}",
                "prod_name": "Blue Jeans",
                "product_type": "Jeans",
                "colour": "blue",
                "gender": "women",
                "score": 0.5,
                "price_inr": 900.0,
                "store": "myntra",
                "detail_desc": "",
            }
            for i in range(25)
        ]
        ethnic = [
            {
                "article_id": "ETH1",
                "prod_name": "Cotton Palazzo",
                "product_type": "Palazzo",
                "colour": "blue",
                "gender": "women",
                "score": 0.5,
                "price_inr": 800.0,
                "store": "myntra",
                "detail_desc": "",
            }
        ]
        return western + ethnic


class TestBiasedRetrieverEthnicShiftPromotesPastTruncation:
    def test_unbiased_top_k_would_exclude_the_ethnic_item(self) -> None:
        """Precondition: naive top_k=20 slicing (no bias) drops the ethnic item —
        it's ranked 26th out of 26 candidates."""
        fake = _LargeWesternPoolFakeRetriever()
        naive_top20 = fake.search("trousers jeans skirt")[:20]
        assert "ETH1" not in {it["article_id"] for it in naive_top20}

    def test_ethnic_shift_bias_promotes_ethnic_item_into_top_k(self) -> None:
        fake = _LargeWesternPoolFakeRetriever()
        biased = _BiasedRetriever(
            retriever=fake,
            bias_mode="ethnic_shift",
            base_complement_colours=set(),
            occasion_slug="casual",
        )
        result = biased.search("trousers jeans skirt", top_k=20)
        assert "ETH1" in {it["article_id"] for it in result}, (
            "ethnic_shift bias must promote the ethnic candidate back into the "
            "truncated top_k candidate set handed to _find_best_candidate"
        )


# ---------------------------------------------------------------------------
# compose_biased_look(bias_mode="ethnic_shift") never bypasses the gender gate
# ---------------------------------------------------------------------------


class _GenderMismatchFakeRetriever:
    """Bottom-slot query returns a men's-only ethnic candidate (must be gender-gated
    out for a women's look) and a women's western candidate."""

    def search(
        self, query: str, top_k: int = 20, filters: dict | None = None  # noqa: ARG002
    ) -> list[dict]:
        if "trousers" in query or "jeans" in query or "skirt" in query:
            return [
                {
                    "article_id": "ETHM",
                    "prod_name": "Ethnic Palazzo",
                    "display_name": "Ethnic Palazzo",
                    "colour": "blue",
                    "product_type": "Palazzo",
                    "gender": "men",  # wrong gender for a women's look
                    "score": 0.99,
                    "price_inr": 899.0,
                    "store": "myntra",
                    "detail_desc": "",
                },
                {
                    "article_id": "WESTW",
                    "prod_name": "Blue Jeans",
                    "display_name": "Blue Jeans",
                    "colour": "blue",
                    "product_type": "Jeans",
                    "gender": "women",
                    "score": 0.5,
                    "price_inr": 999.0,
                    "store": "myntra",
                    "detail_desc": "",
                },
            ]
        return []


class TestComposeBiasedLookEthnicShiftGating:
    def test_ethnic_shift_never_bypasses_gender_gate(self) -> None:
        catalogue_df = _make_seed_catalogue_row("SEED_GATE")
        retriever = _GenderMismatchFakeRetriever()
        base_look = compose_outfit(
            catalogue_df, retriever, seed_article_id="SEED_GATE",
            occasion_slug="casual", gender="women",
        )
        ethnic_look = compose_biased_look(
            catalogue_df=catalogue_df,
            retriever=retriever,
            base_look=base_look,
            seed_article_id="SEED_GATE",
            occasion_slug="casual",
            gender="women",
            budget_inr=None,
            pairing_stats=None,
            brand_gender_default="women",
            bias_mode="ethnic_shift",
        )
        assert ethnic_look is not None
        bottom = next(c for c in ethnic_look["complements"] if c["_slot"] == "bottom")
        assert bottom["article_id"] == "WESTW", (
            "a men's-only candidate must stay gender-gated out even with a strong "
            "ethnic-shift bias in its favour"
        )


# ---------------------------------------------------------------------------
# graph.py router_node + outfit_node integration (real unified index)
# ---------------------------------------------------------------------------

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
    """Deliberately returns a WRONG (non-outfit) router decision so tests only pass
    if the swap-slot / ethnic-shift routing is fully deterministic."""

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
def _unified_index() -> tuple:
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    dense = DenseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    sparse = SparseRetriever.load(_MINIMAL_CONFIG, UNIFIED_DIR)
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _MINIMAL_CONFIG)
    return retriever, catalogue_df


def _blank_state(query: str, memory) -> dict:
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


def _next_turn_state(prior_result: dict, query: str, memory) -> dict:
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
def test_swap_slot_changes_only_named_slot(_unified_index: tuple) -> None:
    """"swap the {slot} in this look" must change ONLY that slot's item, keeping the
    seed and every other complement's article_id identical.
    """
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "casual top"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    turn1_state = _blank_state("put together a casual look for women", memory)
    turn1_result = agent.invoke(turn1_state)
    assert turn1_result.get("look_id"), "precondition: turn 1 must compose a look"

    turn1_items = turn1_result.get("retrieved_items", [])
    complements = [it for it in turn1_items if it.get("_role") == "complement"]
    assert complements, "precondition: at least one complement slot must be filled"
    target_slot = complements[0]["_slot"]
    other_ids_before = {
        it["article_id"] for it in turn1_items
        if not (it.get("_role") == "complement" and it.get("_slot") == target_slot)
    }
    swapped_id_before = complements[0]["article_id"]

    turn2_state = _next_turn_state(
        turn1_result, f"swap the {target_slot} in this look", memory
    )
    turn2_result = agent.invoke(turn2_state)

    assert turn2_result.get("look_id"), (
        f"expected a look_id on the swap turn, tool_calls={turn2_result.get('tool_calls')}"
    )
    turn2_items = turn2_result.get("retrieved_items", [])
    other_ids_after = {
        it["article_id"] for it in turn2_items
        if not (it.get("_role") == "complement" and it.get("_slot") == target_slot)
    }
    swapped_item_after = next(
        (it for it in turn2_items if it.get("_slot") == target_slot), None
    )

    assert other_ids_after == other_ids_before, (
        "swap must leave every non-target item's article_id unchanged"
    )
    assert swapped_item_after is not None, f"expected a new {target_slot} item"
    assert swapped_item_after["article_id"] != swapped_id_before, (
        "the swapped slot's article_id must change"
    )


@pytest.mark.requires_index
def test_swap_unknown_slot_responds_gracefully(_unified_index: tuple) -> None:
    """An unmapped slot word must produce an honest "couldn't find" message rather
    than crashing or silently recomposing the whole look."""
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "casual top"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    turn1_state = _blank_state("put together a casual look for women", memory)
    turn1_result = agent.invoke(turn1_state)
    assert turn1_result.get("look_id"), "precondition: turn 1 must compose a look"

    turn2_state = _next_turn_state(
        turn1_result, "swap the gizmo in this look", memory
    )
    turn2_result = agent.invoke(turn2_state)

    # streaming_mode=True hands the honest message off via current_plan
    # ("pending_answer") rather than final_answer directly — mirrors how
    # outfit_node/respond_node/clarify_node all behave in streaming mode.
    plan = json.loads(turn2_result.get("current_plan") or "{}")
    assert plan.get("action") == "pending_answer", (
        f"expected an honest pending_answer message, not a crash; got {turn2_result}"
    )
    assert "gizmo" in plan.get("text", "").lower()


@pytest.mark.requires_index
def test_make_this_look_more_ethnic_recomposes_without_crashing(_unified_index: tuple) -> None:
    """"Make this look more ethnic" must route through the ethnic_shift bias and
    produce a new look (>=1 item + look_id) rather than collapsing into the
    formality_shift default.
    """
    from src.agents.graph import build_graph
    from src.memory.conversation import ConversationMemory

    retriever, catalogue_df = _unified_index
    llm = _MockLLM([json.dumps({"action": "search", "query": "casual top"})])
    memory = ConversationMemory(llm, _MINIMAL_CONFIG)
    agent = build_graph(retriever, catalogue_df, llm, _MINIMAL_CONFIG, streaming_mode=True)

    turn1_state = _blank_state("put together a casual look for women", memory)
    turn1_result = agent.invoke(turn1_state)
    assert turn1_result.get("look_id"), "precondition: turn 1 must compose a look"

    turn2_state = _next_turn_state(turn1_result, "Make this look more ethnic", memory)
    turn2_result = agent.invoke(turn2_state)

    assert turn2_result.get("look_id"), (
        f"expected a look_id on the ethnic-shift refinement turn, "
        f"tool_calls={turn2_result.get('tool_calls')}"
    )
    turn2_items = turn2_result.get("retrieved_items", [])
    assert len(turn2_items) >= 1
