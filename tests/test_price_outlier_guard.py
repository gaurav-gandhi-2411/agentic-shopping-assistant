"""Absurd price-outlier guard for unbudgeted outfit looks (2026-07-23).

Live-proven bug: an "eid outfit for men" look anchored on a ₹12,632 "Pastel
Seafoam Embroidered Kurta Pajama" picked "THE SHEHENSHAH TRADITIONAL NEHRU
WAISTCOAT" (store=rathore) at ₹3,19,999 for the outerwear slot — 25.3x the
anchor price, pushing the look total to ₹3,33,979 for a user who gave no
budget signal at all.

Root cause: the offending row's product_type_name is mislabeled "Fashion"
(not "waistcoat"), and that catalogue-wide bucket mixes cheap and ultra-
luxury items together (p99 ≈ ₹3,76,729) — a per-product-type percentile cap
would NOT catch this outlier since it sits inside that bucket's own "normal"
range. See src.agents.outfit.composer._PRICE_OUTLIER_FACTOR's module-level
docstring for the full design rationale.

Fixes verified here:
  A. composer._score_candidates rejects any candidate priced above
     price_outlier_cap as a hard gate (never a soft nudge).
  B. compose_outfit computes price_outlier_cap = anchor_price *
     _PRICE_OUTLIER_FACTOR ONLY when budget_inr is None and the anchor has a
     real catalogue price — skipped entirely for budgeted queries (an
     explicit high budget is an intentional request for expensive items) and
     for owned/uploaded anchors with no price to anchor against.
  C. End-to-end: an unbudgeted eid-men look anchored on the real live-repro
     item never contains a slot item priced above the cap; the SAME look
     WITH an explicit high budget can still surface an expensive complement.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.agents.outfit.composer import _PRICE_OUTLIER_FACTOR, _score_candidates

# ---------------------------------------------------------------------------
# A. _score_candidates price_outlier_cap hard gate
# ---------------------------------------------------------------------------


def _waistcoat_candidate(article_id: str, prod_name: str, price_inr: float) -> dict:
    return {
        "article_id": article_id,
        "product_type": "Fashion",  # mirrors the real mislabeled row
        "prod_name": prod_name,
        "display_name": prod_name,
        "detail_desc": "",
        "colour": "black",
        "gender": "men",
        "score": 0.5,
        "price_inr": price_inr,
        "store": "rathore",
    }


_COMMON_KWARGS: dict = {
    "query": "waistcoat nehru jacket",
    "slot_name": "outerwear",
    "occasion_slug": "eid",
    "gender": "men",
    "anchor_colour": "seafoam",
    "seen_ids": set(),
    "seen_prod_colour": set(),
    "budget_remaining": None,
    "pairing_stats": None,
    "anchor_class": "kurta_set",
    "seen_stores": None,
    "neutral_fallback_ids": set(),
}


class TestScoreCandidatesPriceOutlierCap:
    def test_candidate_above_cap_is_rejected(self) -> None:
        absurd = _waistcoat_candidate("W1", "THE SHEHENSHAH TRADITIONAL NEHRU WAISTCOAT", 319999.0)
        scored = _score_candidates([absurd], **_COMMON_KWARGS, price_outlier_cap=101056.0)
        assert scored == []

    def test_candidate_at_or_below_cap_survives(self) -> None:
        reasonable = _waistcoat_candidate("W2", "Cotton Nehru Waistcoat", 4500.0)
        scored = _score_candidates([reasonable], **_COMMON_KWARGS, price_outlier_cap=101056.0)
        assert len(scored) == 1
        assert scored[0][1]["article_id"] == "W2"

    def test_candidate_exactly_at_cap_survives(self) -> None:
        """Boundary check: price == cap must NOT be rejected (only price >
        cap is rejected — see _score_candidates' `if price > price_outlier_cap`)."""
        at_cap = _waistcoat_candidate("W3", "Cotton Nehru Waistcoat", 101056.0)
        scored = _score_candidates([at_cap], **_COMMON_KWARGS, price_outlier_cap=101056.0)
        assert len(scored) == 1

    def test_cap_none_is_full_noop(self) -> None:
        """price_outlier_cap=None (the default, and always the value passed
        for a budgeted query) never rejects anything on price grounds."""
        absurd = _waistcoat_candidate("W1", "THE SHEHENSHAH TRADITIONAL NEHRU WAISTCOAT", 319999.0)
        scored = _score_candidates([absurd], **_COMMON_KWARGS, price_outlier_cap=None)
        assert len(scored) == 1

    def test_mixed_pool_keeps_only_survivors(self) -> None:
        absurd = _waistcoat_candidate("W1", "THE SHEHENSHAH TRADITIONAL NEHRU WAISTCOAT", 319999.0)
        reasonable = _waistcoat_candidate("W2", "Cotton Nehru Waistcoat", 4500.0)
        scored = _score_candidates(
            [absurd, reasonable], **_COMMON_KWARGS, price_outlier_cap=101056.0
        )
        ids = {item["article_id"] for _, item in scored}
        assert ids == {"W2"}


# ---------------------------------------------------------------------------
# C. End-to-end: compose_outfit real-index integration
# ---------------------------------------------------------------------------

UNIFIED_DIR = Path("data/processed/unified")

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

# The exact live-repro anchor: "Pastel Seafoam Embroidered Kurta Pajama |
# TULA" (₹12,632, store=jadeblue) — see
# data/processed/unified/catalogue.parquet.
_EID_ANCHOR_ARTICLE_ID = "9935063286066"


@pytest.fixture(scope="module")
def _unified_index() -> tuple:
    from src.retrieval.dense_search import DenseRetriever
    from src.retrieval.hybrid_search import HybridRetriever
    from src.retrieval.sparse_search import SparseRetriever

    dense = DenseRetriever.load(_CONFIG, str(UNIFIED_DIR))
    sparse = SparseRetriever.load(_CONFIG, str(UNIFIED_DIR))
    catalogue_df = pd.read_parquet(UNIFIED_DIR / "catalogue.parquet")
    retriever = HybridRetriever(dense, sparse, catalogue_df, _CONFIG)
    return retriever, catalogue_df


@pytest.mark.requires_index
class TestComposeOutfitPriceOutlierRealIndex:
    def test_unbudgeted_eid_look_never_exceeds_price_cap(self, _unified_index) -> None:
        """The exact live-repro shape: no budget signal at all."""
        from src.agents.outfit.composer import compose_outfit

        retriever, catalogue_df = _unified_index
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=_EID_ANCHOR_ARTICLE_ID,
            occasion_slug="eid", gender="men", budget_inr=None,
        )
        anchor_price = look["seed_item"]["price_inr"]
        cap = anchor_price * _PRICE_OUTLIER_FACTOR
        for c in look["complements"]:
            assert (c.get("price_inr") or 0.0) <= cap, (
                f"{c.get('prod_name')} priced {c.get('price_inr')} exceeds "
                f"{_PRICE_OUTLIER_FACTOR}x anchor cap of {cap} "
                f"(anchor={anchor_price})"
            )

    def test_budgeted_high_query_can_still_surface_expensive_items(self, _unified_index) -> None:
        """An explicit high budget is an intentional request for expensive
        items — the price-outlier guard must not silently override it."""
        from src.agents.outfit.composer import compose_outfit

        retriever, catalogue_df = _unified_index
        anchor_price = 12632.0  # the real anchor's own catalogue price
        cap_that_would_apply_unbudgeted = anchor_price * _PRICE_OUTLIER_FACTOR
        look = compose_outfit(
            catalogue_df, retriever,
            seed_article_id=_EID_ANCHOR_ARTICLE_ID,
            occasion_slug="eid", gender="men", budget_inr=400000.0,
        )
        prices = [c.get("price_inr") or 0.0 for c in look["complements"]]
        assert prices, "expected at least one complement for a ₹400,000 budget"
        assert any(p > cap_that_would_apply_unbudgeted for p in prices), (
            "expected at least one complement priced above the unbudgeted "
            "cap when the user gave an explicit high budget — the guard "
            "must be a full no-op on the budgeted path"
        )
