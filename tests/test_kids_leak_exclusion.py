"""Kids/juniors-item leak into adult results — P0 trust/safety bug (2026-07-12).

Live-proven bug: "red lehenga bridal" -> girls' lehengas ("Bitiya by Bhama
Girls...", "Cutiekins Girls...") ranked ABOVE adult bridal options. Both
live-reproduced queries are NON-occasion-keyword queries ("bridal" is not in
intent_parser._OCCASION_MAP), so the (now-removed) occasion-gated kids check in
graph.py never fired for them.

Root causes covered here:
  1. src.catalogue.adapter.derive_item_gender used to treat "girl"/"girls"/
     "boy"/"boys" as ADULT gender keywords, converting a kids signal into a
     false adult gender="women"/"men" label.
  2. src.retrieval.hybrid_search.HybridRetriever did not exclude kids items
     from its BM25 retrieval window at all.
  3. src.agents.graph's plain-search node only excluded kids items when an
     occasion keyword was present in the query.

Fixes verified here:
  A. derive_item_gender no longer promotes girl(s)/boy(s)/kid(s)/junior(s) text
     to an adult gender (unless a legitimate garment-noun/gender-word keyword
     is also present, or a brand default applies).
  B. Word-boundary-safe keyword matching so short adult terms ("man") don't
     false-positive inside unrelated words ("Roman").
  C. is_kids_item is importable from src.catalogue.cleaning (promoted from
     src.agents.outfit.slots so the retrieval layer can use it without an
     agents-layer import).
  D. HybridRetriever.search excludes kids items from its BM25 window
     unconditionally, for both occasion and non-occasion queries.
"""
from __future__ import annotations

import unittest.mock as mock

import pandas as pd
import pytest

from src.catalogue.adapter import derive_item_gender
from src.catalogue.cleaning import is_kids_item
from src.retrieval.hybrid_search import HybridRetriever

_DEFAULT_CONFIG: dict = {
    "retrieval": {
        "final_k": 10,
        "top_k": 10,
        "rrf_k": 60,
        "store_diversity": 0.0,
    }
}


def _make_catalogue(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal catalogue_df from a list of row dicts.

    Mirrors tests/test_hybrid_search_store_gender.py's helper of the same name.
    """
    full_rows = []
    for r in rows:
        article_id = r.get("article_id", "unknown")
        defaults = {
            "display_name": f"Item {article_id}",
            "prod_name": f"Item {article_id}",
            "detail_desc": "desc",
            "image_url": None,
            "price_inr": 500.0,
            "pdp_handle": None,
            "pdp_live": True,
            "product_type_name": "lehenga",
            "facets": {
                "colour_group_name": "Red",
                "product_type_name": "lehenga",
                "department_name": "Ladieswear",
            },
        }
        merged = {**defaults, **r}
        full_rows.append(merged)
    return pd.DataFrame(full_rows)


def _make_retriever(cat_df: pd.DataFrame, config: dict | None = None) -> HybridRetriever:
    """Build a HybridRetriever whose dense+sparse mocks return every row in order."""
    article_ids = cat_df["article_id"].tolist()
    dense_mock = mock.MagicMock()
    dense_mock.search.return_value = [(aid, 1.0 / (i + 1)) for i, aid in enumerate(article_ids)]
    sparse_mock = mock.MagicMock()
    sparse_mock.search.return_value = [(aid, 1.0 / (i + 1)) for i, aid in enumerate(article_ids)]
    return HybridRetriever(dense_mock, sparse_mock, cat_df, config or _DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# A. derive_item_gender no longer promotes kids markers to an adult gender
# ---------------------------------------------------------------------------


class TestDeriveItemGenderKidsMarkers:
    def test_girls_item_with_no_other_adult_signal_is_unknown(self) -> None:
        """A pure kids title with no brand default and no garment-noun overlap
        must not be assigned an adult gender just because it says "Girls"."""
        result = derive_item_gender("Girls Blue Cotton Top", "top", "unknown")
        assert result == "unknown"

    def test_boys_item_with_no_other_adult_signal_is_unknown(self) -> None:
        result = derive_item_gender("Boys Blue Cotton Shirt", "shirt", "unknown")
        assert result == "unknown"

    def test_girls_item_still_falls_back_to_explicit_brand_default(self) -> None:
        """Brand-level defaults are a legitimate, separate signal — untouched by
        this fix. Only the girl(s)/boy(s) KEYWORD must stop granting adult
        gender; a real brand default is still honoured."""
        result = derive_item_gender("Girls Blue Cotton Top", "top", "women")
        assert result == "women"

    def test_girls_lehenga_still_women_via_garment_noun(self) -> None:
        """"lehenga" is a legitimate women's-garment keyword independent of the
        kids marker — a girls' lehenga is still labeled gender="women" by
        derive_item_gender itself (that's root cause #1's whole point: the
        gender column alone can't be trusted to exclude kids items, which is
        why is_kids_item exists as a SEPARATE, additional exclusion)."""
        result = derive_item_gender(
            "Bitiya by Bhama Girls Green Net Embellished Lehenga Choli", "lehenga", "unknown"
        )
        assert result == "women"

    def test_boyfriend_jeans_no_longer_misclassified_as_men(self) -> None:
        """Word-boundary-safety regression: "Boyfriend Jeans" contains "boy" as
        a substring but must never be treated as a kids/men signal — this was a
        real collision risk with the old girl/boy adult-keyword membership."""
        result = derive_item_gender("Boyfriend Straight Fit Jeans", "jeans", "women")
        assert result == "women"

    def test_word_boundary_protects_short_adult_keywords_from_substrings(self) -> None:
        """"man" must not match inside "Roman" — regression guard for the
        word-boundary-safe matching introduced alongside the keyword-set fix."""
        result = derive_item_gender("Roman Sandals", "sandals", "unknown")
        assert result == "unknown"

    def test_explicit_women_signal_still_assigns_women(self) -> None:
        result = derive_item_gender("Women Black Solid A-Line Dress", "dress", "unknown")
        assert result == "women"

    def test_explicit_men_signal_still_assigns_men(self) -> None:
        result = derive_item_gender("Men Blue Slim Fit Shirt", "shirt", "unknown")
        assert result == "men"


# ---------------------------------------------------------------------------
# C. is_kids_item promoted to src.catalogue.cleaning
# ---------------------------------------------------------------------------


class TestIsKidsItemPromoted:
    def test_girls_marker_detected(self) -> None:
        assert is_kids_item("Bitiya by Bhama GIRLS Green Net Embellished Lehenga Choli") is True

    def test_boys_marker_detected(self) -> None:
        assert is_kids_item("Boys Blue Cotton Shirt") is True

    def test_kids_marker_detected(self) -> None:
        assert is_kids_item("Kids Party Wear Dress") is True

    def test_junior_marker_detected(self) -> None:
        assert is_kids_item("M&H Juniors Girls Blue Straight Knee Length Denim Skirts") is True

    def test_adult_item_not_flagged(self) -> None:
        assert is_kids_item("Women Black Solid A-Line Dress") is False

    def test_empty_and_none_safe(self) -> None:
        assert is_kids_item("") is False
        assert is_kids_item(None) is False


# ---------------------------------------------------------------------------
# F. is_kids_item detail_desc phrase extension (2026-08-10, occasion-register
#    wave Cluster C) — narrow "kids + garment noun" phrase check, separate
#    from the bare prod_name word scan above.
# ---------------------------------------------------------------------------


class TestIsKidsItemDescPhrase:
    def test_kids_phrase_in_desc_detected(self) -> None:
        """Real miss: name carries no kids marker at all; desc opens with an
        explicit kids+garment-noun phrase."""
        assert is_kids_item(
            "White Multicolor Cotton Blend Printed Kurta Pyjama Set",
            "Kids Printed Kurta Pyjama Set crafted from soft cotton blend.",
        ) is True

    def test_kids_boys_set_phrase_detected(self) -> None:
        assert is_kids_item(
            "Set of 2: Off-White check Printed Shirt with Off-white check Printed Capri",
            "This Off-White Check Printed Shirt & Capri Kids Boys Set of 2 is a fun pick.",
        ) is True

    def test_bare_kids_word_far_from_garment_noun_not_flagged(self) -> None:
        """The narrow phrase pattern requires kids within ~20 chars of a
        tracked garment noun — a distant/unrelated "kids" mention (e.g. an
        age-range disclaimer) must not trigger it."""
        assert is_kids_item(
            "Classic Cotton T-Shirt",
            "Wear for every range of age groups from Boys, Girls, and Gents alike.",
        ) is False

    def test_no_detail_desc_falls_back_to_name_only(self) -> None:
        """Omitting detail_desc preserves the original prod_name-only
        behaviour exactly — backward compatible with every pre-existing
        call site that doesn't pass it."""
        assert is_kids_item("Women Black Solid A-Line Dress") is False
        assert is_kids_item("Boys Blue Cotton Shirt") is True

    def test_empty_detail_desc_safe(self) -> None:
        assert is_kids_item("Women Black Solid A-Line Dress", "") is False
        assert is_kids_item("Women Black Solid A-Line Dress", None) is False


# ---------------------------------------------------------------------------
# D. HybridRetriever excludes kids items unconditionally (non-occasion query)
# ---------------------------------------------------------------------------


class TestHybridSearchKidsExclusion:
    """HybridRetriever.search excludes kids items from the BM25 (sparse)
    retrieval window — mirrors the _not_fabric_mask/_not_inactive_store_mask
    pattern exactly (BM25-window-only; the mock dense retriever below returns
    every row regardless of the mask, same as real FAISS is never pre-filtered
    by this mask in production — dense-path leakage is the reason
    src.agents.graph's unconditional post-hoc is_kids_item strip exists, see
    TestSearchNodeKidsExclusionRealIndex below for the true end-to-end guarantee).
    """

    def test_kids_item_excluded_from_bm25_allowed_ids_no_type_filter(self) -> None:
        """Reproduces the live bug's retrieval-layer half: "red lehenga bridal"
        is a non-occasion-keyword query, and the catalogue mislabels the girls'
        lehenga as gender="women" (root cause #1) — so a naive gender filter
        alone does not exclude it from the BM25 window. Must fail before the
        hybrid_search.py mask fix and pass after."""
        cat_df = _make_catalogue([
            {
                "article_id": "kids1",
                "gender": "women",
                "prod_name": "Bitiya by Bhama Girls Green Net Embellished Lehenga Choli",
                "display_name": "Bitiya by Bhama Girls Green Net Embellished Lehenga Choli",
            },
            {
                "article_id": "adult1",
                "gender": "women",
                "prod_name": "Red Embellished Bridal Lehenga Choli",
                "display_name": "Red Embellished Bridal Lehenga Choli",
            },
        ])
        retriever = _make_retriever(cat_df)
        retriever.search("red lehenga bridal", top_k=10, filters={"gender": "women"})
        allowed_ids = retriever.sparse.search.call_args.kwargs["allowed_ids"]
        assert allowed_ids is not None
        assert "kids1" not in allowed_ids
        assert "adult1" in allowed_ids

    def test_kids_item_excluded_from_bm25_allowed_ids_no_filters_at_all(self) -> None:
        """Exclusion is unconditional — not tied to any filter being set."""
        cat_df = _make_catalogue([
            {
                "article_id": "kids1",
                "gender": "women",
                "prod_name": "Cutiekins Girls Pink Party Lehenga",
                "display_name": "Cutiekins Girls Pink Party Lehenga",
            },
            {
                "article_id": "adult1",
                "gender": "women",
                "prod_name": "Gold Jewellery Set to go with Red Lehenga",
                "display_name": "Gold Jewellery Set to go with Red Lehenga",
            },
        ])
        retriever = _make_retriever(cat_df)
        retriever.search("gold jewellery to go with red lehenga", top_k=10, filters=None)
        allowed_ids = retriever.sparse.search.call_args.kwargs["allowed_ids"]
        assert allowed_ids is not None
        assert "kids1" not in allowed_ids
        assert "adult1" in allowed_ids

    def test_kids_item_excluded_from_bm25_allowed_ids_type_filtered(self) -> None:
        """Kids exclusion also holds on the type-filter branch (product_type_name
        set), not just the no-filter branch."""
        cat_df = _make_catalogue([
            {
                "article_id": "kids1",
                "gender": "men",
                "product_type_name": "sherwani",
                "prod_name": "Boys Cream Sherwani Set",
                "display_name": "Boys Cream Sherwani Set",
            },
            {
                "article_id": "adult1",
                "gender": "men",
                "product_type_name": "sherwani",
                "prod_name": "Cream Silk Wedding Sherwani",
                "display_name": "Cream Silk Wedding Sherwani",
            },
        ])
        retriever = _make_retriever(cat_df)
        retriever.search(
            "cream sherwani for wedding",
            top_k=10,
            filters={"gender": "men", "product_type_name": "sherwani"},
        )
        allowed_ids = retriever.sparse.search.call_args.kwargs["allowed_ids"]
        assert allowed_ids is not None
        assert "kids1" not in allowed_ids
        assert "adult1" in allowed_ids


# ---------------------------------------------------------------------------
# E. End-to-end: search_node strips kids items unconditionally (real index)
# ---------------------------------------------------------------------------


class TestSearchNodeKidsExclusionRealIndex:
    """Reproduces the live bug end-to-end against the real unified catalogue:
    a non-occasion-keyword adult query must never surface a girls'/boys' item,
    even though the BM25-window mask alone (tested above) cannot catch
    dense-path (FAISS) leakage — this is what src.agents.graph's unconditional
    post-_is_material is_kids_item strip exists to close.
    """

    @staticmethod
    def _run_search(query: str):
        import pandas as pd

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

    def _assert_no_kids_items(self, query: str) -> None:
        result = self._run_search(query)
        items = result.get("retrieved_items", [])
        assert items, f"expected items for {query!r}"
        leaked = [
            it for it in items
            if is_kids_item(it.get("prod_name") or it.get("display_name") or "")
        ]
        assert not leaked, (
            f"kids item(s) leaked into {query!r} results: "
            f"{[it.get('prod_name') for it in leaked]}"
        )

    @pytest.mark.requires_index
    def test_red_lehenga_bridal_no_kids_leak(self) -> None:
        """The exact live-reproduced query: "red lehenga bridal" (no occasion
        keyword) must never surface a girls' lehenga."""
        self._assert_no_kids_items("red lehenga bridal")

    @pytest.mark.requires_index
    def test_gold_jewellery_to_go_with_red_lehenga_no_kids_leak(self) -> None:
        """The second live-reproduced query."""
        self._assert_no_kids_items("gold jewellery to go with red lehenga")


class _RealIndexMockLLM:
    """Minimal fixed-response LLM stub — mirrors tests/test_occasion_search_augmentation.py."""

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
