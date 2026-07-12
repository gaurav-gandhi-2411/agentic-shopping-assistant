"""Regression tests for the 2026-07-11 colour-family filter fix.

Root cause (live-proven while investigating a strict-eval "red embellished
lehenga" miss that turned out to be an eval-harness gap, not a real bug):
intent_parser._COLOUR_MAP collapsed 9 real-world colour synonyms (navy,
mustard, burgundy, maroon, charcoal, peach, olive, teal, cream) onto an
approximate NEIGHBOURING catalogue colour_group_name value instead of the
catalogue's own distinct value — e.g. "navy" -> "Dark Blue", silently
excluding all 770 items the catalogue itself tags "Navy Blue" from every
"navy" query (hard exact-match filter in HybridRetriever.search).

Fix: _COLOUR_MAP now targets the catalogue's own exact value for each of
these nine; colour_filter_values() widens the RETRIEVAL FILTER (not
intent.colour, which stays a single display string) to the small real
near-synonym family — applied only inside search_catalogue (the actual
retrieval boundary), never persisted into session state.
"""
from __future__ import annotations

from src.agents.intent_parser import colour_filter_values, parse_intent
from src.agents.tools import search_catalogue
from src.retrieval.hybrid_search import _facet_value_matches


class TestColourMapTargetsCatalogueExactValues:
    def test_navy_maps_to_navy_blue_not_dark_blue(self) -> None:
        assert parse_intent("navy jacket").colour == "Navy Blue"

    def test_mustard_maps_to_mustard_not_yellow(self) -> None:
        assert parse_intent("mustard kurta").colour == "Mustard"

    def test_burgundy_and_maroon_stay_distinct(self) -> None:
        assert parse_intent("burgundy saree").colour == "Burgundy"
        assert parse_intent("maroon saree").colour == "Maroon"

    def test_dark_blue_unaffected(self) -> None:
        # "dark blue" itself was never a fragmented synonym — must not change.
        assert parse_intent("dark blue kurta").colour == "Dark Blue"


class TestColourFilterValuesWidening:
    def test_known_fragmented_colour_widens_to_family(self) -> None:
        assert colour_filter_values("Navy Blue") == ("Navy Blue", "Dark Blue")
        assert colour_filter_values("Mustard") == ("Mustard", "Yellow")

    def test_unfragmented_colour_passes_through_unchanged(self) -> None:
        assert colour_filter_values("Black") == "Black"
        assert colour_filter_values("Red") == "Red"

    def test_none_passes_through(self) -> None:
        assert colour_filter_values(None) is None


class TestFacetValueMatchesIsin:
    def test_scalar_value_exact_match_unchanged(self) -> None:
        assert _facet_value_matches({"colour_group_name": "Black"}, "colour_group_name", "black")
        assert not _facet_value_matches(
            {"colour_group_name": "Black"}, "colour_group_name", "white"
        )

    def test_list_value_matches_any_member(self) -> None:
        facets = {"colour_group_name": "Dark Blue"}
        assert _facet_value_matches(
            facets, "colour_group_name", ("Navy Blue", "Dark Blue")
        )
        assert not _facet_value_matches(
            facets, "colour_group_name", ("Navy Blue", "Black")
        )


class TestSearchCatalogueWideningNeverMutatesCallerFilters:
    def test_caller_filters_dict_unchanged_after_call(self) -> None:
        class _StubRetriever:
            def search(self, query, top_k=20, filters=None):
                self.last_filters = filters
                return []

        retriever = _StubRetriever()
        filters = {"gender": "men", "colour_group_name": "Navy Blue"}
        original = dict(filters)
        search_catalogue("navy kurta", filters, retriever, 10)
        assert filters == original, "search_catalogue must not mutate the caller's filters dict"
        assert retriever.last_filters["colour_group_name"] == ("Navy Blue", "Dark Blue")
