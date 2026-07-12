"""Regression tests for _detect_ooc word-boundary matching.

Live defect (reports/ui_defect_sweep_20260710.md P0-1): plain substring `in`
matched "tea" inside "ins-tea-d", so the colour refinement "show me pastel
colours instead" was refused mid-conversation as a food-and-drink query.
"""
from __future__ import annotations

from src.agents.graph import _detect_ooc


class TestDetectOocWordBoundaries:
    def test_pastel_refinement_is_not_food(self) -> None:
        assert _detect_ooc("show me pastel colours instead") is None

    def test_real_food_query_still_detected(self) -> None:
        assert _detect_ooc("best coffee beans for cold brew") == "food and drink"

    def test_electronics_still_detected(self) -> None:
        assert _detect_ooc("recommend me a good gaming laptop") == "electronics"

    def test_tv_at_string_end_now_detected(self) -> None:
        # The old space-padded " tv " hack missed "tv" at the end of the query.
        assert _detect_ooc("a new tv") == "electronics"

    def test_clothing_queries_pass(self) -> None:
        assert _detect_ooc("wedding lehenga under 15000") is None
        assert _detect_ooc("pastel kurta for haldi") is None
