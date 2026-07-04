"""Unit tests for src/agents/reranker.py.

Covers the live-proven crash: a candidate item can carry an explicit
``colour=None`` key (not merely a missing key) — ``dict.get(key, default)``
only substitutes the default when the key is MISSING, so ``.get("colour", "").
lower()`` still calls ``.lower()`` on ``None`` and raises ``AttributeError``.
This must never crash a WS turn.
"""
from __future__ import annotations

from src.agents.reranker import _enforce_colour_diversity, _format_candidates, rerank


def _item(article_id: str, colour: str | None, **extra: object) -> dict:
    base: dict = {
        "article_id": article_id,
        "colour": colour,
        "display_name": f"Item {article_id}",
        "prod_name": f"Item {article_id}",
        "product_type": "Dress",
        "department": "Women",
    }
    base.update(extra)
    return base


class TestEnforceColourDiversityNoneSafety:
    """`_enforce_colour_diversity` must never raise on colour=None and must
    treat None the same as an empty-string colour."""

    def test_none_colour_in_selected_does_not_crash(self) -> None:
        selected = [_item("1", None), _item("2", None)]
        candidates = selected + [_item("3", "black")]
        # Must not raise AttributeError('NoneType' object has no attribute 'lower')
        result = _enforce_colour_diversity(selected, candidates, "a date night look")
        assert result is not None

    def test_none_colour_swaps_in_a_different_colour_when_available(self) -> None:
        selected = [_item("1", None), _item("2", None)]
        candidates = selected + [_item("3", "black")]
        result = _enforce_colour_diversity(selected, candidates, "date night outfit")
        result_ids = {it["article_id"] for it in result}
        assert "3" in result_ids, "should swap in a distinct-colour candidate"

    def test_none_colour_candidate_pool_does_not_crash(self) -> None:
        """The candidate pool (not just `selected`) may also carry colour=None."""
        selected = [_item("1", "black"), _item("2", "black")]
        candidates = selected + [_item("3", None)]
        result = _enforce_colour_diversity(selected, candidates, "evening out")
        assert result is not None
        result_ids = {it["article_id"] for it in result}
        assert "3" in result_ids

    def test_non_date_night_query_returns_selected_unchanged(self) -> None:
        selected = [_item("1", None), _item("2", None)]
        result = _enforce_colour_diversity(selected, selected, "casual t-shirt")
        assert result == selected

    def test_already_diverse_colours_returns_unchanged(self) -> None:
        selected = [_item("1", "black"), _item("2", "red")]
        result = _enforce_colour_diversity(selected, selected, "date night")
        assert result == selected

    def test_no_alternative_available_keeps_selected(self) -> None:
        """All colours None and no alternative in the pool — must not crash,
        must return the original selection unchanged."""
        selected = [_item("1", None), _item("2", None)]
        result = _enforce_colour_diversity(selected, selected, "date night")
        assert result == selected


class TestFormatCandidatesNoneSafety:
    def test_none_colour_and_fields_do_not_render_literal_none(self) -> None:
        items = [_item("1", None, product_type=None, department=None)]
        text = _format_candidates(items)
        assert "None" not in text


class TestRerankFallsBackOnLlmFailure:
    """Regression guard: rerank() must never crash even with None-colour items,
    whether the LLM call fails or the colour-diversity post-check runs."""

    class _FailingLLM:
        def generate(self, prompt: str, system: str | None = None, **kwargs: object) -> str:
            raise RuntimeError("boom")

    def test_llm_failure_falls_back_without_crashing(self) -> None:
        items = [_item(str(i), None) for i in range(8)]
        result = rerank("date night look", items, self._FailingLLM())  # type: ignore[arg-type]
        assert len(result) == 5
