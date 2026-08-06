"""Regression test for the 2026-07-25 unconditional loungewear-exclusion fix
(src.agents.graph.search_node + scripts/eval_strict.py's mirror), part of the
"occasion-register" strict-eval miss bucket.

Root cause: _apply_loungewear_gate only fires inside the occasion-gated block
(formal-ethnic occasions + gym) — a query with NO occasion at all ("black
dress for my wife") had zero sleepwear protection and could surface a literal
"Black Printed Cotton Night Dress" (kaftan sleepwear). Fixed by adding an
UNCONDITIONAL loungewear strip right after the kids-item strip, exempting
queries that themselves explicitly ask for a night dress/nightgown (mirrors
_apply_loungewear_gate's own documented "a bare night dress query has a
legitimate reason to want these items" design intent).

Exercised via scripts/eval_strict.py's _retrieve_pipeline (the same
production-mirroring retrieval used by the strict gold eval) against the real
unified catalogue — the reliable way to exercise search_node's actual
filter/gate chain without the router-LLM's routing behaviour in the way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from src.catalogue.cleaning import is_loungewear_text  # noqa: E402


def _pipeline_items(query: str, gender: str) -> list[dict]:
    from eval_model import _build_components
    from eval_strict import _retrieve_pipeline

    comps = _build_components(need_agent=False, data_dir=_ROOT / "data" / "processed" / "unified")
    return _retrieve_pipeline(comps["retriever"], query, gender, occasion_gate=True)[:5]


@pytest.mark.requires_index
class TestRetrievePipelineLoungewearExclusion:
    def test_black_dress_no_occasion_excludes_nightdress(self) -> None:
        """The exact live-reproduced miss: "black dress for my wife" (no
        occasion keyword) must never surface a night dress/kaftan sleepwear
        item."""
        items = _pipeline_items("black dress for my wife", "women")
        assert items, "expected items for 'black dress for my wife'"
        leaked = [
            it for it in items
            if is_loungewear_text(it.get("prod_name") or it.get("display_name") or "")
        ]
        assert not leaked, (
            f"loungewear item(s) leaked into results: "
            f"{[it.get('prod_name') for it in leaked]}"
        )

    def test_explicit_night_dress_query_still_returns_night_dresses(self) -> None:
        """The exemption: a query that ITSELF asks for a night dress must
        still be able to return night dresses — the unconditional strip must
        not defeat this legitimate use case."""
        items = _pipeline_items("cotton night dress for women", "women")
        assert items, "expected items for 'cotton night dress for women'"
        matched = [
            it for it in items
            if is_loungewear_text(it.get("prod_name") or it.get("display_name") or "")
        ]
        assert matched, "explicit night dress query returned zero night dresses"
