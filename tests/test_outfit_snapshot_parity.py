"""Selection parity: the outfit compose path picks exactly the pre-change items.

Loads evals/fixtures/outfit_compose_snapshot.json (recorded from the UNMODIFIED compose path
at commit 285e533, public view of the unified index, LLM-free) and recomposes every query with
the current code: same ordered item IDs per variant, same totals, same empty slots.

One entry ("reception outfit for women") was re-recorded when the hash-seed-dependent word order
in slots.split_accessory_query_by_family was fixed: at 285e533 that query was nondeterministic
(on master too) and the fixture held one of its two possible outputs. The other 29 are unchanged.

Needs the built unified index. Point OUTFIT_SNAPSHOT_INDEX_DIR at it (default
data/processed/unified); skipped when absent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import outfit_compose_snapshot as snap

_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))
_FIXTURE = json.loads(snap.FIXTURE_PATH.read_text("utf-8"))


def test_fixture_covers_all_queries() -> None:
    assert list(_FIXTURE) == snap.QUERIES
    assert len(_FIXTURE) == 30


@pytest.fixture(scope="module")
def _index() -> tuple[object, object]:
    if not (_INDEX_DIR / "catalogue.parquet").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("snapshot is the public view; INCLUDE_NON_PUBLIC_STORES is set")
    retriever, df = snap.load_retriever(_INDEX_DIR)
    return retriever, df


@pytest.mark.parametrize("query", snap.QUERIES)
def test_selection_identical_to_snapshot(query: str, _index: tuple[object, object]) -> None:
    retriever, df = _index
    got = snap.summarise(snap.compose_turn(query, df, retriever))
    assert got == _FIXTURE[query]
