"""Selection parity: the outfit compose path picks exactly the recorded items.

Loads evals/fixtures/outfit_compose_snapshot.json (recorded on ONE specific index build, public view,
LLM-free) and recomposes every query with the current code: same ordered item IDs per variant, same
totals, same empty slots.

Selections depend on the exact index build, so this test is OPT-IN and fingerprint-guarded:
  * it runs only when OUTFIT_SNAPSHOT_INDEX_DIR points at an index (there is NO default path: CI
    downloads the deployed bucket's index, which is a different build than the one recorded);
  * before comparing anything it asserts the index fingerprint (rows, sorted-id hash, price sum,
    index file sizes, build id; evals/fixtures/outfit_compose_snapshot.index.json) matches, and
    FAILS with an explicit "not the recorded build" message otherwise (never a confusing diff).

One entry ("reception outfit for women") was re-recorded when the hash-seed-dependent word order
in slots.split_accessory_query_by_family was fixed: at 285e533 that query was nondeterministic
(on master too) and the fixture held one of its two possible outputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import outfit_compose_snapshot as snap

_ENV = os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR")
_FIXTURE = json.loads(snap.FIXTURE_PATH.read_text("utf-8"))
_OPT_IN_REASON = (
    "opt-in: set OUTFIT_SNAPSHOT_INDEX_DIR to the index build the fixture was recorded on "
    "(see evals/fixtures/outfit_compose_snapshot.index.json); selections of any other build differ"
)


def test_fixture_covers_all_queries() -> None:
    assert list(_FIXTURE) == snap.QUERIES
    assert len(_FIXTURE) == 30


def test_fingerprint_sidecar_is_complete() -> None:
    fp = json.loads(snap.FINGERPRINT_PATH.read_text("utf-8"))
    assert set(fp) == {
        "rows", "ids_sha256", "price_inr_sum", "dense_faiss_bytes", "bm25_pkl_bytes", "build_id"
    }
    assert fp["rows"] > 80_000 and len(fp["ids_sha256"]) == 64


@pytest.fixture(scope="module")
def _index() -> tuple[object, object]:
    if not _ENV:
        pytest.skip(_OPT_IN_REASON)
    index_dir = Path(_ENV)
    if not (index_dir / "catalogue.parquet").exists():
        pytest.skip(f"OUTFIT_SNAPSHOT_INDEX_DIR={index_dir} has no built index")
    if os.environ.get("INCLUDE_NON_PUBLIC_STORES"):
        pytest.skip("snapshot is the public view; INCLUDE_NON_PUBLIC_STORES is set")
    retriever, df = snap.load_retriever(index_dir)
    try:
        snap.assert_index_matches(index_dir, df)
    except snap.IndexMismatchError as exc:
        pytest.fail(str(exc), pytrace=False)
    return retriever, df


@pytest.mark.parametrize("query", snap.QUERIES)
def test_selection_identical_to_snapshot(query: str, _index: tuple[object, object]) -> None:
    retriever, df = _index
    got = snap.summarise(snap.compose_turn(query, df, retriever))
    assert got == _FIXTURE[query]
