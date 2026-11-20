"""SparseRetriever's per-query BM25 score-vector memo: hit, miss, bypass, eviction, parity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retrieval.sparse_search import SCORE_CACHE_SIZE, SparseRetriever

_DOCS = [
    "red silk saree wedding",
    "blue cotton kurta men",
    "black leather heels party",
    "gold jhumka earrings festive",
    "navy blazer office",
    "white linen shirt men",
]


@pytest.fixture()
def sparse(tmp_path) -> SparseRetriever:
    df = pd.DataFrame(
        {"article_id": [f"a{i}" for i in range(len(_DOCS))], "search_text": _DOCS}
    )
    r = SparseRetriever({})
    r.build_index(df, tmp_path)
    return r


def _count_calls(r: SparseRetriever, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    real = r.bm25.get_scores

    def spy(tokens: list[str]) -> np.ndarray:
        calls.append(list(tokens))
        return real(tokens)

    monkeypatch.setattr(r.bm25, "get_scores", spy)
    return calls


def test_default_size_is_16(sparse: SparseRetriever) -> None:
    assert SCORE_CACHE_SIZE == 16
    assert sparse._score_cache_size == 16


def test_miss_then_hit_computes_once(sparse, monkeypatch) -> None:
    calls = _count_calls(sparse, monkeypatch)
    first = sparse.search("red saree", top_k=3)
    second = sparse.search("Red  SAREE!", top_k=3)  # same exact token tuple after tokenising
    assert first == second
    assert len(calls) == 1


def test_different_token_order_or_content_is_a_miss(sparse, monkeypatch) -> None:
    calls = _count_calls(sparse, monkeypatch)
    sparse.search("red saree")
    sparse.search("saree red")  # exact tuple differs -> no sharing
    sparse.search("red saree wedding")
    assert len(calls) == 3


def test_cached_search_matches_uncached_including_allowed_ids(sparse) -> None:
    allowed = np.array(["a1", "a5", "a0"])
    cached_all = sparse.search("men shirt kurta", top_k=5)
    cached_masked = sparse.search("men shirt kurta", top_k=5, allowed_ids=allowed)
    again_masked = sparse.search("men shirt kurta", top_k=5, allowed_ids=allowed)
    other_mask = sparse.search("men shirt kurta", top_k=5, allowed_ids=np.array(["a5"]))

    sparse._init_score_cache(0)
    assert sparse.search("men shirt kurta", top_k=5) == cached_all
    assert sparse.search("men shirt kurta", top_k=5, allowed_ids=allowed) == cached_masked
    assert again_masked == cached_masked
    # A different allowed_ids set must see its OWN mask over the shared raw vector.
    assert other_mask == sparse.search("men shirt kurta", top_k=5, allowed_ids=np.array(["a5"]))
    assert {aid for aid, _ in other_mask} == {"a5"}


def test_bypass_when_size_zero(sparse, monkeypatch) -> None:
    sparse._init_score_cache(0)
    calls = _count_calls(sparse, monkeypatch)
    sparse.search("red saree")
    sparse.search("red saree")
    assert len(calls) == 2
    assert len(sparse._score_cache) == 0


def test_instance_without_cache_attrs_bypasses(sparse, monkeypatch) -> None:
    del sparse._score_cache_size  # e.g. object built via __new__ by tooling
    calls = _count_calls(sparse, monkeypatch)
    sparse.search("red saree")
    sparse.search("red saree")
    assert len(calls) == 2


def test_lru_eviction_is_bounded_and_drops_least_recent(sparse, monkeypatch) -> None:
    sparse._init_score_cache(2)
    calls = _count_calls(sparse, monkeypatch)
    sparse.search("red saree")  # A
    sparse.search("blue kurta")  # B
    sparse.search("red saree")  # touch A -> B is now least recent
    sparse.search("black heels")  # C evicts B
    assert len(sparse._score_cache) == 2
    n = len(calls)
    sparse.search("red saree")  # A still cached
    assert len(calls) == n
    sparse.search("blue kurta")  # B was evicted -> recomputed
    assert len(calls) == n + 1


def test_cached_vector_is_read_only(sparse) -> None:
    sparse.search("red saree")
    (vec,) = sparse._score_cache.values()
    with pytest.raises(ValueError):
        vec[0] = 99.0


def test_swapping_the_bm25_object_invalidates(sparse, tmp_path) -> None:
    before = sparse.search("red saree")
    # >=3 docs: BM25Okapi's idf is 0 for a term in half of a 2-doc corpus.
    df = pd.DataFrame(
        {
            "article_id": ["z0", "z1", "z2"],
            "search_text": ["red saree", "blue jeans", "green cap"],
        }
    )
    other = SparseRetriever({})
    other.build_index(df, tmp_path / "other")
    sparse.bm25, sparse.article_ids = other.bm25, other.article_ids
    sparse._id_to_pos = other._id_to_pos
    after = sparse.search("red saree")
    assert after != before
    assert after[0][0] == "z0"
