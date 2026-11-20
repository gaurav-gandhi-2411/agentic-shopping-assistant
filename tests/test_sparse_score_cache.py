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


def test_caller_mutating_returned_array_cannot_corrupt_cache(sparse) -> None:
    tokens = SparseRetriever._tokenize("red saree")
    want = np.array(sparse.bm25.get_scores(tokens))  # independent uncached copy
    for _ in range(2):  # first call = miss path, second = hit path
        got = sparse._get_scores(tokens)
        with pytest.raises(ValueError):
            got[0] = 123.0
        with pytest.raises(ValueError):
            got.flags.writeable = True  # a view of a read-only base cannot be re-enabled
        with pytest.raises(ValueError):
            got *= 0  # in-place ops are writes too
    np.testing.assert_array_equal(sparse._get_scores(tokens), want)


def _swapped_in(sparse: SparseRetriever, tmp_path, docs: list[str]) -> SparseRetriever:
    """Point ``sparse`` at a different bm25 over the SAME ids (a0..), return a fresh reference."""
    df = pd.DataFrame({"article_id": [f"a{i}" for i in range(len(docs))], "search_text": docs})
    other = SparseRetriever({})
    other.build_index(df, tmp_path / "other")
    sparse.bm25, sparse.article_ids = other.bm25, other.article_ids
    sparse._id_to_pos = other._id_to_pos
    return other


def test_swapping_the_bm25_object_invalidates(sparse, tmp_path) -> None:
    # Same length and ids, but "red saree" now matches index 3 (was index 0): a stale vector
    # would silently keep answering a0.
    before = sparse.search("red saree", top_k=1)
    assert before[0][0] == "a0"
    docs = ["blue cotton kurta", "black heels", "navy blazer", "red silk saree", "white shirt",
            "gold earrings"]
    other = _swapped_in(sparse, tmp_path, docs)
    after = sparse.search("red saree", top_k=1)
    assert after[0][0] == "a3"
    other._init_score_cache(0)  # uncached reference answer
    assert after == other.search("red saree", top_k=1)


def test_replaced_bm25_is_not_confused_with_old_one_after_id_reuse(sparse, tmp_path) -> None:
    """The cache pins its bm25 (strong ref), so a replacement can never share its identity."""
    sparse.search("red saree")
    old_ref = sparse._score_cache_owner
    assert old_ref is sparse.bm25
    docs = ["blue cotton kurta", "black heels", "navy blazer", "red silk saree", "white shirt",
            "gold earrings"]
    _swapped_in(sparse, tmp_path, docs)
    del old_ref
    import gc

    gc.collect()
    # Old bm25 is still alive (pinned) until the next search drops it, so its id is not reusable.
    assert sparse._score_cache_owner is not sparse.bm25
    assert sparse.search("red saree", top_k=1)[0][0] == "a3"
    assert sparse._score_cache_owner is sparse.bm25


def test_masked_search_does_not_poison_unmasked_search(sparse) -> None:
    q = "men shirt kurta"
    masked = sparse.search(q, top_k=6, allowed_ids=np.array(["a5"]))
    assert {aid for aid, _ in masked} == {"a5"}
    unmasked = sparse.search(q, top_k=6)  # must come from the RAW vector, not the masked one
    sparse._init_score_cache(0)
    assert unmasked == sparse.search(q, top_k=6)
    assert len({aid for aid, _ in unmasked}) > 1


def test_concurrent_searches_match_uncached(sparse) -> None:
    import threading

    queries = ["red saree", "blue kurta", "black heels", "gold earrings", "navy blazer",
               "white shirt men"]
    sparse._init_score_cache(2)  # tiny: forces constant eviction races
    results: dict[tuple[int, int], list] = {}
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for rnd in range(40):
                q = queries[(tid + rnd) % len(queries)]
                results[(tid, rnd)] = (q, sparse.search(q, top_k=4))
        except BaseException as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    sparse._init_score_cache(0)
    expected = {q: sparse.search(q, top_k=4) for q in queries}
    assert len(results) == 8 * 40
    assert all(res == expected[q] for q, res in results.values())
