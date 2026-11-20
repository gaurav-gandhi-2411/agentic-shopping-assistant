"""Vectorised BM25 (postings-based) must equal rank_bm25's get_scores exactly.

Synthetic-corpus tests run everywhere (CI); the real-index test skips when no unified index is
built (set OUTFIT_SNAPSHOT_INDEX_DIR, default data/processed/unified).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retrieval.sparse_search import SparseRetriever, _CscBm25

_VOCAB = [f"w{i}" for i in range(40)] + ["common", "everywhere"]


def _corpus(n: int = 300) -> list[str]:
    rng = np.random.default_rng(42)
    docs = []
    for _ in range(n):
        words = list(rng.choice(_VOCAB[:40], size=int(rng.integers(1, 25))))
        words += ["common"] * int(rng.integers(0, 3)) + ["everywhere"]  # idf<0 -> epsilon floor
        docs.append(" ".join(words))
    docs[7] = "w1 w1 w1 w1 w1 w1 w1 w1"  # tf >> 1 in a short doc
    return docs


@pytest.fixture()
def sparse(tmp_path: Path) -> SparseRetriever:
    docs = _corpus()
    df = pd.DataFrame({"article_id": [f"a{i}" for i in range(len(docs))], "search_text": docs})
    r = SparseRetriever({})
    r.build_index(df, tmp_path)
    r._init_score_cache(0)
    return r


_QUERIES = [
    ["w1"],
    ["w1", "w2", "w3"],
    ["w1", "w1", "w2"],  # repeated token counts twice in rank_bm25
    ["common"],
    ["everywhere", "w5"],  # negative-idf (epsilon-floored) term
    ["nope"],  # out of vocabulary
    ["nope", "w9", "alsonope"],
    [],
    [f"w{i}" for i in range(40)],
]


@pytest.mark.parametrize("tokens", _QUERIES)
def test_vectorised_scores_bit_identical(sparse: SparseRetriever, tokens: list[str]) -> None:
    csc = _CscBm25(sparse.bm25)
    want = sparse.bm25.get_scores(tokens)
    got = csc.get_scores(tokens)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)  # exact, not allclose


def test_search_identical_in_both_modes_incl_masks_and_k(sparse: SparseRetriever) -> None:
    ids = np.array(sparse.article_ids)
    masks = [None, ids[::2], ids[:50], np.array(["a3", "a7", "missing"])]
    for q in ["w1 w2", "w1 w1 common", "everywhere", "zzz", "w5 w9 w11 w30"]:
        for m in masks:
            for k in (1, 5, 30, 80):
                sparse._vectorised = False
                ref = sparse.search(q, top_k=k, allowed_ids=m)
                sparse._vectorised = True
                assert sparse.search(q, top_k=k, allowed_ids=m) == ref


def test_vectorised_path_is_actually_used(sparse: SparseRetriever, monkeypatch) -> None:
    def boom(_tokens: list[str]) -> np.ndarray:
        raise AssertionError("rank_bm25.get_scores called on the vectorised path")

    monkeypatch.setattr(sparse.bm25, "get_scores", boom)
    assert sparse.search("w1 w2", top_k=3)


def test_switch_off_uses_rank_bm25(sparse: SparseRetriever, monkeypatch) -> None:
    calls: list[list[str]] = []
    real = sparse.bm25.get_scores
    monkeypatch.setattr(sparse.bm25, "get_scores", lambda t: (calls.append(t), real(t))[1])
    sparse._vectorised = False
    sparse.search("w1 w2")
    assert calls == [["w1", "w2"]]


def test_postings_rebuilt_when_bm25_swapped(sparse: SparseRetriever, tmp_path: Path) -> None:
    sparse.search("w1")
    first = sparse._csc
    docs = list(reversed(_corpus()))
    df = pd.DataFrame({"article_id": [f"a{i}" for i in range(len(docs))], "search_text": docs})
    other = SparseRetriever({})
    other.build_index(df, tmp_path / "o")
    sparse.bm25, sparse.article_ids, sparse._id_to_pos = (
        other.bm25,
        other.article_ids,
        other._id_to_pos,
    )
    got = sparse.search("w1 w2", top_k=10)
    assert sparse._csc is not first
    other._init_score_cache(0)
    other._vectorised = False
    assert got == other.search("w1 w2", top_k=10)


def test_memo_still_applies_on_vectorised_path(sparse: SparseRetriever, monkeypatch) -> None:
    sparse._init_score_cache(4)
    n = {"c": 0}
    real = sparse._compute_scores

    def counting(t: list[str]) -> np.ndarray:
        n["c"] += 1
        return real(t)

    monkeypatch.setattr(sparse, "_compute_scores", counting)
    a = sparse.search("w1 w2")
    b = sparse.search("w1 w2")
    assert a == b and n["c"] == 1


_INDEX_DIR = Path(os.environ.get("OUTFIT_SNAPSHOT_INDEX_DIR", "data/processed/unified"))


def test_real_index_bit_identical() -> None:
    if not (_INDEX_DIR / "bm25.pkl").exists():
        pytest.skip(f"no built unified index at {_INDEX_DIR}")
    from scripts import outfit_compose_snapshot as snap

    sp = SparseRetriever.load({}, _INDEX_DIR)
    sp._init_score_cache(0)
    for q in snap.QUERIES + ["bag sling handbag women accessory embellished formal evening"]:
        t = sp._tokenize(q)
        np.testing.assert_array_equal(sp._csc.get_scores(t), sp.bm25.get_scores(t), err_msg=q)
