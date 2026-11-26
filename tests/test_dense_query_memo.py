"""The dense query-vector memo is shared by threadpool workers: no exception, bounded size."""
from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np

from src.retrieval import dense_search
from src.retrieval.dense_search import DenseRetriever


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], **_k: Any) -> np.ndarray:
        self.calls += 1
        time.sleep(0)  # yield the GIL so threads interleave inside the memo
        return np.full((1, 4), float(len(texts[0])), dtype=np.float64)


class _Index:
    def search(self, vec: np.ndarray, top_k: int, params: Any = None) -> tuple[Any, Any]:
        return np.zeros((1, top_k), dtype=np.float32), np.full((1, top_k), -1, dtype=np.int64)


def _retriever() -> DenseRetriever:
    r = DenseRetriever.__new__(DenseRetriever)  # same construction path as ``load``
    r.model = _Model()  # type: ignore[assignment]
    r.index = _Index()  # type: ignore[assignment]
    r.article_ids = np.array([], dtype=object)
    return r


def test_memo_hit_returns_the_same_vector_for_the_exact_text() -> None:
    r = _retriever()
    r.search("red kurta")
    r.search("red kurta")
    r.search("Red kurta")  # exact text is the key: different case is a different entry
    assert r.model.calls == 2  # type: ignore[attr-defined]


def test_memo_survives_16_threads_hammering_it_past_its_bound() -> None:
    r = _retriever()
    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def work(worker: int) -> None:
        barrier.wait()
        try:
            for i in range(600):  # 600 x 16 mostly-distinct texts >> the 256 bound
                r.search(f"query {i} {worker if i % 3 else 0}")
        except BaseException as exc:  # noqa: BLE001 - the test asserts none escapes
            errors.append(exc)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force frequent thread switches
    try:
        threads = [threading.Thread(target=work, args=(w,)) for w in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert errors == []
    assert len(r.__dict__["_query_vec_cache"]) <= dense_search._QUERY_VEC_CACHE_MAX
