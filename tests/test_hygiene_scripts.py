"""Hermetic tests for the V3(b2)(b) measurement scripts (every path is under tmp_path)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

import faiss  # noqa: E402

from scripts import (  # noqa: E402
    catalogue_hygiene_report,
    make_hygiene_index_copy,
    store_miss_rates,
)


def _catalogue() -> pd.DataFrame:
    rows = [
        ("1", "libas", "Red Silk Kurta", "kurta"),
        ("2", "ishhaara", "Brocade Lehenga Box", "lehenga"),  # dropped
        ("3", "ishhaara", "Gold Bangles Lehenga", "Bangles"),  # quarantined
        ("4", "voylla", "Kundan Jhumka Earrings", "earrings"),
    ]
    return pd.DataFrame(
        {
            "article_id": [r[0] for r in rows],
            "store": [r[1] for r in rows],
            "prod_name": [r[2] for r in rows],
            "product_type_name": [r[3] for r in rows],
            "search_text": [f"{r[2]}. {r[3]}" for r in rows],
            "price_inr": 100.0,
        }
    )


def test_wilson_interval_matches_known_values() -> None:
    lo, hi = store_miss_rates.wilson(16, 30)  # globalrepublic example from the brief
    assert 0.36 < lo < 0.37 and 0.69 < hi < 0.70
    assert store_miss_rates.wilson(0, 5)[0] == 0.0
    assert store_miss_rates.wilson(5, 5)[1] == 1.0


def test_per_store_table_counts_and_reasons() -> None:
    labels = [
        {"query_id": "q1", "items": [
            {"article_id": "1", "relevant": True},
            {"article_id": "2", "relevant": False, "reason": "type-confusion"},
            {"article_id": "9", "relevant": False, "reason": "budget"},  # not in catalogue
        ]},
    ]
    table = store_miss_rates.per_store_table(labels, {"1": "libas", "2": "ishhaara"})
    by = table.set_index("store")
    assert by.loc["libas", ["n", "misses"]].tolist() == [1, 0]
    assert by.loc["ishhaara", "reasons"] == "type-confusion 1"
    assert by.loc["?", "n"] == 1


def test_restrict_to_retrieved_keeps_only_labelled_items_in_a_top5() -> None:
    labels = [
        {"query_id": "q1", "items": [
            {"article_id": "1", "relevant": True},
            {"article_id": "2", "relevant": False, "reason": "budget"},  # labelled, not shown
        ]},
        {"query_id": "q2", "items": [{"article_id": "3", "relevant": False}]},  # query not run
    ]
    rows = [{"id": "q1", "top": [{"article_id": "1", "store": "libas"},
                                 {"article_id": "9", "store": "x"}]}]  # 9 is unlabelled
    kept, store_of = store_miss_rates.restrict_to_retrieved(labels, rows)
    assert [(e["query_id"], [i["article_id"] for i in e["items"]]) for e in kept] == [("q1", ["1"])]
    assert store_of == {"1": "libas", "9": "x"}


def test_hygiene_report_lists_dropped_and_quarantined() -> None:
    text = catalogue_hygiene_report.build_report(_catalogue())
    assert "dropped (non_fashion) 1" in text and "quarantined 1" in text
    assert "Brocade Lehenga Box" in text and "Gold Bangles Lehenga" in text


def test_index_copy_keeps_aligned_vectors_and_refuses_to_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    cat = _catalogue()
    cat.to_parquet(src / "catalogue.parquet", index=False)
    vecs = np.eye(4, 8, dtype=np.float32)  # row i has a unique unit vector
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    faiss.write_index(index, str(src / "dense.faiss"))
    np.save(str(src / "dense_article_ids.npy"), cat["article_id"].values.astype(str))

    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--dst", str(dst)])
    make_hygiene_index_copy.main()

    new_ids = np.load(str(dst / "dense_article_ids.npy"), allow_pickle=True).tolist()
    assert new_ids == ["1", "4"]
    new_index = faiss.read_index(str(dst / "dense.faiss"))
    assert new_index.ntotal == 2
    np.testing.assert_array_equal(new_index.reconstruct(0), vecs[0])  # id "1"
    np.testing.assert_array_equal(new_index.reconstruct(1), vecs[3])  # id "4"
    assert (dst / "hygiene_dropped.csv").exists() and (dst / "bm25.pkl").exists()

    with pytest.raises(SystemExit):  # dst exists now: refuse, never overwrite
        make_hygiene_index_copy.main()
