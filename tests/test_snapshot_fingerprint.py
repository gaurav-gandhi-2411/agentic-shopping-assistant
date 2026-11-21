"""Index fingerprint guard for the outfit snapshot: hermetic (tmp_path only, no real index)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.outfit_compose_snapshot import (
    IndexMismatchError,
    assert_index_matches,
    index_fingerprint,
)


def _index(tmp_path: Path, faiss_bytes: int = 100, bm25_bytes: int = 50) -> Path:
    (tmp_path / "dense.faiss").write_bytes(b"x" * faiss_bytes)
    (tmp_path / "bm25.pkl").write_bytes(b"y" * bm25_bytes)
    return tmp_path


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {"article_id": ["a3", "a1", "a2"], "price_inr": [300.0, 100.0, None]}
    )


def test_fingerprint_is_stable_under_row_order(tmp_path: Path) -> None:
    d = _index(tmp_path)
    assert index_fingerprint(d, _df()) == index_fingerprint(d, _df().iloc[::-1])


def test_fingerprint_changes_with_rows_prices_and_index_files(tmp_path: Path) -> None:
    d = _index(tmp_path)
    base = index_fingerprint(d, _df())
    assert index_fingerprint(d, _df().iloc[:2]) != base  # a row dropped
    priced = _df()
    priced.loc[0, "price_inr"] = 301.0
    assert index_fingerprint(d, priced) != base  # a price moved
    renamed = _df().assign(article_id=["a3", "a1", "zz"])
    assert index_fingerprint(d, renamed) != base  # an id swapped, same row count
    assert index_fingerprint(_index(tmp_path, faiss_bytes=101), _df()) != base  # re-embedded
    assert index_fingerprint(_index(tmp_path, bm25_bytes=51), _df()) != base  # re-tokenised
    assert base["rows"] == 3 and base["build_id"] is None and len(base["ids_sha256"]) == 64


def test_matching_index_passes_and_a_different_one_fails_with_a_clear_message(
    tmp_path: Path,
) -> None:
    d = _index(tmp_path)
    want = index_fingerprint(d, _df())
    assert_index_matches(d, _df(), want)  # same build: silent
    with pytest.raises(IndexMismatchError) as exc:
        assert_index_matches(d, _df().iloc[:2], want)
    msg = str(exc.value)
    assert "not the build" in msg and "rows: fixture=3 index=2" in msg
    assert "--record" in msg  # tells the reader what to do
    assert "AssertionError" not in msg  # a build mismatch, never a selection diff
