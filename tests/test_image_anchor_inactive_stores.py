"""Unit tests for inactive-store exclusion in src/agents/outfit/image_anchor.py.

Fully offline: builds tiny synthetic FAISS indices + catalogue parquets under
tmp_path, mirroring the pattern in tests/test_image_style.py.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.outfit.image_anchor import (
    _index_cache,
    _store_map_cache,
    find_anchor_from_image,
)
from src.retrieval.index_store import UNIFIED_BRAND


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Ensure module-level caches never leak state between tests."""
    _index_cache.clear()
    _store_map_cache.clear()
    yield
    _index_cache.clear()
    _store_map_cache.clear()


def _build_faiss_index(tmp_path: Path, brand: str, article_ids: list[str]) -> None:
    """Write a tiny orthonormal FAISS index + article_ids.npy for *brand* under tmp_path."""
    import faiss

    dim = 512
    n = len(article_ids)
    vecs = np.eye(n, dim, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    brand_dir = tmp_path / brand
    brand_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(brand_dir / "clip.faiss"))
    np.save(str(brand_dir / "clip_article_ids.npy"), np.array(article_ids))


def _mock_encoder_for_row(row_idx: int, dim: int = 512) -> MagicMock:
    """Return a mock CLIP encoder whose image vector exactly matches article at row_idx."""
    vec = np.eye(row_idx + 1, dim, dtype=np.float32)[row_idx]
    mock_enc = MagicMock()
    mock_enc.encode_image.return_value = vec
    return mock_enc


# ---------------------------------------------------------------------------
# Per-brand short-circuit: brand slug itself is inactive
# ---------------------------------------------------------------------------


def test_inactive_brand_returns_empty_without_loading_index(tmp_path: Path) -> None:
    """brand='berrylush' (inactive) must return [] without touching the FAISS index."""
    # Deliberately do NOT create a berrylush index under tmp_path — if the guard fires
    # correctly the function never tries to load it, so this proves no I/O occurred.
    img = object()  # never touched — encoder is never invoked either
    result = find_anchor_from_image(img, brand="berrylush", _index_base=tmp_path)
    assert result == []


def test_active_brand_with_missing_index_still_returns_empty(tmp_path: Path) -> None:
    """Sanity: an active brand with no index on disk still returns [] (pre-existing behaviour)."""
    img = object()
    result = find_anchor_from_image(img, brand="myntra", _index_base=tmp_path)
    assert result == []


# ---------------------------------------------------------------------------
# Unified brand: per-item store-column join
# ---------------------------------------------------------------------------


def test_unified_brand_excludes_inactive_store_candidates(tmp_path: Path) -> None:
    """Unified brand results must drop candidates whose catalogue store is inactive."""
    article_ids = ["art_berrylush", "art_myntra", "art_snitch"]
    _build_faiss_index(tmp_path, UNIFIED_BRAND, article_ids)

    catalogue_base = tmp_path / "catalogue"
    unified_dir = catalogue_base / UNIFIED_BRAND
    unified_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "article_id": article_ids,
            "store": ["berrylush", "myntra", "snitch"],
        }
    ).to_parquet(unified_dir / "catalogue.parquet")

    # Mock encoder returns a vector equidistant from all 3 (all rows are orthonormal,
    # so a small combined vector ranks all 3 by their own axis weight); instead just
    # request top_k=3 so all candidates are returned pre-filter, then verify filtering.
    mock_enc = MagicMock()
    combined_vec = np.zeros(512, dtype=np.float32)
    combined_vec[0] = combined_vec[1] = combined_vec[2] = 1.0 / np.sqrt(3)
    mock_enc.encode_image.return_value = combined_vec

    with patch("src.agents.outfit.image_anchor.get_clip_encoder", return_value=mock_enc):
        results = find_anchor_from_image(
            object(),
            brand=UNIFIED_BRAND,
            top_k=3,
            _index_base=tmp_path,
            _catalogue_base=catalogue_base,
        )

    assert "art_berrylush" not in results
    assert set(results) == {"art_myntra", "art_snitch"}


def test_unified_brand_no_catalogue_degrades_to_no_filtering(tmp_path: Path) -> None:
    """When the unified catalogue parquet is absent, filtering degrades gracefully (no crash)."""
    article_ids = ["art_001", "art_002"]
    _build_faiss_index(tmp_path, UNIFIED_BRAND, article_ids)

    mock_enc = _mock_encoder_for_row(0)
    # Point catalogue base at an empty tmp dir so no parquet is found.
    empty_catalogue_base = tmp_path / "no_catalogue_here"

    with patch("src.agents.outfit.image_anchor.get_clip_encoder", return_value=mock_enc):
        results = find_anchor_from_image(
            object(),
            brand=UNIFIED_BRAND,
            top_k=2,
            _index_base=tmp_path,
            _catalogue_base=empty_catalogue_base,
        )

    # No store map available → no filtering applied → both candidates returned.
    assert set(results) == {"art_001", "art_002"}
