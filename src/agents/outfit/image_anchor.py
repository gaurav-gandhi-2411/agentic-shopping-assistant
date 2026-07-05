"""Image-to-anchor retrieval using the per-brand CLIP-text index.

Given an uploaded PIL image and a brand slug, this module:
  1. Encodes the image into the shared CLIP 512-d space.
  2. Searches the brand's pre-built CLIP-text FAISS index (IndexFlatIP).
  3. Returns the top-K candidate article_ids ranked by cosine similarity.

Index layout on disk (built by scripts/build_clip_index.py)::

    data/processed/clip/<brand>/clip.faiss          — IndexFlatIP (float32, 512-d)
    data/processed/clip/<brand>/clip_article_ids.npy — corresponding article_id strings

Privacy contract
----------------
The PIL image is only used for the ``encode_image`` call.  Callers MUST drop
all references to the image bytes after calling ``find_anchor_from_image``.
This module never writes image data to disk, logs, or DB.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.config.stores import get_inactive_stores
from src.retrieval.clip_encoder import get_clip_encoder
from src.retrieval.index_store import UNIFIED_BRAND

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-brand index cache — loaded lazily, kept in memory for the process lifetime.
# ---------------------------------------------------------------------------

_IndexEntry = tuple[object, np.ndarray]  # (faiss.Index, article_ids array)
_index_cache: dict[str, _IndexEntry] = {}
_cache_lock = threading.Lock()

# Repo-root-relative base directory for CLIP indices.
_CLIP_INDEX_BASE = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "clip"

# Repo-root-relative base directory for catalogue parquets (one level up from the CLIP
# index dir).  Only consulted for the unified brand: per-brand CLIP indices already
# contain only that one brand's own article_ids, so an inactive per-brand slug (e.g.
# "berrylush") is handled by short-circuiting before the FAISS index is even loaded —
# see the inactive-store guard in find_anchor_from_image.
_CATALOGUE_BASE = Path(__file__).parent.parent.parent.parent / "data" / "processed"

# article_id -> lowercase store slug, cached per catalogue base path (keyed so tests
# using a tmp_path override don't collide with the real cache entry).
_store_map_cache: dict[str, dict[str, str]] = {}
_store_map_lock = threading.Lock()


def _index_dir(brand: str, base: Path | None = None) -> Path:
    """Return the per-brand CLIP index directory."""
    return (base or _CLIP_INDEX_BASE) / brand


def _load_brand_index(brand: str, base: Path | None = None) -> _IndexEntry | None:
    """Load (or return cached) CLIP index for *brand*.

    Returns ``None`` when the index files do not exist (feature off for brand).
    Raises on corrupt files so the endpoint can surface a 503.
    """
    if brand in _index_cache:
        return _index_cache[brand]

    with _cache_lock:
        if brand in _index_cache:
            return _index_cache[brand]

        idx_dir = _index_dir(brand, base)
        faiss_path = idx_dir / "clip.faiss"
        ids_path = idx_dir / "clip_article_ids.npy"

        if not faiss_path.exists() or not ids_path.exists():
            logger.warning(
                "CLIP index not found for brand=%s (expected %s). "
                "Run scripts/build_clip_index.py to build it.",
                brand,
                idx_dir,
            )
            return None

        import faiss  # noqa: I001  # lazy import — faiss is a heavy dep

        index = faiss.read_index(str(faiss_path))
        article_ids = np.load(str(ids_path), allow_pickle=True)
        entry: _IndexEntry = (index, article_ids)
        _index_cache[brand] = entry
        logger.info(
            "CLIP index loaded: brand=%s vectors=%d", brand, index.ntotal
        )
        return entry


def _load_unified_store_map(base: Path | None = None) -> dict[str, str] | None:
    """Load an article_id -> lowercase store slug map from the unified catalogue.

    Returns ``None`` when the catalogue parquet is absent (e.g. isolated unit tests
    that only stub the FAISS index) so callers degrade to "no store filtering"
    rather than crashing.  Cached for the process lifetime, keyed by base path.
    """
    cache_key = str(base) if base is not None else "__default__"
    if cache_key in _store_map_cache:
        return _store_map_cache[cache_key]

    with _store_map_lock:
        if cache_key in _store_map_cache:
            return _store_map_cache[cache_key]

        path = (base or _CATALOGUE_BASE) / UNIFIED_BRAND / "catalogue.parquet"
        if not path.exists():
            return None

        import pandas as pd  # lazy import — only the unified brand needs this

        try:
            df = pd.read_parquet(path, columns=["article_id", "store"])
        except Exception:
            logger.warning("Failed to load unified catalogue store map from %s", path)
            return None

        mapping = {
            str(aid): str(store).lower()
            for aid, store in zip(df["article_id"], df["store"], strict=True)
            if store
        }
        _store_map_cache[cache_key] = mapping
        return mapping


def warm(brand: str, model_id: str = "clip-ViT-B-32", _index_base: Path | None = None) -> bool:
    """Eagerly load the brand's CLIP FAISS index and the CLIP encoder.

    Intended to be called from a background thread at API startup so the
    first real ``/style/from-image`` request does not pay the cold-load cost
    (index read from disk + CLIP model instantiation).  Safe to call even
    when the brand has no CLIP index — returns False in that case instead
    of raising.

    Args:
        brand:       Brand slug whose CLIP index should be warmed.
        model_id:    CLIP model identifier (must match config.yaml clip.model).
        _index_base: Override the base path for CLIP indices (used in tests).

    Returns:
        True when both the index and encoder loaded successfully, False when
        the brand has no CLIP index on disk.

    Raises:
        RuntimeError: If the CLIP encoder cannot be loaded (propagated from
        ``get_clip_encoder``); callers running this in a background thread
        should catch and log rather than let it crash the thread silently.
    """
    entry = _load_brand_index(brand, _index_base)
    if entry is None:
        return False
    get_clip_encoder(model_id)
    return True


def find_anchor_from_image(
    img: "PILImage",
    brand: str,
    top_k: int = 5,
    *,
    model_id: str = "clip-ViT-B-32",
    _index_base: Path | None = None,
    _catalogue_base: Path | None = None,
) -> list[str]:
    """Find the nearest catalogue anchors for an uploaded image.

    Uses CLIP zero-shot retrieval: the image is projected into the shared
    image+text space and compared against pre-computed CLIP-text embeddings
    of each item's ``search_text``.

    Privacy: the image is encoded in-memory only.  After this function returns
    the caller MUST drop the reference to ``img`` and the underlying bytes.

    Inactive-store exclusion (STORE_CONFIG single source of truth):
      - Per-brand brand slug (e.g. ``"berrylush"``): if that slug is itself
        flagged inactive, every candidate would come from an excluded store,
        so this returns ``[]`` immediately without loading the FAISS index.
      - Unified brand (``"unified"``): candidates are joined against the
        unified catalogue's ``store`` column and any hit from an inactive
        store is dropped before the result list is returned.

    Args:
        img:         In-memory PIL Image (bytes never touched after this call).
        brand:       Brand slug, e.g. ``"hm"`` or ``"myntra"`` or ``"unified"``.
        top_k:       Number of candidate article_ids to return.
        model_id:    CLIP model identifier (must match the one used at index build time).
        _index_base: Override the base path for CLIP indices (used in tests).
        _catalogue_base: Override the base path for catalogue parquets (used in tests).

    Returns:
        List of up to ``top_k`` article_id strings, ranked by descending
        CLIP cosine similarity.  Empty when the brand index is absent or the
        brand itself is an inactive store.

    Raises:
        RuntimeError: If the CLIP encoder cannot be loaded.
    """
    brand_lower = brand.lower()
    inactive_stores = get_inactive_stores()

    # Per-brand CLIP indices are 1:1 with a store slug — if that store is inactive,
    # every candidate the index could return is excluded, so skip loading entirely.
    if brand_lower != UNIFIED_BRAND and brand_lower in inactive_stores:
        logger.info(
            "find_anchor_from_image: brand=%s is an inactive store; returning []", brand
        )
        return []

    entry = _load_brand_index(brand, _index_base)
    if entry is None:
        logger.warning("find_anchor_from_image: no CLIP index for brand=%s; returning []", brand)
        return []

    faiss_index, article_ids = entry

    encoder = get_clip_encoder(model_id)
    img_vec = encoder.encode_image(img)  # (512,) float32, L2-normalised

    query = img_vec.reshape(1, -1).astype(np.float32)
    actual_k = min(top_k, faiss_index.ntotal)
    scores, indices = faiss_index.search(query, actual_k)

    # Unified brand only: join against the catalogue store column so results from
    # an inactive store never surface, even though its rows remain in the index.
    store_map: dict[str, str] | None = None
    if brand_lower == UNIFIED_BRAND:
        store_map = _load_unified_store_map(_catalogue_base)

    results: list[str] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        article_id = str(article_ids[idx])
        if store_map is not None and store_map.get(article_id) in inactive_stores:
            continue
        results.append(article_id)
    logger.debug(
        "find_anchor_from_image: brand=%s top_k=%d returned=%d",
        brand,
        top_k,
        len(results),
    )
    return results
