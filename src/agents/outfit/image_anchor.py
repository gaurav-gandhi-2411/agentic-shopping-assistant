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

from src.retrieval.clip_encoder import get_clip_encoder

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


def find_anchor_from_image(
    img: "PILImage",
    brand: str,
    top_k: int = 5,
    *,
    model_id: str = "clip-ViT-B-32",
    _index_base: Path | None = None,
) -> list[str]:
    """Find the nearest catalogue anchors for an uploaded image.

    Uses CLIP zero-shot retrieval: the image is projected into the shared
    image+text space and compared against pre-computed CLIP-text embeddings
    of each item's ``search_text``.

    Privacy: the image is encoded in-memory only.  After this function returns
    the caller MUST drop the reference to ``img`` and the underlying bytes.

    Args:
        img:         In-memory PIL Image (bytes never touched after this call).
        brand:       Brand slug, e.g. ``"hm"`` or ``"myntra"``.
        top_k:       Number of candidate article_ids to return.
        model_id:    CLIP model identifier (must match the one used at index build time).
        _index_base: Override the base path for CLIP indices (used in tests).

    Returns:
        List of up to ``top_k`` article_id strings, ranked by descending
        CLIP cosine similarity.  Empty when the brand index is absent.

    Raises:
        RuntimeError: If the CLIP encoder cannot be loaded.
    """
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

    results: list[str] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append(str(article_ids[idx]))
    logger.debug(
        "find_anchor_from_image: brand=%s top_k=%d returned=%d",
        brand,
        top_k,
        len(results),
    )
    return results
