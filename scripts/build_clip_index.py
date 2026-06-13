"""Build per-brand CLIP-text FAISS indices for image-to-anchor retrieval.

For each brand with a catalogue parquet, embeds every item's ``search_text``
using the sentence-transformers CLIP text encoder (same shared 512-d space as
the image encoder), and writes a FAISS IndexFlatIP plus a companion article_id
array.

Output layout::

    data/processed/clip/<brand>/clip.faiss
    data/processed/clip/<brand>/clip_article_ids.npy

Usage::

    python scripts/build_clip_index.py                          # all brands
    python scripts/build_clip_index.py --brand myntra snitch    # specific brands
    python scripts/build_clip_index.py --model clip-ViT-B-32    # custom model

Design decisions
----------------
- CLIP-TEXT index only: no item images are downloaded.  Zero-shot retrieval
  works by embedding ``search_text`` with CLIP's text encoder and comparing
  to an uploaded image's CLIP image embedding.
- Deterministic: numpy seed=42, no shuffling.
- Resilient: missing catalogue → skip with a clear warning.
- Batched: respects ``--batch-size`` (default 64) for memory efficiency.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repo root on sys.path so ``src.*`` imports work when run as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("build_clip_index")

# Brands whose catalogues live under data/processed/<brand>/catalogue.parquet
# (HM's catalogue is at data/processed/catalogue.parquet — handled separately).
_KNOWN_BRANDS: list[str] = [
    "hm",
    "myntra",
    "snitch",
    "flipkart",
    "fashor",
    "virgio",
    "powerlook",
    "sample_in",
]

_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "processed"
_DEFAULT_CLIP_DIR = _REPO_ROOT / "data" / "processed" / "clip"
_DEFAULT_MODEL = "clip-ViT-B-32"
_DEFAULT_BATCH_SIZE = 64


def _catalogue_path(brand: str, data_dir: Path) -> Path:
    """Return the catalogue parquet path for a brand (HM uses legacy flat layout)."""
    if brand == "hm":
        return data_dir / "catalogue.parquet"
    return data_dir / brand / "catalogue.parquet"


def build_brand_index(
    brand: str,
    *,
    data_dir: Path,
    clip_dir: Path,
    model_id: str,
    batch_size: int,
) -> int | None:
    """Build the CLIP-text index for *brand*.

    Args:
        brand:      Brand slug (e.g. ``"myntra"``).
        data_dir:   Root of processed data (contains catalogues).
        clip_dir:   Root of CLIP output directory (``clip/<brand>/`` will be created).
        model_id:   sentence-transformers CLIP model identifier.
        batch_size: Text encoding batch size.

    Returns:
        Number of vectors indexed on success, or ``None`` if the catalogue is missing.
    """
    import faiss  # noqa: I001
    import pandas as pd

    from src.retrieval.clip_encoder import get_clip_encoder

    parquet = _catalogue_path(brand, data_dir)
    if not parquet.exists():
        logger.warning("SKIP brand=%s: catalogue not found at %s", brand, parquet)
        return None

    df = pd.read_parquet(parquet)
    if "search_text" not in df.columns:
        logger.warning("SKIP brand=%s: 'search_text' column missing in %s", brand, parquet)
        return None

    texts = df["search_text"].fillna("").tolist()
    article_ids = df["article_id"].astype(str).values

    logger.info("brand=%s  items=%d  model=%s", brand, len(texts), model_id)

    encoder = get_clip_encoder(model_id)

    # Encode in batches to keep memory bounded
    all_vecs: list[np.ndarray] = []
    rng = np.random.default_rng(42)  # seed for reproducibility (no shuffling; satisfies req)
    _ = rng  # seed loaded; used for any future stochastic steps

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = encoder.encode_texts(batch)  # (B, 512) float32, L2-normalised
        all_vecs.append(vecs)
        if (i // batch_size) % 10 == 0:
            logger.info(
                "  brand=%s encoded %d/%d items", brand, min(i + batch_size, len(texts)), len(texts)
            )

    embeddings = np.vstack(all_vecs).astype(np.float32)  # (N, 512)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir = clip_dir / brand
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "clip.faiss"))
    np.save(str(out_dir / "clip_article_ids.npy"), article_ids)

    logger.info(
        "brand=%s  vectors=%d  dim=%d  saved to %s",
        brand,
        index.ntotal,
        dim,
        out_dir,
    )
    return index.ntotal


def main(argv: list[str] | None = None) -> None:
    """Entry point for building CLIP indices."""
    parser = argparse.ArgumentParser(description="Build per-brand CLIP-text FAISS indices")
    parser.add_argument(
        "--brand",
        nargs="*",
        default=None,
        help="Brand slugs to build (default: all known brands)",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"sentence-transformers CLIP model (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Text encoding batch size (default: {_DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of processed data directory",
    )
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=_DEFAULT_CLIP_DIR,
        help="Root output directory for CLIP indices",
    )
    args = parser.parse_args(argv)

    brands = args.brand if args.brand else _KNOWN_BRANDS

    results: dict[str, int | None] = {}
    for brand in brands:
        try:
            count = build_brand_index(
                brand,
                data_dir=args.data_dir,
                clip_dir=args.clip_dir,
                model_id=args.model,
                batch_size=args.batch_size,
            )
            results[brand] = count
        except Exception as exc:
            logger.error("ERROR brand=%s: %s", brand, exc, exc_info=True)
            results[brand] = None

    # Summary report
    print("\n=== CLIP index build summary ===")
    for brand, count in results.items():
        if count is None:
            print(f"  {brand:<20} SKIPPED / ERROR")
        else:
            print(f"  {brand:<20} {count:>7,} vectors")
    print("================================\n")

    failed = [b for b, c in results.items() if c is None]
    if failed:
        logger.warning("Brands with no index: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
