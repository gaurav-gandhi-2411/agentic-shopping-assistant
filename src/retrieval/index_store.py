from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_INDEX_FILES = frozenset(
    ["dense.faiss", "bm25.pkl", "bm25_article_ids.npy", "dense_article_ids.npy", "catalogue.parquet"]
)


def ensure_index_dir(
    brand: str,
    default_dir: Path,
    index_store_uri: str | None,
) -> Path:
    """Return the local directory containing retrieval index files for *brand*.

    Local (dev) mode: INDEX_STORE_URI unset → returns default_dir for hm,
    default_dir/<brand> for all other brands (matches 01_build_retrieval.py --out).

    GCS mode: INDEX_STORE_URI set → downloads brand-keyed blobs from
    gs://<bucket>/<prefix>/<brand>/ into default_dir/<brand>/ and returns that path.
    """
    if not index_store_uri:
        if brand == "hm":
            return default_dir
        return default_dir / brand

    local_brand_dir = default_dir / brand
    local_brand_dir.mkdir(parents=True, exist_ok=True)
    _download_from_gcs(index_store_uri, brand, local_brand_dir)
    return local_brand_dir


def _download_from_gcs(uri: str, brand: str, local_dir: Path) -> None:
    from google.cloud import storage  # type: ignore[import-untyped]  # noqa: I001  # lazy: only when INDEX_STORE_URI is set

    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    path_prefix = parsed.path.strip("/")
    gcs_prefix = f"{path_prefix}/{brand}/" if path_prefix else f"{brand}/"

    logger.info("Downloading index from gs://%s/%s → %s", bucket_name, gcs_prefix, local_dir)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))

    if not blobs:
        raise RuntimeError(
            f"No index files found at gs://{bucket_name}/{gcs_prefix}. "
            f"Run scripts/01_build_retrieval.py --brand {brand} and upload to GCS first."
        )

    for blob in blobs:
        filename = blob.name[len(gcs_prefix):]
        if not filename or "/" in filename:
            continue  # skip directory markers and subdirectory blobs
        dest = local_dir / filename
        logger.info("  gs://%s/%s → %s", bucket_name, blob.name, dest)
        blob.download_to_filename(str(dest))

    logger.info("Index download complete: %d files", len(blobs))
