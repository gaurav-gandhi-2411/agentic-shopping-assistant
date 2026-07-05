from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_INDEX_FILES = frozenset(
    ["dense.faiss", "bm25.pkl", "bm25_article_ids.npy", "dense_article_ids.npy", "catalogue.parquet"]
)

# CLIP index filenames expected under gs://<bucket>/<prefix>/clip/<brand>/
_CLIP_FILES = frozenset(["clip.faiss", "clip_article_ids.npy"])

# Sentinel value: when BRAND is set to this string (or left unset with UNIFIED=1),
# the service loads the cross-store unified index instead of a per-brand index.
UNIFIED_BRAND = "unified"


def ensure_index_dir(
    brand: str,
    default_dir: Path,
    index_store_uri: str | None,
) -> Path:
    """Return the local directory containing retrieval index files for *brand*.

    Unified mode (brand == "unified"):
        - Local: returns default_dir/unified/
        - GCS:   downloads gs://<bucket>/<prefix>/unified/ blobs into default_dir/unified/

    Per-brand mode (legacy):
        - Local: returns default_dir for hm, default_dir/<brand> for all other brands.
        - GCS:   downloads brand-keyed blobs from gs://<bucket>/<prefix>/<brand>/

    This function is additive — the per-brand path is unchanged so existing per-brand
    deployments continue to work without modification.
    """
    if brand == UNIFIED_BRAND:
        # Unified cross-store index
        local_unified_dir = default_dir / UNIFIED_BRAND
        if not index_store_uri:
            return local_unified_dir
        local_unified_dir.mkdir(parents=True, exist_ok=True)
        _download_from_gcs(index_store_uri, UNIFIED_BRAND, local_unified_dir)
        return local_unified_dir

    # --- legacy per-brand path (unchanged) ---
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


def download_supplementary_assets(uri: str, brand: str, repo_root: Path) -> None:
    """Download CLIP index and Shopify variant map for *brand* from GCS.

    Called at startup alongside :func:`ensure_index_dir` when ``INDEX_STORE_URI``
    is set.  Both asset families degrade gracefully when absent (the respective
    feature simply becomes unavailable), so missing blobs are logged as warnings
    rather than raising.

    Assets downloaded:
    - ``gs://<bucket>/<prefix>/clip/<brand>/clip.faiss``          →
      ``<repo_root>/data/processed/clip/<brand>/clip.faiss``
    - ``gs://<bucket>/<prefix>/clip/<brand>/clip_article_ids.npy`` →
      ``<repo_root>/data/processed/clip/<brand>/clip_article_ids.npy``
    - ``gs://<bucket>/<prefix>/shopify_variants/<brand>.json``     →
      ``<repo_root>/data/processed/shopify_variants/<brand>.json``

    For ``brand == UNIFIED_BRAND`` there is no single per-brand variant file —
    cart-link resolution loads variant maps per underlying store slug — so instead
    every ``*.json`` blob found under ``shopify_variants/`` (e.g. ``snitch.json``,
    ``fashor.json``, ``powerlook.json``) is downloaded into the same local directory.

    Args:
        uri:       The ``INDEX_STORE_URI`` value (e.g. ``gs://asa-demo-indices/``).
        brand:     Active brand slug (from the ``BRAND`` env var).
        repo_root: Absolute path to the repository root; used to compute local
                   destination paths that match what the loaders expect.
    """
    from google.cloud import storage  # type: ignore[import-untyped]  # noqa: I001  # lazy: only when INDEX_STORE_URI is set

    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    path_prefix = parsed.path.strip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # ------------------------------------------------------------------
    # 1. CLIP index: gs://<bucket>/<prefix>/clip/<brand>/{clip.faiss,clip_article_ids.npy}
    # ------------------------------------------------------------------
    clip_local_dir = repo_root / "data" / "processed" / "clip" / brand
    clip_local_dir.mkdir(parents=True, exist_ok=True)

    clip_gcs_prefix = f"{path_prefix}/clip/{brand}/" if path_prefix else f"clip/{brand}/"

    logger.info(
        "Downloading CLIP index from gs://%s/%s → %s",
        bucket_name,
        clip_gcs_prefix,
        clip_local_dir,
    )
    for filename in _CLIP_FILES:
        blob_name = f"{clip_gcs_prefix}{filename}"
        blob = bucket.blob(blob_name)
        dest = clip_local_dir / filename
        try:
            blob.download_to_filename(str(dest))
            logger.info("  gs://%s/%s → %s", bucket_name, blob_name, dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CLIP asset not found or download failed — image-search feature will be "
                "unavailable for brand=%s. blob=gs://%s/%s error=%s",
                brand,
                bucket_name,
                blob_name,
                exc,
            )

    # ------------------------------------------------------------------
    # 2. Shopify variant map: gs://<bucket>/<prefix>/shopify_variants/<brand>.json
    # ------------------------------------------------------------------
    variants_local_dir = repo_root / "data" / "processed" / "shopify_variants"
    variants_local_dir.mkdir(parents=True, exist_ok=True)

    if brand == UNIFIED_BRAND:
        _download_all_variant_maps(bucket, bucket_name, path_prefix, variants_local_dir)
    else:
        variants_blob_name = (
            f"{path_prefix}/shopify_variants/{brand}.json"
            if path_prefix
            else f"shopify_variants/{brand}.json"
        )
        variants_dest = variants_local_dir / f"{brand}.json"

        logger.info(
            "Downloading Shopify variant map from gs://%s/%s → %s",
            bucket_name,
            variants_blob_name,
            variants_dest,
        )
        blob = bucket.blob(variants_blob_name)
        try:
            blob.download_to_filename(str(variants_dest))
            logger.info("  gs://%s/%s → %s", bucket_name, variants_blob_name, variants_dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Shopify variant map not found or download failed — cart-link feature will be "
                "unavailable for brand=%s. blob=gs://%s/%s error=%s",
                brand,
                bucket_name,
                variants_blob_name,
                exc,
            )


def _download_all_variant_maps(
    bucket: object,
    bucket_name: str,
    path_prefix: str,
    variants_local_dir: Path,
) -> None:
    """Download every ``*.json`` Shopify variant map blob for the unified brand.

    Unlike per-brand mode (a single known blob name), unified mode has no fixed
    filename to request — the underlying store slugs (snitch, fashor, powerlook,
    virgio, ...) are only discoverable by listing the ``shopify_variants/`` prefix.
    Missing prefix, zero blobs, listing failures, and per-file download failures are
    all non-fatal: they are logged as warnings and the cart-link feature degrades
    for the affected store(s) rather than failing service startup.

    Args:
        bucket:              A ``google.cloud.storage.Bucket``-like object.
        bucket_name:         Bucket name, used only for log messages.
        path_prefix:         Optional path prefix parsed from ``INDEX_STORE_URI``.
        variants_local_dir:  Local directory to download each variant map into.
    """
    variants_gcs_prefix = (
        f"{path_prefix}/shopify_variants/" if path_prefix else "shopify_variants/"
    )
    logger.info("Listing Shopify variant maps at gs://%s/%s", bucket_name, variants_gcs_prefix)

    try:
        variant_blobs = [
            b
            for b in bucket.list_blobs(prefix=variants_gcs_prefix)  # type: ignore[attr-defined]
            if b.name.endswith(".json")
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to list Shopify variant maps — cart-link feature will be unavailable "
            "for brand=%s. prefix=gs://%s/%s error=%s",
            UNIFIED_BRAND,
            bucket_name,
            variants_gcs_prefix,
            exc,
        )
        return

    if not variant_blobs:
        logger.warning(
            "No Shopify variant maps found at gs://%s/%s — cart-link feature will be "
            "unavailable for brand=%s.",
            bucket_name,
            variants_gcs_prefix,
            UNIFIED_BRAND,
        )
        return

    for blob in variant_blobs:
        filename = blob.name[len(variants_gcs_prefix) :]
        if not filename or "/" in filename:
            continue  # skip directory markers and nested paths
        dest = variants_local_dir / filename
        try:
            blob.download_to_filename(str(dest))
            logger.info("  gs://%s/%s → %s", bucket_name, blob.name, dest)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Shopify variant map download failed for %s — cart-link feature may be "
                "degraded for that store. blob=gs://%s/%s error=%s",
                filename,
                bucket_name,
                blob.name,
                exc,
            )

    logger.info("Shopify variant maps download complete: %d files", len(variant_blobs))
