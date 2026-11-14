"""Glue between the API lifespan and the index hot-reload machinery (Wave 1b V3(b)).

Kept out of ``api/main.py`` so it can be tested with a fake store and a fake embedder.

Startup: when a bucket store is configured and ``pointers/`` names a build, the newest build is
loaded instead of the flat layout; no pointer, or ANY failure while loading it, falls back to the
flat layout (the loader works before any publisher, pointer or lifecycle rule exists).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.config.store_registry import STORE_REGISTRY
from src.retrieval.build_manifest import BuildManifest
from src.retrieval.index_reload import (
    GcsIndexSource,
    IndexReloader,
    IndexSource,
    IndexSourceError,
    LoadBuild,
    fetch_and_load_build,
    hot_reload_enabled,
    latest_pointer,
    reload_interval_s,
    sanitize,
)
from src.retrieval.snapshot import IndexSnapshot, load_clip_entry, load_index_parts

logger = logging.getLogger(__name__)


def make_index_source(uri: str | None, environ: dict[str, str] | None = None) -> IndexSource | None:
    """The pointer/build store for ``uri``, or None when builds are not in play.

    Only ``gs://`` URIs are supported, and ``INDEX_HOT_RELOAD=0`` returns None (it is the
    one kill-switch for BOTH the startup pointer lookup and the background reload: the service
    then behaves exactly as before V3(b)).
    """
    if not uri or not uri.startswith("gs://"):
        return None
    if not hot_reload_enabled({**os.environ, **(environ or {}), "INDEX_STORE_URI": uri}):
        return None
    return GcsIndexSource(uri)


def parts_loader(config: dict, *, model: Any = None) -> LoadBuild:
    """LoadBuild that loads retriever + public catalogue + CLIP only (no graphs): startup use,
    where the LLM, hence the graphs, is not ready yet."""

    def load(
        build_dir: Path, manifest: BuildManifest, current: IndexSnapshot | None
    ) -> IndexSnapshot:
        reuse = current.retriever.dense.model if current is not None else model
        retriever, df = load_index_parts(build_dir, config, model=reuse)
        return IndexSnapshot(
            retriever=retriever,
            catalogue_df=df,
            manifest=manifest,
            build_id=manifest.build_id,
            clip=load_clip_entry(build_dir),
            source="build",
        )

    return load


def graph_loader(config: dict, llm: Any) -> LoadBuild:
    """LoadBuild for hot reload: parts plus both agent graphs compiled around the NEW retriever
    and DataFrame (the swap replaces graphs and index together). Reuses the embedding model of the
    snapshot being replaced instead of loading a second MiniLM."""
    import api.deps as deps

    inner = parts_loader(config)

    def load(
        build_dir: Path, manifest: BuildManifest, current: IndexSnapshot | None
    ) -> IndexSnapshot:
        parts = inner(build_dir, manifest, current)
        return deps.make_snapshot(
            parts.retriever,
            parts.catalogue_df,
            llm,
            config,
            manifest=manifest,
            clip=parts.clip,
            source="build",
        )

    return load


def load_newest_build(
    source: IndexSource, config: dict, *, model: Any = None
) -> IndexSnapshot | None:
    """Newest published build as a (graph-less) snapshot, or None to fall back to the flat layout.

    None when there is no pointer; also None, with ONE warning, on any failure.
    """
    try:
        build_id = latest_pointer(source)
        if build_id is None:
            logger.info("index_startup no pointer in the store; using the flat layout")
            return None
        snap = fetch_and_load_build(
            source,
            build_id,
            parts_loader(config, model=model),
            current=None,
            registry_slugs=frozenset(STORE_REGISTRY),
        )
        logger.info("index_startup loaded build_id=%s items=%d", build_id, len(snap.catalogue_df))
        return snap
    except IndexSourceError as exc:
        logger.warning("index_startup_fallback reason=source detail=%s", sanitize(str(exc)))
    except Exception as exc:  # noqa: BLE001 - ANY failure falls back to the flat layout
        reason = getattr(exc, "reason", type(exc).__name__)
        logger.warning("index_startup_fallback reason=%s detail=%s", reason, sanitize(str(exc)))
    return None


def start_hot_reload(source: IndexSource, config: dict, llm: Any) -> IndexReloader:
    """Create, register in ``api.deps`` and start the reload thread."""
    import api.deps as deps

    reloader = IndexReloader(
        source,
        graph_loader(config, llm),
        deps.get_snapshot_or_none,
        deps.install_snapshot,
        interval_s=reload_interval_s(),
        registry_slugs=frozenset(STORE_REGISTRY),
    )
    deps._reloader = reloader
    reloader.start()
    logger.info("index_hot_reload started interval_s=%s", reload_interval_s())
    return reloader
