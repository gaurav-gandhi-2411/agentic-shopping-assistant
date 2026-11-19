"""Glue between the API lifespan and the index loading machinery (Wave 1b V3(b)).

Kept out of ``api/main.py`` so it can be tested with a fake store and a fake embedder.

Revision-pinned build (the rollout design): ``INDEX_BUILD_ID`` pins this revision to one build,
loaded by :func:`load_pinned_snapshot`. The pointer/hot-reload machinery below is DEPRECATED
(default OFF, no further work) and never combined with a pin.

DEPRECATED pointer startup: when a bucket store is configured and ``pointers/`` names a build, the newest build is
loaded instead of the flat layout; no pointer, or ANY failure while loading it, falls back to the
flat layout (the loader works before any publisher, pointer or lifecycle rule exists).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from src.config.store_registry import STORE_REGISTRY
from src.retrieval.build_manifest import BuildManifest, BuildValidationError
from src.retrieval.index_reload import (
    GcsIndexSource,
    IndexReloader,
    IndexSource,
    IndexSourceError,
    LoadBuild,
    LocalDirSource,
    fetch_and_load_build,
    hot_reload_enabled,
    latest_pointer,
    reload_interval_s,
    sanitize,
    verify_snapshot_index,
)
from src.retrieval.pinned_build import (
    PinRequest,
    download_pinned_build,
    make_tmp_dir,
    reason_code,
    remove_tmp_dir,
)
from src.retrieval.snapshot import IndexSnapshot, load_clip_entry, load_index_parts

logger = logging.getLogger(__name__)


def make_index_source(uri: str | None, environ: dict[str, str] | None = None) -> IndexSource | None:
    """The pointer/build store for ``uri``, or None when builds are not in play.

    Only ``gs://`` URIs are supported, and ``INDEX_HOT_RELOAD=1`` must be set explicitly: it is
    the one opt-in for BOTH the startup pointer lookup and the background reload. Unset (the
    default) returns None and the service loads the flat layout exactly as before V3(b); the
    swap is off by default because it does not fit the 4 GiB limit (see DEPLOY.md).
    """
    if not uri or not uri.startswith("gs://"):
        return None
    if not hot_reload_enabled({**os.environ, **(environ or {}), "INDEX_STORE_URI": uri}):
        return None
    return GcsIndexSource(uri)


def make_pinned_source(uri: str | None) -> IndexSource | None:
    """The store a pinned build is read from, or None when ``uri`` cannot hold builds.

    ``gs://<bucket>[/prefix]`` is the production source. ``file:///<dir>`` (a directory laid out
    like the bucket) exists for local development and tests ONLY: it lets the real app boot be
    exercised without a cloud; it is never set on Cloud Run.
    """
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme == "gs" and parsed.netloc:
        return GcsIndexSource(uri)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost") and parsed.path:
        return LocalDirSource(Path(url2pathname(parsed.path)))
    return None


def load_pinned_snapshot(
    pin: PinRequest,
    uri: str | None,
    config: dict,
    tmp_root: Path,
    *,
    source: IndexSource | None = None,
    model: Any = None,
    registry_slugs: frozenset[str] | None = None,
) -> tuple[IndexSnapshot | None, str | None]:
    """Load the pinned build as a (graph-less) snapshot: ``(snapshot, None)`` or ``(None, code)``.

    Downloads ONLY ``builds/<id>/`` (the manifest's files) into a scratch directory under
    ``tmp_root``, verifies, loads, checks vector counts, then deletes the scratch directory on
    EVERY outcome, so one build is on disk only while it is read and one in memory afterwards.
    Never raises: any failure is ONE structured error line (fixed reason code, internal reason
    token, exception class name; never exception text) and a fixed code for /readyz; the caller then serves the legacy
    index. Never touches another build.
    """
    if pin.failure is not None or pin.build_id is None:
        _log_pin_failure(pin.build_id, pin.failure or "id_invalid", "id_invalid", "")
        return None, pin.failure or "id_invalid"
    src = source if source is not None else make_pinned_source(uri)
    if src is None:
        _log_pin_failure(pin.build_id, "no_source", "no_source", "")
        return None, "no_source"
    tmp: Path | None = None
    try:
        tmp = make_tmp_dir(tmp_root)
        build_dir = tmp / pin.build_id
        build_dir.mkdir()
        slugs = registry_slugs if registry_slugs is not None else frozenset(STORE_REGISTRY)
        manifest = download_pinned_build(src, pin.build_id, build_dir, registry_slugs=slugs)
        snap = parts_loader(config, model=model)(build_dir, manifest, None)
        verify_snapshot_index(snap, manifest)
        logger.info(
            "index_pinned_loaded build_id=%s items=%d", pin.build_id, len(snap.catalogue_df)
        )
        return snap, None
    except Exception as exc:  # noqa: BLE001 - ANY failure falls back to legacy, never a crash
        code = reason_code(exc)
        # Only fixed tokens reach the log: BuildValidationError.reason is a short machine code;
        # for anything else just the exception CLASS NAME (never its text, which can carry
        # paths, URLs or credentials).
        internal = exc.reason if isinstance(exc, BuildValidationError) else type(exc).__name__
        _log_pin_failure(pin.build_id, code, internal, type(exc).__name__)
        return None, code
    finally:
        if tmp is not None:
            remove_tmp_dir(tmp)


def _log_pin_failure(build_id: str | None, code: str, reason: str, exc_class: str) -> None:
    """The ONE structured error line for a pinned build that could not be loaded.

    Carries only fixed tokens: the reason code, the validated build id (or none), the internal
    reason token and the exception class name. No exception text, no paths, no URLs.
    """
    logger.error(
        "index_pin_failed pinned_build_id=%s code=%s reason=%s exc=%s",
        build_id,
        code,
        reason,
        exc_class,
    )


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
