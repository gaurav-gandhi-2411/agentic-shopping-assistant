"""In-process hot reload of the retrieval index (Wave 1b V3(b)).

A background daemon thread checks the bucket's ``pointers/`` listing every
``INDEX_RELOAD_INTERVAL_S`` (default 900 s, minimum 60 s). When the newest pointer names a build
other than the one being served it:

  1. downloads that build into a fresh temp directory (never ``/app/data``),
  2. verifies it (manifest sha256 and sizes, counts, store set, +-30% gate vs the served build),
  3. loads it into a NEW :class:`~src.retrieval.snapshot.IndexSnapshot` (graphs compiled from the
     new retriever), checks vector count and dimensions, deletes the temp directory,
  4. runs a cheap smoke query on the new snapshot,
  5. swaps the single snapshot reference atomically, then releases the old snapshot
     (references dropped, ``gc.collect()``; in-flight turns keep theirs until they finish).

ANY failure at ANY step keeps the current snapshot serving, logs ONE structured warning
(``index_reload_failed reason=...``; no URLs, no credentials) and backs off exponentially
(interval, 2x, 4x ... capped at 1 h). The loop never raises and never blocks a request. Reloads
are single-flight: an attempt that finds another one running returns ``busy`` immediately.

The store is behind the small :class:`IndexSource` protocol (GCS in production, a local directory
in tests and for local experiments). Layout, relative to the store root::

    pointers/<UTC timestamp>.json     {"build_id": "<UTC timestamp>"}   (create-only; the largest
                                       name is "latest"; deleting it rolls back to the previous)
    builds/<build_id>/<files>         index files + build_manifest.json (flat)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from src.retrieval.build_manifest import (
    BuildManifest,
    BuildValidationError,
    load_manifest,
    verify_build_dir,
    verify_loaded_index,
)
from src.retrieval.snapshot import IndexSnapshot

logger = logging.getLogger(__name__)

POINTERS_PREFIX = "pointers/"
BUILDS_PREFIX = "builds/"
# Build ids are UTC timestamps; the strict pattern also keeps a hostile pointer from steering a
# download or a temp path outside builds/<id>/ ("../x").
BUILD_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
POINTER_NAME_RE = re.compile(r"^\d{8}T\d{6}Z\.json$")

DEFAULT_INTERVAL_S = 900.0  # 15 min
MIN_INTERVAL_S = 60.0
MAX_BACKOFF_S = 3600.0  # 1 h

INTERVAL_ENV = "INDEX_RELOAD_INTERVAL_S"
SWITCH_ENV = "INDEX_HOT_RELOAD"

_URL_CREDENTIALS = re.compile(r"(://)[^/@\s]+@")


class IndexSourceError(Exception):
    """The store could not be read (listing, pointer or download)."""


class IndexSource(Protocol):
    """Where pointers and builds come from (GCS in production, a directory in tests)."""

    def list_pointer_names(self) -> list[str]:
        """File names under ``pointers/`` (e.g. ``20261115T020000Z.json``)."""
        ...

    def read_pointer(self, name: str) -> Mapping[str, Any]:
        """Parsed JSON of ``pointers/<name>``."""
        ...

    def download_build(self, build_id: str, dest: Path) -> None:
        """Copy every file of ``builds/<build_id>/`` (flat) into the existing directory ``dest``."""
        ...


class LocalDirSource:
    """IndexSource over a local directory laid out like the bucket (tests, local experiments)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list_pointer_names(self) -> list[str]:
        d = self.root / "pointers"
        return sorted(p.name for p in d.iterdir()) if d.is_dir() else []

    def read_pointer(self, name: str) -> Mapping[str, Any]:
        return json.loads((self.root / "pointers" / name).read_text(encoding="utf-8"))

    def download_build(self, build_id: str, dest: Path) -> None:
        src = self.root / "builds" / build_id
        if not src.is_dir():
            raise IndexSourceError("build directory not found")
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)


class GcsIndexSource:
    """IndexSource over ``gs://<bucket>[/<prefix>]`` (the INDEX_STORE_URI bucket).

    Uses the same ``google.cloud.storage`` client construction as ``index_store`` (ambient
    credentials of the runtime service account; read access only).
    """

    def __init__(self, uri: str) -> None:
        parsed = urlparse(uri)
        self._bucket_name = parsed.netloc
        prefix = parsed.path.strip("/")
        self._prefix = f"{prefix}/" if prefix else ""
        self._bucket: Any = None

    def _bkt(self) -> Any:
        if self._bucket is None:
            from google.cloud import storage  # lazy: only when a bucket store is configured

            self._bucket = storage.Client().bucket(self._bucket_name)
        return self._bucket

    def list_pointer_names(self) -> list[str]:
        prefix = f"{self._prefix}{POINTERS_PREFIX}"
        names = []
        for blob in self._bkt().list_blobs(prefix=prefix):
            name = blob.name[len(prefix) :]
            if name and "/" not in name:
                names.append(name)
        return sorted(names)

    def read_pointer(self, name: str) -> Mapping[str, Any]:
        blob = self._bkt().blob(f"{self._prefix}{POINTERS_PREFIX}{name}")
        return json.loads(blob.download_as_text())

    def download_build(self, build_id: str, dest: Path) -> None:
        prefix = f"{self._prefix}{BUILDS_PREFIX}{build_id}/"
        blobs = [
            b for b in self._bkt().list_blobs(prefix=prefix) if "/" not in b.name[len(prefix) :]
        ]
        if not blobs:
            raise IndexSourceError("build has no files")
        for blob in blobs:
            name = blob.name[len(prefix) :]
            if name:
                blob.download_to_filename(str(dest / name))


# --------------------------------------------------------------------------- pointers


def latest_pointer(source: IndexSource) -> str | None:
    """build_id named by the lexicographically largest well-formed pointer, or None if none.

    Malformed pointer file names are ignored (they cannot be a create-only timestamp pointer);
    a well-formed newest pointer with a bad body raises ``IndexSourceError`` rather than falling
    back to an older pointer: the owner's rollback is to DELETE the newest pointer, and silently
    skipping a broken one would hide it.
    """
    names = sorted(n for n in source.list_pointer_names() if POINTER_NAME_RE.match(n))
    if not names:
        return None
    try:
        body = source.read_pointer(names[-1])
        build_id = body.get("build_id") if isinstance(body, Mapping) else None
    except Exception as exc:
        raise IndexSourceError(f"pointer unreadable ({type(exc).__name__})") from exc
    if not isinstance(build_id, str) or not BUILD_ID_RE.match(build_id):
        raise IndexSourceError("pointer has no valid build_id")
    return build_id


# --------------------------------------------------------------------------- load


LoadBuild = Callable[[Path, BuildManifest, "IndexSnapshot | None"], IndexSnapshot]


def verify_snapshot_index(snapshot: IndexSnapshot, manifest: BuildManifest) -> None:
    """Vector count and dimension of the loaded indices against the manifest."""
    dense = snapshot.retriever.dense.index
    clip = snapshot.clip[0] if snapshot.clip else None
    verify_loaded_index(
        manifest,
        dense_ntotal=int(dense.ntotal),
        dense_dim=int(dense.d),
        clip_ntotal=int(clip.ntotal) if clip is not None else None,
        clip_dim=int(clip.d) if clip is not None else None,
    )


def fetch_and_load_build(
    source: IndexSource,
    build_id: str,
    load_build: LoadBuild,
    *,
    current: IndexSnapshot | None = None,
    tmp_root: Path | None = None,
    registry_slugs: frozenset[str] | None = None,
) -> IndexSnapshot:
    """Download ``builds/<build_id>/`` to a temp dir, verify, load, delete the temp dir.

    The temp dir lives under ``tmp_root`` (default: the system temp dir, ``/tmp`` on Cloud Run,
    which is memory-backed: its bytes count against the instance memory) and is removed in
    ``finally``, so it exists only while the files are being verified and read into memory.
    One build is loaded only from ONE complete directory: files from different builds can
    never be mixed.

    Raises:
        BuildValidationError / IndexSourceError / any loader exception; callers treat every
        exception as "keep the current snapshot".
    """
    if not BUILD_ID_RE.match(build_id):
        raise BuildValidationError("build_id_invalid", "build id is not a UTC timestamp")
    tmp = Path(tempfile.mkdtemp(prefix="asa-index-", dir=str(tmp_root) if tmp_root else None))
    try:
        build_dir = tmp / build_id
        build_dir.mkdir()
        source.download_build(build_id, build_dir)
        manifest = load_manifest(build_dir)
        if manifest.build_id != build_id:
            raise BuildValidationError(
                "build_id_mismatch", "manifest build_id differs from pointer"
            )
        previous = current.manifest if current is not None else None
        verify_build_dir(build_dir, manifest, registry_slugs=registry_slugs, previous=previous)
        snapshot = load_build(build_dir, manifest, current)
        verify_snapshot_index(snapshot, manifest)
        return snapshot
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def default_smoke(snapshot: IndexSnapshot) -> None:
    """Cheap smoke query on a freshly loaded snapshot: a search must run and the frame be non-empty."""
    if len(snapshot.catalogue_df) == 0:
        raise BuildValidationError("smoke", "catalogue is empty")
    snapshot.retriever.search("dress", top_k=1)


# --------------------------------------------------------------------------- reloader


def sanitize(text: str, limit: int = 200) -> str:
    """Strip URL credentials and cap the length of text going into a log line."""
    return _URL_CREDENTIALS.sub(r"\1***@", text)[:limit]


def hot_reload_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """INDEX_HOT_RELOAD: ``0``/``false`` off, ``1``/``true`` on; unset: on only when a bucket
    store (``INDEX_STORE_URI=gs://...``) is configured, so local dev and tests are OFF."""
    env = os.environ if environ is None else environ
    raw = (env.get(SWITCH_ENV) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return (env.get("INDEX_STORE_URI") or "").startswith("gs://")


def reload_interval_s(environ: Mapping[str, str] | None = None) -> float:
    """INDEX_RELOAD_INTERVAL_S, default 900, clamped to >= 60; garbage falls back to the default."""
    env = os.environ if environ is None else environ
    try:
        value = float(env.get(INTERVAL_ENV) or DEFAULT_INTERVAL_S)
    except ValueError:
        value = DEFAULT_INTERVAL_S
    return max(MIN_INTERVAL_S, value)


@dataclass(frozen=True)
class ReloadResult:
    """Outcome of one check. ``outcome``: swapped | unchanged | no_pointer | failed | busy."""

    outcome: str
    build_id: str | None = None
    reason: str | None = None


class IndexReloader:
    """Background checker + atomic swapper. See the module docstring for the flow.

    Args:
        source:       where pointers and builds live.
        load_build:   builds an IndexSnapshot (parts + graphs) from a verified build directory;
                      receives the current snapshot so it can reuse the embedding model.
        get_current:  returns the snapshot being served.
        install:      atomically publishes a new snapshot; returns the previous one (or None).
        smoke:        callable run on the new snapshot before the swap (default: one search).
        registry_slugs: allowed store set (the registry) for manifest validation.
    """

    def __init__(
        self,
        source: IndexSource,
        load_build: LoadBuild,
        get_current: Callable[[], IndexSnapshot | None],
        install: Callable[[IndexSnapshot], IndexSnapshot | None],
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        max_backoff_s: float = MAX_BACKOFF_S,
        tmp_root: Path | None = None,
        smoke: Callable[[IndexSnapshot], None] = default_smoke,
        registry_slugs: frozenset[str] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._source = source
        self._load_build = load_build
        self._get_current = get_current
        self._install = install
        self._interval_s = interval_s
        self._max_backoff_s = max_backoff_s
        self._tmp_root = tmp_root
        self._smoke = smoke
        self._registry_slugs = registry_slugs
        self._clock = clock
        self._flight = threading.Lock()  # single-flight: held for the whole of an attempt
        self._state = threading.Lock()  # guards the status fields below
        self._failures = 0
        self._last_check_at: float | None = None
        self._last_swap_at: float | None = None
        self._last_outcome = "never_checked"
        self._last_reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- status -----------------------------------------------------------------------

    @property
    def consecutive_failures(self) -> int:
        with self._state:
            return self._failures

    def next_delay_s(self) -> float:
        """Seconds until the next check: the interval, doubled per consecutive failure, capped."""
        with self._state:
            if self._failures == 0:
                return self._interval_s
            return min(self._interval_s * (2**self._failures), self._max_backoff_s)

    def status(self) -> dict[str, Any]:
        """Reload state for /readyz (no secrets, no paths)."""

        def iso(t: float | None) -> str | None:
            return (
                datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds") if t else None
            )

        with self._state:
            return {
                "hot_reload": "on",
                "interval_s": self._interval_s,
                "last_check_at": iso(self._last_check_at),
                "last_outcome": self._last_outcome,
                "last_reason": self._last_reason,
                "last_swap_at": iso(self._last_swap_at),
                "consecutive_failures": self._failures,
            }

    # -- one attempt ------------------------------------------------------------------

    def check_once(self) -> ReloadResult:
        """One check-download-validate-load-swap attempt. Never raises."""
        if not self._flight.acquire(blocking=False):
            return ReloadResult("busy")
        try:
            result = self._attempt()
            self._record(result)
        finally:
            self._flight.release()
        # Outside _attempt(): its locals (the previous snapshot among them) are gone, so the
        # old index is collectable now unless an in-flight turn still holds it.
        gc.collect()
        return result

    def _record(self, result: ReloadResult) -> None:
        now = self._clock()
        with self._state:
            self._last_check_at = now
            self._last_outcome = result.outcome
            self._last_reason = result.reason
            if result.outcome == "failed":
                self._failures += 1
            else:
                self._failures = 0
            if result.outcome == "swapped":
                self._last_swap_at = now

    def _attempt(self) -> ReloadResult:
        build_id: str | None = None
        try:
            build_id = latest_pointer(self._source)
            if build_id is None:
                return ReloadResult("no_pointer")
            current = self._get_current()
            if current is not None and current.build_id == build_id:
                return ReloadResult("unchanged", build_id)
            new = fetch_and_load_build(
                self._source,
                build_id,
                self._load_build,
                current=current,
                tmp_root=self._tmp_root,
                registry_slugs=self._registry_slugs,
            )
            self._smoke(new)
            self._install(new)  # the atomic swap; the returned old snapshot is dropped here
            logger.info(
                "index_reload_swapped build_id=%s previous_build_id=%s items=%d",
                build_id,
                current.build_id if current is not None else None,
                len(new.catalogue_df),
            )
            return ReloadResult("swapped", build_id)
        except BuildValidationError as exc:
            return self._failed(build_id, exc.reason, exc.message)
        except IndexSourceError as exc:
            return self._failed(build_id, "source", str(exc))
        except Exception as exc:  # noqa: BLE001 - ANY failure must keep the current snapshot
            return self._failed(build_id, type(exc).__name__, str(exc))

    def _failed(self, build_id: str | None, reason: str, detail: str) -> ReloadResult:
        with self._state:
            attempt = self._failures + 1
            delay = min(self._interval_s * (2**attempt), self._max_backoff_s)
        logger.warning(
            "index_reload_failed reason=%s build_id=%s attempt=%d next_check_s=%d detail=%s",
            reason,
            build_id,
            attempt,
            int(delay),
            sanitize(detail),
        )
        return ReloadResult("failed", build_id, reason)

    # -- thread -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the daemon thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="index-reload", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to stop and wait briefly (used by lifespan shutdown and tests)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self.next_delay_s()):
            try:
                self.check_once()
            except Exception:  # pragma: no cover - check_once never raises; belt and braces
                logger.exception("index_reload_loop_error")
