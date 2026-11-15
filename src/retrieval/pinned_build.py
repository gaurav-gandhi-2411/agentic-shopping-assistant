"""Revision-pinned index build: download, verify and load ONE named build (Wave 1b V3(b) rework).

Rollout design (architect decision): each Cloud Run revision is pinned to one build id through
``INDEX_BUILD_ID``. At startup the API loads exactly ``builds/<id>/`` and nothing else: no
pointer lookup, no listing, no background thread, never a second build. Old and new instances run
side by side during a rollout; no instance ever holds two builds.

Failure never crashes startup and never serves a DIFFERENT build: the caller falls back to the
flat legacy index and reports the failure in ``/readyz`` with one of the fixed ``PIN_*`` reason
codes below (so the tag-URL deploy smoke fails before the traffic shift).
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.retrieval.build_manifest import (
    CLIP_FILES,
    MANIFEST_NAME,
    REQUIRED_FILES,
    BuildManifest,
    BuildValidationError,
    is_flat_file_name,
    load_manifest,
    verify_build_dir,
)
from src.retrieval.index_reload import (
    BUILD_ID_RE,
    BuildNotFoundError,
    BuildTooLargeError,
    IndexSource,
    IndexSourceError,
)

PIN_ENV = "INDEX_BUILD_ID"

# Size caps: the live build is about 0.57 GiB in total (largest file, CLIP, 0.23 GiB).
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 768 * 1024 * 1024
MAX_BUILD_BYTES = 1536 * 1024 * 1024
ALLOWED_FILES = frozenset((*REQUIRED_FILES, *CLIP_FILES))

# Fixed reason codes shown in /readyz (never free text from an exception).
PIN_REASONS = (
    "id_invalid",  # INDEX_BUILD_ID is not a build id
    "no_source",  # no usable INDEX_STORE_URI, or not the unified brand
    "not_found",  # the build or a file of it is not in the store
    "source_error",  # the store could not be read
    "too_large",  # a file (or the build) exceeds the size caps
    "manifest_invalid",  # unreadable manifest / unsupported schema / unexpected file list
    "integrity",  # missing file, size or sha256 differs, unlisted required file
    "inconsistent",  # counts, stores, vector count, dimension or build id disagree
    "load_failed",  # the loader itself raised
    "build_mismatch",  # a build other than the pinned one is being served
)

_INTEGRITY = {"file_missing", "file_size", "file_hash", "unlisted_file", "clip_mismatch"}
_MANIFEST = {"manifest_invalid", "manifest_schema", "unexpected_file"}
_INCONSISTENT = {
    "count_mismatch",
    "store_unknown",
    "vector_count",
    "dim_mismatch",
    "build_id_mismatch",
    "count_gate",
    "build_id_invalid",
}


class PinnedBuildError(Exception):
    """The pinned build could not be loaded. ``code`` is one of :data:`PIN_REASONS`."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class PinRequest:
    """What ``INDEX_BUILD_ID`` asked for. ``requested`` False means legacy (unset/blank)."""

    requested: bool
    build_id: str | None = None  # only a VALID id is ever kept (never echo a hostile value)
    failure: str | None = None  # "id_invalid" when requested but malformed


def read_pin(environ: Mapping[str, str]) -> PinRequest:
    """Parse ``INDEX_BUILD_ID``: unset or blank = not requested; malformed = ``id_invalid``."""
    raw = (environ.get(PIN_ENV) or "").strip()
    if not raw:
        return PinRequest(False)
    if not BUILD_ID_RE.fullmatch(raw):
        return PinRequest(True, None, "id_invalid")
    return PinRequest(True, raw, None)


def reason_code(exc: BaseException) -> str:
    """Map any exception raised while loading a pinned build to a fixed reason code."""
    if isinstance(exc, PinnedBuildError):
        return exc.code
    if isinstance(exc, BuildNotFoundError):
        return "not_found"
    if isinstance(exc, BuildTooLargeError):
        return "too_large"
    if isinstance(exc, BuildValidationError):
        if exc.reason == "manifest_missing":
            return "not_found"
        if exc.reason in _MANIFEST:
            return "manifest_invalid"
        if exc.reason in _INTEGRITY:
            return "integrity"
        if exc.reason in _INCONSISTENT:
            return "inconsistent"
        return "load_failed"
    if isinstance(exc, IndexSourceError | OSError):
        return "source_error"
    return "load_failed"


def download_pinned_build(
    source: IndexSource,
    build_id: str,
    dest: Path,
    *,
    registry_slugs: frozenset[str] | None = None,
) -> BuildManifest:
    """Fetch ONLY the files ``builds/<build_id>/build_manifest.json`` lists into ``dest`` and
    verify them (sha256, sizes, counts, store set).

    The manifest is fetched first (size-capped); every name it lists must be a plain file name
    (the verifier's guard) of a known index file; per-file and total sizes are capped before any
    data file is requested. ``dest`` must exist; the caller deletes it on any outcome.
    """
    if not BUILD_ID_RE.fullmatch(build_id):
        raise PinnedBuildError("id_invalid")
    source.download_file(build_id, MANIFEST_NAME, dest / MANIFEST_NAME, MAX_MANIFEST_BYTES)
    manifest = load_manifest(dest)
    if manifest.build_id != build_id:
        raise BuildValidationError("build_id_mismatch", "manifest build_id differs from the pin")
    names = [f.name for f in manifest.files]
    if len(set(names)) != len(names):
        raise BuildValidationError("manifest_invalid", "duplicate file entries")
    for entry in manifest.files:
        if not is_flat_file_name(entry.name) or entry.name not in ALLOWED_FILES:
            raise BuildValidationError("unexpected_file", "manifest lists an unexpected file")
        if entry.size_bytes > MAX_FILE_BYTES:
            raise BuildTooLargeError("a build file exceeds the size cap")
    if sum(f.size_bytes for f in manifest.files) > MAX_BUILD_BYTES:
        raise BuildTooLargeError("the build exceeds the size cap")
    for entry in manifest.files:
        # Cap at the manifest's declared size: a larger object is refused before it is written.
        source.download_file(build_id, entry.name, dest / entry.name, entry.size_bytes)
    return verify_build_dir(dest, manifest, registry_slugs=registry_slugs)


def make_tmp_dir(tmp_root: Path | None) -> Path:
    """Scratch directory for one download (under ``tmp_root``, created if needed)."""
    if tmp_root is not None:
        tmp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="asa-pinned-", dir=str(tmp_root) if tmp_root else None))


def remove_tmp_dir(path: Path) -> None:
    """Delete a scratch directory (partial or complete downloads never outlive the attempt)."""
    shutil.rmtree(path, ignore_errors=True)
