"""Index build manifest: ``build_manifest.json`` and its verifier (Wave 1b V3(b)).

A published index build is one directory holding the index files plus ``build_manifest.json``.
The manifest says what the build is (id, time, git sha, per-store counts, vector counts, dims)
and carries a sha256 and size for every file, so a loader can prove it holds ONE complete,
untampered build before it serves a single query from it.

Layout in the bucket (see DEPLOY.md, "Index builds and hot reload")::

    builds/<UTC timestamp>/{dense.faiss, bm25.pkl, bm25_article_ids.npy,
                            dense_article_ids.npy, catalogue.parquet,
                            [clip.faiss, clip_article_ids.npy,] build_manifest.json}
    pointers/<UTC timestamp>.json      -> {"build_id": "<UTC timestamp>"}

The JSON schema is generated from :class:`BuildManifest` into ``docs/build_manifest.schema.json``
(a test keeps the file in sync with the model).

Verification is split in two so the cheap part can run before any heavy load:

  - :func:`verify_build_dir`   hashes, sizes, row counts of the files on disk, store set vs the
                               registry, count gate vs the previous manifest.
  - :func:`verify_loaded_index` vector count and dimension of the FAISS indices once loaded.

Both raise :class:`BuildValidationError` with a short machine ``reason`` code; the hot-reload
loop logs the code and keeps serving the current index.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

MANIFEST_NAME = "build_manifest.json"
SCHEMA_VERSION = 1

DENSE_DIM = 384  # MiniLM (config.yaml retrieval.dense_model)
CLIP_DIM = 512  # CLIP ViT-B/32

# Files every build must carry; CLIP files are optional (the image feature degrades without them).
REQUIRED_FILES: tuple[str, ...] = (
    "dense.faiss",
    "bm25.pkl",
    "bm25_article_ids.npy",
    "dense_article_ids.npy",
    "catalogue.parquet",
)
CLIP_FILES: tuple[str, ...] = ("clip.faiss", "clip_article_ids.npy")

# V3c gate: total items must be within +-30% of the previous build.
COUNT_GATE_TOLERANCE = 0.30


class BuildValidationError(Exception):
    """A build failed validation. ``reason`` is a short machine code, safe to log."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason
        self.message = message


class FileEntry(BaseModel):
    """One file of the build."""

    name: str = Field(description="File name inside the build directory (no sub-directories).")
    sha256: str = Field(min_length=64, max_length=64, description="Lower-case hex sha256.")
    size_bytes: int = Field(ge=0)


class StoreEntry(BaseModel):
    """One store's contribution to the build."""

    slug: str
    items: int = Field(ge=0)
    source_updated_at: str | None = Field(
        default=None,
        description="Newest feed `updated_at` of the store's rows, else the feed fetch time (ISO UTC).",
    )


class BuyLinkSample(BaseModel):
    """Result of the Buy-link sample check (filled by the publisher; placeholder until V3c)."""

    checked: int = Field(default=0, ge=0)
    ok: int = Field(default=0, ge=0)


class ValidationBlock(BaseModel):
    """Publisher-side validation results recorded in the manifest (informational for the API)."""

    count_delta_pct: float | None = Field(
        default=None, description="(total_items - previous total) / previous total * 100."
    )
    schema_ok: bool | None = None
    buy_link_sample: BuyLinkSample | None = None


class BuildManifest(BaseModel):
    """``build_manifest.json``."""

    schema_version: int = SCHEMA_VERSION
    build_id: str = Field(
        description="UTC timestamp, e.g. 20261115T020000Z; also the directory name."
    )
    built_at: datetime = Field(description="UTC build time (timezone-aware).")
    git_sha: str
    stores: list[StoreEntry]
    total_items: int = Field(ge=0)
    vector_count: int = Field(ge=0, description="Dense vectors; must equal total_items.")
    dense_dim: int = DENSE_DIM
    clip_dim: int | None = Field(default=None, description="512 when CLIP files are present.")
    files: list[FileEntry]
    validation: ValidationBlock = Field(default_factory=ValidationBlock)
    previous_build_id: str | None = None


# --------------------------------------------------------------------------- helpers


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (1 MiB chunks: bounded memory on a 170 MB index)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_entry(path: Path) -> FileEntry:
    """FileEntry (name, sha256, size) for ``path``."""
    return FileEntry(name=path.name, sha256=sha256_file(path), size_bytes=path.stat().st_size)


def load_manifest(build_dir: Path) -> BuildManifest:
    """Parse ``<build_dir>/build_manifest.json``.

    Raises:
        BuildValidationError: ``manifest_missing`` or ``manifest_invalid``.
    """
    path = Path(build_dir) / MANIFEST_NAME
    if not path.is_file():
        raise BuildValidationError("manifest_missing", f"{MANIFEST_NAME} not found")
    try:
        return BuildManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError, OSError) as exc:
        # Type name only: a pydantic error echoes input values.
        raise BuildValidationError("manifest_invalid", type(exc).__name__) from exc


def try_load_manifest(build_dir: Path) -> BuildManifest | None:
    """Manifest of a flat/legacy directory if one is present and parseable, else None.

    No hash verification (the flat layout has no integrity guarantee to preserve); /readyz
    reports ``legacy (no manifest)`` when this returns None.
    """
    try:
        return load_manifest(build_dir)
    except BuildValidationError:
        return None


def write_manifest(build_dir: Path, manifest: BuildManifest) -> Path:
    """Write the manifest next to the index files (used by tests and the V3c publisher)."""
    path = Path(build_dir) / MANIFEST_NAME
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def check_count_gate(
    manifest: BuildManifest,
    previous: BuildManifest | None,
    tolerance: float = COUNT_GATE_TOLERANCE,
) -> float | None:
    """Item-count delta vs the previous build in percent; raise when outside +-tolerance.

    Returns None when there is no previous build (nothing to compare against: first publish).
    """
    if previous is None or previous.total_items <= 0:
        return None
    delta = (manifest.total_items - previous.total_items) / previous.total_items
    if abs(delta) > tolerance:
        raise BuildValidationError(
            "count_gate",
            f"items {manifest.total_items} vs previous {previous.total_items} "
            f"({delta * 100:+.1f}%, limit +-{tolerance * 100:.0f}%)",
        )
    return delta * 100.0


def now_build_id(now: datetime | None = None) -> str:
    """UTC timestamp build id, lexicographically sortable (e.g. ``20261115T020000Z``)."""
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- verification


def _parquet_store_counts(path: Path) -> tuple[int, dict[str, int]]:
    """(row count, rows per lower-case store) from the parquet, reading only the store column."""
    import pandas as pd
    import pyarrow.parquet as pq

    n_rows = int(pq.ParquetFile(path).metadata.num_rows)
    stores = pd.read_parquet(path, columns=["store"])["store"]
    counts = stores.fillna("").astype(str).str.lower().value_counts().to_dict()
    return n_rows, {str(k): int(v) for k, v in counts.items()}


def _npy_len(path: Path) -> int:
    """Length of a 1-d .npy without loading the data (header only, mmap)."""
    import numpy as np

    # Header only: works for object-dtype arrays too (no pickle is read, unlike np.load).
    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
    return int(shape[0])


def verify_build_dir(
    build_dir: Path,
    manifest: BuildManifest | None = None,
    *,
    registry_slugs: frozenset[str] | None = None,
    previous: BuildManifest | None = None,
    check_store_counts: bool = True,
) -> BuildManifest:
    """Verify that ``build_dir`` is one complete, untampered build; return its manifest.

    Checks, in order (first failure raises):
      ``manifest_*``      manifest present and parseable, supported ``schema_version``
      ``file_missing``    every required file (and every file the manifest lists) exists
      ``file_size`` / ``file_hash``   size and sha256 of every listed file
      ``unlisted_file``   a required file that the manifest does not list
      ``count_mismatch``  total_items == vector_count == sum(store counts) == parquet rows ==
                          ids-file lengths; with ``check_store_counts`` the per-store counts
                          also equal the parquet's ``store`` column
      ``store_unknown``   every manifest store is in the registry
      ``clip_mismatch``   clip_dim set iff both CLIP files are listed
      ``count_gate``      total items within +-30% of ``previous`` (when supplied)
    """
    build_dir = Path(build_dir)
    manifest = manifest or load_manifest(build_dir)
    if manifest.schema_version != SCHEMA_VERSION:
        raise BuildValidationError(
            "manifest_schema", f"schema_version {manifest.schema_version} != {SCHEMA_VERSION}"
        )

    listed = {f.name: f for f in manifest.files}
    for name in REQUIRED_FILES:
        if name not in listed:
            raise BuildValidationError("unlisted_file", f"{name} is not listed in the manifest")
    clip_listed = [n for n in CLIP_FILES if n in listed]
    if clip_listed and len(clip_listed) != len(CLIP_FILES):
        raise BuildValidationError("clip_mismatch", "only one of the two CLIP files is listed")
    if bool(clip_listed) != (manifest.clip_dim is not None):
        raise BuildValidationError("clip_mismatch", "clip_dim and the CLIP file list disagree")

    for name, entry in listed.items():
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise BuildValidationError("file_missing", "manifest lists a non-flat file name")
        path = build_dir / name
        if not path.is_file():
            raise BuildValidationError("file_missing", f"{name} not found")
        if path.stat().st_size != entry.size_bytes:
            raise BuildValidationError("file_size", f"{name} size differs from the manifest")
        if sha256_file(path) != entry.sha256:
            raise BuildValidationError("file_hash", f"{name} sha256 differs from the manifest")

    store_sum = sum(s.items for s in manifest.stores)
    if not (manifest.total_items == manifest.vector_count == store_sum):
        raise BuildValidationError(
            "count_mismatch",
            f"total_items={manifest.total_items} vector_count={manifest.vector_count} "
            f"sum(stores)={store_sum}",
        )
    n_rows, parquet_counts = _parquet_store_counts(build_dir / "catalogue.parquet")
    id_lens = {
        n: _npy_len(build_dir / n)
        for n in ("dense_article_ids.npy", "bm25_article_ids.npy")
        + (("clip_article_ids.npy",) if clip_listed else ())
    }
    if n_rows != manifest.total_items or any(v != manifest.total_items for v in id_lens.values()):
        raise BuildValidationError(
            "count_mismatch", f"parquet rows={n_rows} id files={id_lens} vs {manifest.total_items}"
        )
    if check_store_counts:
        declared = {s.slug.lower(): s.items for s in manifest.stores}
        if declared != parquet_counts:
            raise BuildValidationError(
                "count_mismatch", "per-store counts differ from the catalogue"
            )

    if registry_slugs is not None:
        unknown = sorted({s.slug.lower() for s in manifest.stores} - registry_slugs)
        if unknown:
            raise BuildValidationError("store_unknown", f"not in the registry: {unknown}")

    check_count_gate(manifest, previous)
    return manifest


def verify_loaded_index(
    manifest: BuildManifest,
    *,
    dense_ntotal: int,
    dense_dim: int,
    clip_ntotal: int | None = None,
    clip_dim: int | None = None,
) -> None:
    """Check the FAISS indices AFTER loading: vector count == items, dimensions match."""
    if dense_ntotal != manifest.total_items:
        raise BuildValidationError(
            "vector_count",
            f"dense index holds {dense_ntotal}, manifest says {manifest.total_items}",
        )
    if dense_dim != manifest.dense_dim:
        raise BuildValidationError(
            "dim_mismatch", f"dense dim {dense_dim} != manifest {manifest.dense_dim}"
        )
    if manifest.clip_dim is not None:
        if clip_ntotal != manifest.total_items:
            raise BuildValidationError(
                "vector_count",
                f"clip index holds {clip_ntotal}, manifest says {manifest.total_items}",
            )
        if clip_dim != manifest.clip_dim:
            raise BuildValidationError(
                "dim_mismatch", f"clip dim {clip_dim} != manifest {manifest.clip_dim}"
            )


def build_manifest_for_dir(
    build_dir: Path,
    *,
    build_id: str,
    git_sha: str,
    stores: list[StoreEntry],
    built_at: datetime | None = None,
    dense_dim: int = DENSE_DIM,
    clip_dim: int | None = None,
    previous: BuildManifest | None = None,
) -> BuildManifest:
    """Assemble (not write) the manifest for the index files already in ``build_dir``.

    Used by the V3c publisher and by tests. Hashes every file present among
    REQUIRED_FILES + CLIP_FILES; total/vector counts come from ``stores``.
    """
    build_dir = Path(build_dir)
    names = list(REQUIRED_FILES) + [n for n in CLIP_FILES if (build_dir / n).is_file()]
    total = sum(s.items for s in stores)
    delta: float | None = None
    if previous is not None and previous.total_items > 0:
        delta = (total - previous.total_items) / previous.total_items * 100.0
    return BuildManifest(
        build_id=build_id,
        built_at=built_at or datetime.now(timezone.utc),
        git_sha=git_sha,
        stores=stores,
        total_items=total,
        vector_count=total,
        dense_dim=dense_dim,
        clip_dim=clip_dim if any(n in CLIP_FILES for n in names) else None,
        files=[file_entry(build_dir / n) for n in names],
        validation=ValidationBlock(count_delta_pct=delta),
        previous_build_id=previous.build_id if previous else None,
    )


def schema_json() -> dict[str, Any]:
    """The JSON schema of build_manifest.json (checked in as docs/build_manifest.schema.json)."""
    return BuildManifest.model_json_schema()


def schema_text() -> str:
    """Canonical text of :func:`schema_json` (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(schema_json(), indent=2, sort_keys=True) + "\n"
