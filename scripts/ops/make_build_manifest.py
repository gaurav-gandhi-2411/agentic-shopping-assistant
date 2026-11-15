"""Write ``build_manifest.json`` for a directory of index files (Wave 1b V3(b) rework).

    python scripts/ops/make_build_manifest.py --index-dir <dir> [--build-id 20261115T020000Z]
        [--previous <dir or build_manifest.json>] [--git-sha <sha>] [--built-at <ISO UTC>]
        [--overwrite]

Used by the V3(c) refresh job and by hand. Reads what the loader needs from ``--index-dir``
(dense.faiss, bm25.pkl, bm25_article_ids.npy, dense_article_ids.npy, catalogue.parquet and, when
present, clip.faiss + clip_article_ids.npy), derives everything the manifest asserts FROM those
files (per-store counts from the catalogue ``store`` column, vector counts and dimensions from
the FAISS files, sha256 + size per file), checks it for consistency and the store registry, runs
the +-30 % count gate when ``--previous`` is given, writes the manifest next to the files and
re-verifies the result with the loader's own verifier. Exit code 0 only when a manifest was
written; any inconsistency prints ``REFUSED: <reason>`` and exits 1 without writing anything.

Local, read-only apart from the one manifest file: no network, no cloud.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config.store_registry import STORE_REGISTRY  # noqa: E402
from src.retrieval.build_manifest import (  # noqa: E402
    CLIP_FILES,
    MANIFEST_NAME,
    REQUIRED_FILES,
    BuildManifest,
    BuildValidationError,
    StoreEntry,
    ValidationBlock,
    build_manifest_for_dir,
    check_count_gate,
    load_manifest,
    now_build_id,
    verify_build_dir,
    verify_loaded_index,
    write_manifest,
)
from src.retrieval.index_reload import BUILD_ID_RE  # noqa: E402

GIT_SHA_ENVS = ("GIT_SHA", "GITHUB_SHA")  # first one set wins when --git-sha is not given


class Refused(Exception):
    """The build is inconsistent or invalid; nothing is written."""


def _faiss_shape(path: Path) -> tuple[int, int]:
    """(ntotal, dim) of a FAISS index file."""
    import faiss

    index = faiss.read_index(str(path))
    return int(index.ntotal), int(index.d)


def store_counts(catalogue: Path) -> list[StoreEntry]:
    """Per-store item counts (lower-case slug) from the catalogue's ``store`` column."""
    import pandas as pd

    df = pd.read_parquet(catalogue, columns=["store"])
    stores = df["store"].fillna("").astype(str).str.lower()
    counts = stores.value_counts().to_dict()
    return [StoreEntry(slug=str(s), items=int(n)) for s, n in sorted(counts.items())]


def load_previous(path: Path) -> BuildManifest:
    """Previous manifest from a build directory or a ``build_manifest.json`` path."""
    directory = path.parent if path.is_file() else path
    return load_manifest(directory)


def make_manifest(
    index_dir: Path,
    *,
    build_id: str,
    git_sha: str,
    built_at: datetime,
    previous: BuildManifest | None,
) -> BuildManifest:
    """Assemble and fully check the manifest for ``index_dir`` (writes nothing).

    Raises:
        Refused: the files are incomplete, inconsistent, outside the registry or fail the gate.
    """
    if not BUILD_ID_RE.match(build_id):
        raise Refused(f"build id {build_id!r} is not a UTC timestamp like 20261115T020000Z")
    if not git_sha.strip():
        raise Refused("git sha is empty (pass --git-sha or set GIT_SHA / GITHUB_SHA)")
    missing = [n for n in REQUIRED_FILES if not (index_dir / n).is_file()]
    if missing:
        raise Refused(f"missing required file(s): {', '.join(missing)}")
    clip_present = [n for n in CLIP_FILES if (index_dir / n).is_file()]
    if clip_present and len(clip_present) != len(CLIP_FILES):
        raise Refused("only one of the two CLIP files is present")

    stores = store_counts(index_dir / "catalogue.parquet")
    unknown = sorted({s.slug for s in stores} - set(STORE_REGISTRY))
    if unknown:
        raise Refused(f"stores not in the registry: {unknown}")
    total = sum(s.items for s in stores)
    if total == 0:
        raise Refused("the catalogue is empty")

    dense_n, dense_dim = _faiss_shape(index_dir / "dense.faiss")
    clip_dim: int | None = None
    clip_n: int | None = None
    if clip_present:
        clip_n, clip_dim = _faiss_shape(index_dir / "clip.faiss")

    manifest = build_manifest_for_dir(
        index_dir,
        build_id=build_id,
        git_sha=git_sha,
        stores=stores,
        built_at=built_at,
        dense_dim=dense_dim,
        clip_dim=clip_dim,
        previous=previous,
    )
    try:
        delta = check_count_gate(manifest, previous)
        manifest.validation = ValidationBlock(count_delta_pct=delta, schema_ok=True)
        # Same checks the API runs at startup: files, counts (parquet rows, id files, per-store
        # counts), registry, gate; then vector counts and dimensions of the FAISS files.
        verify_build_dir(
            index_dir, manifest, registry_slugs=frozenset(STORE_REGISTRY), previous=previous
        )
        verify_loaded_index(
            manifest,
            dense_ntotal=dense_n,
            dense_dim=dense_dim,
            clip_ntotal=clip_n,
            clip_dim=clip_dim,
        )
    except BuildValidationError as exc:
        raise Refused(str(exc)) from exc
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index-dir", required=True, type=Path)
    ap.add_argument("--build-id", default=None, help="default: UTC now")
    ap.add_argument("--previous", type=Path, default=None, help="previous build dir or manifest")
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--built-at", default=None, help="ISO UTC time; default: now")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing manifest")
    args = ap.parse_args(argv)

    index_dir: Path = args.index_dir
    try:
        if not index_dir.is_dir():
            raise Refused(f"{index_dir} is not a directory")
        if (index_dir / MANIFEST_NAME).exists() and not args.overwrite:
            raise Refused(f"{MANIFEST_NAME} already exists (builds are create-only; --overwrite)")
        git_sha = args.git_sha or next(
            (os.environ[e] for e in GIT_SHA_ENVS if os.environ.get(e)), ""
        )
        built_at = (
            datetime.fromisoformat(args.built_at).astimezone(timezone.utc)
            if args.built_at
            else datetime.now(timezone.utc)
        )
        previous = None
        if args.previous is not None:
            try:
                previous = load_previous(args.previous)
            except BuildValidationError as exc:
                raise Refused(f"previous manifest unusable: {exc}") from exc
        manifest = make_manifest(
            index_dir,
            build_id=args.build_id or now_build_id(),
            git_sha=git_sha,
            built_at=built_at,
            previous=previous,
        )
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    path = write_manifest(index_dir, manifest)
    print(
        f"wrote {path.name}: build_id={manifest.build_id} items={manifest.total_items} "
        f"stores={len(manifest.stores)} files={len(manifest.files)} "
        f"count_delta_pct={manifest.validation.count_delta_pct}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
