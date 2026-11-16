"""Build store used by the daily refresh: CREATE-ONLY publish of ``builds/<id>/`` (Wave 1b V3(c)).

Layout (same as ``src/retrieval/pinned_build.py`` reads): ``<root>/builds/<UTC id>/<files>``.

Create-only is enforced twice: (1) the target prefix must be empty before the first byte is
written, and (2) every object is written with ``if_generation_match=0`` (GCS) or an exclusive create
(local), so an existing object is NEVER overwritten, whatever races. There is no delete method:
the publisher identity has no ``storage.objects.delete`` and old builds are removed only by the
bucket lifecycle rule after ``customTime`` is set (PR B, ``expire_builds.py``).

``build_manifest.json`` is uploaded LAST: the API loads a build only through its manifest, so a
half-uploaded build is invisible ("not_found") instead of wrong.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

MANIFEST_NAME = "build_manifest.json"
BUILD_ID_RE = re.compile(r"\d{8}T\d{6}Z", re.ASCII)  # same pattern as src/retrieval/index_reload.py
PREFIX = "builds/"
PRECONDITION_FAILED = 412


class StoreError(Exception):
    """A store operation failed (message is safe to log)."""


class CreateOnlyViolation(StoreError):
    """The build (or one of its objects) already exists; nothing was overwritten."""


class BuildStore(Protocol):
    """What the refresh needs from the bucket (reads, create-only writes, customTime)."""

    def list_build_ids(self) -> list[str]:
        """Build ids that have at least one object, ascending (oldest first)."""
        ...

    def list_objects(self, build_id: str) -> dict[str, int]:
        """``name -> size_bytes`` for one build."""
        ...

    def read_object(self, build_id: str, name: str, max_bytes: int) -> bytes:
        """Bytes of one object; refuses anything larger than ``max_bytes``."""
        ...

    def create_build(self, src_dir: Path, build_id: str) -> None:
        """Publish every file of ``src_dir`` under ``builds/<id>/``, create-only, manifest last."""
        ...

    def set_custom_time(self, build_id: str, when: datetime) -> int:
        """Set ``customTime`` on every object of the build; returns the object count."""
        ...


def _check_id(build_id: str) -> None:
    if not BUILD_ID_RE.fullmatch(build_id):
        raise StoreError(f"not a build id: {build_id!r}")


def _upload_order(src_dir: Path) -> list[Path]:
    files = sorted(p for p in src_dir.iterdir() if p.is_file())
    if not any(p.name == MANIFEST_NAME for p in files):
        raise StoreError(f"{MANIFEST_NAME} missing from {src_dir}")
    return [p for p in files if p.name != MANIFEST_NAME] + [src_dir / MANIFEST_NAME]


class LocalBuildStore:
    """Directory-backed store (tests, local dry runs). Same create-only semantics."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, build_id: str) -> Path:
        _check_id(build_id)
        return self.root / "builds" / build_id

    def list_build_ids(self) -> list[str]:
        base = self.root / "builds"
        if not base.is_dir():
            return []
        return sorted(
            p.name for p in base.iterdir() if p.is_dir() and BUILD_ID_RE.fullmatch(p.name)
        )

    def list_objects(self, build_id: str) -> dict[str, int]:
        d = self._dir(build_id)
        return {p.name: p.stat().st_size for p in d.iterdir() if p.is_file()} if d.is_dir() else {}

    def read_object(self, build_id: str, name: str, max_bytes: int) -> bytes:
        path = self._dir(build_id) / Path(name).name
        if not path.is_file():
            raise StoreError(f"{build_id}/{name} not found")
        if path.stat().st_size > max_bytes:
            raise StoreError(f"{build_id}/{name} exceeds {max_bytes} bytes")
        return path.read_bytes()

    def create_build(self, src_dir: Path, build_id: str) -> None:
        d = self._dir(build_id)
        if self.list_objects(build_id):
            raise CreateOnlyViolation(f"builds/{build_id}/ already has objects")
        d.mkdir(parents=True, exist_ok=True)
        for path in _upload_order(Path(src_dir)):
            try:
                with (d / path.name).open("xb") as out:  # exclusive create: never overwrite
                    out.write(path.read_bytes())
            except FileExistsError as exc:
                raise CreateOnlyViolation(f"builds/{build_id}/{path.name} exists") from exc

    def set_custom_time(self, build_id: str, when: datetime) -> int:
        # Local stand-in: customTime is stored as the file mtime so tests can observe it.
        n = 0
        for p in self._dir(build_id).iterdir():
            if p.is_file():
                os.utime(p, (when.timestamp(), when.timestamp()))
                n += 1
        return n


class GcsBuildStore:
    """``gs://<bucket>/builds/<id>/`` through google-cloud-storage (client injectable for tests)."""

    def __init__(self, bucket_name: str, client: Any = None) -> None:
        if not bucket_name or "/" in bucket_name:
            raise StoreError("bucket name must be a bare bucket name")
        self._bucket_name = bucket_name
        self._client = client
        self._bucket: Any = None

    def _bkt(self) -> Any:
        if self._bucket is None:
            if self._client is None:
                from google.cloud import storage  # lazy: not needed for tests or dry fixtures

                self._client = storage.Client()
            self._bucket = self._client.bucket(self._bucket_name)
        return self._bucket

    def list_build_ids(self) -> list[str]:
        it = self._bkt().list_blobs(prefix=PREFIX, delimiter="/")
        list(it)  # prefixes are populated only after the iterator is consumed
        ids = (p[len(PREFIX) :].rstrip("/") for p in it.prefixes)
        return sorted(i for i in ids if BUILD_ID_RE.fullmatch(i))

    def list_objects(self, build_id: str) -> dict[str, int]:
        _check_id(build_id)
        prefix = f"{PREFIX}{build_id}/"
        return {
            b.name[len(prefix) :]: int(b.size or 0) for b in self._bkt().list_blobs(prefix=prefix)
        }

    def read_object(self, build_id: str, name: str, max_bytes: int) -> bytes:
        _check_id(build_id)
        blob = self._bkt().get_blob(f"{PREFIX}{build_id}/{Path(name).name}")
        if blob is None:
            raise StoreError(f"{build_id}/{name} not found")
        if int(blob.size or 0) > max_bytes:
            raise StoreError(f"{build_id}/{name} exceeds {max_bytes} bytes")
        return bytes(blob.download_as_bytes())

    def create_build(self, src_dir: Path, build_id: str) -> None:
        _check_id(build_id)
        if self.list_objects(build_id):
            raise CreateOnlyViolation(f"builds/{build_id}/ already has objects")
        ordered = _upload_order(Path(src_dir))
        for path in ordered:
            blob = self._bkt().blob(f"{PREFIX}{build_id}/{path.name}")
            try:
                blob.upload_from_filename(str(path), if_generation_match=0)
            except Exception as exc:  # noqa: BLE001 - google exceptions are imported lazily
                if getattr(exc, "code", None) == PRECONDITION_FAILED:
                    raise CreateOnlyViolation(f"builds/{build_id}/{path.name} exists") from exc
                raise StoreError(f"upload of {path.name} failed: {type(exc).__name__}") from exc
        sizes = self.list_objects(build_id)
        expected = {p.name: p.stat().st_size for p in ordered}
        if sizes != expected:
            raise StoreError("uploaded object list or sizes differ from the local build")

    def set_custom_time(self, build_id: str, when: datetime) -> int:
        _check_id(build_id)
        prefix = f"{PREFIX}{build_id}/"
        n = 0
        for blob in self._bkt().list_blobs(prefix=prefix):
            blob.custom_time = when
            blob.patch()  # metadata only; needs storage.objects.update, never delete
            n += 1
        return n


class ReadOnlyStore:
    """Wraps a store for dry runs: reads pass through, any write raises. Belt and braces."""

    def __init__(self, inner: BuildStore) -> None:
        self._inner = inner

    def list_build_ids(self) -> list[str]:
        return self._inner.list_build_ids()

    def list_objects(self, build_id: str) -> dict[str, int]:
        return self._inner.list_objects(build_id)

    def read_object(self, build_id: str, name: str, max_bytes: int) -> bytes:
        return self._inner.read_object(build_id, name, max_bytes)

    def create_build(self, src_dir: Path, build_id: str) -> None:
        raise StoreError("read-only store (dry run): create_build refused")

    def set_custom_time(self, build_id: str, when: datetime) -> int:
        raise StoreError("read-only store (dry run): set_custom_time refused")
