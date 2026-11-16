"""scripts/ops/refresh_store.py: create-only publish, manifest last, read-only dry-run wrapper.

Hermetic: a local directory store and an in-memory fake of the google-cloud-storage surface.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import refresh_store as rs  # noqa: E402

B1 = "20261114T020000Z"
B2 = "20261115T020000Z"
WHEN = datetime(2026, 11, 16, 3, 0, tzinfo=timezone.utc)


def _src(tmp_path: Path, name: str = "src", manifest: bool = True) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "catalogue.parquet").write_bytes(b"cat")
    (d / "dense.faiss").write_bytes(b"dense!")
    if manifest:
        (d / rs.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    return d


# --- local store ---------------------------------------------------------------------------


def test_local_create_lists_and_reads_back(tmp_path: Path) -> None:
    store = rs.LocalBuildStore(tmp_path / "bucket")
    assert store.list_build_ids() == []
    store.create_build(_src(tmp_path), B1)
    assert store.list_build_ids() == [B1]
    assert store.list_objects(B1) == {"catalogue.parquet": 3, "dense.faiss": 6, "build_manifest.json": 2}
    assert store.read_object(B1, "dense.faiss", 100) == b"dense!"
    with pytest.raises(rs.StoreError):
        store.read_object(B1, "dense.faiss", 2)


def test_local_create_only_refuses_an_existing_build_and_leaves_it_untouched(tmp_path: Path) -> None:
    store = rs.LocalBuildStore(tmp_path / "bucket")
    store.create_build(_src(tmp_path), B1)
    other = _src(tmp_path, "other")
    (other / "dense.faiss").write_bytes(b"DIFFERENT")
    with pytest.raises(rs.CreateOnlyViolation):
        store.create_build(other, B1)
    assert store.read_object(B1, "dense.faiss", 100) == b"dense!"


def test_build_without_manifest_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    store = rs.LocalBuildStore(tmp_path / "bucket")
    with pytest.raises(rs.StoreError):
        store.create_build(_src(tmp_path, manifest=False), B1)
    assert store.list_objects(B1) == {}


@pytest.mark.parametrize("bad", ["../x", "20261115", "2026-11-15", "", "20261115T020000z"])
def test_bad_build_ids_are_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(rs.StoreError):
        rs.LocalBuildStore(tmp_path).list_objects(bad)


def test_read_only_wrapper_passes_reads_and_refuses_every_write(tmp_path: Path) -> None:
    inner = rs.LocalBuildStore(tmp_path / "bucket")
    inner.create_build(_src(tmp_path), B1)
    ro = rs.ReadOnlyStore(inner)
    assert ro.list_build_ids() == [B1]
    with pytest.raises(rs.StoreError):
        ro.create_build(_src(tmp_path, "s2"), B2)
    with pytest.raises(rs.StoreError):
        ro.set_custom_time(B1, WHEN)
    assert inner.list_build_ids() == [B1]


# --- GCS (fake client) ---------------------------------------------------------------------


class PreconditionFailed(Exception):
    code = 412


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self.bucket, self.name = bucket, name
        self.custom_time: datetime | None = None

    @property
    def size(self) -> int | None:
        data = self.bucket.objects.get(self.name)
        return None if data is None else len(data)

    def upload_from_filename(self, path: str, if_generation_match: int | None = None) -> None:
        self.bucket.upload_log.append((self.name, if_generation_match))
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise PreconditionFailed("exists")
        self.bucket.objects[self.name] = Path(path).read_bytes()

    def download_as_bytes(self) -> bytes:
        return self.bucket.objects[self.name]

    def patch(self) -> None:
        self.bucket.custom_times[self.name] = self.custom_time


class FakeIterator(list):
    prefixes: set[str]


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.custom_times: dict[str, datetime | None] = {}
        self.upload_log: list[tuple[str, int | None]] = []

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)

    def get_blob(self, name: str) -> FakeBlob | None:
        return FakeBlob(self, name) if name in self.objects else None

    def list_blobs(self, prefix: str = "", delimiter: str | None = None) -> FakeIterator:
        names = sorted(n for n in self.objects if n.startswith(prefix))
        it = FakeIterator([FakeBlob(self, n) for n in names] if delimiter is None else [])
        it.prefixes = set()
        if delimiter:
            for n in names:
                rest = n[len(prefix) :]
                if delimiter in rest:
                    it.prefixes.add(prefix + rest.split(delimiter)[0] + delimiter)
        return it


class FakeClient:
    def __init__(self) -> None:
        self.b = FakeBucket()

    def bucket(self, name: str) -> FakeBucket:
        assert name == "bkt"
        return self.b


def _gcs() -> tuple[rs.GcsBuildStore, FakeBucket]:
    client = FakeClient()
    return rs.GcsBuildStore("bkt", client=client), client.b


def test_gcs_create_uses_generation_zero_on_every_object_and_manifest_last(tmp_path: Path) -> None:
    store, bucket = _gcs()
    store.create_build(_src(tmp_path), B1)
    assert [n for n, _ in bucket.upload_log][-1] == f"builds/{B1}/{rs.MANIFEST_NAME}"
    assert {gen for _, gen in bucket.upload_log} == {0}
    assert len(bucket.upload_log) == 3
    assert store.list_build_ids() == [B1]


def test_gcs_existing_prefix_is_refused_before_any_upload(tmp_path: Path) -> None:
    store, bucket = _gcs()
    store.create_build(_src(tmp_path), B1)
    before = dict(bucket.objects)
    bucket.upload_log.clear()
    with pytest.raises(rs.CreateOnlyViolation):
        store.create_build(_src(tmp_path, "s2"), B1)
    assert bucket.upload_log == [] and bucket.objects == before


def test_gcs_race_precondition_failed_is_create_only_violation_not_an_overwrite(
    tmp_path: Path,
) -> None:
    store, bucket = _gcs()
    # An object appears between the empty-prefix check and the upload (a concurrent publisher).
    real_list = store.list_objects
    calls = {"n": 0}

    def racy(build_id: str) -> dict[str, int]:
        calls["n"] += 1
        if calls["n"] == 1:
            out = real_list(build_id)
            bucket.objects[f"builds/{build_id}/dense.faiss"] = b"theirs"
            return out
        return real_list(build_id)

    store.list_objects = racy  # type: ignore[method-assign]
    with pytest.raises(rs.CreateOnlyViolation):
        store.create_build(_src(tmp_path), B1)
    assert bucket.objects[f"builds/{B1}/dense.faiss"] == b"theirs"


def test_gcs_post_upload_size_mismatch_fails(tmp_path: Path) -> None:
    store, bucket = _gcs()
    src = _src(tmp_path)
    orig = FakeBlob.upload_from_filename

    def truncating(self: FakeBlob, path: str, if_generation_match: int | None = None) -> None:
        orig(self, path, if_generation_match)
        self.bucket.objects[self.name] = self.bucket.objects[self.name][:-1]

    FakeBlob.upload_from_filename = truncating  # type: ignore[method-assign]
    try:
        with pytest.raises(rs.StoreError):
            store.create_build(src, B1)
    finally:
        FakeBlob.upload_from_filename = orig  # type: ignore[method-assign]


def test_gcs_other_upload_errors_are_wrapped_without_leaking_details(tmp_path: Path) -> None:
    store, bucket = _gcs()

    def boom(self: FakeBlob, path: str, if_generation_match: Any = None) -> None:
        raise RuntimeError("secret-token-xyz")

    orig = FakeBlob.upload_from_filename
    FakeBlob.upload_from_filename = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(rs.StoreError) as ei:
            store.create_build(_src(tmp_path), B1)
    finally:
        FakeBlob.upload_from_filename = orig  # type: ignore[method-assign]
    assert "secret-token-xyz" not in str(ei.value)


def test_gcs_list_ids_ignores_other_prefixes_and_set_custom_time_touches_only_that_build(
    tmp_path: Path,
) -> None:
    store, bucket = _gcs()
    store.create_build(_src(tmp_path), B1)
    store.create_build(_src(tmp_path, "s2"), B2)
    bucket.objects["unified/dense.faiss"] = b"legacy"
    bucket.objects["builds/not-a-build/x"] = b"junk"
    assert store.list_build_ids() == [B1, B2]
    assert store.set_custom_time(B1, WHEN) == 3
    assert {n.split("/")[1] for n in bucket.custom_times} == {B1}


def test_gcs_rejects_a_bucket_with_a_path() -> None:
    with pytest.raises(rs.StoreError):
        rs.GcsBuildStore("bkt/builds")
