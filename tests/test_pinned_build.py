"""Revision-pinned index build (INDEX_BUILD_ID): startup load, failure semantics, wiring.

Hermetic: a local directory laid out like the bucket (``LocalDirSource``) holds real, tiny index
builds (tests/_index_fixtures.py: real FAISS + BM25, hashing embedder). No network, no GCS; every
root is a ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import api.deps as deps
from api import index_runtime
from src.config.store_registry import STORE_REGISTRY
from src.retrieval import pinned_build
from src.retrieval.build_manifest import (
    FileEntry,
    load_manifest,
    write_manifest,
)
from src.retrieval.index_reload import (
    BuildNotFoundError,
    BuildTooLargeError,
    GcsIndexSource,
    IndexSourceError,
    LocalDirSource,
    hot_reload_enabled,
)
from src.retrieval.pinned_build import PIN_REASONS, PinRequest, read_pin, reason_code
from src.retrieval.snapshot import load_index_parts as _REAL_PARTS  # captured before patching
from tests._index_fixtures import CONFIG, FakeModel, write_build

PINNED = "20261116T020000Z"
OTHER = "20261115T020000Z"
LOGGER = "api.index_runtime"


class SpySource:
    """LocalDirSource wrapper recording every request (build id, file name)."""

    def __init__(self, inner: LocalDirSource, fail_after: int | None = None) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str]] = []
        self.fail_after = fail_after

    def download_file(self, build_id: str, name: str, dest: Path, max_bytes: int) -> None:
        self.calls.append((build_id, name))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise OSError("connection reset by https://user:s3cret-pw@store.example/x")
        self.inner.download_file(build_id, name, dest, max_bytes)

    def list_pointer_names(self) -> list[str]:  # pragma: no cover - must never be called
        raise AssertionError("pointers must not be read by a pinned startup")

    def read_pointer(self, name: str) -> Any:  # pragma: no cover
        raise AssertionError("pointers must not be read by a pinned startup")

    def download_build(self, build_id: str, dest: Path) -> None:  # pragma: no cover
        raise AssertionError("the whole-directory download must not be used")


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    root.mkdir()
    write_build(root, OTHER, {"libas": 4}, publish_pointer=False)
    write_build(root, PINNED, {"libas": 6, "fashor": 3}, publish_pointer=False)
    return root


@pytest.fixture()
def scratch(tmp_path: Path) -> Path:
    return tmp_path / "data" / "pinned"


def _load(
    store: Path, scratch: Path, build_id: str | None = PINNED, **kw: Any
) -> tuple[Any, str | None]:
    pin = PinRequest(True, build_id, None) if build_id else PinRequest(True, None, "id_invalid")
    kw.setdefault("source", LocalDirSource(store))
    return index_runtime.load_pinned_snapshot(
        pin,
        None,
        CONFIG,
        scratch,
        model=FakeModel(),
        registry_slugs=frozenset(STORE_REGISTRY),
        **kw,
    )


def _leftovers(scratch: Path) -> list[Path]:
    return list(scratch.iterdir()) if scratch.exists() else []


# --------------------------------------------------------------------------- happy path


def test_pinned_build_loads_and_reports_its_identity(store: Path, scratch: Path) -> None:
    snap, code = _load(store, scratch)
    assert code is None and snap is not None
    assert snap.source == "build" and snap.build_id == PINNED
    assert snap.manifest is not None and snap.manifest.build_id == PINNED
    assert snap.retriever.dense.index.ntotal == 9 == snap.manifest.total_items


def test_the_scratch_copy_is_deleted_after_a_successful_load(store: Path, scratch: Path) -> None:
    _load(store, scratch)
    assert _leftovers(scratch) == []  # one build in RAM, none on disk


def test_only_the_pinned_build_is_requested_and_only_manifest_files(
    store: Path, scratch: Path
) -> None:
    spy = SpySource(LocalDirSource(store))
    snap, _ = _load(store, scratch, source=spy)
    assert snap is not None
    assert {b for b, _ in spy.calls} == {PINNED}  # never OTHER, never a listing or a pointer
    listed = {f.name for f in load_manifest(store / "builds" / PINNED).files}
    assert {n for _, n in spy.calls} == listed | {"build_manifest.json"}


# --------------------------------------------------------------------------- failures


def _flip_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))


def test_a_missing_build_falls_back_with_not_found(store: Path, scratch: Path) -> None:
    snap, code = _load(store, scratch, build_id="20261120T020000Z")
    assert snap is None and code == "not_found"
    assert _leftovers(scratch) == []


def test_a_corrupt_file_is_an_integrity_failure(store: Path, scratch: Path) -> None:
    _flip_byte(store / "builds" / PINNED / "dense.faiss")  # same size, different sha256
    snap, code = _load(store, scratch)
    assert snap is None and code == "integrity"
    assert _leftovers(scratch) == []


def test_a_wrong_dimension_in_the_manifest_is_inconsistent(store: Path, scratch: Path) -> None:
    bdir = store / "builds" / PINNED
    m = load_manifest(bdir)
    write_manifest(bdir, m.model_copy(update={"dense_dim": 384}))  # files untouched, hashes fine
    snap, code = _load(store, scratch)
    assert snap is None and code == "inconsistent"
    assert _leftovers(scratch) == []


def test_a_manifest_for_another_build_is_inconsistent(store: Path, scratch: Path) -> None:
    bdir = store / "builds" / PINNED
    write_manifest(bdir, load_manifest(bdir).model_copy(update={"build_id": OTHER}))
    snap, code = _load(store, scratch)
    assert snap is None and code == "inconsistent"


def test_an_oversized_file_is_refused_before_it_is_downloaded(
    store: Path, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pinned_build, "MAX_FILE_BYTES", 100)
    spy = SpySource(LocalDirSource(store))
    snap, code = _load(store, scratch, source=spy)
    assert snap is None and code == "too_large"
    assert spy.calls == [(PINNED, "build_manifest.json")]  # no data file was requested


def test_the_source_refuses_an_object_larger_than_its_manifest_entry(store: Path) -> None:
    src = LocalDirSource(store)
    with pytest.raises(BuildTooLargeError):
        src.download_file(PINNED, "dense.faiss", store / "out", 10)
    assert not (store / "out").exists()


@pytest.mark.parametrize("bad_name", ["../evil", "/abs/x", "a\\b", "C:x", "evil.sh", ".."])
def test_a_manifest_naming_an_unexpected_or_non_flat_file_is_refused(
    store: Path, scratch: Path, bad_name: str
) -> None:
    bdir = store / "builds" / PINNED
    m = load_manifest(bdir)
    entry = FileEntry(name=bad_name, sha256="0" * 64, size_bytes=1)
    write_manifest(bdir, m.model_copy(update={"files": [*m.files, entry]}))
    spy = SpySource(LocalDirSource(store))
    snap, code = _load(store, scratch, source=spy)
    assert snap is None and code == "manifest_invalid"
    assert [n for _, n in spy.calls] == ["build_manifest.json"]


def test_a_partial_download_is_deleted_and_the_error_line_has_no_credentials(
    store: Path, scratch: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    spy = SpySource(LocalDirSource(store), fail_after=3)  # manifest + 2 files, then it breaks
    snap, code = _load(store, scratch, source=spy)
    assert snap is None and code == "source_error"
    assert _leftovers(scratch) == []
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # ONE structured error line
    text = errors[0].getMessage()
    assert "index_pin_failed" in text and f"pinned_build_id={PINNED}" in text
    assert "code=source_error" in text and "s3cret-pw" not in text


def test_a_loader_exception_never_crashes_and_reports_load_failed(
    store: Path, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("segfault-ish")

    monkeypatch.setattr(index_runtime, "load_index_parts", boom)
    snap, code = _load(store, scratch)
    assert snap is None and code == "load_failed"
    assert _leftovers(scratch) == []


def test_no_usable_store_is_no_source(scratch: Path) -> None:
    pin = PinRequest(True, PINNED, None)
    assert index_runtime.load_pinned_snapshot(pin, None, CONFIG, scratch) == (None, "no_source")
    assert index_runtime.load_pinned_snapshot(pin, "ftp://x/y", CONFIG, scratch) == (
        None,
        "no_source",
    )


# --------------------------------------------------------------------------- id validation


@pytest.mark.parametrize(
    "raw",
    ["../x", "/etc/passwd", "C:\\x", "20261116T020000Z/../x", "x" * 300, "20261116t020000z", "abc"],
)
def test_malformed_ids_are_rejected_and_never_echoed(raw: str) -> None:
    pin = read_pin({"INDEX_BUILD_ID": raw})
    assert pin == PinRequest(True, None, "id_invalid")


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_is_not_a_pin(raw: str | None) -> None:
    environ = {} if raw is None else {"INDEX_BUILD_ID": raw}
    assert read_pin(environ) == PinRequest(False)


def test_a_valid_id_is_accepted_and_trimmed() -> None:
    assert read_pin({"INDEX_BUILD_ID": f" {PINNED} "}) == PinRequest(True, PINNED, None)


def test_an_invalid_pin_touches_no_store(scratch: Path, store: Path) -> None:
    spy = SpySource(LocalDirSource(store))
    snap, code = index_runtime.load_pinned_snapshot(
        PinRequest(True, None, "id_invalid"), None, CONFIG, scratch, source=spy
    )
    assert (snap, code) == (None, "id_invalid") and spy.calls == []


def test_download_refuses_an_invalid_id_even_when_called_directly(tmp_path: Path) -> None:
    with pytest.raises(pinned_build.PinnedBuildError) as ei:
        pinned_build.download_pinned_build(LocalDirSource(tmp_path), "../x", tmp_path)
    assert ei.value.code == "id_invalid"


def test_every_reason_code_is_in_the_fixed_list() -> None:
    from src.retrieval.build_manifest import BuildValidationError

    reasons = [
        "manifest_missing", "manifest_invalid", "manifest_schema", "unexpected_file",
        "file_missing", "file_size", "file_hash", "unlisted_file", "clip_mismatch",
        "count_mismatch", "store_unknown", "vector_count", "dim_mismatch",
        "build_id_mismatch", "count_gate", "weird",
    ]  # fmt: skip
    codes = {reason_code(BuildValidationError(r, "x")) for r in reasons}
    codes |= {reason_code(e) for e in (BuildNotFoundError("x"), BuildTooLargeError("x"))}
    codes |= {reason_code(e) for e in (IndexSourceError("x"), OSError("x"), ValueError("x"))}
    assert codes <= set(PIN_REASONS)


# --------------------------------------------------------------------------- sources


def test_make_pinned_source_accepts_gs_and_dev_file_uris(tmp_path: Path) -> None:
    assert isinstance(index_runtime.make_pinned_source("gs://bucket/prefix"), GcsIndexSource)
    src = index_runtime.make_pinned_source(tmp_path.as_uri())
    assert isinstance(src, LocalDirSource) and src.root == tmp_path
    for bad in (None, "", "ftp://x/y", "file://remote-host/x", "/just/a/path", "gs://"):
        assert index_runtime.make_pinned_source(bad) is None


def test_gcs_download_file_reads_one_named_object_and_never_lists(tmp_path: Path) -> None:
    class Blob:
        def __init__(self, size: int | None) -> None:
            self.size = size
            self.downloaded: str | None = None

        def download_to_filename(self, path: str) -> None:
            self.downloaded = path

    class Bucket:
        def __init__(self) -> None:
            self.asked: list[str] = []
            self.blobs = {"p/builds/B/a.npy": Blob(5), "p/builds/B/big": Blob(10**9)}

        def get_blob(self, name: str) -> Blob | None:
            self.asked.append(name)
            return self.blobs.get(name)

        def list_blobs(self, **_: Any) -> Any:  # pragma: no cover
            raise AssertionError("no listing")

    src = GcsIndexSource("gs://bucket/p")
    bucket = Bucket()
    src._bucket = bucket
    src.download_file("B", "a.npy", tmp_path / "a.npy", 100)
    assert bucket.blobs["p/builds/B/a.npy"].downloaded == str(tmp_path / "a.npy")
    with pytest.raises(BuildTooLargeError):
        src.download_file("B", "big", tmp_path / "big", 100)
    with pytest.raises(BuildNotFoundError):
        src.download_file("B", "missing", tmp_path / "m", 100)
    assert bucket.asked == ["p/builds/B/a.npy", "p/builds/B/big", "p/builds/B/missing"]


def test_hot_reload_stays_off_by_default() -> None:
    assert hot_reload_enabled({}) is False
    assert hot_reload_enabled({"INDEX_BUILD_ID": PINNED}) is False  # a pin does not enable it


# --------------------------------------------------------------------------- lifespan wiring


class Boot:
    """Runs api.main.lifespan with the heavy parts stubbed; captures what reaches deps._init."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: Path) -> None:
        import api.main as main

        self.main = main
        self.store = store
        self.data = tmp_path / "boot-data"
        self.init_kwargs: dict[str, Any] = {}
        self.reloads_started = 0
        monkeypatch.setattr(main, "_DATA_DIR", self.data)
        monkeypatch.setattr(main, "_REPO_ROOT", tmp_path / "boot-root")
        monkeypatch.setattr(
            "src.llm.client.get_llm_client", lambda _cfg: SimpleNamespace(), raising=True
        )
        monkeypatch.setattr("src.agents.outfit.image_anchor.warm", lambda *a, **k: False)
        monkeypatch.setattr(index_runtime, "load_index_parts", self._parts)
        monkeypatch.setattr("src.retrieval.snapshot.load_index_parts", self._parts)

        def fake_init(retriever: Any, df: Any, llm: Any, config: Any, **kw: Any) -> None:
            self.init_kwargs = {"retriever": retriever, "df": df, **kw}

        monkeypatch.setattr(deps, "_init", fake_init)
        monkeypatch.setattr(
            index_runtime,
            "start_hot_reload",
            lambda *a, **k: self.__setattr__("reloads_started", self.reloads_started + 1),
        )
        for name in ("INDEX_STORE_URI", "INDEX_BUILD_ID", "INDEX_HOT_RELOAD", "BRAND", "UNIFIED"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("DEMO_MODE", "false")

    @staticmethod
    def _parts(index_dir: Path, config: dict, *, model: Any = None) -> Any:
        return _REAL_PARTS(index_dir, config, model=model or FakeModel())

    def run(self) -> dict[str, Any]:
        async def go() -> None:
            async with self.main.lifespan(self.main.app):
                pass

        asyncio.run(go())
        return self.init_kwargs


@pytest.fixture()
def boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: Path) -> Boot:
    from tests._index_fixtures import make_rows, write_index_files

    b = Boot(tmp_path, monkeypatch, store)
    write_index_files(make_rows({"libas": 5}), b.data / "unified")  # the flat legacy index
    monkeypatch.setenv("INDEX_STORE_URI", store.as_uri())
    return b


def test_lifespan_pinned_happy_path(boot: Boot, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_BUILD_ID", PINNED)
    kw = boot.run()
    assert kw["source"] == "build" and kw["manifest"].build_id == PINNED
    assert kw["pinned_build_id"] == PINNED and kw["pin_failure"] is None
    assert len(kw["df"]) == 9


def test_lifespan_unset_is_exactly_the_legacy_flat_load(boot: Boot) -> None:
    kw = boot.run()
    assert kw["source"] == "flat" and kw["manifest"] is None
    assert kw["pinned_build_id"] is None and kw["pin_failure"] is None
    assert len(kw["df"]) == 5


def test_lifespan_missing_build_serves_legacy_and_records_the_failure(
    boot: Boot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INDEX_BUILD_ID", "20300101T000000Z")
    seen: list[str] = []

    class Grab(logging.Handler):  # attached to the logger itself: setup_logging() resets root
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    handler = Grab()
    index_runtime.logger.addHandler(handler)
    try:
        kw = boot.run()  # does not raise
    finally:
        index_runtime.logger.removeHandler(handler)
    assert kw["source"] == "flat" and len(kw["df"]) == 5  # the legacy index, NOT another build
    assert kw["pinned_build_id"] == "20300101T000000Z" and kw["pin_failure"] == "not_found"
    assert sum("index_pin_failed" in m for m in seen) == 1


def test_lifespan_corrupt_build_falls_back_and_never_serves_it(
    boot: Boot, monkeypatch: pytest.MonkeyPatch, store: Path
) -> None:
    _flip_byte(store / "builds" / PINNED / "bm25.pkl")
    monkeypatch.setenv("INDEX_BUILD_ID", PINNED)
    kw = boot.run()
    assert kw["source"] == "flat" and kw["pin_failure"] == "integrity"
    assert len(kw["df"]) == 5


@pytest.mark.parametrize("raw", ["../x", "20261116t020000z", "x" * 300])
def test_lifespan_bad_id_falls_back_without_echoing_it(
    boot: Boot, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("INDEX_BUILD_ID", raw)
    kw = boot.run()
    assert kw["source"] == "flat" and kw["pinned_build_id"] is None
    assert kw["pin_failure"] == "id_invalid"


def test_lifespan_a_pin_disables_hot_reload_even_when_switched_on(
    boot: Boot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INDEX_BUILD_ID", PINNED)
    monkeypatch.setenv("INDEX_HOT_RELOAD", "1")
    boot.run()
    assert boot.reloads_started == 0


def test_lifespan_pin_on_a_non_unified_brand_is_no_source(
    boot: Boot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests._index_fixtures import make_rows, write_index_files

    monkeypatch.setenv("BRAND", "snitch")
    monkeypatch.setenv("INDEX_BUILD_ID", PINNED)
    write_index_files(make_rows({"libas": 5}), boot.data / "snitch")
    kw = boot.run()
    assert kw["pin_failure"] == "no_source" and kw["source"] == "flat"


# --------------------------------------------------------------------------- log/readyz hygiene


def test_exception_text_reaches_neither_the_log_nor_readyz(
    store: Path, scratch: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd
    from fastapi.testclient import TestClient

    from api.main import app
    from src.retrieval.snapshot import IndexSnapshot

    secret_path = r"C:\Users\victim\secret-dir\leak.bin"
    secret_url = "https://admin:hunter2-pw@bucket.example/o?token=abc123"

    class Boom(SpySource):
        def download_file(self, build_id: str, name: str, dest: Path, max_bytes: int) -> None:
            raise OSError(f"cannot write {secret_path} via {secret_url}")

    caplog.set_level(logging.DEBUG)
    snap, code = _load(store, scratch, source=Boom(LocalDirSource(store)))
    assert snap is None and code == "source_error"
    records = [r for r in caplog.records if "index_pin_failed" in r.getMessage()]
    assert len(records) == 1
    line = records[0].getMessage()
    assert line == (
        f"index_pin_failed pinned_build_id={PINNED} code=source_error reason=OSError exc=OSError"
    )
    everything = " ".join(r.getMessage() + str(r.args) + str(r.exc_info) for r in caplog.records)
    for leak in (
        "victim",
        "secret-dir",
        "leak.bin",
        "hunter2",
        "admin",
        "abc123",
        "bucket.example",
    ):
        assert leak not in everything

    retriever = SimpleNamespace(dense=SimpleNamespace(index=SimpleNamespace(ntotal=1)))
    monkeypatch.setattr(
        deps,
        "_snapshot",
        IndexSnapshot(
            retriever=retriever,
            catalogue_df=pd.DataFrame({"article_id": ["a"]}),
            pinned_build_id=PINNED,
            pin_failure=code,
        ),
    )
    monkeypatch.setattr(deps, "_llm", SimpleNamespace())
    monkeypatch.setattr(deps, "_db_engine", None)
    body = TestClient(app).get("/readyz").text
    assert code in body
    for leak in (
        "victim",
        "secret-dir",
        "leak.bin",
        "hunter2",
        "admin",
        "abc123",
        "bucket.example",
    ):
        assert leak not in body
