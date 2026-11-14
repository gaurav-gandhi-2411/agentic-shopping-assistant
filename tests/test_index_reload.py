"""In-process index hot reload (Wave 1b V3(b)): swap, release, in-flight consistency, failures.

Hermetic: a local directory laid out like the bucket (LocalDirSource) holds real, tiny index
builds (tests/_index_fixtures.py: real FAISS + BM25, hashing embedder). No network, no GCS.
The served snapshot lives in api.deps exactly as in production (install_snapshot).
"""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import pytest

import api.deps as deps
from api import index_runtime
from src.config.store_registry import STORE_REGISTRY
from src.retrieval.build_manifest import load_manifest, write_manifest
from src.retrieval.index_reload import (
    DEFAULT_INTERVAL_S,
    IndexReloader,
    IndexSourceError,
    LocalDirSource,
    default_smoke,
    fetch_and_load_build,
    hot_reload_enabled,
    latest_pointer,
    reload_interval_s,
    sanitize,
)
from src.retrieval.snapshot import IndexSnapshot
from tests._index_fixtures import (
    CONFIG,
    FakeModel,
    load_build_fn,
    make_snapshot,
    publish,
    write_build,
)

OLD = "20261115T020000Z"
NEW = "20261116T020000Z"
NEWER = "20261117T020000Z"
LOGGER = "src.retrieval.index_reload"


class Env:
    """A store root with an OLD build already served, plus helpers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tmp = root.parent / "reload-tmp"
        self.tmp.mkdir()
        self.old_manifest = write_build(root, OLD, {"libas": 6})
        deps.install_snapshot(
            make_snapshot(root / "builds" / OLD, manifest=self.old_manifest, model=FakeModel())
        )

    def reloader(self, **kw: Any) -> IndexReloader:
        kw.setdefault("tmp_root", self.tmp)
        kw.setdefault("registry_slugs", frozenset(STORE_REGISTRY))
        return IndexReloader(
            kw.pop("source", LocalDirSource(self.root)),
            load_build_fn,
            deps.get_snapshot_or_none,
            deps.install_snapshot,
            **kw,
        )

    def served(self) -> IndexSnapshot:
        snap = deps.get_snapshot_or_none()
        assert snap is not None
        return snap

    def new_build(
        self, build_id: str = NEW, stores: dict[str, int] | None = None, **kw: Any
    ) -> None:
        write_build(self.root, build_id, stores or {"fashor": 5}, previous=self.old_manifest, **kw)


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.setattr(deps, "_snapshot", None)
    root = tmp_path / "store"
    root.mkdir()
    return Env(root)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno >= logging.WARNING]


# --------------------------------------------------------------------------- swap


def test_swap_happens_only_on_a_new_valid_pointer(env: Env) -> None:
    rl = env.reloader()
    served0 = env.served()
    assert rl.check_once().outcome == "unchanged"  # pointer == served build
    assert env.served() is served0

    env.new_build()
    res = rl.check_once()
    assert (res.outcome, res.build_id) == ("swapped", NEW)
    assert env.served() is not served0 and env.served().build_id == NEW
    assert set(env.served().catalogue_df["store"]) == {"fashor"}
    assert rl.check_once().outcome == "unchanged"  # idempotent
    st = rl.status()
    assert st["last_outcome"] == "unchanged" and st["last_swap_at"] is not None
    assert st["consecutive_failures"] == 0


def test_no_pointer_is_not_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deps, "_snapshot", None)
    (tmp_path / "pointers").mkdir()
    rl = IndexReloader(
        LocalDirSource(tmp_path), load_build_fn, deps.get_snapshot_or_none, deps.install_snapshot
    )
    assert rl.check_once().outcome == "no_pointer"
    assert rl.consecutive_failures == 0


def test_the_old_snapshot_is_released_after_the_swap(env: Env) -> None:
    old_ref = weakref.ref(env.served())
    old_retriever = weakref.ref(env.served().retriever)
    old_df = weakref.ref(env.served().catalogue_df)
    env.new_build()
    rl = env.reloader()  # kept alive: the reloader itself must not retain the old snapshot
    assert rl.check_once().outcome == "swapped"
    gc.collect()
    assert old_ref() is None, "old IndexSnapshot still referenced after the swap"
    assert old_retriever() is None and old_df() is None  # faiss/BM25/DataFrame freed with it


def test_in_flight_request_keeps_its_snapshot_while_a_swap_happens(env: Env) -> None:
    env.new_build()
    taken, proceed = threading.Event(), threading.Event()
    seen: dict[str, Any] = {}

    def in_flight_turn() -> None:
        snap = deps.get_snapshot()  # taken ONCE at the start of the turn
        taken.set()
        proceed.wait(5)  # the swap happens here
        seen["build_id"] = snap.build_id
        seen["search_stores"] = {r["store"] for r in snap.retriever.search("kurta", top_k=10)}
        seen["df_stores"] = set(snap.catalogue_df["store"])

    t = threading.Thread(target=in_flight_turn)
    t.start()
    assert taken.wait(5)
    assert env.reloader().check_once().outcome == "swapped"
    # a request that starts AFTER the swap sees the new build
    assert deps.get_snapshot().build_id == NEW
    proceed.set()
    t.join(5)
    assert not t.is_alive()
    # the in-flight turn finished entirely on the OLD build, consistently
    assert seen["build_id"] == OLD
    assert seen["search_stores"] == {"libas"} and seen["df_stores"] == {"libas"}


def test_in_flight_snapshot_is_freed_once_its_turn_ends(env: Env) -> None:
    held = env.served()
    old_ref = weakref.ref(held)
    env.new_build()
    rl = env.reloader()
    rl.check_once()
    gc.collect()
    assert old_ref() is not None  # still referenced by the in-flight "turn"
    del held
    gc.collect()
    assert old_ref() is None


def test_rollback_by_deleting_the_newest_pointer_restores_the_previous_build(env: Env) -> None:
    rl = env.reloader()
    env.new_build()
    assert rl.check_once().outcome == "swapped"
    (env.root / "pointers" / f"{NEW}.json").unlink()  # the owner's rollback
    res = rl.check_once()
    assert (res.outcome, res.build_id) == ("swapped", OLD)
    assert env.served().build_id == OLD


# --------------------------------------------------------------------------- failures


def _corrupt_hash(env: Env) -> None:
    p = env.root / "builds" / NEW / "bm25.pkl"
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF
    p.write_bytes(bytes(data))


def _truncate(env: Env) -> None:
    p = env.root / "builds" / NEW / "dense.faiss"
    p.write_bytes(p.read_bytes()[:-8])


def _wrong_dim(env: Env) -> None:
    m = load_manifest(env.root / "builds" / NEW)
    m.dense_dim = 32  # files are untouched (hashes fine); the loaded index has d=16
    write_manifest(env.root / "builds" / NEW, m)


def _missing_manifest(env: Env) -> None:
    (env.root / "builds" / NEW / "build_manifest.json").unlink()


def _missing_file(env: Env) -> None:
    (env.root / "builds" / NEW / "catalogue.parquet").unlink()


def _count_gate(env: Env) -> None:
    import shutil

    shutil.rmtree(env.root / "builds" / NEW)
    (env.root / "pointers" / f"{NEW}.json").unlink()
    write_build(env.root, NEW, {"fashor": 2}, previous=env.old_manifest)  # 6 -> 2 items


def _unknown_store(env: Env) -> None:
    import shutil

    shutil.rmtree(env.root / "builds" / NEW)
    write_build(env.root, NEW, {"fashor": 3, "no-such-store": 2}, previous=env.old_manifest)


def _bad_pointer_body(env: Env) -> None:
    (env.root / "pointers" / f"{NEW}.json").write_text('{"build_id": "../../etc"}', "utf-8")


@pytest.mark.parametrize(
    ("break_it", "reason"),
    [
        (_corrupt_hash, "file_hash"),
        (_truncate, "file_size"),
        (_wrong_dim, "dim_mismatch"),
        (_missing_manifest, "manifest_missing"),
        (_missing_file, "file_missing"),
        (_count_gate, "count_gate"),
        (_unknown_store, "store_unknown"),
        (_bad_pointer_body, "source"),
    ],
)
def test_every_validation_failure_keeps_the_old_snapshot_and_logs_once(
    env: Env, caplog: pytest.LogCaptureFixture, break_it: Any, reason: str
) -> None:
    env.new_build()
    break_it(env)
    served0 = env.served()
    rl = env.reloader()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        res = rl.check_once()
    assert (res.outcome, res.reason) == ("failed", reason)
    assert env.served() is served0 and env.served().build_id == OLD
    warns = _warnings(caplog)
    assert len(warns) == 1, [w.getMessage() for w in warns]
    msg = warns[0].getMessage()
    assert msg.startswith("index_reload_failed") and f"reason={reason}" in msg
    assert "swapped" not in " ".join(r.getMessage() for r in caplog.records)
    assert rl.consecutive_failures == 1
    assert list(env.tmp.iterdir()) == []  # temp dir deleted even on failure


def test_exception_in_smoke_keeps_the_old_snapshot(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    env.new_build()
    served0 = env.served()

    def bad_smoke(_: IndexSnapshot) -> None:
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        res = env.reloader(smoke=bad_smoke).check_once()
    assert (res.outcome, res.reason) == ("failed", "RuntimeError")
    assert env.served() is served0
    assert len(_warnings(caplog)) == 1


def test_exception_in_download_keeps_the_old_snapshot(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    env.new_build()
    served0 = env.served()

    class Flaky(LocalDirSource):
        def download_build(self, build_id: str, dest: Path) -> None:
            raise OSError("connection reset by peer https://user:s3cret@storage.example/x")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        res = env.reloader(source=Flaky(env.root)).check_once()
    assert (res.outcome, res.reason) == ("failed", "OSError")
    assert env.served() is served0
    warns = _warnings(caplog)
    assert len(warns) == 1
    assert "s3cret" not in warns[0].getMessage()  # credentials in a URL never reach the log
    assert list(env.tmp.iterdir()) == []


def test_exception_in_the_loader_keeps_the_old_snapshot(env: Env) -> None:
    env.new_build()
    served0 = env.served()

    def broken_loader(*_: Any) -> IndexSnapshot:
        raise MemoryError("simulated")

    rl = IndexReloader(
        LocalDirSource(env.root),
        broken_loader,
        deps.get_snapshot_or_none,
        deps.install_snapshot,
        tmp_root=env.tmp,
    )
    assert rl.check_once().outcome == "failed"
    assert env.served() is served0


def test_a_mixed_torn_build_is_rejected_by_the_manifest_hashes(env: Env) -> None:
    """A store that serves build NEW's manifest with one file from build OLD (a torn publish)
    can never be loaded: every file is hash-checked against the manifest of ONE build."""
    env.new_build()
    old_dir, new_dir = env.root / "builds" / OLD, env.root / "builds" / NEW
    (new_dir / "dense_article_ids.npy").write_bytes(
        (old_dir / "dense_article_ids.npy").read_bytes()
    )
    res = env.reloader().check_once()
    assert res.outcome == "failed" and res.reason in ("file_size", "file_hash")


def test_each_attempt_uses_a_fresh_directory_so_files_never_mix_across_attempts(env: Env) -> None:
    env.new_build()
    p = env.root / "builds" / NEW / "bm25.pkl"
    good = p.read_bytes()
    p.write_bytes(good[:-4])
    rl = env.reloader()
    assert rl.check_once().outcome == "failed"
    p.write_bytes(good)  # the publisher finishes uploading
    assert rl.check_once().outcome == "swapped"
    assert list(env.tmp.iterdir()) == []


def test_a_build_with_a_manifest_naming_another_build_is_rejected(env: Env) -> None:
    env.new_build()
    m = load_manifest(env.root / "builds" / NEW)
    m.build_id = OLD
    write_manifest(env.root / "builds" / NEW, m)
    assert env.reloader().check_once().reason == "build_id_mismatch"


# --------------------------------------------------------------------------- backoff / flight


def test_backoff_doubles_up_to_one_hour_and_resets_on_success(env: Env) -> None:
    env.new_build()
    _corrupt_hash(env)
    rl = env.reloader(interval_s=900.0)
    assert rl.next_delay_s() == 900.0
    delays = []
    for _ in range(4):
        assert rl.check_once().outcome == "failed"
        delays.append(rl.next_delay_s())
    assert delays == [1800.0, 3600.0, 3600.0, 3600.0]  # exponential, capped at 1 h
    (env.root / "pointers" / f"{NEW}.json").unlink()  # owner removes the bad pointer
    assert rl.check_once().outcome == "unchanged"
    assert rl.next_delay_s() == 900.0


def test_reloads_are_single_flight(env: Env) -> None:
    env.new_build()
    started, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def slow_loader(build_dir: Path, manifest: Any, current: Any) -> IndexSnapshot:
        calls.append(1)
        started.set()
        release.wait(5)
        return load_build_fn(build_dir, manifest, current)

    rl = IndexReloader(
        LocalDirSource(env.root),
        slow_loader,
        deps.get_snapshot_or_none,
        deps.install_snapshot,
        tmp_root=env.tmp,
    )
    results: list[str] = []
    t = threading.Thread(target=lambda: results.append(rl.check_once().outcome))
    t.start()
    assert started.wait(5)
    assert rl.check_once().outcome == "busy"  # overlapping attempt returns immediately
    release.set()
    t.join(10)
    assert results == ["swapped"] and len(calls) == 1


# --------------------------------------------------------------------------- switches


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),  # default: OFF
        ({"INDEX_STORE_URI": "gs://bucket"}, False),  # a bucket alone does NOT enable it
        ({"INDEX_STORE_URI": "gs://bucket", "INDEX_HOT_RELOAD": "0"}, False),  # explicit off
        ({"INDEX_STORE_URI": "gs://bucket", "INDEX_HOT_RELOAD": "false"}, False),
        ({"INDEX_HOT_RELOAD": "banana"}, False),  # anything but an explicit yes is off
        ({"INDEX_HOT_RELOAD": ""}, False),
        ({"INDEX_HOT_RELOAD": "1"}, True),  # explicit on
        ({"INDEX_STORE_URI": "gs://bucket", "INDEX_HOT_RELOAD": "true"}, True),
    ],
)
def test_hot_reload_switch(environ: dict[str, str], expected: bool) -> None:
    assert hot_reload_enabled(environ) is expected


def test_default_is_off_no_source_no_thread_even_with_a_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INDEX_HOT_RELOAD", raising=False)
    monkeypatch.setenv("INDEX_STORE_URI", "gs://bucket")
    assert index_runtime.make_index_source("gs://bucket") is None  # flat layout as before V3b


def test_explicit_off_and_explicit_on_switch_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_HOT_RELOAD", "0")
    assert index_runtime.make_index_source("gs://bucket") is None
    monkeypatch.setenv("INDEX_HOT_RELOAD", "1")
    assert index_runtime.make_index_source("gs://bucket") is not None
    assert index_runtime.make_index_source(None) is None
    assert index_runtime.make_index_source("file:///x") is None


def test_interval_is_tunable_with_a_60_second_floor() -> None:
    assert reload_interval_s({}) == DEFAULT_INTERVAL_S == 900.0
    assert reload_interval_s({"INDEX_RELOAD_INTERVAL_S": "300"}) == 300.0
    assert reload_interval_s({"INDEX_RELOAD_INTERVAL_S": "5"}) == 60.0
    assert reload_interval_s({"INDEX_RELOAD_INTERVAL_S": "banana"}) == 900.0


def test_background_thread_swaps_without_blocking_and_stops(env: Env) -> None:
    rl = env.reloader(interval_s=0.05)
    rl.start()
    try:
        assert rl._thread is not None and rl._thread.daemon
        env.new_build()
        deadline = time.monotonic() + 10
        while deps.get_snapshot().build_id != NEW and time.monotonic() < deadline:
            time.sleep(0.05)
        assert deps.get_snapshot().build_id == NEW
    finally:
        rl.stop()
    assert rl._thread is not None and not rl._thread.is_alive()


# --------------------------------------------------------------------------- pointers / startup


def test_latest_pointer_is_the_lexicographically_largest_well_formed_name(tmp_path: Path) -> None:
    src = LocalDirSource(tmp_path)
    assert latest_pointer(src) is None
    for name in (OLD, NEW):
        publish(tmp_path, name)
    (tmp_path / "pointers" / "zzz-readme.txt").write_text("ignored", "utf-8")
    (tmp_path / "pointers" / "99999999T999999Z.json.tmp").write_text("{}", "utf-8")
    assert latest_pointer(src) == NEW
    publish(tmp_path, NEWER)
    assert latest_pointer(src) == NEWER


def test_a_hostile_pointer_cannot_steer_the_download_path(tmp_path: Path) -> None:
    publish(tmp_path, OLD)
    (tmp_path / "pointers" / f"{OLD}.json").write_text('{"build_id": "../builds/x"}', "utf-8")
    with pytest.raises(IndexSourceError):
        latest_pointer(LocalDirSource(tmp_path))
    with pytest.raises(Exception) as exc:
        fetch_and_load_build(LocalDirSource(tmp_path), "../x", load_build_fn)
    assert getattr(exc.value, "reason", "") == "build_id_invalid"


def test_startup_loads_the_newest_build_and_falls_back_when_there_is_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = LocalDirSource(tmp_path)
    assert index_runtime.load_newest_build(src, CONFIG, model=FakeModel()) is None  # no pointer
    write_build(tmp_path, OLD, {"libas": 6})
    snap = index_runtime.load_newest_build(src, CONFIG, model=FakeModel())
    assert snap is not None and snap.build_id == OLD and snap.source == "build"
    assert snap.manifest is not None and snap.agent_sync is None  # graphs come with deps._init
    # a corrupt newest build: ONE warning, flat fallback (None)
    write_build(tmp_path, NEW, {"libas": 6})
    p = tmp_path / "builds" / NEW / "bm25.pkl"
    p.write_bytes(p.read_bytes()[:-3])
    with caplog.at_level(logging.INFO, logger="api.index_runtime"):
        assert index_runtime.load_newest_build(src, CONFIG, model=FakeModel()) is None
    fallbacks = [r for r in caplog.records if "index_startup_fallback" in r.getMessage()]
    assert len(fallbacks) == 1 and "reason=file_size" in fallbacks[0].getMessage()


def test_start_hot_reload_registers_the_reloader_for_readyz(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deps, "_reloader", None)
    monkeypatch.setenv("INDEX_RELOAD_INTERVAL_S", "3600")
    rl = index_runtime.start_hot_reload(LocalDirSource(env.root), CONFIG, llm=object())
    try:
        assert deps.get_reloader() is rl
        assert rl.status()["hot_reload"] == "on" and rl.status()["interval_s"] == 3600.0
    finally:
        rl.stop()


def test_default_smoke_rejects_an_empty_catalogue_and_runs_a_search(env: Env) -> None:
    default_smoke(env.served())  # does not raise
    empty = IndexSnapshot(
        retriever=env.served().retriever, catalogue_df=env.served().catalogue_df.iloc[0:0]
    )
    with pytest.raises(Exception, match="smoke"):
        default_smoke(empty)


def test_sanitize_masks_url_credentials_and_caps_length() -> None:
    assert "pw" not in sanitize("GET https://bob:pw@host/x failed")
    assert len(sanitize("x" * 1000)) == 200


def test_pointer_file_is_plain_json_naming_the_build(env: Env) -> None:
    body = json.loads((env.root / "pointers" / f"{OLD}.json").read_text("utf-8"))
    assert body == {"build_id": OLD}
