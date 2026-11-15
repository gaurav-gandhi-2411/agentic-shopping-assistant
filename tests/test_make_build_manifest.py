"""scripts/ops/make_build_manifest.py: golden manifest, loader round-trip, refusals.

Hermetic: tiny synthetic builds (real FAISS + BM25, hashing embedder) under ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from api import index_runtime
from src.config.store_registry import STORE_REGISTRY
from src.retrieval.build_manifest import (
    MANIFEST_NAME,
    load_manifest,
    sha256_file,
    verify_build_dir,
    write_manifest,
)
from src.retrieval.pinned_build import PinRequest
from tests._index_fixtures import CONFIG, FakeModel, make_rows, write_index_files

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "make_build_manifest.py"
_spec = importlib.util.spec_from_file_location("make_build_manifest", _PATH)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

GOLDEN = Path(__file__).parent / "fixtures" / "build_manifest_golden.json"
BUILD = "20261115T020000Z"
BUILT_AT = "2026-11-15T02:00:00+00:00"


def _index(tmp_path: Path, stores: dict[str, int] | None = None, **kw: Any) -> Path:
    d = tmp_path / "idx"
    write_index_files(make_rows(stores or {"libas": 4, "fashor": 2}), d, **kw)
    return d


def _run(d: Path, *extra: str) -> int:
    return tool.main(
        [
            "--index-dir",
            str(d),
            "--build-id",
            BUILD,
            "--git-sha",
            "deadbeef",
            "--built-at",
            BUILT_AT,
        ]
        + list(extra)
    )


def _normalised(manifest: dict[str, Any]) -> dict[str, Any]:
    """Drop the byte-level values that depend on the FAISS/pickle/parquet library versions."""
    out = json.loads(json.dumps(manifest))
    for f in out["files"]:
        f["sha256"] = "<sha256>"
        f["size_bytes"] = "<size>"
    return out


def test_golden_manifest_for_a_tiny_synthetic_build(tmp_path: Path) -> None:
    d = _index(tmp_path, with_clip=True)
    assert _run(d) == 0
    written = json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert _normalised(written) == json.loads(GOLDEN.read_text(encoding="utf-8"))
    for f in written["files"]:  # the values dropped above are the real ones
        assert f["sha256"] == sha256_file(d / f["name"])
        assert f["size_bytes"] == (d / f["name"]).stat().st_size


def test_the_written_manifest_passes_the_loaders_verifier_and_loads(tmp_path: Path) -> None:
    d = _index(tmp_path, with_clip=True)
    assert _run(d) == 0
    manifest = load_manifest(d)
    verify_build_dir(d, manifest, registry_slugs=frozenset(STORE_REGISTRY))
    # Full round trip through the API's pinned startup path.
    store = tmp_path / "store"
    (store / "builds").mkdir(parents=True)
    d.rename(store / "builds" / BUILD)
    snap, code = index_runtime.load_pinned_snapshot(
        PinRequest(True, BUILD, None),
        store.as_uri(),
        CONFIG,
        tmp_path / "scratch",
        model=FakeModel(),
    )
    assert code is None and snap is not None
    assert snap.build_id == BUILD and snap.retriever.dense.index.ntotal == manifest.total_items


def test_manifest_facts_come_from_the_files(tmp_path: Path) -> None:
    d = _index(tmp_path)
    assert _run(d) == 0
    m = load_manifest(d)
    assert {s.slug: s.items for s in m.stores} == {"fashor": 2, "libas": 4}
    assert m.total_items == m.vector_count == 6 and m.dense_dim == 16 and m.clip_dim is None
    assert m.git_sha == "deadbeef" and m.built_at == datetime(2026, 11, 15, 2, tzinfo=timezone.utc)
    assert m.previous_build_id is None and m.validation.schema_ok is True


def test_previous_manifest_sets_the_lineage_and_the_gate_result(tmp_path: Path) -> None:
    prev_dir = tmp_path / "prev"
    write_index_files(make_rows({"libas": 5, "fashor": 2}), prev_dir)  # 7 items
    assert tool.main(["--index-dir", str(prev_dir), "--build-id", "20261114T020000Z",
                      "--git-sha", "a"]) == 0  # fmt: skip
    d = _index(tmp_path)  # 6 items: -14.3 %
    assert _run(d, "--previous", str(prev_dir)) == 0
    m = load_manifest(d)
    assert m.previous_build_id == "20261114T020000Z"
    assert m.validation.count_delta_pct == pytest.approx(-14.2857, abs=1e-3)


def test_previous_may_be_given_as_the_manifest_file(tmp_path: Path) -> None:
    prev_dir = tmp_path / "prev"
    write_index_files(make_rows({"libas": 6}), prev_dir)
    assert tool.main(["--index-dir", str(prev_dir), "--build-id", "20261114T020000Z",
                      "--git-sha", "a"]) == 0  # fmt: skip
    d = _index(tmp_path)
    assert _run(d, "--previous", str(prev_dir / MANIFEST_NAME)) == 0


# --------------------------------------------------------------------------- refusals


def _refused(d: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> str:
    assert _run(d, *extra) == 1
    assert not (d / MANIFEST_NAME).exists()  # nothing written on a refusal
    return capsys.readouterr().err


def test_refuses_when_the_count_gate_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prev_dir = tmp_path / "prev"
    write_index_files(make_rows({"libas": 20}), prev_dir)
    assert tool.main(["--index-dir", str(prev_dir), "--build-id", "20261114T020000Z",
                      "--git-sha", "a"]) == 0  # fmt: skip
    err = _refused(_index(tmp_path), capsys, "--previous", str(prev_dir))
    assert "REFUSED" in err and "count_gate" in err


def test_refuses_a_missing_required_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path)
    (d / "bm25.pkl").unlink()
    assert "bm25.pkl" in _refused(d, capsys)


def test_refuses_a_store_outside_the_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path, {"libas": 3, "not-a-store": 2})
    assert "not-a-store" in _refused(d, capsys)


def test_refuses_when_the_id_files_disagree_with_the_catalogue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path)
    np.save(str(d / "dense_article_ids.npy"), np.array(["a", "b"]))
    assert "count_mismatch" in _refused(d, capsys)


def test_refuses_when_the_vector_count_differs_from_the_catalogue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path)
    other = tmp_path / "other"
    write_index_files(make_rows({"libas": 9}), other)
    (d / "dense.faiss").write_bytes((other / "dense.faiss").read_bytes())
    assert "vector_count" in _refused(d, capsys)


def test_refuses_one_clip_file_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = _index(tmp_path, with_clip=True)
    (d / "clip_article_ids.npy").unlink()
    assert "CLIP" in _refused(d, capsys)


@pytest.mark.parametrize("bad", ["../x", "2026-11-15", "20261115t020000z", ""])
def test_refuses_a_malformed_build_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    d = _index(tmp_path)
    code = tool.main(["--index-dir", str(d), "--build-id", bad, "--git-sha", "a"])
    # an empty --build-id means "default to now"; every other value must be refused
    assert code == (0 if bad == "" else 1)
    if bad:
        assert not (d / MANIFEST_NAME).exists()
        assert "build id" in capsys.readouterr().err


def test_refuses_without_a_git_sha(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in tool.GIT_SHA_ENVS:
        monkeypatch.delenv(name, raising=False)
    d = _index(tmp_path)
    assert tool.main(["--index-dir", str(d), "--build-id", BUILD]) == 1
    assert "git sha" in capsys.readouterr().err


def test_git_sha_is_taken_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in tool.GIT_SHA_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "cafe1234")
    d = _index(tmp_path)
    assert tool.main(["--index-dir", str(d), "--build-id", BUILD]) == 0
    assert load_manifest(d).git_sha == "cafe1234"


def test_an_existing_manifest_is_not_replaced_without_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path)
    assert _run(d) == 0
    before = (d / MANIFEST_NAME).read_text(encoding="utf-8")
    assert _run(d) == 1 and "already exists" in capsys.readouterr().err
    assert (d / MANIFEST_NAME).read_text(encoding="utf-8") == before
    assert _run(d, "--overwrite") == 0


def test_a_missing_directory_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert tool.main(["--index-dir", str(tmp_path / "nope"), "--git-sha", "a"]) == 1
    assert "not a directory" in capsys.readouterr().err


def test_an_unusable_previous_manifest_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path)
    assert "previous manifest unusable" in _refused(d, capsys, "--previous", str(tmp_path))


def test_write_manifest_roundtrip_is_stable(tmp_path: Path) -> None:
    d = _index(tmp_path)
    assert _run(d) == 0
    m = load_manifest(d)
    write_manifest(d, m)
    assert load_manifest(d) == m


# --------------------------------------------------------------------------- bad arguments


@pytest.mark.parametrize("bad", ["garbage", "2026-13-45T00:00:00", "yesterday", "2026-11-15T25:00"])
def test_a_bad_built_at_is_refused_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    d = _index(tmp_path)
    code = tool.main(
        ["--index-dir", str(d), "--build-id", BUILD, "--git-sha", "a", "--built-at", bad]
    )
    err = capsys.readouterr().err
    assert code == 1 and err.startswith("REFUSED: --built-at") and "Traceback" not in err
    assert not (d / MANIFEST_NAME).exists()


def test_a_naive_built_at_is_read_as_utc_explicitly(tmp_path: Path) -> None:
    d = _index(tmp_path)
    assert tool.main(["--index-dir", str(d), "--build-id", BUILD, "--git-sha", "a",
                      "--built-at", "2026-11-15T02:00:00"]) == 0  # fmt: skip
    assert load_manifest(d).built_at == datetime(2026, 11, 15, 2, tzinfo=timezone.utc)


def test_an_offset_built_at_is_converted_to_utc(tmp_path: Path) -> None:
    d = _index(tmp_path)
    assert tool.main(["--index-dir", str(d), "--build-id", BUILD, "--git-sha", "a",
                      "--built-at", "2026-11-15T02:00:00+05:30"]) == 0  # fmt: skip
    assert load_manifest(d).built_at == datetime(2026, 11, 14, 20, 30, tzinfo=timezone.utc)


def test_parse_built_at_does_not_depend_on_the_machine_time_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    monkeypatch.setenv("TZ", "Asia/Kolkata")
    if hasattr(time, "tzset"):
        time.tzset()
    assert tool.parse_built_at("2026-11-15T02:00:00") == datetime(
        2026, 11, 15, 2, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("kind", ["missing_dir", "garbage_manifest", "empty_dir"])
def test_a_bad_previous_is_refused_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str
) -> None:
    d = _index(tmp_path)
    prev = tmp_path / "prev"
    if kind != "missing_dir":
        prev.mkdir()
    if kind == "garbage_manifest":
        (prev / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    code = tool.main(["--index-dir", str(d), "--build-id", BUILD, "--git-sha", "a",
                      "--previous", str(prev)])  # fmt: skip
    err = capsys.readouterr().err
    assert code == 1 and "REFUSED: previous manifest unusable" in err and "Traceback" not in err


def test_a_corrupt_input_file_is_refused_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _index(tmp_path)
    (d / "catalogue.parquet").write_bytes(b"not a parquet file")
    assert _run(d) == 1
    err = capsys.readouterr().err
    assert err.startswith("REFUSED: unexpected") and "Traceback" not in err
    assert not (d / MANIFEST_NAME).exists()


def test_the_cli_process_prints_refused_and_no_traceback(tmp_path: Path) -> None:
    import subprocess
    import sys

    d = _index(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(_PATH),
            "--index-dir",
            str(d),
            "--built-at",
            "garbage",
            "--git-sha",
            "a",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1 and "REFUSED:" in r.stderr and "Traceback" not in r.stderr
