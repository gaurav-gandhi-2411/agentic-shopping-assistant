"""build_manifest.json: schema, loader and tamper detection (Wave 1b V3(b)).

Every build here is written by tests/_index_fixtures.py under ``tmp_path`` (hermetic, no network).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config.store_registry import STORE_REGISTRY
from src.retrieval.build_manifest import (
    COUNT_GATE_TOLERANCE,
    MANIFEST_NAME,
    BuildManifest,
    BuildValidationError,
    check_count_gate,
    load_manifest,
    now_build_id,
    schema_text,
    try_load_manifest,
    verify_build_dir,
    verify_loaded_index,
    write_manifest,
)
from tests._index_fixtures import write_build

REGISTRY = frozenset(STORE_REGISTRY)
STORES = {"libas": 6, "fashor": 4}
BID = "20261115T020000Z"


@pytest.fixture()
def build(tmp_path: Path) -> tuple[Path, BuildManifest]:
    m = write_build(tmp_path, BID, STORES)
    return tmp_path / "builds" / BID, m


def _reason(exc: pytest.ExceptionInfo[BuildValidationError]) -> str:
    return exc.value.reason


def test_valid_build_verifies_and_round_trips(build: tuple[Path, BuildManifest]) -> None:
    bdir, written = build
    got = verify_build_dir(bdir, registry_slugs=REGISTRY)
    assert got == written
    assert got.total_items == got.vector_count == 10
    assert {f.name for f in got.files} >= {"dense.faiss", "bm25.pkl", "catalogue.parquet"}
    assert got.built_at.tzinfo is not None
    assert got.previous_build_id is None


def test_bad_hash_with_unchanged_size_is_rejected(build: tuple[Path, BuildManifest]) -> None:
    bdir, _ = build
    path = bdir / "bm25.pkl"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # flip one byte: size identical, sha256 differs
    path.write_bytes(bytes(data))
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "file_hash"


def test_truncated_file_is_rejected_by_size(build: tuple[Path, BuildManifest]) -> None:
    bdir, _ = build
    path = bdir / "dense.faiss"
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "file_size"


def test_missing_file_is_rejected(build: tuple[Path, BuildManifest]) -> None:
    bdir, _ = build
    (bdir / "catalogue.parquet").unlink()
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "file_missing"


def test_required_file_absent_from_the_manifest_is_rejected(
    build: tuple[Path, BuildManifest],
) -> None:
    bdir, m = build
    m.files = [f for f in m.files if f.name != "bm25.pkl"]
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "unlisted_file"


def test_count_mismatch_between_manifest_and_catalogue_is_rejected(
    build: tuple[Path, BuildManifest],
) -> None:
    bdir, m = build
    m.total_items = m.vector_count = 11  # rows on disk: 10
    m.stores[0].items += 1
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "count_mismatch"


def test_vector_count_differing_from_items_is_rejected(build: tuple[Path, BuildManifest]) -> None:
    bdir, m = build
    m.vector_count = m.total_items - 1
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "count_mismatch"


def test_per_store_counts_must_match_the_catalogue(build: tuple[Path, BuildManifest]) -> None:
    bdir, m = build
    m.stores[0].items += 1
    m.stores[1].items -= 1  # sums still agree; the split does not
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "count_mismatch"


def test_store_outside_the_registry_is_rejected(tmp_path: Path) -> None:
    write_build(tmp_path, BID, {"libas": 3, "not-a-store": 2})
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(tmp_path / "builds" / BID, registry_slugs=REGISTRY)
    assert _reason(exc) == "store_unknown"


def test_wrong_dimension_and_wrong_vector_count_are_caught_after_load(
    build: tuple[Path, BuildManifest],
) -> None:
    _, m = build
    verify_loaded_index(m, dense_ntotal=10, dense_dim=m.dense_dim)  # the good case
    with pytest.raises(BuildValidationError) as exc:
        verify_loaded_index(m, dense_ntotal=10, dense_dim=m.dense_dim + 1)
    assert _reason(exc) == "dim_mismatch"
    with pytest.raises(BuildValidationError) as exc:
        verify_loaded_index(m, dense_ntotal=9, dense_dim=m.dense_dim)
    assert _reason(exc) == "vector_count"


def test_clip_files_and_dims_are_checked_together(tmp_path: Path) -> None:
    m = write_build(tmp_path, BID, STORES, with_clip=True)
    bdir = tmp_path / "builds" / BID
    assert verify_build_dir(bdir).clip_dim == 8
    verify_loaded_index(m, dense_ntotal=10, dense_dim=m.dense_dim, clip_ntotal=10, clip_dim=8)
    with pytest.raises(BuildValidationError) as exc:
        verify_loaded_index(m, dense_ntotal=10, dense_dim=m.dense_dim, clip_ntotal=10, clip_dim=7)
    assert _reason(exc) == "dim_mismatch"
    m.clip_dim = None  # clip files listed but no clip_dim
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "clip_mismatch"


def test_manifest_missing_invalid_and_unsupported_version(
    build: tuple[Path, BuildManifest],
) -> None:
    bdir, m = build
    assert try_load_manifest(bdir) is not None
    m.schema_version = 99
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "manifest_schema"
    (bdir / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(BuildValidationError) as exc:
        load_manifest(bdir)
    assert _reason(exc) == "manifest_invalid"
    assert try_load_manifest(bdir) is None
    (bdir / MANIFEST_NAME).unlink()
    with pytest.raises(BuildValidationError) as exc:
        load_manifest(bdir)
    assert _reason(exc) == "manifest_missing"
    assert try_load_manifest(bdir) is None  # legacy layout: no manifest, not an error


def test_manifest_listing_a_path_outside_the_build_dir_is_rejected(
    build: tuple[Path, BuildManifest],
) -> None:
    bdir, m = build
    evil = m.files[0].model_copy(update={"name": "../escape.bin"})
    m.files = [*m.files, evil]
    write_manifest(bdir, m)
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "file_missing"


@pytest.mark.parametrize(
    ("items", "ok"),
    [(70, True), (130, True), (69, False), (131, False)],  # previous = 100, gate = +-30%
)
def test_count_gate_boundaries(items: int, ok: bool) -> None:
    prev = BuildManifest.model_construct(total_items=100, build_id="p")
    cur = BuildManifest.model_construct(total_items=items, build_id="c")
    assert COUNT_GATE_TOLERANCE == 0.30
    if ok:
        delta = check_count_gate(cur, prev)
        assert delta == pytest.approx((items - 100) / 100 * 100)
    else:
        with pytest.raises(BuildValidationError) as exc:
            check_count_gate(cur, prev)
        assert _reason(exc) == "count_gate"


def test_count_gate_is_skipped_without_a_previous_build() -> None:
    cur = BuildManifest.model_construct(total_items=5, build_id="c")
    assert check_count_gate(cur, None) is None


def test_verify_applies_the_count_gate_against_the_previous_manifest(tmp_path: Path) -> None:
    prev = write_build(tmp_path, "20261114T020000Z", {"libas": 10})
    write_build(tmp_path, BID, {"libas": 4}, previous=prev)  # 10 -> 4 items: -60%
    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(tmp_path / "builds" / BID, previous=prev)
    assert _reason(exc) == "count_gate"
    ok = write_build(tmp_path, "20261116T020000Z", {"libas": 9}, previous=prev)
    assert ok.previous_build_id == prev.build_id
    assert ok.validation.count_delta_pct == pytest.approx(-10.0)
    verify_build_dir(tmp_path / "builds" / "20261116T020000Z", previous=prev)


def test_now_build_id_is_a_sortable_utc_timestamp() -> None:
    a = now_build_id(datetime(2026, 11, 15, 2, 0, 0, tzinfo=timezone.utc))
    b = now_build_id(datetime(2026, 11, 15, 14, 0, 0, tzinfo=timezone.utc))
    assert a == "20261115T020000Z" and a < b


def test_checked_in_json_schema_matches_the_model() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "build_manifest.schema.json"
    assert path.read_text(encoding="utf-8").replace("\r\n", "\n") == schema_text(), (
        "regenerate docs/build_manifest.schema.json from src.retrieval.build_manifest.schema_text()"
    )


_HOSTILE_NAMES = [
    "../sentinel.bin",
    "..\\sentinel.bin",
    "SENTINEL_ABS",  # replaced by the absolute path of the sentinel file
    "C:\\sentinel.bin",
    "C:sentinel.bin",
    "sub/dir.bin",
    "",
    ".",
    "..",
    "bad\x00name.bin",
    "bad\nname.bin",
    "x" * 300,
]


@pytest.mark.parametrize("hostile", range(len(_HOSTILE_NAMES)))
def test_hostile_manifest_file_names_are_rejected_before_anything_is_read_outside(
    build: tuple[Path, BuildManifest], monkeypatch: pytest.MonkeyPatch, hostile: int
) -> None:
    """A manifest entry naming a file outside the build dir must be refused BEFORE any file is
    opened or hashed. A sentinel beside the build dir has a matching hash and size, so a
    verifier without the guard would accept the entry (and the test fail)."""
    import hashlib

    from src.retrieval import build_manifest as bm

    bdir, m = build
    sentinel = bdir.parent / "sentinel.bin"
    payload = b"outside the build dir"
    sentinel.write_bytes(payload)
    (bdir / "sub").mkdir()
    (bdir / "sub" / "dir.bin").write_bytes(payload)
    name = str(sentinel) if _HOSTILE_NAMES[hostile] == "SENTINEL_ABS" else _HOSTILE_NAMES[hostile]
    evil = m.files[0].model_copy(
        update={
            "name": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    )
    m.files = [*m.files, evil]
    write_manifest(bdir, m)

    hashed: list[Path] = []
    real = bm.sha256_file
    monkeypatch.setattr(
        bm, "sha256_file", lambda path, *a, **k: (hashed.append(path), real(path))[1]
    )

    with pytest.raises(BuildValidationError) as exc:
        verify_build_dir(bdir)
    assert _reason(exc) == "file_missing"
    assert hashed == [], f"files were hashed before the name was rejected: {hashed}"
