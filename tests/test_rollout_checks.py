"""scripts/ops/rollout_checks.py: pre-rollout preflight and keep-last-N revision selection.

Hermetic: local directory store, JSON fixtures; nothing touches gcloud or a bucket.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import refresh_store as rs  # noqa: E402
import rollout_checks as rc  # noqa: E402

PREV = "20261115T020000Z"


def _build(tmp_path: Path, store: rs.LocalBuildStore, build_id: str = PREV) -> Path:
    src = tmp_path / f"src-{build_id}"
    src.mkdir()
    (src / "dense.faiss").write_bytes(b"12345")
    (src / "catalogue.parquet").write_bytes(b"123")
    manifest = {
        "files": [
            {"name": "dense.faiss", "sha256": "0" * 64, "size_bytes": 5},
            {"name": "catalogue.parquet", "sha256": "0" * 64, "size_bytes": 3},
        ]
    }
    (src / rs.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    store.create_build(src, build_id)
    return store.root / "builds" / build_id


@pytest.fixture()
def store(tmp_path: Path) -> rs.LocalBuildStore:
    return rs.LocalBuildStore(tmp_path / "bucket")


# --- preflight -----------------------------------------------------------------------------


def test_preflight_ok_when_the_previous_build_is_complete(
    tmp_path: Path, store: rs.LocalBuildStore
) -> None:
    _build(tmp_path, store)
    assert rc.preflight_problem(PREV, store) is None


def test_preflight_ok_for_a_legacy_previous_revision(store: rs.LocalBuildStore) -> None:
    assert rc.preflight_problem(None, store) is None  # unified/ is never touched by retention


def test_preflight_refuses_when_the_previous_build_was_deleted(store: rs.LocalBuildStore) -> None:
    assert "gone" in (rc.preflight_problem(PREV, store) or "")


def test_preflight_refuses_a_missing_or_resized_file(
    tmp_path: Path, store: rs.LocalBuildStore
) -> None:
    bdir = _build(tmp_path, store)
    (bdir / "dense.faiss").write_bytes(b"1234")  # one byte short
    assert "dense.faiss" in (rc.preflight_problem(PREV, store) or "")
    (bdir / "dense.faiss").unlink()
    assert "dense.faiss" in (rc.preflight_problem(PREV, store) or "")


def test_preflight_refuses_an_unreadable_manifest_and_a_malformed_pin(
    tmp_path: Path, store: rs.LocalBuildStore
) -> None:
    bdir = _build(tmp_path, store)
    (bdir / rs.MANIFEST_NAME).write_text("{nope", encoding="utf-8")
    assert "unreadable manifest" in (rc.preflight_problem(PREV, store) or "")
    assert "malformed" in (rc.preflight_problem("<invalid>", store) or "")


def test_revision_pin_reads_the_env_and_flags_garbage() -> None:
    def rev(env: list[dict[str, Any]]) -> dict[str, Any]:
        return {"spec": {"containers": [{"env": env}]}}

    assert rc.revision_pin(rev([{"name": "INDEX_BUILD_ID", "value": PREV}])) == PREV
    assert rc.revision_pin(rev([{"name": "X", "value": "1"}])) is None
    assert rc.revision_pin(rev([{"name": "INDEX_BUILD_ID", "value": "../x"}])) == "<invalid>"
    with pytest.raises(rc.CheckInputError):
        rc.revision_pin({})


# --- revisions to delete -------------------------------------------------------------------


def revs(n: int) -> list[dict[str, Any]]:
    return [
        {"metadata": {"name": f"r-{i:03d}", "creationTimestamp": f"2026-11-{i:02d}T02:00:00Z"}}
        for i in range(1, n + 1)
    ]


def svc(serving: str, tagged: str | None = None) -> dict[str, Any]:
    traffic: list[dict[str, Any]] = [{"revisionName": serving, "percent": 100}]
    if tagged:
        traffic.append({"revisionName": tagged, "percent": 0, "tag": "ix-test"})
    return {"status": {"traffic": traffic, "latestReadyRevisionName": serving}}


def test_keeps_the_five_newest_and_deletes_the_rest_oldest_first() -> None:
    assert rc.select_revisions_to_delete(revs(9), svc("r-009")) == ["r-001", "r-002", "r-003", "r-004"]


def test_fewer_than_keep_revisions_deletes_nothing() -> None:
    assert rc.select_revisions_to_delete(revs(5), svc("r-005")) == []


def test_serving_tagged_latest_and_rollback_target_are_never_deleted() -> None:
    out = rc.select_revisions_to_delete(
        revs(12), svc("r-002", tagged="r-003"), protect=["r-001"]
    )  # serving, tag and rollback target are all OLD
    assert "r-001" not in out and "r-002" not in out and "r-003" not in out
    assert out == ["r-004", "r-005", "r-006", "r-007"]


def test_latest_revision_traffic_entry_resolves_via_latest_ready() -> None:
    service = {
        "status": {
            "traffic": [{"latestRevision": True, "percent": 100}],
            "latestReadyRevisionName": "r-001",
        }
    }
    assert "r-001" not in rc.select_revisions_to_delete(revs(9), service)


def test_a_zero_percent_untagged_traffic_entry_does_not_protect_a_revision() -> None:
    service = svc("r-009")
    service["status"]["traffic"].append({"revisionName": "r-001", "percent": 0})
    assert "r-001" in rc.select_revisions_to_delete(revs(9), service)


def test_bad_input_fails_closed_instead_of_deleting() -> None:
    with pytest.raises(ValueError):
        rc.select_revisions_to_delete(revs(9), svc("r-009"), keep=0)
    with pytest.raises(rc.CheckInputError):
        rc.select_revisions_to_delete(revs(9), {"status": {}})
    with pytest.raises(rc.CheckInputError):
        rc.select_revisions_to_delete([{"metadata": {"name": "x"}}], svc("r-009"))


def test_cli_prints_names_and_fails_closed_on_empty_or_broken_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    r, s = tmp_path / "r.json", tmp_path / "s.json"
    r.write_text(json.dumps(revs(7)), encoding="utf-8")
    s.write_text(json.dumps(svc("r-007")), encoding="utf-8")
    base = ["revisions-to-delete", "--revisions-json", str(r), "--service-json", str(s)]
    assert rc.main(base) == 0
    assert capsys.readouterr().out.split() == ["r-001", "r-002"]
    assert rc.main([*base, "--protect", "r-001"]) == 0
    assert capsys.readouterr().out.split() == ["r-002"]
    r.write_text("[]", encoding="utf-8")
    assert rc.main(base) == 2  # an empty listing must never read as "nothing to protect"
    r.write_text("{broken", encoding="utf-8")
    assert rc.main(base) == 2
