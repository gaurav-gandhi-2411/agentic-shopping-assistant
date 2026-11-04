"""Unit tests for scripts/ops/assert_min_scale.py (hermetic: files in tmp_path only)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

# ASSERT_MIN_SCALE_PATH lets a mutation check point these tests at a mutated copy of the script.
_PATH = Path(
    os.environ.get("ASSERT_MIN_SCALE_PATH")
    or Path(__file__).resolve().parents[1] / "scripts" / "ops" / "assert_min_scale.py"
)
_spec = importlib.util.spec_from_file_location("assert_min_scale", _PATH)
assert _spec is not None and _spec.loader is not None
ams = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ams)

REV_KEY = "autoscaling.knative.dev/minScale"
SVC_KEY = "run.googleapis.com/minScale"


def _service(traffic: list[dict], template_min: str | None = None, svc_min: str | None = None):
    tpl_ann = {REV_KEY: template_min} if template_min is not None else {}
    svc_ann = {SVC_KEY: svc_min} if svc_min is not None else {}
    return {
        "metadata": {"annotations": svc_ann},
        "spec": {"template": {"metadata": {"annotations": tpl_ann}}},
        "status": {"traffic": traffic},
    }


def _write(tmp_path: Path, service: dict, revs: dict[str, str | None]) -> list[str]:
    (tmp_path / "service.json").write_text(json.dumps(service), encoding="utf-8")
    d = tmp_path / "revs"
    d.mkdir()
    for name, min_scale in revs.items():
        ann = {REV_KEY: min_scale} if min_scale is not None else {}
        (d / f"{name}.json").write_text(
            json.dumps({"metadata": {"annotations": ann}}), encoding="utf-8"
        )
    return ["--service", str(tmp_path / "service.json"), "--revisions-dir", str(d)]


NEW_ONLY = [{"revisionName": "svc-new", "percent": 100}]
SPLIT = [
    {"revisionName": "svc-new", "percent": 90},
    {"revisionName": "svc-old", "percent": 10},
]
TAGGED = [
    {"revisionName": "svc-new", "percent": 100},
    {"revisionName": "svc-tagged", "percent": 0, "tag": "c-abc1234"},
]


def test_all_zero_passes(tmp_path, capsys) -> None:
    args = _write(
        tmp_path, _service(SPLIT + TAGGED[1:]), {"svc-new": "0", "svc-old": "0", "svc-tagged": "0"}
    )
    assert ams.main(args) == 0
    assert "svc-old minScale=0" in capsys.readouterr().out


def test_missing_annotation_means_zero(tmp_path) -> None:
    assert ams.main(_write(tmp_path, _service(NEW_ONLY), {"svc-new": None})) == 0


def test_new_revision_min1_fails(tmp_path, capsys) -> None:
    assert ams.main(_write(tmp_path, _service(NEW_ONLY), {"svc-new": "1"})) == 1
    assert "svc-new" in capsys.readouterr().err


def test_old_serving_revision_in_split_min1_fails(tmp_path, capsys) -> None:
    args = _write(tmp_path, _service(SPLIT), {"svc-new": "0", "svc-old": "1"})
    assert ams.main(args) == 1
    assert "svc-old" in capsys.readouterr().err


def test_tagged_zero_percent_revision_min1_fails(tmp_path, capsys) -> None:
    args = _write(tmp_path, _service(TAGGED), {"svc-new": "0", "svc-tagged": "1"})
    assert ams.main(args) == 1
    assert "svc-tagged" in capsys.readouterr().err


def test_untagged_zero_percent_revision_is_not_checked(tmp_path) -> None:
    traffic = NEW_ONLY + [{"revisionName": "svc-dead", "percent": 0}]
    assert ams.main(_write(tmp_path, _service(traffic), {"svc-new": "0", "svc-dead": "5"})) == 0


def test_template_level_min1_fails(tmp_path, capsys) -> None:
    args = _write(tmp_path, _service(NEW_ONLY, template_min="1"), {"svc-new": "0"})
    assert ams.main(args) == 1
    assert "template" in capsys.readouterr().err


def test_service_level_min1_fails(tmp_path) -> None:
    assert ams.main(_write(tmp_path, _service(NEW_ONLY, svc_min="1"), {"svc-new": "0"})) == 1


def test_unparsable_value_fails_closed(tmp_path) -> None:
    assert ams.main(_write(tmp_path, _service(NEW_ONLY), {"svc-new": "lots"})) == 1


def test_missing_revision_file_fails_closed(tmp_path) -> None:
    assert ams.main(_write(tmp_path, _service(SPLIT), {"svc-new": "0"})) == 1


@pytest.mark.parametrize("body", ["", "{}", "not json", "[]", '{"status": {"traffic": []}}'])
def test_malformed_or_empty_service_fails_closed(tmp_path, body) -> None:
    (tmp_path / "revs").mkdir()
    (tmp_path / "service.json").write_text(body, encoding="utf-8")
    argv = ["--service", str(tmp_path / "service.json"), "--revisions-dir", str(tmp_path / "revs")]
    assert ams.main(argv) == 1


def test_empty_revision_json_fails_closed(tmp_path) -> None:
    args = _write(tmp_path, _service(NEW_ONLY), {"svc-new": "0"})
    (tmp_path / "revs" / "svc-new.json").write_text("", encoding="utf-8")
    assert ams.main(args) == 1


def test_list_revisions_includes_tagged_and_split(tmp_path, capsys) -> None:
    args = _write(tmp_path, _service(SPLIT + TAGGED[1:]), {})
    assert ams.main([args[0], args[1], "--list-revisions"]) == 0
    assert capsys.readouterr().out.split() == ["svc-new", "svc-old", "svc-tagged"]
