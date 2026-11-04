"""Unit tests for scripts/ops/pin_images.py (hermetic: fake gcloud, files in tmp_path only)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# PIN_IMAGES_PATH lets a mutation check point these tests at a mutated copy of the script.
_PATH = Path(
    os.environ.get("PIN_IMAGES_PATH")
    or Path(__file__).resolve().parents[1] / "scripts" / "ops" / "pin_images.py"
)
_spec = importlib.util.spec_from_file_location("pin_images", _PATH)
assert _spec is not None and _spec.loader is not None
pin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pin)

BASE = "asia-south1-docker.pkg.dev/stylemaitri-prod-260813/cloud-run-source-deploy/asa-stylist-api"
D_NEW = "sha256:" + "fb" * 32
D_PREV = "sha256:" + "ab" * 32


def _rev(tmp_path: Path, name: str, image: str, status: str | None) -> Path:
    doc: dict = {"spec": {"containers": [{"image": image}]}, "status": {}}
    if status is not None:
        doc["status"]["imageDigest"] = status
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class _Runner:
    """Fake gcloud: records argv, returns scripted return codes."""

    def __init__(self, rcs: list[int] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._rcs = rcs or []

    def __call__(self, argv: list[str]) -> int:
        self.calls.append(argv)
        return self._rcs[len(self.calls) - 1] if len(self._rcs) >= len(self.calls) else 0


def _cmd(digest: str, tag: str) -> list[str]:
    return ["artifacts", "docker", "tags", "add", f"{BASE}@{digest}", f"{BASE}:{tag}"]


def test_real_shape_full_reference_in_both_fields(tmp_path) -> None:
    """Regression: Cloud Run's status.imageDigest is the FULL ref, not a bare sha256:..."""
    new = _rev(tmp_path, "new.json", f"{BASE}@{D_NEW}", f"{BASE}@{D_NEW}")
    prev = _rev(tmp_path, "prev.json", f"{BASE}@{D_PREV}", f"{BASE}@{D_PREV}")
    runner = _Runner()
    assert pin.pin_all(new, prev, runner, None) == 0
    assert runner.calls == [_cmd(D_NEW, "live"), _cmd(D_PREV, "prev")]
    for call in runner.calls:
        assert call[4].count("@") == 1 and call[5].count(":") == 1


@pytest.mark.parametrize(
    ("image", "status"),
    [
        (f"{BASE}@{D_NEW}", D_NEW),  # bare digest in status
        (f"{BASE}@{D_NEW}", f"{BASE}@{D_NEW}"),  # full ref in status
        (f"{BASE}:abc123@{D_NEW}", None),  # tag + digest in image, no status
        (f"{BASE}:abc123@{D_NEW}", f"{BASE}:abc123@{D_NEW}"),  # tag + digest everywhere
        (f"{BASE}:abc123", D_NEW),  # tag-only image, bare digest from status
        (f"{BASE}:abc123", f"{BASE}@{D_NEW}"),  # tag-only image, full ref from status
    ],
)
def test_shapes_resolve_to_tagless_base_and_bare_digest(tmp_path, image, status) -> None:
    rev = _rev(tmp_path, "r.json", image, status)
    assert pin.resolve(rev) == (BASE, D_NEW)


@pytest.mark.parametrize(
    ("image", "status"),
    [
        (f"{BASE}:abc123", None),  # tag only, no digest anywhere: never tag by mutable tag
        (f"{BASE}:abc123", ""),
        (f"{BASE}@sha256:deadbeef", None),  # short/garbage digest
        (f"{BASE}@{D_NEW}", "garbage"),  # bad status falls back to the valid image digest
    ],
)
def test_no_full_digest_hard_fails(tmp_path, image, status) -> None:
    rev = _rev(tmp_path, "r.json", image, status)
    if status == "garbage":
        assert pin.resolve(rev) == (BASE, D_NEW)
        return
    with pytest.raises(pin.PinError):
        pin.resolve(rev)


def test_unreadable_json_fails(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("", encoding="utf-8")
    with pytest.raises(pin.PinError):
        pin.resolve(bad)


def test_no_digest_means_no_gcloud_call_and_nonzero(tmp_path) -> None:
    new = _rev(tmp_path, "new.json", f"{BASE}@{D_NEW}", None)
    prev = _rev(tmp_path, "prev.json", f"{BASE}:abc123", None)
    runner = _Runner()
    assert pin.pin_all(new, prev, runner, None) != 0
    assert runner.calls == []  # resolve everything before mutating anything


def test_gcloud_failure_on_live_skips_prev(tmp_path) -> None:
    new = _rev(tmp_path, "new.json", f"{BASE}@{D_NEW}", f"{BASE}@{D_NEW}")
    prev = _rev(tmp_path, "prev.json", f"{BASE}@{D_PREV}", f"{BASE}@{D_PREV}")
    runner = _Runner([3])
    assert pin.pin_all(new, prev, runner, None) == 3
    assert runner.calls == [_cmd(D_NEW, "live")]


def test_prev_equals_new_adds_both_tags(tmp_path) -> None:
    new = _rev(tmp_path, "new.json", f"{BASE}@{D_NEW}", f"{BASE}@{D_NEW}")
    runner = _Runner()
    assert pin.pin_all(new, new, runner, None) == 0
    assert runner.calls == [_cmd(D_NEW, "live"), _cmd(D_NEW, "prev")]


def test_project_flag_is_appended(tmp_path) -> None:
    new = _rev(tmp_path, "new.json", f"{BASE}@{D_NEW}", None)
    runner = _Runner()
    assert pin.pin_all(new, new, runner, "my-proj") == 0
    assert all(c[-2:] == ["--project", "my-proj"] for c in runner.calls)


def test_cli_uses_gcloud_env_hook(tmp_path) -> None:
    fake = tmp_path / "fake_gcloud.py"
    log = tmp_path / "argv.log"
    fake.write_text(
        "import json, sys\n"
        f"open({log.as_posix()!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    new = _rev(tmp_path, "new.json", f"{BASE}@{D_NEW}", f"{BASE}@{D_NEW}")
    prev = _rev(tmp_path, "prev.json", f"{BASE}@{D_PREV}", f"{BASE}@{D_PREV}")
    env = {**os.environ, "GCLOUD": f'"{Path(sys.executable).as_posix()}" "{fake.as_posix()}"'}
    env["GCP_PROJECT"] = "p"
    res = subprocess.run(  # noqa: S603
        [sys.executable, str(_PATH), str(new), str(prev)], env=env, check=False
    )
    assert res.returncode == 0
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        [*_cmd(D_NEW, "live"), "--project", "p"],
        [*_cmd(D_PREV, "prev"), "--project", "p"],
    ]
