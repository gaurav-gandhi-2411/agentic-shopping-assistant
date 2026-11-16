"""scripts/ops/check_parity.py: the deploy.yml parity step, now testable (it was inline jq).

Hermetic: revision JSON fixtures built in code.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import check_parity as cp  # noqa: E402

OLD, NEW = "20261115T020000Z", "20261116T020000Z"


def rev(pin: str | None = OLD, **over: Any) -> dict[str, Any]:
    env: list[dict[str, Any]] = [
        {"name": "ZED", "value": "z"},
        {"name": "GROQ_API_KEY", "valueFrom": {"secretKeyRef": {"name": "groq", "key": "latest"}}},
        {"name": "INDEX_STORE_URI", "value": "gs://asa-demo-indices-stylemaitri-prod"},
    ]
    if pin:
        env.append({"name": "INDEX_BUILD_ID", "value": pin})
    doc: dict[str, Any] = {
        "metadata": {"annotations": {"autoscaling.knative.dev/maxScale": "3"}},
        "spec": {
            "containers": [{"image": "img@sha256:" + "a" * 64, "env": env,
                            "resources": {"limits": {"cpu": "2", "memory": "4Gi"}}}],
            "containerConcurrency": 80,
            "timeoutSeconds": 300,
            "serviceAccountName": "runtime@x.iam.gserviceaccount.com",
        },
    }
    doc.update(over)
    return doc


def test_identical_revisions_pass_in_both_modes() -> None:
    assert cp.problems(rev(), rev(), None) == []
    assert cp.problems(rev(OLD), rev(NEW), NEW) == []


def test_env_order_does_not_matter() -> None:
    new = rev()
    new["spec"]["containers"][0]["env"].reverse()
    assert cp.problems(rev(), new, None) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["spec"]["containers"][0]["env"].append({"name": "EXTRA", "value": "1"}),
        lambda d: d["spec"]["containers"][0]["env"][0].update(value="changed"),
        lambda d: d["spec"]["containers"][0]["env"][1]["valueFrom"]["secretKeyRef"].update(key="2"),
        lambda d: d["spec"]["containers"][0]["resources"]["limits"].update(memory="2Gi"),
        lambda d: d["spec"].update(containerConcurrency=10),
        lambda d: d["spec"].update(timeoutSeconds=60),
        lambda d: d["spec"].update(serviceAccountName="other@x"),
        lambda d: d["metadata"]["annotations"].update({"autoscaling.knative.dev/maxScale": "9"}),
    ],
)
def test_any_non_pin_difference_fails_even_in_rollout_mode(mutate: Any) -> None:
    new = rev(NEW)
    mutate(new)
    assert cp.problems(rev(OLD), new, NEW)
    assert cp.problems(rev(OLD), copy.deepcopy(new), None)


def test_code_deploy_must_inherit_the_pin_unchanged() -> None:
    assert cp.problems(rev(OLD), rev(NEW), None)  # pin changed without a rollout
    assert cp.problems(rev(OLD), rev(None), None)  # pin dropped
    assert cp.problems(rev(None), rev(None), None) == []  # legacy stays legacy


def test_rollout_requires_the_exact_expected_pin() -> None:
    assert any("expected" in p for p in cp.problems(rev(OLD), rev(OLD), NEW))  # not applied
    assert cp.problems(rev(OLD), rev("20261117T020000Z"), NEW)  # applied a different id
    assert cp.problems(rev(OLD), rev(None), NEW)


def test_rollout_from_a_legacy_revision_is_allowed() -> None:
    assert cp.problems(rev(None), rev(NEW), NEW) == []


def test_rollout_refuses_a_downgrade_and_a_malformed_id() -> None:
    assert any("DOWNGRADE" in p for p in cp.problems(rev(NEW), rev(OLD), OLD))
    assert any("not a build id" in p for p in cp.problems(rev(OLD), rev("../x"), "../x"))


def test_view_equals_the_old_jq_shape_minus_the_pin() -> None:
    v = cp.view(rev())
    assert set(v) == {"env", "limits", "concurrency", "timeout", "serviceAccount", "maxScale"}
    assert [e["name"] for e in v["env"]] == ["GROQ_API_KEY", "INDEX_STORE_URI", "ZED"]


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(rev(OLD)), encoding="utf-8")
    b.write_text(json.dumps(rev(NEW)), encoding="utf-8")
    assert cp.main([str(a), str(b), "--expect-index-build-id", NEW]) == 0
    assert cp.main([str(a), str(b)]) == 1  # pin changed on a "code deploy"
    assert cp.main([str(a), str(tmp_path / "missing.json")]) == 2
    b.write_text("{}", encoding="utf-8")
    assert cp.main([str(a), str(b)]) == 2  # no spec: fail closed, never "equal"
    assert "cannot compare" in capsys.readouterr().err
