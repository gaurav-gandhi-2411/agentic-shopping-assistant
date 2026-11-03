"""Tests for scripts/check_advisory_allowlist.py (time-boxed advisory allow-list).

Hermetic: every file lives under tmp_path, `npm audit` is a monkeypatched fake (no real npm).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_advisory_allowlist.py"
REAL_FILE = Path(__file__).resolve().parents[1] / "security" / "advisory-allowlist.yaml"
TODAY = "2026-11-10"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_advisory_allowlist", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def entry(**over: Any) -> dict[str, Any]:
    """A valid pip entry; override fields per test (a None value drops the field)."""
    base: dict[str, Any] = {
        "id": "PYSEC-2026-1234",
        "ecosystem": "pip",
        "package": "examplepkg",
        "justification": "Vulnerable function is never imported or called; verified by grep.",
        "added": "2026-11-02",
        "expires": "2026-12-02",
        "approver": "GG",
    }
    base.update(over)
    return {k: v for k, v in base.items() if v is not None}


def write(tmp_path: Path, entries: list[Any]) -> Path:
    p = tmp_path / "allow.yaml"
    p.write_text(yaml.safe_dump({"entries": entries}), encoding="utf-8")
    return p


def run(args: list[str], path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = 0
    try:
        mod.main([*args, "--file", str(path), "--today", TODAY])
    except SystemExit as exc:
        code = 1 if isinstance(exc.code, str) else int(exc.code or 0)
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_shipped_file_is_valid_and_empty(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["check"], REAL_FILE, capsys)
    assert code == 0 and "0 entries" in out
    assert yaml.safe_load(REAL_FILE.read_text(encoding="utf-8"))["entries"] == []


def test_empty_file_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["check"], write(tmp_path, []), capsys)
    assert code == 0 and "allowlist ok: 0 entries" in out


def test_valid_entry_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["check"], write(tmp_path, [entry()]), capsys)
    assert code == 0 and "1 entries" in out


@pytest.mark.parametrize("content", ["entries: [", "just a string", "entries: notalist", "- a\n- b"])
def test_malformed_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(content, encoding="utf-8")
    code, _, err = run(["check"], p, capsys)
    assert code == 1 and ("unreadable" in err or "malformed" in err)


def test_unreadable_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["check"], tmp_path / "missing.yaml", capsys)
    assert code == 1 and "unreadable" in err
    binary = tmp_path / "bin.yaml"
    binary.write_bytes(b"\xff\xfe\x00bad")
    code, _, err = run(["check"], binary, capsys)
    assert code == 1 and "unreadable" in err


def test_entry_not_a_mapping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["check"], write(tmp_path, ["PYSEC-2026-1"]), capsys)
    assert code == 1 and "not a mapping" in err


@pytest.mark.parametrize("field", ["id", "ecosystem", "package", "justification", "added",
                                   "expires", "approver"])
def test_missing_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(**{field: None})]), capsys)
    assert code == 1 and "missing" in err and field in err


def test_bad_ecosystem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(ecosystem="cargo")]), capsys)
    assert code == 1 and "ecosystem must be pip|npm" in err


@pytest.mark.parametrize("approver", ["gg", "someone", "Claude"])
def test_non_gg_approver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], approver: str
) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(approver=approver)]), capsys)
    assert code == 1 and "approver must be GG" in err


def test_short_justification(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(justification="not reachable")]), capsys)
    assert code == 1 and "justification too short" in err


def test_expired(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    e = entry(added="2026-09-01", expires="2026-10-01")
    code, _, err = run(["check"], write(tmp_path, [e]), capsys)
    assert code == 1 and "EXPIRED" in err


def test_expiry_day_itself_is_still_valid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    e = entry(added="2026-10-11", expires=TODAY)
    code, _, _ = run(["check"], write(tmp_path, [e]), capsys)
    assert code == 0


def test_over_30_day_window(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    e = entry(added="2026-11-02", expires="2026-12-03")  # 31 days
    code, _, err = run(["check"], write(tmp_path, [e]), capsys)
    assert code == 1 and "within 30 days" in err
    ok = entry(added="2026-11-02", expires="2026-12-02")  # exactly 30
    assert run(["check"], write(tmp_path, [ok]), capsys)[0] == 0
    backwards = entry(added="2026-11-02", expires="2026-11-02")
    assert run(["check"], write(tmp_path, [backwards]), capsys)[0] == 1


def test_future_added_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    e = entry(added="2026-11-11", expires="2026-12-01")  # today is 2026-11-10
    code, _, err = run(["check"], write(tmp_path, [e]), capsys)
    assert code == 1 and "in the future" in err
    same_day = entry(added=TODAY, expires="2026-12-01")
    assert run(["check"], write(tmp_path, [same_day]), capsys)[0] == 0


def test_bad_dates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(added="yesterday")]), capsys)
    assert code == 1 and "YYYY-MM-DD" in err


@pytest.mark.parametrize(("eco", "adv_id"), [
    ("pip", "PYSEC-2026-1"),
    ("pip", "PYSEC-2026-123456"),
    ("pip", "GHSA-jfh8-c2jp-5v3q"),
    ("pip", "CVE-2026-1234"),
    ("pip", "CVE-2026-12345678"),
    ("npm", "GHSA-jfh8-c2jp-5v3q"),
    ("npm", "1098765"),
    ("npm", "42"),
])
def test_good_ids_per_ecosystem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], eco: str, adv_id: str
) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(ecosystem=eco, id=adv_id)]), capsys)
    assert code == 0, err


@pytest.mark.parametrize(("eco", "adv_id"), [
    ("pip", "PYSEC-26-1"),
    ("pip", "PYSEC-2026-"),
    ("pip", "pysec-2026-1234"),
    ("pip", "CVE-2026-123"),           # CVE sequence needs >= 4 digits
    ("pip", "GHSA-jfh8-c2jp-5v3"),     # short group
    ("pip", "GHSA-jfh8-c2jp-5v3a"),    # 'a' is not in the GHSA alphabet
    ("pip", "GHSA-JFH8-C2JP-5V3Q"),    # uppercase
    ("pip", "GHSA-jfh8-c2jp-5v3q; rm -rf /"),
    ("pip", "1098765"),                # numeric ids are npm-only
    ("pip", "--ignore-vuln"),
    ("pip", "*"),
    ("npm", "PYSEC-2026-1234"),        # pip-style id under npm
    ("npm", "CVE-2026-1234"),
    ("npm", "0123"),
    ("npm", "12345678"),               # 8 digits
    ("npm", "GHSA-jfh8-c2jp-5v30"),    # '0' is not in the alphabet
    ("npm", "GHSA-jfh8-c2jp"),
])
def test_bad_ids_per_ecosystem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], eco: str, adv_id: str
) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(ecosystem=eco, id=adv_id)]), capsys)
    assert code == 1 and "not a valid" in err


def test_duplicates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(["check"], write(tmp_path, [entry(), entry()]), capsys)
    assert code == 1 and "duplicate" in err
    # same id under a different ecosystem is a different key, not a duplicate
    other = entry(ecosystem="npm", id="GHSA-jfh8-c2jp-5v3q")
    two = entry(id="GHSA-jfh8-c2jp-5v3q")
    assert run(["check"], write(tmp_path, [two, other]), capsys)[0] == 0


def test_all_errors_reported_at_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = entry(approver="x", justification="short", id="nope")
    code, _, err = run(["check"], write(tmp_path, [bad]), capsys)
    assert code == 1 and "approver" in err and "justification" in err and "not a valid" in err


def test_pip_args_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["pip-args"], write(tmp_path, []), capsys)
    assert code == 0 and out.strip() == ""
    entries = [
        entry(id="PYSEC-2026-1234"),
        entry(id="CVE-2026-9999"),
        entry(ecosystem="npm", id="GHSA-jfh8-c2jp-5v3q"),
    ]
    code, out, _ = run(["pip-args"], write(tmp_path, entries), capsys)
    assert code == 0
    assert out.strip() == "--ignore-vuln PYSEC-2026-1234 --ignore-vuln CVE-2026-9999"


def test_pip_args_refuse_invalid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(["pip-args"], write(tmp_path, [entry(approver="x")]), capsys)
    assert code == 1 and out.strip() == ""  # never prints ignore flags from an invalid file


# ----------------------------------------------------------------- npm-audit with a fake npm


def _vuln(severity: str, *urls: str | None) -> dict[str, Any]:
    via: list[Any] = [{"url": u} if u else "transitive-pkg" for u in urls]
    return {"severity": severity, "via": via}


def fake_npm(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any] | str, seen: list[Any] | None = None
) -> None:
    stdout = payload if isinstance(payload, str) else json.dumps(payload)

    def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        if seen is not None:
            seen.append((cmd, kw.get("cwd")))
        return subprocess.CompletedProcess(cmd, 1, stdout=stdout, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _run)


GHSA = "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q"


def test_npm_audit_clean_and_low_severity_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    fake_npm(monkeypatch, {"vulnerabilities": {"a": _vuln("moderate", GHSA), "b": _vuln("low", GHSA)}}, seen)
    code, out, _ = run(["npm-audit"], write(tmp_path, []), capsys)
    assert code == 0 and "npm audit ok" in out
    assert seen == [(["npm", "audit", "--json"], "frontend")]


@pytest.mark.parametrize("severity", ["high", "critical"])
def test_npm_audit_high_critical_fail_when_not_allowlisted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, severity: str
) -> None:
    fake_npm(monkeypatch, {"vulnerabilities": {"lodash": _vuln(severity, GHSA)}})
    code, _, err = run(["npm-audit"], write(tmp_path, []), capsys)
    assert code == 1 and "lodash" in err and "GHSA-jfh8-c2jp-5v3q" in err


def test_npm_audit_allowlisted_advisory_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_npm(monkeypatch, {"vulnerabilities": {"lodash": _vuln("high", GHSA)}})
    allow = entry(ecosystem="npm", id="GHSA-jfh8-c2jp-5v3q", package="lodash")
    code, out, _ = run(["npm-audit"], write(tmp_path, [allow]), capsys)
    assert code == 0 and "npm audit ok" in out
    # a pip entry with the same id must NOT cover an npm finding
    pip_only = entry(id="GHSA-jfh8-c2jp-5v3q", package="lodash")
    assert run(["npm-audit"], write(tmp_path, [pip_only]), capsys)[0] == 1


def test_npm_audit_partial_allowlist_still_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = "https://github.com/advisories/GHSA-2222-3333-4444"
    fake_npm(monkeypatch, {"vulnerabilities": {"x": _vuln("critical", GHSA, other)}})
    allow = entry(ecosystem="npm", id="GHSA-jfh8-c2jp-5v3q", package="x")
    code, _, err = run(["npm-audit"], write(tmp_path, [allow]), capsys)
    assert code == 1 and "GHSA-2222-3333-4444" in err


def test_npm_audit_high_with_no_advisory_ids_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_npm(monkeypatch, {"vulnerabilities": {"x": _vuln("high", None)}})
    assert run(["npm-audit"], write(tmp_path, []), capsys)[0] == 1


@pytest.mark.parametrize("payload", ["not json", "{}", '{"error": {"code": "ENOTFOUND"}}',
                                     '{"vulnerabilities": []}', '{"vulnerabilities": {"a": 1}}'])
def test_npm_audit_unparseable_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    fake_npm(monkeypatch, payload)
    code, _, err = run(["npm-audit"], write(tmp_path, []), capsys)
    assert code == 1 and "unparseable" in err


def test_npm_audit_missing_npm_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: Any, **k: Any) -> None:
        raise FileNotFoundError("npm")

    monkeypatch.setattr(mod.subprocess, "run", _boom)
    code, _, err = run(["npm-audit"], write(tmp_path, []), capsys)
    assert code == 1 and "could not run" in err


def test_npm_audit_invalid_allowlist_never_runs_npm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    fake_npm(monkeypatch, {"vulnerabilities": {}}, seen)
    code, _, _ = run(["npm-audit"], write(tmp_path, [entry(approver="x")]), capsys)
    assert code == 1 and seen == []
