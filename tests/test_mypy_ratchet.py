"""Tests for scripts/mypy_ratchet.py using a fake mypy (a tiny script printing canned output)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.mypy_ratchet import normalize_line, run_ratchet

ERR_A = 'src\\a.py:10: error: Incompatible types in assignment (expression has type "str")  [assignment]'
ERR_A_MOVED = 'src\\a.py:99: error: Incompatible types in assignment (expression has type "str")  [assignment]'
ERR_B = 'src/b.py:5:3: error: Item "None" of "X | None" has no attribute "y"  [union-attr]'
KEY_A = 'src/a.py: [assignment] Incompatible types in assignment (expression has type "str")'
KEY_B = 'src/b.py: [union-attr] Item "None" of "X | None" has no attribute "y"'


def _fake(tmp_path: Path, stdout: str, code: int, stderr: str = "") -> list[str]:
    """Build a command that prints `stdout`/`stderr` and exits with `code`."""
    out = tmp_path / "out.txt"
    out.write_text(stdout, encoding="utf-8")
    script = tmp_path / "fake_mypy.py"
    script.write_text(
        "import sys\n"
        f"sys.stdout.write(open({str(out)!r}, encoding='utf-8').read())\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({code})\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _found(*errs: str, checked: int = 7) -> str:
    return "\n".join([*errs, f"Found {len(errs)} errors in 1 file (checked {checked} source files)", ""])


def _baseline(tmp_path: Path, *keys: str) -> Path:
    p = tmp_path / "baseline.txt"
    p.write_text("".join(f"{k}\n" for k in sorted(keys)), encoding="utf-8")
    return p


def test_normalize_strips_line_numbers_and_separators() -> None:
    assert normalize_line(ERR_A) == KEY_A
    assert normalize_line(ERR_A) == normalize_line(ERR_A_MOVED)
    assert normalize_line(ERR_B) == KEY_B
    assert normalize_line("src/a.py:3: note: something") is None
    assert normalize_line("Found 1 error in 1 file (checked 2 source files)") is None


def test_known_errors_pass_even_when_lines_move(tmp_path: Path) -> None:
    cmd = _fake(tmp_path, _found(ERR_A_MOVED, ERR_B), 1)
    assert run_ratchet(cmd, _baseline(tmp_path, KEY_A, KEY_B), update=False) == 0


def test_new_error_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cmd = _fake(tmp_path, _found(ERR_A, ERR_B), 1)
    assert run_ratchet(cmd, _baseline(tmp_path, KEY_A), update=False) == 1
    assert f"new: {KEY_B}" in capsys.readouterr().out


def test_duplicate_of_baselined_error_fails(tmp_path: Path) -> None:
    # A second identical error in the same file is still new (multiset semantics).
    cmd = _fake(tmp_path, _found(ERR_A, ERR_A_MOVED), 1)
    assert run_ratchet(cmd, _baseline(tmp_path, KEY_A), update=False) == 1


def test_fixed_error_passes_with_notice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cmd = _fake(tmp_path, _found(ERR_A), 1)
    assert run_ratchet(cmd, _baseline(tmp_path, KEY_A, KEY_B), update=False) == 0
    out = capsys.readouterr().out
    assert "NOTICE" in out and f"fixed: {KEY_B}" in out


def test_clean_run_with_empty_baseline_passes(tmp_path: Path) -> None:
    cmd = _fake(tmp_path, "Success: no issues found in 7 source files\n", 0)
    assert run_ratchet(cmd, _baseline(tmp_path), update=False) == 0


@pytest.mark.parametrize(
    ("stdout", "code", "stderr"),
    [
        ("", 2, "Traceback: boom"),  # crash / usage error
        ("src/a.py: error: Source file found twice under different module names\n"
         "Found 1 error in 1 file (errors prevented further checking)\n", 2, ""),
        ("some garbage\n", 1, ""),  # unparsable: no summary
        ("Success: no issues found in 0 source files\n", 0, ""),  # checked nothing
        (_found(ERR_A, checked=0), 1, ""),  # checked nothing
        ("Found 3 errors in 1 file (checked 4 source files)\n" + ERR_A + "\n", 1, ""),  # count mismatch
        (_found(ERR_A), 0, ""),  # exit status contradicts output
    ],
)
def test_untrustworthy_mypy_fails_closed(tmp_path: Path, stdout: str, code: int, stderr: str) -> None:
    cmd = _fake(tmp_path, stdout, code, stderr)
    assert run_ratchet(cmd, _baseline(tmp_path, KEY_A), update=False) == 2


def test_missing_baseline_fails_closed(tmp_path: Path) -> None:
    cmd = _fake(tmp_path, _found(ERR_A), 1)
    assert run_ratchet(cmd, tmp_path / "nope.txt", update=False) == 2


def test_missing_mypy_binary_fails_closed(tmp_path: Path) -> None:
    cmd = [str(tmp_path / "does-not-exist")]
    assert run_ratchet(cmd, _baseline(tmp_path), update=False) == 2


def test_update_writes_sorted_baseline_and_does_not_write_on_crash(tmp_path: Path) -> None:
    target = tmp_path / "b.txt"
    assert run_ratchet(_fake(tmp_path, _found(ERR_B, ERR_A), 1), target, update=True) == 0
    assert target.read_text(encoding="utf-8").splitlines() == sorted([KEY_A, KEY_B])
    before = target.read_text(encoding="utf-8")
    assert run_ratchet(_fake(tmp_path, "", 2), target, update=True) == 2
    assert target.read_text(encoding="utf-8") == before
