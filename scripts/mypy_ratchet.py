"""Run mypy and fail on any error that is not in the committed baseline (a ratchet).

Why: mypy over this repo reports hundreds of pre-existing errors. Fixing them all in one
change is out of scope, but a non-blocking mypy catches nothing. The ratchet makes CI fail on
every *new* error while the baseline (``mypy-baseline.txt``) shrinks over time.

Baseline format: one normalized error per line, ``<path>: [<code>] <message>``, sorted. A
repeated line means that many occurrences (multiset), so adding a second identical error in
the same file is still caught. Line numbers are stripped so unrelated edits do not churn it.

Exit codes: 0 = no new errors, 1 = new errors, 2 = could not verify (fail closed): mypy
crashed, exited with an unexpected status, output was unparsable/inconsistent, or it checked
zero files.

Usage:
    python scripts/mypy_ratchet.py            # check against baseline
    python scripts/mypy_ratchet.py --update   # rewrite baseline from current mypy output
Mypy's own settings (files, excludes, package bases) live in pyproject.toml [tool.mypy].
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / "mypy-baseline.txt"
DEFAULT_MYPY_CMD = [sys.executable, "-m", "mypy", "--no-color-output"]

_ERROR_RE = re.compile(r"^(?P<path>[^\s:][^:]*?):\d+(?::\d+)?: error: (?P<msg>.*)$")
_FOUND_RE = re.compile(r"^Found (\d+) errors? in \d+ files? \(checked (\d+) source files?\)")
_OK_RE = re.compile(r"^Success: no issues found in (\d+) source files?")


class RatchetError(Exception):
    """mypy could not be run or its result could not be trusted (exit status 2)."""


def normalize_line(line: str) -> str | None:
    """Return the line-number-free key for a mypy error line, or None if it is not an error."""
    m = _ERROR_RE.match(line.rstrip())
    if m is None:
        return None
    # Backslash -> slash so a baseline made on Windows matches Linux CI output.
    path = m.group("path").replace("\\", "/")
    msg = re.sub(r"\bline \d+", "line N", m.group("msg"))
    msg = re.sub(r"(\.pyi?):\d+(:\d+)?", r"\1", msg).replace("\\", "/")
    msg = re.sub(r"\s+", " ", msg).strip()
    # The code stays in the key so a changed error code counts as a new error.
    code = re.search(r"\s\[([\w-]+)\]$", msg)
    if code is None:
        return f"{path}: [none] {msg}"
    return f"{path}: [{code.group(1)}] {msg[: code.start()]}"


def collect_errors(mypy_cmd: list[str]) -> Counter[str]:
    """Run mypy and return a multiset of normalized errors; raise RatchetError if untrustworthy."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            mypy_cmd, capture_output=True, text=True, check=False, encoding="utf-8"
        )
    except OSError as exc:
        raise RatchetError(f"could not launch mypy: {exc}") from exc
    tail = f"stderr tail: {proc.stderr[-400:]!r} stdout tail: {proc.stdout[-400:]!r}"
    if proc.returncode not in (0, 1):
        raise RatchetError(f"mypy exited with status {proc.returncode}; {tail}")
    lines = proc.stdout.splitlines()
    if any("prevented further checking" in ln for ln in lines):
        raise RatchetError(f"mypy aborted before checking all files; {tail}")
    errors: Counter[str] = Counter()
    for ln in lines:
        key = normalize_line(ln)
        if key is not None:
            errors[key] += 1
    n_errors = sum(errors.values())
    summary_errors, checked = None, None
    for ln in lines:
        if m := _FOUND_RE.match(ln):
            summary_errors, checked = int(m.group(1)), int(m.group(2))
        elif m := _OK_RE.match(ln):
            summary_errors, checked = 0, int(m.group(1))
    if checked is None:
        raise RatchetError(f"no mypy summary line found; {tail}")
    if checked == 0:
        raise RatchetError("mypy checked 0 source files (bad 'files'/'exclude' config?)")
    if summary_errors != n_errors:
        raise RatchetError(
            f"parsed {n_errors} error lines but mypy summary says {summary_errors}; {tail}"
        )
    if (proc.returncode == 0) != (n_errors == 0):
        raise RatchetError(f"exit status {proc.returncode} inconsistent with {n_errors} errors")
    return errors


def load_baseline(path: Path) -> Counter[str]:
    """Read the baseline file into a multiset; a missing file is an error (fail closed)."""
    if not path.is_file():
        raise RatchetError(f"baseline file missing: {path} (run with --update to create it)")
    return Counter(ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def run_ratchet(mypy_cmd: list[str], baseline_path: Path, update: bool) -> int:
    """Compare current mypy errors against the baseline; return the process exit status."""
    try:
        current = collect_errors(mypy_cmd)
        if update:
            text = "".join(f"{k}\n" for k in sorted(current.elements()))
            baseline_path.write_text(text, encoding="utf-8", newline="\n")
            print(f"mypy-ratchet: baseline updated: {sum(current.values())} errors")
            return 0
        baseline = load_baseline(baseline_path)
    except RatchetError as exc:
        print(f"mypy-ratchet: FAIL (cannot verify): {exc}", file=sys.stderr)
        return 2
    new = current - baseline
    fixed = baseline - current
    if fixed:
        print(
            f"mypy-ratchet: NOTICE: {sum(fixed.values())} baseline error(s) no longer occur; "
            "shrink the baseline with `python scripts/mypy_ratchet.py --update`:"
        )
        for k in sorted(fixed.elements()):
            print(f"  fixed: {k}")
    if new:
        print(f"mypy-ratchet: FAIL: {sum(new.values())} new mypy error(s) not in baseline:")
        for k in sorted(new.elements()):
            print(f"  new: {k}")
        return 1
    print(
        f"mypy-ratchet: OK: {sum(current.values())} errors, none new "
        f"({sum(baseline.values())} baselined)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="mypy ratchet against mypy-baseline.txt")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline")
    ap.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    ap.add_argument("--mypy-cmd", nargs=argparse.REMAINDER, help="override mypy command (tests)")
    args = ap.parse_args(argv)
    cmd = args.mypy_cmd if args.mypy_cmd else DEFAULT_MYPY_CMD
    return run_ratchet(cmd, args.baseline, args.update)


if __name__ == "__main__":
    sys.exit(main())
