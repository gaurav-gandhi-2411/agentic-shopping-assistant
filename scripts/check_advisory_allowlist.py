"""Validate security/advisory-allowlist.yaml and feed the CI audit tools.

Usage:
  python scripts/check_advisory_allowlist.py check [--file PATH] [--today YYYY-MM-DD]
  python scripts/check_advisory_allowlist.py pip-args    # prints: --ignore-vuln ID ... (pip entries)
  python scripts/check_advisory_allowlist.py npm-audit   # runs `npm audit --json` in frontend/,
                                                         # fails on high/critical advisories that
                                                         # are not allow-listed
Exit code 1 on any expired / malformed entry or un-allow-listed finding (fail closed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT = Path("security/advisory-allowlist.yaml")
REQUIRED = ("id", "ecosystem", "package", "justification", "added", "expires", "approver")
MAX_DAYS = 30
MIN_JUSTIFICATION = 40
# The real file is a handful of entries; a huge file is either an error or a parser DoS (a ~10 MB
# file took ~250 s in yaml.safe_load), so refuse it before parsing.
MAX_BYTES = 64 * 1024
STR_FIELDS = ("id", "ecosystem", "package", "approver", "justification")
DATE_FIELDS = ("added", "expires")

# GitHub advisory ids use a 20-char alphabet (no 0, 1, a, b, d, e, i, k, l, n, o, s, t, u, y, z).
# Checked against 100 real ids from `gh api /advisories` (all 100 match).
_GHSA = r"GHSA(?:-[23456789cfghjmpqrvwx]{4}){3}"
ID_PATTERNS: dict[str, re.Pattern[str]] = {
    # re.ASCII: without it `\d` matches Unicode digits (e.g. Arabic-Indic) and they would pass.
    "pip": re.compile(rf"(?:PYSEC-\d{{4}}-\d+|{_GHSA}|CVE-\d{{4}}-\d{{4,}})", re.ASCII),
    # npm audit reports GHSA urls; legacy advisories have a plain numeric id.
    "npm": re.compile(rf"(?:{_GHSA}|[1-9]\d{{0,6}})", re.ASCII),
}


def validate(data: Any, today: dt.date) -> list[str]:
    """Return every problem found in the parsed allow-list (empty list means valid)."""
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return ["allowlist malformed: top level must be a mapping with an 'entries' list"]
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for i, e in enumerate(data["entries"]):
        if not isinstance(e, dict):
            errors.append(f"entries[{i}]: not a mapping")
            continue
        tag = f"entries[{i}]({e.get('id')})"
        missing = [f for f in REQUIRED if not e.get(f)]
        if missing:
            errors.append(f"{tag}: missing {missing}")
            continue
        bad_types = [f"{f} must be a string" for f in STR_FIELDS if not isinstance(e[f], str)]
        bad_types += [
            f"{f} must be a date (YYYY-MM-DD)"
            for f in DATE_FIELDS
            if not isinstance(e[f], (dt.date, str))
        ]
        if bad_types:
            errors.append(f"{tag}: " + "; ".join(bad_types))
            continue
        eco, adv_id = e["ecosystem"], e["id"]
        if eco not in ID_PATTERNS:
            errors.append(f"{tag}: ecosystem must be pip|npm")
        elif not ID_PATTERNS[eco].fullmatch(adv_id):
            errors.append(f"{tag}: id is not a valid {eco} advisory id")
        if e["approver"] != "GG":
            errors.append(f"{tag}: approver must be GG")
        if len(str(e["justification"]).strip()) < MIN_JUSTIFICATION:
            errors.append(f"{tag}: justification too short (<{MIN_JUSTIFICATION} chars)")
        try:
            added = dt.date.fromisoformat(str(e["added"]))
            exp = dt.date.fromisoformat(str(e["expires"]))
        except ValueError:
            errors.append(f"{tag}: added/expires must be YYYY-MM-DD")
            continue
        if added > today:
            errors.append(f"{tag}: added {added} is in the future (today {today})")
        if exp <= added or (exp - added).days > MAX_DAYS:
            errors.append(f"{tag}: expires must be within {MAX_DAYS} days after added")
        if exp < today:
            errors.append(f"{tag}: EXPIRED on {exp} (today {today})")
        key = (str(eco), adv_id)
        if key in seen:
            errors.append(f"{tag}: duplicate")
        seen.add(key)
    return errors


def load(path: Path, today: dt.date) -> list[dict[str, Any]]:
    """Load and validate entries; exit 1 listing every problem (fail closed)."""
    try:
        with path.open("rb") as fh:
            raw = fh.read(MAX_BYTES + 1)  # bounded read: never slurp an arbitrarily large file
        if len(raw) > MAX_BYTES:
            sys.exit(f"allowlist too large: over {MAX_BYTES // 1024} KiB limit, refusing to parse")
        data = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        sys.exit(f"allowlist unreadable: {exc}")
    errors = validate(data, today)
    if errors:
        sys.exit("allowlist invalid:\n  " + "\n  ".join(errors))
    return list(data["entries"])


def npm_audit(entries: list[dict[str, Any]]) -> None:
    """Run `npm audit --json` in frontend/; exit 1 on any high/critical not allow-listed.

    Same threshold as the previous `npm audit --audit-level=high` step. Anything unparseable or
    a missing npm fails closed.
    """
    allowed = {str(e["id"]) for e in entries if e["ecosystem"] == "npm"}
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["npm", "audit", "--json"], cwd="frontend", capture_output=True, text=True, check=False
        )
    except OSError as exc:
        sys.exit(f"npm audit could not run: {exc}")
    try:
        vulns = json.loads(out.stdout)["vulnerabilities"]
        bad: set[str] = set()
        for name, v in vulns.items():
            if v["severity"] not in ("high", "critical"):
                continue
            ids = {x["url"].rsplit("/", 1)[-1] for x in v["via"] if isinstance(x, dict) and "url" in x}
            if not ids or not ids <= allowed:
                bad.add(f"{name} ({v['severity']}): {sorted(ids - allowed)}")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        sys.exit("npm audit output unparseable")
    if bad:
        sys.exit("npm audit: un-allow-listed high/critical:\n  " + "\n  ".join(sorted(bad)))
    print("npm audit ok")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("cmd", choices=("check", "pip-args", "npm-audit"))
    p.add_argument("--file", type=Path, default=DEFAULT)
    p.add_argument("--today", type=dt.date.fromisoformat, default=None,
                   help="YYYY-MM-DD; override today's date (deterministic tests)")
    a = p.parse_args(argv)
    entries = load(a.file, a.today or dt.datetime.now(dt.timezone.utc).date())
    if a.cmd == "check":
        print(f"allowlist ok: {len(entries)} entries")
    elif a.cmd == "pip-args":
        print(" ".join(f"--ignore-vuln {e['id']}" for e in entries if e["ecosystem"] == "pip"))
    else:
        npm_audit(entries)


if __name__ == "__main__":
    main()
