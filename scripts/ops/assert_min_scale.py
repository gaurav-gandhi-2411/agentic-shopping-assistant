"""Fail a deploy if any revision that can receive traffic keeps warm instances (minScale > 0).

Stdlib only, reads `gcloud run ... --format=json` output from files. Fails CLOSED: unreadable,
malformed, or incomplete input is a failure, never a pass.

Checked surfaces (a leftover on any one of them bills idle instances):
  * every revision in service `.status.traffic` with percent > 0 OR a non-empty tag (a tagged
    0% revision is still routable through its tag URL), read from `<revisions-dir>/<name>.json`
    annotation `autoscaling.knative.dev/minScale`;
  * the service template annotation `autoscaling.knative.dev/minScale` (what the NEXT revision
    inherits; today's template carried 1 from a retired revision);
  * the service-level annotation `run.googleapis.com/minScale`.
A missing annotation means 0.

Usage:
  assert_min_scale.py --service service.json --list-revisions      # prints names to check
  assert_min_scale.py --service service.json --revisions-dir DIR    # asserts; exit 1 on failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REV_KEY = "autoscaling.knative.dev/minScale"
SERVICE_KEY = "run.googleapis.com/minScale"


class AssertionFailure(Exception):
    """Raised for any violation or unverifiable input; the message names the offender."""


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssertionFailure(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise AssertionFailure(f"{path} is empty or not a JSON object")
    return data


def _annotation(doc: dict[str, Any], *path: str, key: str, where: str) -> int:
    """Return the integer minScale at doc[path...].annotations[key]; missing means 0."""
    node: Any = doc
    for part in path:
        node = node.get(part) if isinstance(node, dict) else None
    annotations = node.get("annotations") if isinstance(node, dict) else None
    if annotations is None:
        return 0
    if not isinstance(annotations, dict):
        raise AssertionFailure(f"{where}: annotations is not an object")
    raw = annotations.get(key)
    if raw is None:
        return 0
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise AssertionFailure(f"{where}: unparsable {key}={raw!r}") from exc


def routable_revisions(service: dict[str, Any]) -> list[str]:
    """Revision names in status.traffic with percent > 0 or a tag (sorted, unique)."""
    traffic = (service.get("status") or {}).get("traffic")
    if not isinstance(traffic, list) or not traffic:
        raise AssertionFailure(
            "service status.traffic is missing or empty; cannot resolve revisions"
        )
    names: set[str] = set()
    for entry in traffic:
        if not isinstance(entry, dict):
            raise AssertionFailure(f"malformed traffic entry: {entry!r}")
        if (entry.get("percent") or 0) > 0 or (entry.get("tag") or ""):
            name = entry.get("revisionName")
            if not name:
                raise AssertionFailure(f"routable traffic entry without revisionName: {entry!r}")
            names.add(name)
    if not names:
        raise AssertionFailure("no revision has traffic or a tag; refusing to pass vacuously")
    return sorted(names)


def check(service: dict[str, Any], revisions_dir: Path) -> list[str]:
    """Return human-readable OK lines, or raise AssertionFailure naming the first offender."""
    ok: list[str] = []
    svc_min = _annotation(service, "metadata", key=SERVICE_KEY, where="service")
    if svc_min > 0:
        raise AssertionFailure(f"service-level {SERVICE_KEY}={svc_min}, expected 0")
    tpl_min = _annotation(
        service, "spec", "template", "metadata", key=REV_KEY, where="service template"
    )
    if tpl_min > 0:
        raise AssertionFailure(
            f"service template {REV_KEY}={tpl_min}, expected 0 (next revision would inherit it)"
        )
    ok.append("service-level and template minScale = 0")
    for name in routable_revisions(service):
        rev = _load(revisions_dir / f"{name}.json")
        value = _annotation(rev, "metadata", key=REV_KEY, where=f"revision {name}")
        if value > 0:
            raise AssertionFailure(f"revision {name} (serving or tagged) has minScale={value}")
        ok.append(f"revision {name} minScale=0")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--revisions-dir", type=Path)
    parser.add_argument("--list-revisions", action="store_true")
    args = parser.parse_args(argv)
    try:
        service = _load(args.service)
        if args.list_revisions:
            print("\n".join(routable_revisions(service)))
            return 0
        if args.revisions_dir is None:
            raise AssertionFailure("--revisions-dir is required unless --list-revisions")
        for line in check(service, args.revisions_dir):
            print(line)
    except AssertionFailure as exc:
        print(f"::error::minScale assertion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
