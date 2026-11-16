"""Index-rollout guards used by the refresh workflow and deploy.yml (Wave 1b V3(c), part C).

    # pre-rollout: the revision we would roll back to must still be able to load its pinned build
    python scripts/ops/rollout_checks.py preflight --prev-revision-json rev-prev.json --bucket <name>

    # keep-last-N revision cleanup: print the revision names that may be deleted
    python scripts/ops/rollout_checks.py revisions-to-delete --revisions-json revisions.json \
        --service-json service.json [--protect <name> ...] [--keep 5]

``preflight`` needs bucket read access (publisher identity), ``revisions-to-delete`` only reads JSON
files; neither writes anything. Exit 0 ok, 1 a check failed (refuse the rollout), 2 unreadable input.

Pre-rollout rationale: auto-rollback sends traffic back to the previous revision, which is pinned to
the previous build. If the bucket lifecycle rule has already deleted that build, the "rollback" would
land on a revision that can only serve the legacy fallback and fails /readyz. Refuse the rollout
instead (and alert) while the rollback is still safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refresh_store as rs  # noqa: E402

PIN_ENV = "INDEX_BUILD_ID"
KEEP_REVISIONS = 5
MAX_MANIFEST_BYTES = 8 * 1024 * 1024


class CheckInputError(Exception):
    """Unreadable or malformed input (fail closed)."""


def revision_pin(revision: Mapping[str, Any]) -> str | None:
    """``INDEX_BUILD_ID`` of a revision JSON; None when unset (legacy) or not a valid build id."""
    try:
        env = revision["spec"]["containers"][0].get("env") or []
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise CheckInputError("revision JSON has no spec.containers[0]") from exc
    for item in env:
        if item.get("name") == PIN_ENV:
            value = item.get("value")
            return value if isinstance(value, str) and rs.BUILD_ID_RE.fullmatch(value) else "<invalid>"
    return None


def preflight_problem(prev_pin: str | None, store: rs.BuildStore) -> str | None:
    """Why the previous revision could not reload its pinned build, or None when it can.

    ``prev_pin`` None (the previous revision serves the legacy flat index) is fine: ``unified/`` is
    never touched by retention. Otherwise the build must have a readable manifest and EVERY file the
    manifest lists, with the listed size.
    """
    if prev_pin is None:
        return None
    if prev_pin == "<invalid>" or not rs.BUILD_ID_RE.fullmatch(prev_pin):
        return "previous revision has a malformed INDEX_BUILD_ID"
    try:
        objects = store.list_objects(prev_pin)
        if rs.MANIFEST_NAME not in objects:
            return f"previous build {prev_pin} is gone from the bucket (no manifest)"
        manifest = json.loads(store.read_object(prev_pin, rs.MANIFEST_NAME, MAX_MANIFEST_BYTES))
        listed = {f["name"]: int(f["size_bytes"]) for f in manifest["files"]}
    except rs.StoreError as exc:
        return f"cannot read previous build {prev_pin}: {exc}"
    except (ValueError, KeyError, TypeError) as exc:
        return f"previous build {prev_pin} has an unreadable manifest ({type(exc).__name__})"
    for name, size in sorted(listed.items()):
        if objects.get(name) != size:
            return f"previous build {prev_pin}: {name} missing or wrong size"
    return None


def select_revisions_to_delete(
    revisions: Iterable[Mapping[str, Any]],
    service: Mapping[str, Any],
    protect: Collection[str] = (),
    keep: int = KEEP_REVISIONS,
) -> list[str]:
    """Revision names older than the ``keep`` newest, minus anything that can receive traffic.

    Never returned: a revision in ``status.traffic`` with traffic > 0 or a tag, the latest ready
    revision, any name in ``protect`` (the rollback target and the new revision), and the ``keep``
    newest by ``metadata.creationTimestamp``. Oldest first.

    Raises:
        ValueError: ``keep`` < 1.
        CheckInputError: a revision has no name/timestamp, or the service has no ``status.traffic``.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    status = service.get("status") or {}
    traffic = status.get("traffic")
    if not isinstance(traffic, list):
        raise CheckInputError("service JSON has no status.traffic")
    protected = set(protect)
    for entry in traffic:
        if (entry.get("percent") or 0) > 0 or entry.get("tag"):
            name = entry.get("revisionName") or (
                status.get("latestReadyRevisionName") if entry.get("latestRevision") else None
            )
            if not name:
                raise CheckInputError(f"traffic entry without a revision name: {entry!r}")
            protected.add(name)
    if status.get("latestReadyRevisionName"):
        protected.add(status["latestReadyRevisionName"])
    dated: list[tuple[str, str]] = []
    for rev in revisions:
        meta = rev.get("metadata") or {}
        name, stamp = meta.get("name"), meta.get("creationTimestamp")
        if not name or not stamp:
            raise CheckInputError("revision without metadata.name / creationTimestamp")
        dated.append((str(stamp), str(name)))
    dated.sort()  # RFC 3339 UTC timestamps sort lexicographically
    newest = {name for _, name in dated[-keep:]}
    return [name for _, name in dated if name not in newest and name not in protected]


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckInputError(f"cannot read {path.name}: {type(exc).__name__}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("preflight")
    pf.add_argument("--prev-revision-json", type=Path, required=True)
    pf.add_argument("--bucket", required=True)
    rv = sub.add_parser("revisions-to-delete")
    rv.add_argument("--revisions-json", type=Path, required=True)
    rv.add_argument("--service-json", type=Path, required=True)
    rv.add_argument("--protect", action="append", default=[])
    rv.add_argument("--keep", type=int, default=KEEP_REVISIONS)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "preflight":
            pin = revision_pin(_load(args.prev_revision_json))
            problem = preflight_problem(pin, rs.ReadOnlyStore(rs.GcsBuildStore(args.bucket)))
            if problem:
                print(f"::error::pre-rollout check failed, refusing the rollout: {problem}")
                return 1
            print(f"pre-rollout check OK (previous pin: {pin or 'none, legacy index'})")
            return 0
        revisions = _load(args.revisions_json)
        if not isinstance(revisions, list) or not revisions:
            raise CheckInputError("revisions JSON is not a non-empty list")
        names = select_revisions_to_delete(revisions, _load(args.service_json), args.protect, args.keep)
    except (CheckInputError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    print("\n".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
