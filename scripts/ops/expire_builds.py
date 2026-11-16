"""Index-build retention: mark old builds for the bucket lifecycle rule (Wave 1b V3(c), part B).

    python scripts/ops/expire_builds.py --bucket <name> --service-json service.json \
        --revisions-dir revs/ [--keep 3] [--apply]

Runs ONLY after a successful rollout (new build serving 100 %, post-shift check passed). It never
deletes anything: the publisher identity has no ``storage.objects.delete``. For every build that is
(a) not among the ``keep`` newest and (b) not referenced by any Cloud Run revision, it sets
``customTime = now`` on the build's objects. The bucket lifecycle rule then deletes objects under
``builds/`` whose ``daysSinceCustomTime >= 2``. Without ``--apply`` it prints the plan and writes
nothing (the store is wrapped read-only).

A build is "referenced" when ANY of these names it through ``INDEX_BUILD_ID``: a revision in
``status.traffic`` with traffic > 0 or a tag, the service TEMPLATE (what the next code deploy
inherits), or any other revision whose JSON is supplied (the rollback targets that the revision
cleanup keeps). A revision in ``status.traffic`` whose JSON is missing fails closed.

Fail-closed everywhere: unreadable service/revision JSON, an unparseable build list or ``--keep < 1``
exit non-zero with nothing marked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Collection, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refresh_store as rs  # noqa: E402

PIN_ENV = "INDEX_BUILD_ID"
DEFAULT_KEEP = 3
EXIT_OK, EXIT_USAGE, EXIT_INPUT, EXIT_STORE = 0, 2, 3, 4


class InputError(Exception):
    """Service or revision JSON is missing or malformed (fail closed)."""


def select_builds_to_expire(
    builds: Iterable[str], referenced: Collection[str], keep: int = DEFAULT_KEEP
) -> list[str]:
    """Build ids to mark for expiry: everything except the ``keep`` newest and the referenced.

    Ids are UTC timestamps, so the lexicographic order is the time order. Strings that are not
    build ids are never returned (they are not ours to touch). The result is ascending.

    Raises:
        ValueError: ``keep`` < 1 (the newest build is never expirable).
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    valid = sorted({b for b in builds if rs.BUILD_ID_RE.fullmatch(b)})
    protected = set(valid[-keep:]) | set(referenced)
    return [b for b in valid if b not in protected]


def without_marked(expire: Iterable[str], marked: Collection[str]) -> list[str]:
    """Drop builds that already carry a ``customTime``.

    Re-marking on every run would push their expiry back each day and they would never be
    deleted (found by the simulation test): a build is marked once, the lifecycle rule does the rest.
    """
    return [b for b in expire if b not in marked]


def pin_of(spec: Mapping[str, Any]) -> str | None:
    """``INDEX_BUILD_ID`` of a container spec (a revision's ``.spec``, a template's ``.spec.template.spec``)."""
    containers = spec.get("containers") or []
    for env in (containers[0].get("env") or []) if containers else []:
        if env.get("name") == PIN_ENV:
            value = env.get("value")
            return value if isinstance(value, str) and rs.BUILD_ID_RE.fullmatch(value) else None
    return None


def referenced_build_ids(
    service: Mapping[str, Any], revisions: Mapping[str, Mapping[str, Any]]
) -> set[str]:
    """Build ids pinned by traffic-bearing/tagged revisions, the service template and ``revisions``.

    Raises:
        InputError: a revision in ``status.traffic`` (percent > 0 or tagged) has no JSON in
            ``revisions``, or the service has no ``status.traffic``.
    """
    traffic = (service.get("status") or {}).get("traffic")
    if not isinstance(traffic, list):
        raise InputError("service JSON has no status.traffic")
    latest_ready = (service.get("status") or {}).get("latestReadyRevisionName")
    found: set[str] = set()
    for entry in traffic:
        if not ((entry.get("percent") or 0) > 0 or entry.get("tag")):
            continue
        name = entry.get("revisionName") or (latest_ready if entry.get("latestRevision") else None)
        if not name or name not in revisions:
            raise InputError(f"no revision JSON for traffic entry {name or entry!r}")
    # Every supplied revision counts (rollback targets), including all traffic-bearing ones above.
    for rev in revisions.values():
        pin = pin_of(rev.get("spec") or {})
        if pin:
            found.add(pin)
    template_pin = pin_of(((service.get("spec") or {}).get("template") or {}).get("spec") or {})
    if template_pin:
        found.add(template_pin)
    return found


def load_revisions(directory: Path) -> dict[str, dict[str, Any]]:
    """``<name>.json`` files of ``gcloud run revisions describe --format=json`` keyed by name."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InputError(f"unreadable revision JSON {path.name}") from exc
        if not isinstance(data, dict):
            raise InputError(f"revision JSON {path.name} is not an object")
        out[path.stem] = data
    return out


def plan(
    store: rs.BuildStore,
    service: Mapping[str, Any],
    revisions: Mapping[str, Mapping[str, Any]],
    keep: int = DEFAULT_KEEP,
) -> dict[str, Any]:
    """Compute what would be expired (complete builds only: a half upload is never counted)."""
    complete = [b for b in store.list_build_ids() if rs.MANIFEST_NAME in store.list_objects(b)]
    referenced = referenced_build_ids(service, revisions)
    selected = select_builds_to_expire(complete, referenced, keep)
    marked = {b for b in selected if store.custom_time_of(b) is not None}
    return {
        "builds": complete,
        "keep_newest": complete[-keep:],
        "referenced": sorted(referenced),
        "expire": without_marked(selected, marked),  # newly marked by this run
        "already_marked": sorted(marked),  # waiting for the lifecycle rule
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--service-json", type=Path, required=True)
    ap.add_argument("--revisions-dir", type=Path, required=True)
    ap.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    ap.add_argument("--apply", action="store_true", help="set customTime (default: plan only)")
    args = ap.parse_args(argv)
    if args.keep < 1:
        print("REFUSED: --keep must be >= 1", file=sys.stderr)
        return EXIT_USAGE
    try:
        service = json.loads(args.service_json.read_text(encoding="utf-8"))
        revisions = load_revisions(args.revisions_dir)
        base: rs.BuildStore = rs.GcsBuildStore(args.bucket)
        result = plan(base if args.apply else rs.ReadOnlyStore(base), service, revisions, args.keep)
        if args.apply:
            now = datetime.now(timezone.utc)
            result["marked"] = {b: base.set_custom_time(b, now) for b in result["expire"]}
    except (InputError, OSError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except rs.StoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_STORE
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
