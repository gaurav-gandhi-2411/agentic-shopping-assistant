"""Config-parity check between the serving revision and the new one (used by deploy.yml).

    python scripts/ops/check_parity.py REV_PREV.json REV_NEW.json [--expect-index-build-id <id>]

Same view the workflow used to compute with jq (env incl. secret refs, resource limits, concurrency,
timeout, service account, maxScale), with ONE deliberate change for index rollouts: the
``INDEX_BUILD_ID`` env var is excluded from the env diff and asserted separately.

* ordinary code deploy (no ``--expect-index-build-id``): the new revision's pin must EQUAL the
  serving revision's pin (an unset pin on both counts as equal): the pin was inherited, not changed;
* rollout (``--expect-index-build-id <id>``): the new revision's pin must equal ``<id>``, and ``<id>``
  must be a valid build id and not older than the serving revision's pin (no silent downgrade).

Exit 0 when identical, 1 with a unified diff / reason otherwise, 2 on unreadable input.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

PIN_ENV = "INDEX_BUILD_ID"
BUILD_ID_RE = re.compile(r"\d{8}T\d{6}Z", re.ASCII)  # same pattern as src/retrieval/index_reload.py


class ParityInputError(Exception):
    """A revision JSON is unreadable or not the expected shape (fail closed)."""


def _spec(doc: dict[str, Any]) -> dict[str, Any]:
    spec = doc.get("spec")
    if not isinstance(spec, dict) or not (spec.get("containers") or [None])[0]:
        raise ParityInputError("revision JSON has no spec.containers[0]")
    return spec


def pin(doc: dict[str, Any]) -> str | None:
    """``INDEX_BUILD_ID`` value of a revision, None when unset."""
    for env in _spec(doc)["containers"][0].get("env") or []:
        if env.get("name") == PIN_ENV:
            value = env.get("value")
            return value if isinstance(value, str) and value else None
    return None


def view(doc: dict[str, Any]) -> dict[str, Any]:
    """The fields that must not change between two revisions of one service."""
    spec = _spec(doc)
    container = spec["containers"][0]
    env = [e for e in (container.get("env") or []) if e.get("name") != PIN_ENV]
    return {
        "env": sorted(env, key=lambda e: str(e.get("name"))),
        "limits": (container.get("resources") or {}).get("limits"),
        "concurrency": spec.get("containerConcurrency"),
        "timeout": spec.get("timeoutSeconds"),
        "serviceAccount": spec.get("serviceAccountName"),
        "maxScale": ((doc.get("metadata") or {}).get("annotations") or {}).get(
            "autoscaling.knative.dev/maxScale"
        ),
    }


def problems(prev: dict[str, Any], new: dict[str, Any], expect_id: str | None) -> list[str]:
    """Every parity violation (empty list = OK)."""
    out: list[str] = []
    a = json.dumps(view(prev), indent=2, sort_keys=True).splitlines()
    b = json.dumps(view(new), indent=2, sort_keys=True).splitlines()
    if a != b:
        diff = "\n".join(difflib.unified_diff(a, b, "serving", "new", lineterm=""))
        out.append(
            "new revision differs from the serving revision in env/secrets/cpu/memory/"
            f"concurrency/timeout/maxScale/SA:\n{diff}"
        )
    prev_pin, new_pin = pin(prev), pin(new)
    if expect_id is None:
        if new_pin != prev_pin:
            out.append(f"{PIN_ENV} changed on a code deploy: {prev_pin!r} -> {new_pin!r}")
        return out
    if not BUILD_ID_RE.fullmatch(expect_id):
        out.append(f"expected build id {expect_id!r} is not a build id")
    elif prev_pin is not None and expect_id < prev_pin:
        out.append(f"rollout to {expect_id} would DOWNGRADE the serving pin {prev_pin}")
    if new_pin != expect_id:
        out.append(f"{PIN_ENV} on the new revision is {new_pin!r}, expected {expect_id!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prev", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--expect-index-build-id", default=None)
    args = ap.parse_args(argv)
    try:
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in (args.prev, args.new)]
        found = problems(docs[0], docs[1], args.expect_index_build_id)
    except (OSError, ValueError, ParityInputError, AttributeError) as exc:
        print(f"::error::cannot compare revisions: {exc}", file=sys.stderr)
        return 2
    for text in found:
        print(f"::error::{text}")
    if not found:
        mode = f"rollout to {args.expect_index_build_id}" if args.expect_index_build_id else "code deploy"
        print(f"config parity OK ({mode})")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
