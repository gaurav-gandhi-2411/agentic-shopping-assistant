"""Tag the serving image `live` and the previously serving image `prev` in Artifact Registry.

Called by the last step of .github/workflows/deploy.yml, after a fully successful deploy, so the
AR cleanup policy's tag-pin Keep rule (scripts/ops/ar_cleanup_policy.json) protects both images.

Usage: pin_images.py REV_NEW.json REV_PREV.json
Env:   GCP_PROJECT (passed as --project), GCLOUD (command, default "gcloud"; tests inject a fake).

Cloud Run reports the digest as the FULL reference (`REPO/pkg@sha256:<64 hex>`), not a bare
`sha256:...`, in both `.status.imageDigest` and `.spec.containers[0].image`; either may also carry
a `:tag` before the `@`. The bare digest and the tag-less base are therefore derived here, and a
revision with no full 64-hex digest is a hard failure (never guess, never tag by mutable tag).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

_DIGEST_RE = re.compile(r"(?:^|@)(sha256:[0-9a-f]{64})$")
_TAG_RE = re.compile(r"(/[^/:@]+):[^/@]*$")


class PinError(Exception):
    """Unresolvable revision JSON (no digest, no image)."""


def resolve(revision_json: Path) -> tuple[str, str]:
    """Return (base, digest): `base` is the image path without tag/digest, `digest` is sha256:<64>."""
    try:
        doc = json.loads(revision_json.read_text(encoding="utf-8"))
        image = doc["spec"]["containers"][0]["image"]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise PinError(f"{revision_json}: cannot read spec.containers[0].image: {exc}") from exc
    if not isinstance(image, str) or not image:
        raise PinError(f"{revision_json}: empty image")
    candidates = [(doc.get("status") or {}).get("imageDigest"), image]
    for cand in candidates:
        match = _DIGEST_RE.search(cand) if isinstance(cand, str) else None
        if match:
            base = _TAG_RE.sub(r"\1", image.split("@", 1)[0])
            return base, match.group(1)
    raise PinError(f"{revision_json}: no full sha256 digest found (image={image!r})")


def pin_all(
    new_json: Path, prev_json: Path, runner: Callable[[list[str]], int], project: str | None
) -> int:
    """Add `live` for the new image, then `prev` for the previous one. Stops at the first failure."""
    try:
        plan = [("live", *resolve(new_json)), ("prev", *resolve(prev_json))]
    except PinError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    for tag, base, digest in plan:
        argv = ["artifacts", "docker", "tags", "add", f"{base}@{digest}", f"{base}:{tag}"]
        if project:
            argv += ["--project", project]
        print(f"tag {base}:{tag} -> {digest}")
        rc = runner(argv)
        if rc != 0:
            print(f"::error::gcloud tags add for '{tag}' failed (rc={rc})", file=sys.stderr)
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: pin_images.py REV_NEW.json REV_PREV.json", file=sys.stderr)
        return 2
    gcloud = shlex.split(os.environ.get("GCLOUD") or "gcloud")

    def runner(cmd: list[str]) -> int:
        return subprocess.run([*gcloud, *cmd], check=False).returncode  # noqa: S603

    return pin_all(Path(args[0]), Path(args[1]), runner, os.environ.get("GCP_PROJECT"))


if __name__ == "__main__":
    sys.exit(main())
