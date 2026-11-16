"""Daily index refresh orchestrator (Wave 1b V3(c)).

    python scripts/ops/refresh_index.py --work-dir <dir> (--bucket <name> | --local-store <dir>)
        [--publish] [--build-cmd "<command with {raw_dir} {previous_dir} {out_dir}>"]
        [--feed-fixtures <dir>] [--summary-out refresh-summary.json]

Stages: plan (stores with refresh_enabled) -> previous (newest build with a manifest) -> fetch
(polite, ``refresh_feed.py``) -> fetch gate -> build (pluggable command) -> manifest (the existing
``make_build_manifest`` verifier, incl. the +-30 % total gate) -> validate (per-store counts) ->
publish (create-only, ``refresh_store.py``; ONLY with ``--publish``).

DRY RUN IS THE DEFAULT: without ``--publish`` the store is wrapped read-only, nothing is uploaded and
nothing is rolled out. Any failure leaves the bucket untouched (publish is the last stage, manifest
last within it), writes ``--summary-out`` with the failing stage and exits non-zero.

Exit codes: 0 ok | 2 usage/config | 3 fetch gate | 4 build | 5 validation | 6 publish | 7 internal.

Build seam: the repo has no in-CI incremental build yet (the per-brand ``data/processed`` dirs and
CLIP image vectors do not exist on a fresh runner). ``--build-cmd`` is the contract such a builder
must meet: read ``{raw_dir}/<slug>/products.csv`` for stores that fetched, carry over the rows of
every other store from the previous build (``{previous_dir}``, id in ``{previous_build_id}``), and
write the index files into ``{out_dir}``. Without ``--build-cmd`` the build stage fails with
``build_not_configured``: a refresh that cannot build must not publish.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[1]), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import refresh_feed as rf  # noqa: E402
import refresh_store as rs  # noqa: E402

from src.config.store_registry import STORE_REGISTRY, refreshable_stores  # noqa: E402
from src.retrieval.build_manifest import (  # noqa: E402
    MANIFEST_NAME,
    BuildManifest,
    BuildValidationError,
    load_manifest,
    now_build_id,
    write_manifest,
)

EXIT_OK, EXIT_USAGE, EXIT_FETCH, EXIT_BUILD, EXIT_VALIDATE, EXIT_PUBLISH, EXIT_INTERNAL = (
    0,
    2,
    3,
    4,
    5,
    6,
    7,
)
MIN_FRESH_STORE_RATIO = 0.5  # fewer fresh stores than this and the "refresh" is mostly a re-publish
STORE_DROP_LIMIT = 0.5  # a store losing more than half its items is a broken feed, not a sale
STORE_DROP_MIN_ITEMS = 20  # tiny stores swing too much for a ratio gate
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
BUILD_TIMEOUT_S = 3 * 3600


class RefreshError(Exception):
    """A stage failed. ``code`` is a short machine code, ``exit_code`` the process exit code."""

    def __init__(self, stage: str, code: str, message: str, exit_code: int) -> None:
        super().__init__(f"{stage}/{code}: {message}")
        self.stage, self.code, self.message, self.exit_code = stage, code, message, exit_code


@dataclass
class BuildContext:
    """What a builder gets."""

    raw_dir: Path
    previous_dir: Path
    out_dir: Path
    previous_build_id: str | None
    fetched: frozenset[str]  # slugs whose fresh CSV is in raw_dir
    carried: frozenset[str]  # refreshable slugs to carry over from the previous build


Builder = Callable[[BuildContext], None]


@dataclass
class Summary:
    """Machine-readable result, written on success AND failure."""

    ok: bool = False
    dry_run: bool = True
    stage: str = "plan"
    exit_code: int = EXIT_INTERNAL
    build_id: str | None = None
    previous_build_id: str | None = None
    published: bool = False
    failure: dict[str, str] | None = None
    fresh_stores: int = 0
    refreshable_stores: int = 0
    total_items: int | None = None
    count_delta_pct: float | None = None
    stores: list[dict[str, Any]] = field(default_factory=list)
    store_counts: dict[str, dict[str, int | None]] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class CommandBuilder:
    """Runs ``--build-cmd`` (no shell) with the placeholders filled in."""

    def __init__(self, template: str) -> None:
        self.template = template

    def __call__(self, ctx: BuildContext) -> None:
        values = {
            "raw_dir": str(ctx.raw_dir),
            "previous_dir": str(ctx.previous_dir),
            "out_dir": str(ctx.out_dir),
            "previous_build_id": ctx.previous_build_id or "",
        }
        argv = [part.format(**values) for part in shlex.split(self.template, posix=True)]
        env = {**os.environ, "REFRESH_FETCHED": ",".join(sorted(ctx.fetched))}
        env["REFRESH_CARRIED"] = ",".join(sorted(ctx.carried))
        try:
            done = subprocess.run(  # noqa: S603 - argv list, operator-supplied command, no shell
                argv, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S, env=env, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RefreshError("build", "build_failed", type(exc).__name__, EXIT_BUILD) from exc
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "")[-300:].replace("\n", " ")
            raise RefreshError("build", "build_failed", f"exit {done.returncode}: {tail}", EXIT_BUILD)


def fixture_fetcher(fixtures: Path) -> Callable[[Any], rf.FetchResult]:
    """Fetch stand-in reading ``<fixtures>/<slug>.json`` = ``{"status": ..., "products": [...]}``."""

    def fetch(record: Any) -> rf.FetchResult:
        path = fixtures / f"{record.slug}.json"
        if not path.is_file():
            return rf.FetchResult(record.slug, rf.NETWORK_ERROR, detail="no fixture")
        data = json.loads(path.read_text(encoding="utf-8"))
        status = data.get("status", rf.OK)
        products = data.get("products", []) if status == rf.OK else []
        return rf.FetchResult(
            record.slug, status, pages=1, products=products, fetched_at=data.get("fetched_at", "")
        )

    return fetch


def write_raw_csvs(results: list[rf.FetchResult], raw_dir: Path) -> frozenset[str]:
    """Normalise fresh stores with ``download_shopify.normalize`` (same CSV schema as today)."""
    sys.path.insert(0, str(_HERE.parent))  # scripts/ for download_shopify
    import download_shopify  # noqa: PLC0415 - heavy (requests, pandas): only when fetching

    fetched = set()
    for res in results:
        if res.status != rf.OK:
            continue
        domain = STORE_REGISTRY[res.slug].domain or ""
        frame = download_shopify.normalize(res.products, domain)
        if frame.empty:
            res.status, res.detail = rf.EMPTY, "no purchasable products after normalisation"
            continue
        out = raw_dir / res.slug
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "products.csv", index=False)
        fetched.add(res.slug)
    return frozenset(fetched)


def check_fetch_health(results: list[rf.FetchResult], refreshable: frozenset[str]) -> None:
    """Fail when too few refreshable stores produced fresh rows."""
    fresh = sum(1 for r in results if r.status == rf.OK and r.slug in refreshable)
    if not refreshable or fresh / len(refreshable) < MIN_FRESH_STORE_RATIO:
        raise RefreshError(
            "fetch",
            "too_few_fresh_stores",
            f"{fresh}/{len(refreshable)} refreshable stores fetched (need >= "
            f"{MIN_FRESH_STORE_RATIO:.0%})",
            EXIT_FETCH,
        )


def validate_store_counts(
    manifest: BuildManifest, previous: BuildManifest | None
) -> dict[str, dict[str, int | None]]:
    """Per-store gate against the previous build. Returns ``slug -> {previous, new}``.

    Fails when a store that had items disappears, or one with >= ``STORE_DROP_MIN_ITEMS`` items
    loses more than ``STORE_DROP_LIMIT`` of them. (The +-30 % total gate already ran in
    ``make_manifest``.)
    """
    new = {s.slug: s.items for s in manifest.stores}
    old = {s.slug: s.items for s in previous.stores} if previous else {}
    problems = []
    for slug in sorted(set(new) | set(old)):
        p, n = old.get(slug, 0), new.get(slug, 0)
        if p > 0 and n == 0:
            problems.append(f"{slug}: {p} -> 0")
        elif p >= STORE_DROP_MIN_ITEMS and n < p * (1 - STORE_DROP_LIMIT):
            problems.append(f"{slug}: {p} -> {n}")
    if problems:
        raise RefreshError(
            "validate", "store_count_drop", "; ".join(problems[:10]), EXIT_VALIDATE
        )
    return {s: {"previous": old.get(s), "new": new.get(s)} for s in sorted(set(new) | set(old))}


def previous_build(
    store: rs.BuildStore, work: Path
) -> tuple[str | None, BuildManifest | None, Path]:
    """Newest build that has a manifest (its manifest is copied to ``work/previous``)."""
    prev_dir = work / "previous"
    prev_dir.mkdir(parents=True, exist_ok=True)
    for build_id in reversed(store.list_build_ids()):
        if MANIFEST_NAME not in store.list_objects(build_id):
            continue  # half-published build: invisible to the loader, ignored here too
        try:
            raw = store.read_object(build_id, MANIFEST_NAME, MAX_MANIFEST_BYTES)
            (prev_dir / MANIFEST_NAME).write_bytes(raw)
            return build_id, load_manifest(prev_dir), prev_dir
        except (rs.StoreError, BuildValidationError) as exc:
            # Fail closed: a previous build we cannot read means the gates cannot run.
            raise RefreshError("previous", "previous_unreadable", str(exc)[:200], EXIT_VALIDATE) from exc
    return None, None, prev_dir


def run_refresh(
    *,
    work: Path,
    store: rs.BuildStore,
    fetch: Callable[[Any], rf.FetchResult],
    builder: Builder | None,
    publish: bool,
    git_sha: str,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    only: frozenset[str] | None = None,
) -> Summary:
    """Run every stage; never raises for a stage failure (it is in the returned summary)."""
    import make_build_manifest as mbm  # noqa: PLC0415 - needs faiss/pandas

    s = Summary(dry_run=not publish)
    t0 = time.monotonic()

    def stage(name: str) -> None:
        s.stage = name
        s.timings_s[name] = round(time.monotonic() - t0, 2)

    try:
        stage("plan")
        refreshable = refreshable_stores() if only is None else refreshable_stores() & only
        s.refreshable_stores = len(refreshable)
        if not refreshable:
            raise RefreshError("plan", "no_stores", "no refreshable stores selected", EXIT_USAGE)
        work.mkdir(parents=True, exist_ok=True)

        stage("previous")
        prev_id, prev, prev_dir = previous_build(store, work)
        s.previous_build_id = prev_id
        if prev is None:
            s.notes.append("no previous build: count gates skipped (first publish)")

        stage("fetch")
        results = [fetch(STORE_REGISTRY[slug]) for slug in sorted(refreshable)]
        s.stores = [r.summary() for r in results]
        raw_dir = work / "raw"
        fetched = write_raw_csvs(results, raw_dir)
        s.stores = [r.summary() for r in results]
        s.fresh_stores = len(fetched)
        check_fetch_health(results, refreshable)

        stage("build")
        if builder is None:
            raise RefreshError("build", "build_not_configured", "no --build-cmd given", EXIT_BUILD)
        build_id = now_build_id(now())
        if prev_id is not None and build_id <= prev_id:
            raise RefreshError("build", "build_id_not_newer", f"{build_id} <= {prev_id}", EXIT_BUILD)
        s.build_id = build_id
        out_dir = work / "out" / build_id
        out_dir.mkdir(parents=True, exist_ok=False)
        builder(
            BuildContext(raw_dir, prev_dir, out_dir, prev_id, fetched, refreshable - fetched)
        )

        stage("manifest")
        try:
            manifest = mbm.make_manifest(
                out_dir, build_id=build_id, git_sha=git_sha, built_at=now(), previous=prev
            )
        except mbm.Refused as exc:
            raise RefreshError("manifest", "manifest_refused", str(exc)[:300], EXIT_VALIDATE) from exc
        fetched_at = {r.slug: r.fetched_at for r in results if r.slug in fetched}
        old_at = {e.slug: e.source_updated_at for e in (prev.stores if prev else [])}
        for entry in manifest.stores:
            entry.source_updated_at = fetched_at.get(entry.slug) or old_at.get(entry.slug)
        write_manifest(out_dir, manifest)
        s.total_items = manifest.total_items
        s.count_delta_pct = manifest.validation.count_delta_pct

        stage("validate")
        s.store_counts = validate_store_counts(manifest, prev)

        stage("publish")
        if publish:
            try:
                store.create_build(out_dir, build_id)
            except rs.StoreError as exc:
                raise RefreshError("publish", "publish_failed", str(exc)[:300], EXIT_PUBLISH) from exc
            s.published = True
        else:
            s.notes.append("dry run: nothing uploaded, no rollout")
        s.ok, s.exit_code = True, EXIT_OK
    except RefreshError as exc:
        s.exit_code = exc.exit_code
        s.failure = {"stage": exc.stage, "code": exc.code, "message": exc.message}
    except Exception as exc:  # noqa: BLE001 - the summary must exist even for a bug
        s.exit_code = EXIT_INTERNAL
        s.failure = {"stage": s.stage, "code": "internal", "message": type(exc).__name__}
    s.timings_s["total"] = round(time.monotonic() - t0, 2)
    return s


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    """CLI entry point; returns the exit code and always writes the summary when asked."""
    env = os.environ if environ is None else environ
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, required=True)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--bucket", help="GCS bucket name (production store)")
    where.add_argument("--local-store", type=Path, help="directory laid out like the bucket")
    ap.add_argument("--publish", action="store_true", help="really publish (default: dry run)")
    ap.add_argument("--build-cmd", default=None)
    ap.add_argument("--feed-fixtures", type=Path, default=None, help="no network: read feeds here")
    ap.add_argument("--stores", default=None, help="comma-separated subset (debug)")
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args(argv)

    git_sha = args.git_sha or env.get("GITHUB_SHA") or env.get("GIT_SHA") or ""
    if args.publish and args.bucket and env.get("GITHUB_REF") != "refs/heads/master":
        print("REFUSED: --publish to a bucket only runs on refs/heads/master", file=sys.stderr)
        return EXIT_USAGE
    if not git_sha.strip():
        print("REFUSED: no git sha (--git-sha or GITHUB_SHA)", file=sys.stderr)
        return EXIT_USAGE

    store: rs.BuildStore = (
        rs.GcsBuildStore(args.bucket) if args.bucket else rs.LocalBuildStore(args.local_store)
    )
    if not args.publish:
        store = rs.ReadOnlyStore(store)
    fetch = (
        fixture_fetcher(args.feed_fixtures)
        if args.feed_fixtures
        else rf.StoreFetcher().fetch_store
    )
    only = frozenset(args.stores.split(",")) if args.stores else None
    summary = run_refresh(
        work=args.work_dir,
        store=store,
        fetch=fetch,
        builder=CommandBuilder(args.build_cmd) if args.build_cmd else None,
        publish=args.publish,
        git_sha=git_sha,
        only=only,
    )
    text = json.dumps(summary.__dict__, indent=2, sort_keys=True)
    if args.summary_out:
        args.summary_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
