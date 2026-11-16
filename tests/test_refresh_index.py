"""scripts/ops/refresh_index.py: stages, gates, dry run, create-only publish, exit codes.

Hermetic: tiny synthetic builds (real FAISS + BM25, hashing embedder) in ``tmp_path``, a local
directory store standing in for the bucket, feeds from in-memory fixtures. No network, no cloud.
"""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import refresh_feed as rf  # noqa: E402
import refresh_index as ri  # noqa: E402
import refresh_store as rs  # noqa: E402

from src.config.store_registry import STORE_REGISTRY, StoreRecord, refreshable_stores  # noqa: E402
from src.retrieval.build_manifest import load_manifest  # noqa: E402
from tests._index_fixtures import make_rows, write_build, write_index_files  # noqa: E402

PREV_ID = "20261115T020000Z"
NOW = datetime(2026, 11, 16, 2, 0, tzinfo=timezone.utc)
NEW_ID = "20261116T020000Z"
REFRESHABLE = sorted(refreshable_stores())
N = 25


def _prev_counts() -> dict[str, int]:
    return {**dict.fromkeys(REFRESHABLE, N), "myntra": 10}  # myntra: indexed, never refreshed


@pytest.fixture()
def bucket(tmp_path: Path) -> Path:
    root = tmp_path / "bucket"
    write_build(root, PREV_ID, _prev_counts(), publish_pointer=False)
    return root


def _products(n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "title": f"Kurta {i}",
            "product_type": "Kurta",
            "vendor": "v",
            "body_html": "<p>x</p>",
            "variants": [{"price": "999.0", "available": True}],
            "images": [{"src": "https://cdn.example/i.jpg"}],
            "handle": f"h{i}",
        }
        for i in range(n)
    ]


def _feeds(tmp_path: Path, counts: dict[str, int] | None = None, failing: dict[str, str] | None = None):
    d = tmp_path / "feeds"
    d.mkdir(exist_ok=True)
    for slug in REFRESHABLE:
        status = (failing or {}).get(slug, rf.OK)
        n = (counts or {}).get(slug, N)
        body = {"status": status, "products": _products(n), "fetched_at": "2026-11-16T01:50:00+00:00"}
        (d / f"{slug}.json").write_text(json.dumps(body), encoding="utf-8")
    return ri.fixture_fetcher(d)


def fake_builder(ctx: ri.BuildContext) -> None:
    """Fresh stores from the fetched CSVs; every other store carried over from the previous build."""
    counts: dict[str, int] = {}
    if ctx.previous_build_id:
        counts = {s.slug: s.items for s in load_manifest(ctx.previous_dir).stores}
    for slug in ctx.fetched:
        counts[slug] = len(pd.read_csv(ctx.raw_dir / slug / "products.csv"))
    write_index_files(make_rows(counts), ctx.out_dir)


def _run(tmp_path: Path, bucket: Path, fetch: Any, *, publish: bool = False, builder: Any = fake_builder):
    store: rs.BuildStore = rs.LocalBuildStore(bucket)
    if not publish:
        store = rs.ReadOnlyStore(store)
    return ri.run_refresh(
        work=tmp_path / "work",
        store=store,
        fetch=fetch,
        builder=builder,
        publish=publish,
        git_sha="deadbeef",
        now=lambda: NOW,
    )


def _ids(bucket: Path) -> list[str]:
    return rs.LocalBuildStore(bucket).list_build_ids()


def test_dry_run_does_everything_except_publish(tmp_path: Path, bucket: Path) -> None:
    s = _run(tmp_path, bucket, _feeds(tmp_path))
    assert (s.ok, s.exit_code, s.dry_run, s.published) == (True, 0, True, False)
    assert s.build_id == NEW_ID and s.previous_build_id == PREV_ID
    assert s.fresh_stores == s.refreshable_stores == len(REFRESHABLE) == 38
    assert s.total_items == 38 * N + 10 and s.count_delta_pct == 0.0
    assert _ids(bucket) == [PREV_ID]  # nothing uploaded
    assert (tmp_path / "work" / "out" / NEW_ID / "build_manifest.json").is_file()  # built locally


def test_publish_creates_the_new_build_and_never_touches_the_previous_one(
    tmp_path: Path, bucket: Path
) -> None:
    before = rs.LocalBuildStore(bucket).list_objects(PREV_ID)
    s = _run(tmp_path, bucket, _feeds(tmp_path), publish=True)
    assert s.ok and s.published and s.exit_code == 0
    assert _ids(bucket) == [PREV_ID, NEW_ID]
    assert rs.LocalBuildStore(bucket).list_objects(PREV_ID) == before
    manifest = load_manifest(bucket / "builds" / NEW_ID)
    assert manifest.previous_build_id == PREV_ID and manifest.git_sha == "deadbeef"
    fresh = {e.slug: e.source_updated_at for e in manifest.stores}
    assert fresh["libas"] == "2026-11-16T01:50:00+00:00"


def test_failed_stores_are_carried_over_and_the_summary_names_them(
    tmp_path: Path, bucket: Path
) -> None:
    failing = {slug: rf.CHALLENGE for slug in REFRESHABLE[:10]}
    s = _run(tmp_path, bucket, _feeds(tmp_path, failing=failing), publish=True)
    assert s.ok and s.fresh_stores == 28
    assert {r["slug"] for r in s.stores if r["status"] == rf.CHALLENGE} == set(REFRESHABLE[:10])
    assert s.store_counts[REFRESHABLE[0]] == {"previous": N, "new": N}  # carried, not lost


def test_too_few_fresh_stores_fails_with_exit_3_and_changes_nothing(
    tmp_path: Path, bucket: Path
) -> None:
    failing = {slug: rf.RATE_LIMITED for slug in REFRESHABLE[:20]}  # 18/38 fresh < 50%
    s = _run(tmp_path, bucket, _feeds(tmp_path, failing=failing), publish=True)
    assert (s.ok, s.exit_code, s.published) == (False, ri.EXIT_FETCH, False)
    assert s.failure and s.failure["code"] == "too_few_fresh_stores"
    assert _ids(bucket) == [PREV_ID]


def test_per_store_count_drop_fails_validation_and_nothing_is_published(
    tmp_path: Path, bucket: Path
) -> None:
    s = _run(tmp_path, bucket, _feeds(tmp_path, counts={"libas": 5}), publish=True)
    assert (s.exit_code, s.published) == (ri.EXIT_VALIDATE, False)
    assert s.failure and s.failure["code"] == "store_count_drop" and "libas: 25 -> 5" in s.failure["message"]
    assert _ids(bucket) == [PREV_ID]


def test_total_count_gate_of_the_existing_verifier_applies(tmp_path: Path, bucket: Path) -> None:
    s = _run(tmp_path, bucket, _feeds(tmp_path, counts=dict.fromkeys(REFRESHABLE, 10)), publish=True)
    assert s.exit_code == ri.EXIT_VALIDATE and s.failure["code"] == "manifest_refused"  # type: ignore[index]
    assert "count_gate" in s.failure["message"]  # type: ignore[index]
    assert _ids(bucket) == [PREV_ID]


def test_no_builder_configured_refuses_to_publish(tmp_path: Path, bucket: Path) -> None:
    s = _run(tmp_path, bucket, _feeds(tmp_path), publish=True, builder=None)
    assert s.exit_code == ri.EXIT_BUILD and s.failure["code"] == "build_not_configured"  # type: ignore[index]
    assert _ids(bucket) == [PREV_ID]


def test_command_builder_failure_is_a_build_failure_not_a_crash(tmp_path: Path, bucket: Path) -> None:
    cmd = shlex.join([sys.executable, "-c", "raise SystemExit(3)"])
    s = _run(tmp_path, bucket, _feeds(tmp_path), publish=True, builder=ri.CommandBuilder(cmd))
    assert s.exit_code == ri.EXIT_BUILD and s.failure["code"] == "build_failed"  # type: ignore[index]
    assert "exit 3" in s.failure["message"]  # type: ignore[index]
    assert _ids(bucket) == [PREV_ID]


def test_command_builder_substitutes_placeholders_without_a_shell(tmp_path: Path) -> None:
    out = tmp_path / "o"
    out.mkdir()
    code = "import sys, pathlib; pathlib.Path(sys.argv[1], 'x.txt').write_text(sys.argv[2])"
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} {{out_dir}} {{previous_build_id}}"
    ctx = ri.BuildContext(tmp_path, tmp_path, out, "20261115T020000Z", frozenset(), frozenset())
    ri.CommandBuilder(cmd)(ctx)
    assert (out / "x.txt").read_text() == "20261115T020000Z"


def test_a_build_id_not_newer_than_the_previous_one_is_refused(tmp_path: Path, bucket: Path) -> None:
    s = ri.run_refresh(
        work=tmp_path / "w",
        store=rs.ReadOnlyStore(rs.LocalBuildStore(bucket)),
        fetch=_feeds(tmp_path),
        builder=fake_builder,
        publish=False,
        git_sha="x",
        now=lambda: datetime(2026, 11, 15, 2, 0, tzinfo=timezone.utc),
    )
    assert s.exit_code == ri.EXIT_BUILD and s.failure["code"] == "build_id_not_newer"  # type: ignore[index]


def test_publish_onto_an_existing_prefix_is_refused_and_overwrites_nothing(
    tmp_path: Path, bucket: Path
) -> None:
    # A half-published build (no manifest) already sits at the id this run would use.
    stray = bucket / "builds" / NEW_ID
    stray.mkdir(parents=True)
    (stray / "dense.faiss").write_bytes(b"theirs")
    s = _run(tmp_path, bucket, _feeds(tmp_path), publish=True)
    assert (s.exit_code, s.published) == (ri.EXIT_PUBLISH, False)
    assert s.failure["code"] == "publish_failed"  # type: ignore[index]
    assert (stray / "dense.faiss").read_bytes() == b"theirs"


def test_unreadable_previous_manifest_fails_closed(tmp_path: Path, bucket: Path) -> None:
    (bucket / "builds" / PREV_ID / "build_manifest.json").write_text("{not json", encoding="utf-8")
    s = _run(tmp_path, bucket, _feeds(tmp_path))
    assert s.exit_code == ri.EXIT_VALIDATE and s.failure["code"] == "previous_unreadable"  # type: ignore[index]


def test_first_publish_with_no_previous_build_skips_count_gates(tmp_path: Path) -> None:
    empty = tmp_path / "empty-bucket"
    s = _run(tmp_path, empty, _feeds(tmp_path), publish=True)
    assert s.ok and s.previous_build_id is None and s.count_delta_pct is None
    assert any("first publish" in n for n in s.notes)


def test_opted_out_store_is_never_fetched(
    tmp_path: Path, bucket: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opted = frozenset(REFRESHABLE) - {"libas"}
    monkeypatch.setattr(ri, "refreshable_stores", lambda: opted)
    seen: list[str] = []
    inner = _feeds(tmp_path)

    def spy(rec: StoreRecord) -> rf.FetchResult:
        seen.append(rec.slug)
        return inner(rec)

    s = _run(tmp_path, bucket, spy)
    assert "libas" not in seen and s.refreshable_stores == 37 and s.ok
    assert s.store_counts["libas"] == {"previous": N, "new": N}  # carried over, still served


def test_validate_store_counts_boundaries() -> None:
    from src.retrieval.build_manifest import StoreEntry

    def load_like(counts: dict[str, int]):
        class _M:
            stores = [StoreEntry(slug=k, items=v) for k, v in counts.items()]

        return _M()

    prev = load_like({"libas": 100, "tiny": 10, "gone": 5})
    ri.validate_store_counts(load_like({"libas": 50, "tiny": 1, "gone": 1}), prev)  # exactly -50%: ok
    with pytest.raises(ri.RefreshError):
        ri.validate_store_counts(load_like({"libas": 49, "tiny": 10, "gone": 1}), prev)
    with pytest.raises(ri.RefreshError):
        ri.validate_store_counts(load_like({"libas": 100, "tiny": 10}), prev)  # 'gone' vanished


def test_cli_refuses_a_bucket_publish_outside_master_and_writes_no_upload(tmp_path: Path) -> None:
    rc = ri.main(
        ["--work-dir", str(tmp_path / "w"), "--bucket", "b", "--publish", "--git-sha", "x"],
        environ={"GITHUB_REF": "refs/heads/feature"},
    )
    assert rc == ri.EXIT_USAGE


def test_cli_dry_run_is_default_writes_summary_json_even_on_failure(
    tmp_path: Path, bucket: Path
) -> None:
    out = tmp_path / "summary.json"
    rc = ri.main(
        [
            "--work-dir", str(tmp_path / "w"),
            "--local-store", str(bucket),
            "--feed-fixtures", str(tmp_path / "no-such-dir"),
            "--git-sha", "x",
            "--summary-out", str(out),
        ],
        environ={},
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert rc == ri.EXIT_FETCH and data["ok"] is False and data["dry_run"] is True
    assert data["failure"]["stage"] == "fetch" and data["published"] is False
    assert _ids(bucket) == [PREV_ID]


def test_cli_requires_a_git_sha(tmp_path: Path, bucket: Path) -> None:
    rc = ri.main(["--work-dir", str(tmp_path), "--local-store", str(bucket)], environ={})
    assert rc == ri.EXIT_USAGE


def test_registry_has_the_38_refreshable_stores_the_tests_assume() -> None:
    assert len(REFRESHABLE) == 38 and all(STORE_REGISTRY[s].refresh_enabled for s in REFRESHABLE)
