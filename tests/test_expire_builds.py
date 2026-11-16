"""scripts/ops/expire_builds.py: selection, references, and a 10-day lifecycle simulation.

Hermetic: pure functions, a local directory store and an in-process model of the bucket lifecycle
rule ("delete objects under builds/ with daysSinceCustomTime >= 2"). No cloud.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import expire_builds as eb  # noqa: E402
import refresh_store as rs  # noqa: E402


def bid(day: int) -> str:
    """Build id for day ``day`` of a fictional November (02:00 UTC)."""
    return f"202611{day:02d}T020000Z"


# --- select_builds_to_expire ---------------------------------------------------------------


def test_keeps_the_three_newest_and_expires_the_rest_ascending() -> None:
    builds = [bid(d) for d in (5, 1, 4, 2, 3, 6)]  # deliberately unsorted
    assert eb.select_builds_to_expire(builds, set()) == [bid(1), bid(2), bid(3)]


def test_fewer_than_keep_builds_expires_nothing() -> None:
    assert eb.select_builds_to_expire([bid(1), bid(2), bid(3)], set()) == []
    assert eb.select_builds_to_expire([], set()) == []


def test_a_referenced_build_is_never_expired_even_when_it_is_not_among_the_newest() -> None:
    builds = [bid(d) for d in range(1, 9)]
    assert eb.select_builds_to_expire(builds, {bid(2)}) == [bid(1), bid(3), bid(4), bid(5)]


def test_references_to_unknown_builds_are_ignored_and_invalid_ids_are_never_returned() -> None:
    builds = [bid(1), bid(2), bid(3), bid(4), "not-a-build", "../x", "20261105"]
    assert eb.select_builds_to_expire(builds, {"20300101T000000Z"}) == [bid(1)]


def test_keep_below_one_is_refused() -> None:
    with pytest.raises(ValueError):
        eb.select_builds_to_expire([bid(1)], set(), keep=0)


def test_duplicates_in_the_listing_do_not_shift_the_newest_window() -> None:
    assert eb.select_builds_to_expire([bid(1), bid(2), bid(3), bid(3), bid(4)], set()) == [bid(1)]


# --- referenced_build_ids ------------------------------------------------------------------


def rev(pin: str | None) -> dict[str, Any]:
    env = [{"name": "OTHER", "value": "x"}] + (
        [{"name": "INDEX_BUILD_ID", "value": pin}] if pin else []
    )
    return {"spec": {"containers": [{"env": env}]}}


def service(traffic: list[dict[str, Any]], template_pin: str | None = None) -> dict[str, Any]:
    env = [{"name": "INDEX_BUILD_ID", "value": template_pin}] if template_pin else []
    return {
        "status": {"traffic": traffic, "latestReadyRevisionName": "r-new"},
        "spec": {"template": {"spec": {"containers": [{"env": env}]}}},
    }


def test_traffic_tagged_template_and_other_revisions_are_all_referenced() -> None:
    svc = service(
        [
            {"revisionName": "r-serving", "percent": 100},
            {"revisionName": "r-tagged", "percent": 0, "tag": "ix-canary"},
            {"revisionName": "r-idle", "percent": 0},
        ],
        template_pin=bid(7),
    )
    revs = {
        "r-serving": rev(bid(6)),
        "r-tagged": rev(bid(2)),
        "r-idle": rev(bid(1)),  # no traffic, no tag: skipped as a traffic entry...
        "r-rollback": rev(bid(5)),  # ...but any supplied revision still protects its build
        "r-legacy": rev(None),
    }
    assert eb.referenced_build_ids(svc, revs) == {bid(6), bid(2), bid(1), bid(5), bid(7)}


def test_latest_revision_traffic_entry_resolves_through_latest_ready() -> None:
    svc = service([{"latestRevision": True, "percent": 100}])
    assert eb.referenced_build_ids(svc, {"r-new": rev(bid(3))}) == {bid(3)}


def test_missing_json_for_a_traffic_bearing_revision_fails_closed() -> None:
    svc = service([{"revisionName": "r-serving", "percent": 100}])
    with pytest.raises(eb.InputError):
        eb.referenced_build_ids(svc, {})
    tagged = service([{"revisionName": "r-t", "percent": 0, "tag": "t"}])
    with pytest.raises(eb.InputError):
        eb.referenced_build_ids(tagged, {})
    with pytest.raises(eb.InputError):
        eb.referenced_build_ids({"status": {}}, {})


def test_a_malformed_pin_value_references_nothing() -> None:
    assert eb.pin_of(rev("../etc")["spec"]) is None


def test_traffic_referenced_old_build_survives_the_plan(tmp_path: Path) -> None:
    root = tmp_path / "bucket"
    store = rs.LocalBuildStore(root)
    for d in range(1, 8):
        src = tmp_path / f"s{d}"
        src.mkdir()
        (src / "dense.faiss").write_bytes(b"x")
        (src / rs.MANIFEST_NAME).write_text("{}", encoding="utf-8")
        store.create_build(src, bid(d))
    half = root / "builds" / bid(9)  # half-published: no manifest, must not count as a newest build
    half.mkdir()
    (half / "dense.faiss").write_bytes(b"x")
    svc = service([{"revisionName": "r-old", "percent": 100}])
    result = eb.plan(store, svc, {"r-old": rev(bid(1))})
    assert result["expire"] == [bid(2), bid(3), bid(4)]
    assert result["keep_newest"] == [bid(5), bid(6), bid(7)] and bid(9) not in result["builds"]


# --- lifecycle simulation ------------------------------------------------------------------


class SimBucket:
    """Objects under builds/ with the lifecycle rule: delete iff customTime set and age >= 2 d."""

    def __init__(self) -> None:
        self.custom_time: dict[str, datetime | None] = {}

    def publish(self, build_id: str) -> None:
        assert build_id not in self.custom_time, "create-only"
        self.custom_time[build_id] = None

    def mark(self, build_ids: list[str], now: datetime) -> None:
        for b in build_ids:
            assert b in self.custom_time
            self.custom_time[b] = now

    def lifecycle(self, now: datetime) -> None:
        for b, ct in list(self.custom_time.items()):
            if ct is not None and now - ct >= timedelta(days=2):
                del self.custom_time[b]


class SimWorld:
    """Daily job: refresh -> rollout -> (only after a good rollout) retention -> lifecycle."""

    def __init__(self) -> None:
        self.bucket = SimBucket()
        self.serving: str | None = None
        self.previous: str | None = None
        self.start = datetime(2026, 11, 1, 2, tzinfo=timezone.utc)

    def day(self, d: int, *, refresh_ok: bool, rollout_ok: bool = True) -> None:
        now = self.start + timedelta(days=d - 1)
        if refresh_ok:
            self.bucket.publish(bid(d))
            if rollout_ok:
                self.previous, self.serving = self.serving, bid(d)
                referenced = {b for b in (self.serving, self.previous) if b}
                expire = eb.select_builds_to_expire(list(self.bucket.custom_time), referenced)
                marked = {b for b, ct in self.bucket.custom_time.items() if ct is not None}
                self.bucket.mark(eb.without_marked(expire, marked), now)
        self.bucket.lifecycle(now)

    def newest(self, n: int = 3) -> list[str]:
        return sorted(self.bucket.custom_time)[-n:]


def test_simulation_refresh_fails_ten_days_in_a_row_nothing_is_marked_or_lost() -> None:
    world = SimWorld()
    for d in (1, 2, 3):  # three builds: nothing is old enough to be marked yet
        world.day(d, refresh_ok=True)
    before = dict(world.bucket.custom_time)
    assert all(ct is None for ct in before.values())
    for d in range(4, 14):  # ten failing days: no new builds, retention never runs
        world.day(d, refresh_ok=False)
    assert world.bucket.custom_time == before  # no new build, no customTime set, nothing deleted
    assert world.serving in world.bucket.custom_time
    assert set(world.newest()) == {bid(1), bid(2), bid(3)}
    assert world.serving == bid(3) and world.previous == bid(2)


def test_simulation_failures_after_a_marking_only_lose_what_was_already_marked() -> None:
    world = SimWorld()
    for d in (1, 2, 3, 4):  # day 4 marks bid(1) (older than the 3 newest)
        world.day(d, refresh_ok=True)
    marked = {b for b, ct in world.bucket.custom_time.items() if ct is not None}
    assert marked == {bid(1)}
    for d in range(5, 15):
        world.day(d, refresh_ok=False)
    # no NEW marks during the failing days; the already-marked build aged out as designed
    assert set(world.bucket.custom_time) == {bid(2), bid(3), bid(4)}
    assert all(ct is None for ct in world.bucket.custom_time.values())
    assert world.serving == bid(4) and world.serving in world.bucket.custom_time


def test_simulation_daily_healthy_runs_actually_delete_old_builds_no_remark_loop() -> None:
    # Regression found by this simulation: re-marking an already-marked build every day pushed its
    # expiry back forever, so nothing was ever deleted.
    world = SimWorld()
    for d in range(1, 9):
        world.day(d, refresh_ok=True)
    # b1, b2, b3 aged out (marked days 4, 5, 6, deleted 2 days later); b4, b5 await deletion
    assert set(world.bucket.custom_time) == {bid(d) for d in range(4, 9)}
    assert world.serving == bid(8)


def test_simulation_only_ever_failing_still_keeps_the_pinned_serving_build() -> None:
    world = SimWorld()
    world.bucket.publish(bid(1))
    world.serving = bid(1)  # the one pinned build, nothing else ever published
    for d in range(2, 12):
        world.day(d, refresh_ok=False)
    assert list(world.bucket.custom_time) == [bid(1)]
    assert world.bucket.custom_time[bid(1)] is None


def test_simulation_healthy_week_then_ten_failing_days() -> None:
    world = SimWorld()
    for d in range(1, 8):
        world.day(d, refresh_ok=True)
    assert set(world.bucket.custom_time) == {bid(d) for d in (3, 4, 5, 6, 7)}  # 3, 4 still pending
    for d in range(8, 18):
        world.day(d, refresh_ok=False)
    # the pending ones age out; the serving build and the 3 newest remain
    assert set(world.bucket.custom_time) == {bid(5), bid(6), bid(7)} and world.serving == bid(7)


def test_simulation_builds_publish_but_rollout_keeps_failing_serving_build_survives() -> None:
    world = SimWorld()
    world.day(1, refresh_ok=True)  # serving = bid(1)
    for d in range(2, 12):  # ten days: published every day, every rollout fails -> no retention
        world.day(d, refresh_ok=True, rollout_ok=False)
    assert world.serving == bid(1) and bid(1) in world.bucket.custom_time
    assert all(ct is None for ct in world.bucket.custom_time.values())
    # even a retention run that DID happen would keep it: the serving build is referenced
    later = eb.select_builds_to_expire(list(world.bucket.custom_time), {bid(1)})
    assert bid(1) not in later and len(later) == 11 - 3 - 1


def test_simulation_a_marked_build_is_not_deleted_before_two_days() -> None:
    world = SimWorld()
    for d in range(1, 6):
        world.day(d, refresh_ok=True)
    # day 4 marked bid(1), day 5 marked bid(2); the rule deletes 2 days after the mark
    assert world.bucket.custom_time[bid(2)] is not None
    world.day(6, refresh_ok=True)
    assert bid(2) in world.bucket.custom_time  # 1 day old: still there
    world.day(7, refresh_ok=True)
    assert bid(2) not in world.bucket.custom_time  # 2 days old: gone


# --- CLI -----------------------------------------------------------------------------------


def test_cli_refuses_keep_zero_and_unreadable_inputs(tmp_path: Path) -> None:
    svc = tmp_path / "service.json"
    svc.write_text(json.dumps(service([])), encoding="utf-8")
    revs = tmp_path / "revs"
    revs.mkdir()
    base = ["--bucket", "b", "--service-json", str(svc), "--revisions-dir", str(revs)]
    assert eb.main([*base, "--keep", "0"]) == eb.EXIT_USAGE
    svc.write_text("{nope", encoding="utf-8")
    assert eb.main(base) == eb.EXIT_INPUT
    (revs / "r.json").write_text("[]", encoding="utf-8")
    svc.write_text(json.dumps(service([])), encoding="utf-8")
    assert eb.main(base) == eb.EXIT_INPUT  # a revision file that is not an object
