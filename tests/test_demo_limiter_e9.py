"""E9: the DB and in-memory limiters must never count separately.

Prod failure: psycopg3 auto-prepare against the pgbouncer pooler made the DB limiter error
intermittently; each error switched to a separate in-memory window, so one client got `limit`
admits from the DB window and another `limit` from memory (~2x, and a reset on every flap).
No network, no real DB: a fake engine models demo_rate_limits / demo_daily_stats.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import defaultdict
from typing import Any

import pytest
from starlette.requests import Request

import api.demo.guards as guards
from api.demo.client_ip import get_client_ip

LIMIT = 5


class FakeDb:
    """Minimal pgbouncer-ish engine: counts per ip_hash and per day; can be switched down."""

    def __init__(self) -> None:
        self.up = True
        self.ip_counts: dict[str, int] = defaultdict(int)
        self.daily = 0

    def begin(self) -> FakeDb:
        if not self.up:
            raise ConnectionError("prepared statement already exists")
        return self

    def __enter__(self) -> FakeDb:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        sql = str(stmt)
        params = params or {}
        res = type("R", (), {})()
        if sql.startswith("SELECT count FROM demo_rate_limits"):
            n = self.ip_counts.get(params["ih"])
            res.fetchone = lambda: (n,) if n is not None else None
        elif sql.startswith("SELECT request_count"):
            res.fetchone = lambda: (self.daily,)
        elif "INTO demo_rate_limits" in sql:
            self.ip_counts[params["ih"]] += 1
        elif "INTO demo_daily_stats" in sql:
            self.daily += params["req"]
        return res


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_PER_IP_HOUR_LIMIT", str(LIMIT))
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "1000")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setattr(guards, "_DB_RETRY_S", 0.0)  # retry the DB on every call


def _req(xff: str) -> Request:
    return Request(
        {"type": "http", "headers": [(b"x-forwarded-for", xff.encode())], "client": ("10.8.0.1", 1)}
    )


def _admits(db: FakeDb, ip: str, n: int) -> int:
    return sum(guards.check_ip_rate_limit(ip, "b", db)[0] for _ in range(n))


def test_db_flap_mid_window_never_exceeds_limit() -> None:
    db = FakeDb()
    admitted = 0
    for up in (True, False, True, False, True):  # errors, recovery, errors, recovery
        db.up = up
        admitted += _admits(db, "203.0.113.9", 3)
    assert admitted == LIMIT


def test_spoofed_multi_value_xff_maps_to_one_key_and_is_limited() -> None:
    db = FakeDb()
    admitted = 0
    for i in range(3 * LIMIT):
        db.up = i % 3 != 0  # flapping DB as well
        ip = get_client_ip(_req(f"9.9.{i}.1, 8.8.{i}.2, 203.0.113.9"))
        assert ip == "203.0.113.9"
        admitted += guards.check_ip_rate_limit(ip, "b", db)[0]
    assert admitted == LIMIT


def test_db_and_fallback_paths_key_on_the_same_derived_ip() -> None:
    db = FakeDb()
    ip = get_client_ip(_req("6.6.6.6, 203.0.113.9:4000"))
    guards.check_ip_rate_limit(ip, "b", db)  # DB path
    db.up = False
    guards.check_ip_rate_limit(ip, "b", db)  # fallback path
    assert set(db.ip_counts) == {guards._ip_hash("203.0.113.9")}
    assert list(guards._fallback._windows) == ["203.0.113.9"]
    assert len(guards._fallback._windows["203.0.113.9"]) == 2  # both paths counted in memory


def test_distinct_clients_are_independent() -> None:
    db = FakeDb()
    assert _admits(db, "203.0.113.9", LIMIT + 3) == LIMIT
    assert _admits(db, "203.0.113.10", LIMIT) == LIMIT
    db.up = False
    assert _admits(db, "203.0.113.11", LIMIT + 2) == LIMIT


def test_daily_cap_db_flap_does_not_reset_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_DAILY_REQUEST_CAP", "6")
    db = FakeDb()
    served = 0
    for up in (True, False, True, False, True):
        db.up = up
        for _ in range(3):
            if guards.check_daily_cap("b", db):
                guards.record_request("b", db)
                served += 1
    assert served == 6


def test_restart_with_healthy_db_still_enforces_from_db_count() -> None:
    db = FakeDb()
    assert _admits(db, "203.0.113.9", LIMIT) == LIMIT
    guards.reset_fallback_state()  # new instance: empty memory
    assert guards.check_ip_rate_limit("203.0.113.9", "b", db)[0] is False
    db.daily = 1000
    guards.reset_fallback_state()
    assert guards.check_daily_cap("b", db) is False


def test_concurrent_requests_same_ip_never_exceed_limit() -> None:
    """8 threads released together at limit 5: the memory admit is atomic, so <= LIMIT admits."""
    # Tiny switch interval forces thread interleaving inside the check-then-append, so a
    # missing lock in FallbackLimiter.admit_ip shows up as > LIMIT admits (mutation-checked).
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    n_threads, trials = 8, 150
    try:
        for trial in range(trials):
            guards.reset_fallback_state()
            db = FakeDb()
            barrier = threading.Barrier(n_threads)
            results: list[bool] = []

            def worker(db: FakeDb = db, barrier: threading.Barrier = barrier) -> None:
                barrier.wait()
                results.append(guards.check_ip_rate_limit("203.0.113.9", "b", db)[0])

            threads = [threading.Thread(target=worker) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len(results) == n_threads
            assert sum(results) <= LIMIT, f"trial {trial}: {sum(results)} admits > {LIMIT}"
            assert len(guards._fallback._windows["203.0.113.9"]) <= LIMIT
    finally:
        sys.setswitchinterval(old)


class UpsertFailsDb(FakeDb):
    """Peek (SELECT) succeeds, but the INSERT ... ON CONFLICT upsert raises."""

    fail_upsert = True

    def execute(self, stmt: Any, params: dict[str, Any] | None = None) -> Any:
        if self.fail_upsert and "INTO demo_rate_limits" in str(stmt):
            raise ConnectionError("prepared statement already exists")
        return super().execute(stmt, params)


def test_db_upsert_failure_after_memory_admit_counts_once_and_recovers_with_max(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ip = "203.0.113.9"
    db = UpsertFailsDb()
    caplog.set_level(logging.INFO, logger="api.demo.guards")

    # Memory admitted, then the DB upsert raised: stays admitted, exactly once.
    assert guards.check_ip_rate_limit(ip, "b", db) == (True, 0)
    assert len(guards._fallback._windows[ip]) == 1
    assert db.ip_counts.get(guards._ip_hash(ip), 0) == 0  # the DB never counted it
    assert guards._health.degraded is True
    transitions = [r for r in caplog.records if "demo_limiter_transition state=memory" in r.message]
    assert len(transitions) == 1

    # DB recovers; another instance already counted 3 for this IP. max(DB=3, memory=1) < 5.
    db.fail_upsert = False
    db.ip_counts[guards._ip_hash(ip)] = 3
    later = [guards.check_ip_rate_limit(ip, "b", db)[0] for _ in range(4)]
    # DB count 3 -> admit (DB 4), 4 -> admit (DB 5), then DB 5 >= limit blocks: no reset, no
    # double count of the first request (memory window holds exactly the 3 admitted so far).
    assert later == [True, True, False, False]
    assert len(guards._fallback._windows[ip]) == 3
    assert guards._health.degraded is False
    recovered = [r for r in caplog.records if "reason=recovered" in r.message]
    assert len(recovered) == 1
    assert len([r for r in caplog.records if "state=memory" in r.message]) == 1
