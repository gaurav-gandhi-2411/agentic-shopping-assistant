"""UTC day rollover of the in-memory daily-cost accumulator (api/demo/guards.py).

The clock is faked through ``guards._today_utc`` (never the process TZ).
"""

from __future__ import annotations

import datetime
import logging
import threading
from typing import Any

import pytest

import api.demo.guards as guards

DAY1 = "2030-01-01"
DAY2 = "2030-01-02"


class _Result:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row


class _Conn:
    def __init__(self, eng: _Engine) -> None:
        self._eng = eng

    def execute(self, stmt: Any, params: dict[str, Any]) -> _Result:
        self._eng.params.append(params)
        sql = str(stmt)
        if sql.lstrip().upper().startswith("SELECT"):
            if "request_count" in sql:
                return _Result((self._eng.request_count,))
            return _Result((self._eng.db_cost,) if self._eng.db_cost is not None else None)
        return _Result(None)


class _Engine:
    """Minimal fake: records bind params; returns canned SELECT rows."""

    def __init__(self, db_cost: float | None = None, request_count: int = 0) -> None:
        self.db_cost = db_cost
        self.request_count = request_count
        self.params: list[dict[str, Any]] = []

    def begin(self) -> Any:
        eng = self

        class _Ctx:
            def __enter__(self) -> _Conn:
                return _Conn(eng)

            def __exit__(self, *a: Any) -> None:
                return None

        return _Ctx()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    state = {"day": DAY1}
    monkeypatch.setattr(guards, "_today_utc", lambda: state["day"])
    monkeypatch.setattr(guards, "_daily_cost_accumulated", 0.0)
    monkeypatch.setattr(guards, "_current_day", DAY1)
    monkeypatch.setattr(guards, "_reload_pending", False)
    return state


def test_crossing_midnight_resets_accumulator(clock: dict[str, str]) -> None:
    guards._daily_cost_accumulated = 0.30
    clock["day"] = DAY2
    assert guards.check_daily_cost("b") is True
    assert guards._daily_cost_accumulated == 0.0
    assert guards._current_day == DAY2


def test_yesterday_over_cap_does_not_block_today(clock: dict[str, str]) -> None:
    guards._daily_cost_accumulated = 0.99
    assert guards.check_daily_cost("b") is False  # still day 1: blocked
    clock["day"] = DAY2
    assert guards.check_daily_cost("b") is True


def test_yesterdays_spend_does_not_count_toward_todays_cap(clock: dict[str, str]) -> None:
    guards._daily_cost_accumulated = 0.45
    clock["day"] = DAY2
    eng: Any = _Engine(db_cost=None)  # no row today beyond what record_cost writes
    guards.record_cost("b", 0.10, eng)
    # 0.45 + 0.10 would exceed the 0.50 cap if yesterday leaked into today
    assert guards._daily_cost_accumulated == pytest.approx(0.10)
    assert guards.check_daily_cost("b") is True


def test_rollover_reloads_todays_db_total_after_record_cost(clock: dict[str, str]) -> None:
    guards._daily_cost_accumulated = 0.45
    clock["day"] = DAY2
    eng: Any = _Engine(db_cost=0.55)  # other instances already spent 0.55 today (incl. this call)
    guards.record_cost("b", 0.10, eng)
    assert guards._daily_cost_accumulated == pytest.approx(0.55)
    assert guards.check_daily_cost("b") is False
    assert guards._reload_pending is False


def test_record_path_alone_rolls_over(clock: dict[str, str]) -> None:
    guards._daily_cost_accumulated = 0.45
    clock["day"] = DAY2
    guards.record_cost("b", 0.05, None)  # engine failure path still rolls the day
    assert guards._current_day == DAY2
    assert guards._daily_cost_accumulated == pytest.approx(0.05)


def test_no_double_reset_under_threads(
    clock: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    guards._daily_cost_accumulated = 0.30
    clock["day"] = DAY2
    caplog.set_level(logging.INFO, logger=guards.logger.name)
    barrier = threading.Barrier(16)

    def worker() -> None:
        barrier.wait()
        guards.check_daily_cost("b")

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rolled = [r for r in caplog.records if "rolled over" in r.getMessage()]
    assert len(rolled) == 1
    assert guards._daily_cost_accumulated == 0.0


def test_spend_after_rollover_is_not_wiped_by_later_checks(clock: dict[str, str]) -> None:
    guards._daily_cost_accumulated = 0.30
    clock["day"] = DAY2
    guards.check_daily_cost("b")
    guards.record_cost("b", 0.20, None)
    guards.check_daily_cost("b")
    assert guards._daily_cost_accumulated == pytest.approx(0.20)


def test_day_key_is_the_utc_clock_in_every_db_path(clock: dict[str, str]) -> None:
    clock["day"] = "2031-07-04"
    eng: Any = _Engine(db_cost=0.0, request_count=0)
    guards.check_daily_cap("b", eng)
    guards.record_request("b", eng)
    guards.record_cost("b", 0.01, eng)
    guards.init_demo_guards(eng, "b")
    assert eng.params, "no DB call was made"
    assert {p["date"] for p in eng.params} == {"2031-07-04"}


def test_today_utc_is_utc_not_local() -> None:
    before = datetime.datetime.now(datetime.UTC).date().isoformat()
    got = guards._today_utc()
    after = datetime.datetime.now(datetime.UTC).date().isoformat()
    assert got in {before, after}
