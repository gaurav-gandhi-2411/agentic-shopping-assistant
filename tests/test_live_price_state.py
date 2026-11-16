"""PoliteState: TTL cache, circuit breaker, per-domain token bucket (fake clock, no I/O)."""
from __future__ import annotations

import urllib.robotparser

from src.pricing.guards import LiveConfig, LiveResult
from src.pricing.state import PoliteState


def _state(**cfg) -> tuple[PoliteState, list[float]]:
    now = [1000.0]
    return PoliteState(LiveConfig(enabled=True, **cfg), clock=lambda: now[0]), now


def test_cache_hit_then_expiry_at_ttl() -> None:
    st, now = _state(cache_ttl_s=3600)
    st.cache_put(("libas", "a"), LiveResult(100.0, True))
    now[0] += 3599
    assert st.cache_get(("libas", "a")) == LiveResult(100.0, True)
    now[0] += 2
    assert st.cache_get(("libas", "a")) is None


def test_cache_is_keyed_by_store_and_handle() -> None:
    st, _ = _state()
    st.cache_put(("libas", "a"), LiveResult(100.0, True))
    assert st.cache_get(("snitch", "a")) is None
    assert st.cache_get(("libas", "b")) is None


def test_cache_evicts_oldest_at_capacity() -> None:
    st = PoliteState(LiveConfig(cache_max_entries=2))
    for h in "abc":
        st.cache_put(("libas", h), LiveResult(1.0, True))
    assert st.cache_get(("libas", "a")) is None
    assert st.cache_get(("libas", "c")) is not None


def test_breaker_opens_after_n_failures_and_closes_after_window() -> None:
    st, now = _state(breaker_failures=3, breaker_open_s=300)
    for _ in range(2):
        st.record("libas", False)
    assert st.breaker_open("libas") is False
    st.record("libas", False)
    assert st.breaker_open("libas") is True
    assert st.breaker_open("snitch") is False
    now[0] += 301
    assert st.breaker_open("libas") is False


def test_success_resets_streak() -> None:
    st, _ = _state(breaker_failures=3)
    st.record("libas", False)
    st.record("libas", False)
    st.record("libas", True)
    st.record("libas", False)
    assert st.breaker_open("libas") is False


def test_hold_opens_breaker_immediately() -> None:
    st, now = _state()
    st.record("libas", False, hold_s=120)
    assert st.breaker_open("libas") is True
    now[0] += 121
    assert st.breaker_open("libas") is False


def test_token_bucket_burst_then_one_per_interval() -> None:
    st, now = _state(burst=2, min_interval_s=1.0)
    assert st.reserve("libas.in", 10) == 0.0
    assert st.reserve("libas.in", 10) == 0.0
    assert st.reserve("libas.in", 10) == 1.0  # third request waits one interval
    assert st.reserve("libas.in", 10) == 2.0  # and the next one two: average stays 1/s
    assert st.reserve("snitch.co.in", 10) == 0.0  # domains are independent
    now[0] += 5
    assert st.reserve("libas.in", 10) == 0.0  # refilled


def test_reserve_returns_none_and_reserves_nothing_when_wait_exceeds_max() -> None:
    st, _ = _state(burst=1, min_interval_s=1.0)
    assert st.reserve("libas.in", 0.5) == 0.0
    assert st.reserve("libas.in", 0.5) is None
    assert st.reserve("libas.in", 5.0) == 1.0  # the refused call did not consume a token


def test_robots_cache_respects_ttl() -> None:
    st, now = _state(cache_ttl_s=100)
    rp = urllib.robotparser.RobotFileParser()
    st.robots_put("libas.in", rp)
    assert st.robots_get("libas.in") is rp
    now[0] += 101
    assert st.robots_get("libas.in") is None
