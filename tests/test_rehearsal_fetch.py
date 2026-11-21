"""scripts/ops/rehearsal_fetch.py: hermetic, every HTTP call is a fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import refresh_feed as rf  # noqa: E402
import rehearsal_fetch as rh  # noqa: E402

ROBOTS = rf.HttpResponse(200, {}, "User-agent: *\nAllow: /\n")


def _net(feed_by_host: dict[str, rf.HttpResponse]) -> tuple[rf.HttpGet, list[str]]:
    calls: list[str] = []

    def get(url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        calls.append(url)
        host = url.split("/")[2]
        return ROBOTS if url.endswith("/robots.txt") else feed_by_host[host]

    return get, calls


def _page() -> rf.HttpResponse:
    products = [
        {"id": 1, "title": "SECRET", "variants": [{"price": "499.00"}]},
        {"id": 2, "title": "SECRET", "variants": [{"price": "1299.50"}]},
    ]
    return rf.HttpResponse(200, {}, json.dumps({"products": products}))


def test_run_counts_only_and_stops_a_429_store_without_followups() -> None:
    d1 = rh.STORE_REGISTRY["libas"].domain
    d2 = rh.STORE_REGISTRY["theamethyststore"].domain
    get, calls = _net({d1: _page(), d2: rf.HttpResponse(429, {"retry-after": "30"}, "")})
    throttle = rf.DomainThrottle(sleep=lambda s: None, global_max_per_second=0.5)
    out = rh.run(["libas", "theamethyststore"], {"libas": 5}, get, throttle)
    by = {s["slug"]: s for s in out["stores"]}
    assert by["libas"]["status"] == rf.OK and by["libas"]["items"] == 2
    assert by["libas"]["price_range"] == [499.0, 1299.5] and by["libas"]["delta_vs_live"] == -3
    assert by["theamethyststore"]["status"] == rf.RATE_LIMITED
    assert by["theamethyststore"]["retry_after"] == "30"
    assert sum(1 for c in calls if d2 in c and "products.json" in c) == 1  # no retry
    assert out["http_status_counts"]["429"] == 1 and out["rate_limited_429"] == [d2]
    assert "SECRET" not in json.dumps(out)


def test_markdown_lists_every_store() -> None:
    get, _ = _net({rh.STORE_REGISTRY["libas"].domain: _page()})
    out = rh.run(["libas"], {}, get, rf.DomainThrottle(sleep=lambda s: None))
    assert "| libas | ok |" in rh.to_markdown(out)
