"""One-off DRY-RUN rehearsal of the daily feed fetch from a CI runner (architect item c).

Fetches the 38 refreshable public store feeds through ``refresh_feed.StoreFetcher`` and writes a
counts-only JSON summary plus a markdown table. It never builds, publishes, authenticates or
touches any cloud service, and the JSON never contains product bodies: per store only the item
count, distinct id count and price range are kept.

Politeness on top of the fetcher: one request per 2 s across ALL stores, <= 1 req/s per domain,
robots.txt, and a 429 stops that store for the run (the ``Retry-After`` value is recorded but not
waited on: the response handed to the fetcher has the header removed, so it takes its "stop the
store" branch instead of its wait-and-retry branch, and no further request goes to that host).

Usage: python scripts/ops/rehearsal_fetch.py --out rehearsal_summary.json [--md-out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import refresh_feed as rf  # noqa: E402

from src.config.store_registry import STORE_REGISTRY, refreshable_stores  # noqa: E402

GLOBAL_MAX_PER_SECOND = 0.5  # architect spec: 1 request per 2 s across all stores
COUNTS_FIXTURE = ROOT / "tests" / "fixtures" / "indexed_stores.json"


def load_baseline(path: Path = COUNTS_FIXTURE) -> dict[str, int]:
    """Per-store item counts of the live index (committed fixture)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["slug"]: int(s["items"]) for s in data["stores"]}


def price_range(products: list[dict]) -> list[float] | None:
    """[min, max] over every variant price, or None when no price parses."""
    prices: list[float] = []
    for p in products:
        for v in p.get("variants") or []:
            try:
                prices.append(float(v.get("price")))
            except (TypeError, ValueError):
                continue
    return [min(prices), max(prices)] if prices else None


class Recorder:
    """Wraps an ``http_get``: counts statuses, records 429s, blocks hosts after a 429."""

    def __init__(self, inner: rf.HttpGet = rf.urllib_get) -> None:
        self._inner = inner
        self.status_counts: Counter[str] = Counter()
        self.by_domain: dict[str, Counter[str]] = {}
        self.retry_after: dict[str, str | None] = {}
        self.blocked: set[str] = set()

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        host = url.split("/")[2]
        if host in self.blocked:  # defence in depth: the fetcher already stops the store
            raise rf.NetworkError("host stopped after 429")
        try:
            resp = self._inner(url, headers, timeout)
        except rf.NetworkError:
            self._count(host, "network_error")
            raise
        self._count(host, str(resp.status))
        if resp.status == 429:
            self.retry_after[host] = resp.headers.get("retry-after")
            self.blocked.add(host)
            stripped = {k: v for k, v in resp.headers.items() if k != "retry-after"}
            return rf.HttpResponse(429, stripped, "")
        return resp

    def _count(self, host: str, key: str) -> None:
        self.status_counts[key] += 1
        self.by_domain.setdefault(host, Counter())[key] += 1


def run(
    slugs: list[str],
    baseline: dict[str, int],
    http_get: rf.HttpGet = rf.urllib_get,
    throttle: rf.DomainThrottle | None = None,
) -> dict:
    """Fetch ``slugs`` sequentially; return the counts-only summary."""
    rec = Recorder(http_get)
    throttle = throttle or rf.DomainThrottle(global_max_per_second=GLOBAL_MAX_PER_SECOND)
    fetcher = rf.StoreFetcher(rec, throttle=throttle)
    t0 = time.monotonic()
    stores: list[dict] = []
    for slug in slugs:
        record = STORE_REGISTRY[slug]
        started = time.monotonic()
        res = fetcher.fetch_store(record)
        ids = {p.get("id") for p in res.products}
        base = baseline.get(slug)
        stores.append(
            {
                "slug": slug,
                "domain": record.domain,
                "status": res.status,
                "detail": res.detail,
                "pages": res.pages,
                "requests": res.requests,
                "items": len(res.products),
                "ids": len(ids),
                "price_range": price_range(res.products),
                "live_index_items": base,
                "delta_vs_live": (len(res.products) - base) if base is not None else None,
                "retry_after": rec.retry_after.get(record.domain or ""),
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        res.products = []  # free memory; bodies are never kept
    return {
        "dry_run": True,
        "counts_source": "tests/fixtures/indexed_stores.json (V3a inventory, 2026-11-14)",
        "stores_attempted": len(stores),
        "outcomes": dict(Counter(s["status"] for s in stores)),
        "pages_total": sum(s["pages"] for s in stores),
        "requests_total": sum(s["requests"] for s in stores),
        "http_status_counts": dict(rec.status_counts),
        "rate_limited_429": sorted(rec.retry_after),
        "challenge_stores": [s["slug"] for s in stores if s["status"] == rf.CHALLENGE],
        "items_total": sum(s["items"] for s in stores),
        "live_index_items_total_for_these_stores": sum(s["live_index_items"] or 0 for s in stores),
        "runtime_seconds": round(time.monotonic() - t0, 1),
        "stores": stores,
    }


def to_markdown(summary: dict) -> str:
    """Markdown report for ``$GITHUB_STEP_SUMMARY``."""
    lines = [
        "## Feed fetch rehearsal (dry run, no publish, no cloud)",
        "",
        f"- stores attempted: {summary['stores_attempted']}; outcomes: `{summary['outcomes']}`",
        f"- pages: {summary['pages_total']}; requests (incl. robots.txt): "
        f"{summary['requests_total']}; runtime: {summary['runtime_seconds']} s",
        f"- HTTP status counts: `{summary['http_status_counts']}`",
        f"- 429 hosts: `{summary['rate_limited_429']}`; challenge stores: "
        f"`{summary['challenge_stores']}`",
        f"- items fetched: {summary['items_total']} vs live index (same stores): "
        f"{summary['live_index_items_total_for_these_stores']}",
        f"- counts source: {summary['counts_source']}",
        "",
        "| store | status | pages | req | items | ids | live | delta | detail |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary["stores"]:
        lines.append(
            f"| {s['slug']} | {s['status']} | {s['pages']} | {s['requests']} | {s['items']} "
            f"| {s['ids']} | {s['live_index_items']} | {s['delta_vs_live']} | {s['detail']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry. Exit 0 even with per-store failures: the point is the measurement."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--md-out", type=Path, default=None)
    args = ap.parse_args()
    slugs = sorted(refreshable_stores())
    if len(slugs) != 38:
        print(f"expected 38 refreshable stores, found {len(slugs)}", file=sys.stderr)
        return 2
    summary = run(slugs, load_baseline())
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    env_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    target = args.md_out or (Path(env_summary) if env_summary else None)
    if target:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
