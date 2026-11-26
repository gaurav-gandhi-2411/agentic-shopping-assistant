"""One polite measurement pass over the refreshable store feeds (R3): counts, request log, rates.

Runs ``refresh_feed.StoreFetcher`` (global per-domain limiter, robots.txt, 429 / 5xx policy) over
every fetchable store in the registry, sequentially, ONE pass. It writes nothing into the catalogue
or index. Per-store results and a request log (timestamp, domain, path, status, Retry-After) are
kept; with ``--raw-dir`` each feed page body is also saved gzip-compressed for offline analysis. The
raw pages are for a scratch directory OUTSIDE git (they are item payloads).

Usage:
    python scripts/ops/fetch_pass.py --out summary.json --md-out table.md [--raw-dir DIR]
        [--log-out request_log.csv] [--only slug1,slug2]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import refresh_feed as rf  # noqa: E402

from src.config.store_registry import STORE_REGISTRY, refreshable_stores  # noqa: E402


class Recorder:
    """Wraps an ``http_get``: logs every request start, status and Retry-After; saves page bodies."""

    def __init__(self, inner: rf.HttpGet = rf.urllib_get, raw_dir: Path | None = None) -> None:
        self._inner = inner
        self._raw_dir = raw_dir
        self.log: list[dict] = []
        self.status_counts: Counter[str] = Counter()
        self.by_slug_status: dict[str, Counter[str]] = defaultdict(Counter)
        self.current_slug = ""

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> rf.HttpResponse:
        parts = urlsplit(url)
        entry = {
            "t": time.monotonic(),
            "domain": parts.hostname or "",
            "path": parts.path + ("?" + parts.query if parts.query else ""),
            "status": "network_error",
            "retry_after": "",
        }
        self.log.append(entry)
        try:
            resp = self._inner(url, headers, timeout)
        except rf.NetworkError:
            self.status_counts["network_error"] += 1
            self.by_slug_status[self.current_slug]["network_error"] += 1
            raise
        entry["status"] = str(resp.status)
        entry["retry_after"] = resp.headers.get("retry-after", "")
        self.status_counts[str(resp.status)] += 1
        self.by_slug_status[self.current_slug][str(resp.status)] += 1
        if self._raw_dir and parts.path == "/products.json" and resp.status == 200:
            page = parts.query.rsplit("page=", 1)[-1]
            out = self._raw_dir / f"{self.current_slug}__p{int(page):04d}.json.gz"
            out.write_bytes(gzip.compress(resp.body.encode("utf-8")))
        return resp


def request_rate_by_domain(log: list[dict]) -> dict[str, dict[str, float]]:
    """Per domain: request count, minimum gap between consecutive request STARTS (s), and the
    maximum observed rate = 1 / min gap (requests per second)."""
    times: dict[str, list[float]] = defaultdict(list)
    for e in log:
        times[e["domain"]].append(e["t"])
    out: dict[str, dict[str, float]] = {}
    for dom, ts in times.items():
        gaps = [b - a for a, b in zip(ts, ts[1:], strict=False)]
        min_gap = min(gaps) if gaps else float("inf")
        out[dom] = {
            "requests": len(ts),
            "min_gap_s": round(min_gap, 3) if gaps else float("nan"),
            "max_rate_per_s": round(1.0 / min_gap, 3) if gaps and min_gap > 0 else float("inf"),
        }
    return out


def run(slugs: list[str], raw_dir: Path | None = None, http_get: rf.HttpGet = rf.urllib_get) -> dict:
    """Fetch ``slugs`` sequentially (one pass); return the summary (no product payloads)."""
    rec = Recorder(http_get, raw_dir)
    fetcher = rf.StoreFetcher(rec)
    t0 = time.monotonic()
    stores: list[dict] = []
    for slug in slugs:
        record = STORE_REGISTRY[slug]
        rec.current_slug = slug
        started = time.monotonic()
        res = fetcher.fetch_store(record)
        stores.append(
            {
                "slug": slug,
                "domain": record.domain,
                "status": res.status,
                "detail": res.detail,
                "pages": res.pages,
                "requests": res.requests,
                "items": res.items_fetched,
                "capped": res.status == rf.CAPPED,
                "http_429": rec.by_slug_status[slug].get("429", 0),
                "http_403": rec.by_slug_status[slug].get("403", 0),
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        res.products = []  # free memory
    rates = request_rate_by_domain(rec.log)
    return {
        "stores_attempted": len(stores),
        "outcomes": dict(Counter(s["status"] for s in stores)),
        "requests_total": len(rec.log),
        "http_status_counts": dict(rec.status_counts),
        "items_total": sum(s["items"] for s in stores),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "max_rate_per_s_any_domain": max(r["max_rate_per_s"] for r in rates.values()),
        "min_gap_s_any_domain": min(
            r["min_gap_s"] for r in rates.values() if r["min_gap_s"] == r["min_gap_s"]
        ),
        "rate_by_domain": rates,
        "stores": stores,
        "_log": rec.log,
    }


def main() -> int:
    """CLI entry. Exit 0 even with per-store stops: the point is the measurement."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--log-out", type=Path, default=None)
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--only", default="", help="comma-separated slugs (default: all refreshable)")
    args = ap.parse_args()
    slugs = sorted(refreshable_stores())
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in slugs]
        if unknown:
            print(f"not refreshable / unknown: {unknown}", file=sys.stderr)
            return 2
        slugs = wanted
    if args.raw_dir:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
    summary = run(slugs, args.raw_dir)
    log = summary.pop("_log")
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.log_out:
        with args.log_out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["t", "domain", "path", "status", "retry_after"])
            w.writeheader()
            w.writerows(log)
    if args.md_out:
        args.md_out.write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return 0


def to_markdown(summary: dict) -> str:
    """Per-store table (counts only)."""
    rates = summary["rate_by_domain"]
    lines = [
        "| store | status | pages | requests | items | capped | 429 | 403 | seconds | min gap s | detail |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary["stores"]:
        gap = rates.get(s["domain"], {}).get("min_gap_s", "")
        lines.append(
            f"| {s['slug']} | {s['status']} | {s['pages']} | {s['requests']} | {s['items']} "
            f"| {'yes' if s['capped'] else 'no'} | {s['http_429']} | {s['http_403']} "
            f"| {s['seconds']} | {gap} | {s['detail']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
