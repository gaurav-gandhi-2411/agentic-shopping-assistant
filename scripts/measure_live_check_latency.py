"""Added latency of the live price check under SIMULATED store latencies (mock transport).

No real store is contacted: every request is answered by an in-process ``httpx.MockTransport``
that sleeps for a seeded log-normal delay. This measures the check's own machinery (parallelism,
rate limiting, budget cap), NOT real-store latency, which is NOT MEASURED.

Run: python scripts/measure_live_check_latency.py [--trials 30]
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx

from api.schemas import ItemSummary
from src.pricing.guards import LiveConfig
from src.pricing.live_check import LivePriceChecker

SEED = 42  # repo-wide convention: seeded everywhere stochastic
# Per-request (median_s, sigma) of the simulated log-normal store latency.
LATENCY = {"fast": (0.15, 0.4), "typical": (0.40, 0.6), "slow-tail": (0.90, 0.8)}
STORES = ["libas", "snitch", "voylla", "jompers", "tjori", "kisah", "fashor", "virgio"]


def _cards(n_stores: int) -> list[ItemSummary]:
    return [
        ItemSummary(
            article_id=f"c{i}", prod_name="K", display_name="K", colour="r", product_type="k",
            department="w", price_inr=1000.0, pdp_handle=f"h{i}", store=STORES[i % n_stores],
            pdp_url="https://x/",
        )
        for i in range(8)
    ]


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


async def _trial(rng: random.Random, dist: tuple[float, float], n_stores: int, warm: bool) -> float:
    import math

    median, sigma = dist

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(rng.lognormvariate(math.log(median), sigma))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, json={"variants": [{"price": 100000, "available": True}]})

    checker = LivePriceChecker(LiveConfig(enabled=True), transport=httpx.MockTransport(handler))
    if warm:
        await checker.check(_cards(n_stores))
    t0 = time.perf_counter()
    await checker.check(_cards(n_stores))
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    rng = random.Random(SEED)
    print(f"budget_s=1.5 trials={args.trials} seed={SEED} (SIMULATED latencies, mock transport)")
    print("| scenario | stores | p50 ms | p95 ms | max ms | trials at budget cap |")
    print("|---|---|---|---|---|---|")
    for name, dist in LATENCY.items():
        for n_stores, warm in ((8, False), (3, False), (8, True)):
            xs = [asyncio.run(_trial(rng, dist, n_stores, warm)) for _ in range(args.trials)]
            capped = sum(x >= 1.45 for x in xs)
            label = f"{name}{' (warm cache)' if warm else ''}"
            print(
                f"| {label} | {n_stores} | {statistics.median(xs) * 1000:.0f} | "
                f"{_pct(xs, 0.95) * 1000:.0f} | {max(xs) * 1000:.0f} | {capped}/{args.trials} |"
            )


if __name__ == "__main__":
    main()
