"""Post-deploy smoke test for asa-stylist-api (used by .github/workflows/deploy.yml).

Stdlib only, so the CI job needs no pip install. Makes exactly ONE LLM-backed call
(POST /chat) per run; the deploy workflow runs it twice per deploy (tag URL, then main URL).

    python scripts/ops/deploy_smoke.py --base-url https://c-abc1234---asa-stylist-api-....run.app

Checks, in order (first failure exits 1):
  1. GET <live-path>   -> 200            (default /api/brand, see note below)
  2. GET /readyz       -> 200, status == "ready"
  3. POST /demo/session -> 200, has session_token
  4. POST /chat "black kurta set for haldi" -> 200, non-empty `response`, >= 1 item

Why the liveness path defaults to /api/brand and not /healthz: uptime-alert.yml records that
/healthz 404s at the infrastructure level on this service, while /api/brand shows up as HTTP 200
219 of 224 requests in the 30-day Cloud Run request log (reports/w8-inventory-2026-11-03.md). Override with
--live-path /healthz once that has been verified live.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

QUERY = "black kurta set for haldi"
# A cold start measured 36-88 s (p50 71 s) over the last 30 days; leave headroom above the max.
COLD_START_TIMEOUT_S = 150
CHAT_TIMEOUT_S = 120


def request(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = COLD_START_TIMEOUT_S,
) -> tuple[int, str]:
    """Issue one HTTP request; return (status, body text). Never raises on HTTP errors."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Accept": "application/json", "User-Agent": "asa-deploy-smoke/1"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def fail(msg: str) -> None:
    """Print a failure line and exit non-zero."""
    print(f"FAIL  {msg}")
    sys.exit(1)


def get_with_retry(url: str, attempts: int = 2) -> tuple[int, str]:
    """GET with one retry: the first request to a no-traffic revision can hit a cold start."""
    status, text = 0, ""
    for attempt in range(1, attempts + 1):
        status, text = request("GET", url)
        if status == 200:
            return status, text
        print(f"  attempt {attempt}/{attempts}: HTTP {status} {text[:120]!r}")
        time.sleep(5)
    return status, text


def main() -> None:
    """Run the four checks against --base-url."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--live-path", default="/api/brand")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    if not base.startswith("https://"):
        fail(f"base url must be https, got {base!r}")
    print(f"=== deploy smoke: {base} ===")

    status, text = get_with_retry(base + args.live_path)
    if status != 200:
        fail(f"GET {args.live_path} -> {status}: {text[:200]}")
    print(f"PASS  GET {args.live_path} -> 200")

    status, text = get_with_retry(base + "/readyz")
    if status != 200:
        fail(f"GET /readyz -> {status}: {text[:300]}")
    try:
        ready = json.loads(text).get("status")
    except json.JSONDecodeError:
        ready = None
    if ready != "ready":
        fail(f"/readyz status != ready: {text[:300]}")
    print("PASS  GET /readyz -> 200 ready")

    status, text = request("POST", base + "/demo/session", body={"brand": "unified"}, timeout=60)
    if status != 200:
        fail(f"POST /demo/session -> {status}: {text[:200]}")
    token = json.loads(text).get("session_token")
    if not token:
        fail("POST /demo/session returned no session_token")
    print("PASS  POST /demo/session -> 200 (token received, not printed)")

    started = time.monotonic()
    status, text = request(
        "POST", base + "/chat", body={"message": QUERY}, token=token, timeout=CHAT_TIMEOUT_S
    )
    elapsed = time.monotonic() - started
    if status != 200:
        fail(f"POST /chat -> {status}: {text[:300]}")
    payload = json.loads(text)
    answer = (payload.get("response") or "").strip()
    items = payload.get("items") or []
    if not answer:
        fail("POST /chat returned an empty response")
    if len(items) < 1:
        fail(f"POST /chat returned no products for {QUERY!r}")
    print(f"PASS  POST /chat {QUERY!r} -> 200 in {elapsed:.1f}s, {len(items)} items")
    print("=== smoke OK ===")


if __name__ == "__main__":
    main()
