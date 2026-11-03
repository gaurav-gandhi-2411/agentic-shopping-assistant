"""Post-deploy smoke test for asa-stylist-api (used by .github/workflows/deploy.yml).

Stdlib only, so the CI job needs no pip install.

    python scripts/ops/deploy_smoke.py --base-url https://c-abc1234---asa-stylist-api-....run.app
    python scripts/ops/deploy_smoke.py --base-url https://asa-stylist-api-....run.app --no-chat

Checks, in order (first failure exits 1):
  1. GET <live-path>    -> 200            (default /api/brand, see note below)
  2. GET /readyz        -> 200, status == "ready"
  3. POST /demo/session -> 200, has session_token          (skipped with --no-chat)
  4. POST /chat "black kurta set for haldi" -> 200, non-empty `response`, >= 1 item  (skipped with --no-chat)

LLM calls (MEASURED, not estimated): one POST /chat for this query made exactly 2 LLM calls
(src/agents/reranker.py:170 and src/agents/graph.py:3993) when run in-process against the real
local index with a counting stub LLM; the router LLM (router.py:58) was NOT consulted because the
deterministic pre-router handled the query, and no outfit rationale call (outfit/rationale.py:446,
one per look) fired. If an LLM-routed or outfit turn is ever hit, expect more: static upper bound per
fresh turn is router (1 per iteration) + reranker (1) + respond (1) + 1 rationale per look.
Today /readyz makes 0 LLM calls; once W4 (#77) lands it makes 1. deploy.yml therefore runs the chat
check ONLY against the tag URL (before any traffic moves) and the post-shift main-URL check is
liveness + /readyz only (--no-chat).
See reports/w8-inventory-2026-11-03.md for the measurement.

Why the liveness path defaults to /api/brand and not /healthz: uptime-alert.yml records that
/healthz 404s at the infrastructure level on this service, while /api/brand shows up as HTTP 200
for 219 of 224 requests in the 30-day Cloud Run request log. Override with --live-path /healthz
once that has been verified live.
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
RETRY_SLEEP_S = 5  # module-level so tests can zero it


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


def parse_json(text: str, what: str) -> dict[str, Any]:
    """Parse a JSON object body or fail with a readable message (never a traceback)."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        fail(f"{what}: body is not valid JSON ({exc}): {text[:200]!r}")
        raise  # unreachable; fail() exits
    if not isinstance(parsed, dict):
        fail(f"{what}: expected a JSON object, got {type(parsed).__name__}: {text[:200]!r}")
    return parsed


def get_with_retry(url: str, attempts: int = 2) -> tuple[int, str]:
    """GET with one retry: the first request to a no-traffic revision can hit a cold start."""
    status, text = 0, ""
    for attempt in range(1, attempts + 1):
        status, text = request("GET", url)
        if status == 200:
            return status, text
        print(f"  attempt {attempt}/{attempts}: HTTP {status} {text[:120]!r}")
        if attempt < attempts:
            time.sleep(RETRY_SLEEP_S)
    return status, text


def main(argv: list[str] | None = None, *, require_https: bool = True) -> None:
    """Run the checks against --base-url. `require_https=False` exists for loopback tests only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--live-path", default="/api/brand")
    ap.add_argument("--no-chat", action="store_true", help="liveness + /readyz only, no LLM call")
    args = ap.parse_args(argv)
    base = args.base_url.rstrip("/")
    if require_https and not base.startswith("https://"):
        fail(f"base url must be https, got {base!r}")
    print(f"=== deploy smoke: {base} (chat={'off' if args.no_chat else 'on'}) ===")

    status, text = get_with_retry(base + args.live_path)
    if status != 200:
        fail(f"GET {args.live_path} -> {status}: {text[:200]}")
    print(f"PASS  GET {args.live_path} -> 200")

    status, text = get_with_retry(base + "/readyz")
    if status != 200:
        fail(f"GET /readyz -> {status}: {text[:300]}")
    ready = parse_json(text, "GET /readyz").get("status")
    if ready != "ready":
        fail(f"/readyz status != ready: {text[:300]}")
    print("PASS  GET /readyz -> 200 ready")

    if args.no_chat:
        print("=== smoke OK (no chat) ===")
        return

    status, text = request("POST", base + "/demo/session", body={"brand": "unified"}, timeout=60)
    if status != 200:
        fail(f"POST /demo/session -> {status}: {text[:200]}")
    token = parse_json(text, "POST /demo/session").get("session_token")
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
    payload = parse_json(text, "POST /chat")
    answer = str(payload.get("response") or "").strip()
    items = payload.get("items") or []
    if not answer:
        fail("POST /chat returned an empty response")
    if len(items) < 1:
        fail(f"POST /chat returned no products for {QUERY!r}")
    print(f"PASS  POST /chat {QUERY!r} -> 200 in {elapsed:.1f}s, {len(items)} items")
    print("=== smoke OK ===")


if __name__ == "__main__":
    main()
