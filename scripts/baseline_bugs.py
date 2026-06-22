#!/usr/bin/env python3
"""Baseline failing checks for 4 browser bugs.

Run BEFORE any fixes to confirm RED state on the live Cloud Run URL.
All checks hit the WS /chat/stream path (same path as the browser).

Usage:
    python scripts/baseline_bugs.py [backend_url]
"""
from __future__ import annotations

import asyncio
import json
import sys
from urllib.parse import urlparse

import httpx
import websockets

BASE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "https://asa-stylist-api-657468372797.asia-south1.run.app"
)
WS_BASE = BASE.replace("https://", "wss://").replace("http://", "ws://")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


# ── helpers ──────────────────────────────────────────────────────────────────

async def get_ticket(http: httpx.AsyncClient) -> str:
    r = await http.post("/demo/session")
    r.raise_for_status()
    d = r.json()
    ticket = d.get("ws_ticket", "")
    if not ticket:
        token = d.get("session_token") or d.get("token", "")
        r2 = await http.post("/auth/ws-ticket", headers={"Authorization": f"Bearer {token}"})
        r2.raise_for_status()
        ticket = r2.json()["ticket"]
    return ticket


async def ws_turn(
    http: httpx.AsyncClient,
    user_msg: str,
    conv_id: str | None = None,
) -> tuple[str | None, list[dict], str]:
    """One WS turn → (conv_id, items, prose)."""
    ticket = await get_ticket(http)
    url = f"{WS_BASE}/chat/stream?ticket={ticket}"
    async with websockets.connect(url, open_timeout=30, close_timeout=10) as ws:
        payload: dict = {"type": "user_message", "message": user_msg}
        if conv_id:
            payload["conversation_id"] = conv_id
        await ws.send(json.dumps(payload))

        items: list[dict] = []
        prose_parts: list[str] = []
        new_conv_id = conv_id

        async for raw in ws:
            frame = json.loads(raw)
            ftype = frame.get("type")
            if ftype == "session":
                new_conv_id = frame["conversation_id"]
            elif ftype == "items":
                items = frame.get("items", [])
            elif ftype == "token":
                prose_parts.append(frame.get("text", ""))
            elif ftype == "done":
                break
            elif ftype == "error":
                raise RuntimeError(f"WS server error: {frame.get('message', frame)}")

    return new_conv_id, items, "".join(prose_parts)


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return condition


# ── Bug #1: Wrong buy links ───────────────────────────────────────────────────

async def test_bug1_buy_links(http: httpx.AsyncClient) -> int:
    """Items from berrylush/globalrepublic/libas must carry THEIR OWN pdp_url."""
    print("\n-- Bug #1: Wrong Buy Links --")
    failures = 0

    STORE_EXPECTED_DOMAINS = {
        "berrylush":     "www.berrylush.com",
        "globalrepublic": "globalrepublic.in",
        "libas":         "libas.in",
    }

    for store_slug, expected_domain in STORE_EXPECTED_DOMAINS.items():
        # Trigger a broad search that should include items from this store
        query_map = {
            "berrylush": "dress berrylush",
            "globalrepublic": "top globalrepublic",
            "libas": "kurta libas",
        }
        query = query_map[store_slug]
        try:
            _, items, _ = await ws_turn(http, query)
        except Exception as e:
            print(f"  [{FAIL}] {store_slug}: WS error: {e}")
            failures += 1
            continue

        # Filter to this specific store
        store_items = [it for it in items if it.get("store") == store_slug]

        if not store_items:
            print(f"  [{FAIL}] {store_slug}: 0 items returned")
            failures += 1
            continue

        # Check first item's pdp_url domain
        it0 = store_items[0]
        pdp_url = it0.get("pdp_url")
        if not pdp_url:
            ok = False
            detail = f"pdp_url=None  prod={it0.get('prod_name','')[:50]}"
        else:
            parsed_domain = urlparse(pdp_url).netloc
            ok = expected_domain in parsed_domain
            detail = f"pdp_url={pdp_url[:80]}"

        if not check(f"{store_slug}: pdp_url domain is {expected_domain}", ok, detail):
            failures += 1
        else:
            # Extra: store_display should not be None
            disp = it0.get("store_display")
            if not check(f"{store_slug}: store_display set", bool(disp), f"store_display={disp}"):
                failures += 1

    return failures


# ── Bug #2: Cards not rendering ───────────────────────────────────────────────

async def test_bug2_cards_render(http: httpx.AsyncClient) -> int:
    """These 3 first-turn queries must return ≥1 item card via WS."""
    print("\n-- Bug #2: Cards Not Rendering --")
    failures = 0
    QUERIES = ["saree", "black dress for women", "white shirt men"]

    for q in QUERIES:
        try:
            _, items, prose = await ws_turn(http, q)
        except Exception as e:
            print(f"  [{FAIL}] '{q}': WS error: {e}")
            failures += 1
            continue

        ok = len(items) > 0
        detail = f"{len(items)} cards  prose={prose[:80].strip()!r}"
        if not check(f"'{q}' → cards > 0", ok, detail):
            failures += 1

    return failures


# ── Bug #3: Refinement drops context ─────────────────────────────────────────

async def test_bug3_refinement_context(http: httpx.AsyncClient) -> int:
    """'white shirt men' then 'in blue now' must return MEN'S SHIRT items in Blue."""
    print("\n-- Bug #3: Refinement Drops Context --")
    failures = 0

    try:
        conv_id, items1, _ = await ws_turn(http, "white shirt men")
        print(f"  Turn 1 'white shirt men': {len(items1)} items, conv={conv_id}")

        _, items2, prose2 = await ws_turn(http, "in blue now", conv_id=conv_id)
        print(f"  Turn 2 'in blue now':     {len(items2)} items")
        print(f"  prose: {prose2[:120].strip()!r}")
    except Exception as e:
        print(f"  [{FAIL}] WS error: {e}")
        return 1

    if not items2:
        check("refinement: ≥1 item returned", False, "0 items")
        return 1

    # Check gender
    genders = [it.get("gender") for it in items2]
    non_men = [g for g in genders if g not in ("men", "unisex", None)]
    gender_ok = len(non_men) == 0
    check(
        "refinement: all items gender=men/unisex",
        gender_ok,
        f"genders={set(genders)}" + (f" non-men={non_men[:3]}" if non_men else ""),
    )
    if not gender_ok:
        failures += 1

    # Check product_type
    types = [it.get("product_type") for it in items2]
    non_shirt = [t for t in types if t and t.lower() not in ("shirt", "top", "polo")]
    shirt_ok = len(non_shirt) < len(types) / 2  # majority should be shirts
    check(
        "refinement: majority product_type=shirt",
        shirt_ok,
        f"types={set(types)}" + (f" non-shirt={non_shirt[:3]}" if non_shirt else ""),
    )
    if not shirt_ok:
        failures += 1

    # Check colour
    colours = [it.get("colour") for it in items2]
    non_blue = [c for c in colours if c and c.lower() not in ("blue", "navy", "indigo", "cobalt")]
    colour_ok = len(non_blue) < len(colours) / 2  # majority blue
    check(
        "refinement: majority colour=blue",
        colour_ok,
        f"colours={set(colours)}" + (f" non-blue={non_blue[:3]}" if non_blue else ""),
    )
    if not colour_ok:
        failures += 1

    return failures


# ── Bug #4b: CLIP index on live service ──────────────────────────────────────

async def test_bug4_clip_status(http: httpx.AsyncClient) -> int:
    """POST /style/from-image with a dummy PNG must NOT return 404 (CLIP index missing)."""
    print("\n-- Bug #4b: CLIP Index Status --")
    failures = 0

    # 1×1 white PNG in bytes
    PNG_1X1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    try:
        r = await http.post(
            "/style/from-image",
            content=PNG_1X1,
            headers={"Content-Type": "image/png"},
            timeout=20,
        )
    except Exception as e:
        print(f"  [{FAIL}] /style/from-image request failed: {e}")
        return 1

    status_code = r.status_code
    body = {}
    try:
        body = r.json()
    except Exception:
        pass

    # 404 = CLIP index missing on live service
    not_404 = status_code != 404
    check(
        "CLIP index loaded (not 404)",
        not_404,
        f"status={status_code}  body={json.dumps(body)[:200]}",
    )
    if not not_404:
        failures += 1

    # Any response other than 404/503 suggests the index is present
    clip_ok = status_code in (200, 400, 422)  # 200=found, 400/422=bad input but index exists
    check(
        "CLIP returns usable response (200/400/422)",
        clip_ok,
        f"status={status_code}",
    )
    if not clip_ok:
        failures += 1

    return failures


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"Backend : {BASE}")
    print(f"WS base : {WS_BASE}")
    print(f"{'=' * 65}")

    total_failures = 0
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as http:
        total_failures += await test_bug1_buy_links(http)
        total_failures += await test_bug2_cards_render(http)
        total_failures += await test_bug3_refinement_context(http)
        total_failures += await test_bug4_clip_status(http)

    print(f"\n{'=' * 65}")
    status = FAIL if total_failures else PASS
    print(f"[{status}]  {total_failures} check(s) failed")
    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
