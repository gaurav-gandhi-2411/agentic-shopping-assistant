#!/usr/bin/env python3
"""Live WS proof: hit /chat/stream on the deployed service via the real browser path.

Two-turn flow:
  1. "black dress for women"  -> must return WSItemsMessage with Dress items + images
  2. "in blue"                -> must return WSItemsMessage with Blue Dress items

The WS handler is single-message-per-connection; we reconnect with a fresh ticket
per turn but pass the conversation_id to preserve context.

Usage:
  python scripts/ws_live_proof.py [backend_url]

Default backend: https://asa-stylist-api-657468372797.asia-south1.run.app
"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx
import websockets

BASE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "https://asa-stylist-api-657468372797.asia-south1.run.app"
)
WS_BASE = BASE.replace("https://", "wss://").replace("http://", "ws://")

TURNS = [
    "black dress for women",
    "in blue",
]


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


async def collect_turn(
    ws: websockets.WebSocketClientProtocol,
    message: str,
    conv_id: str | None,
) -> tuple[str | None, list[dict], str]:
    payload: dict = {"type": "user_message", "message": message}
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
            raise RuntimeError(f"server error: {frame.get('message', frame)}")

    return new_conv_id, items, "".join(prose_parts)


async def main() -> None:
    print(f"Backend : {BASE}")
    print(f"WS base : {WS_BASE}\n")

    conv_id: str | None = None

    async with httpx.AsyncClient(base_url=BASE, timeout=30) as http:
        for turn_idx, user_msg in enumerate(TURNS, 1):
            ticket = await get_ticket(http)
            ws_url = f"{WS_BASE}/chat/stream?ticket={ticket}"

            print(f"{'=' * 70}")
            print(f"TURN {turn_idx}  USER: {user_msg}")
            print(f"ticket : {ticket[:20]}...")
            print(f"{'=' * 70}")

            async with websockets.connect(ws_url, open_timeout=30, close_timeout=10) as ws:
                conv_id, items, prose = await collect_turn(ws, user_msg, conv_id)

            print(f"\n[prose] {prose.strip()[:300]}")
            print(f"\n[items] {len(items)} items received")

            if not items:
                print("  *** NO ITEMS — bug ***")
            else:
                for it in items:
                    name = it.get("prod_name", "")[:55]
                    print(
                        f"  {it.get('article_id')} | {it.get('product_type')} | "
                        f"{it.get('colour')} | {it.get('store')} | "
                        f"img={'YES' if it.get('image_url') else 'NO'} | {name}"
                    )

            if items:
                print(f"\n--- RAW items[0] ---")
                print(json.dumps(items[0], indent=2))
            print()


if __name__ == "__main__":
    asyncio.run(main())
