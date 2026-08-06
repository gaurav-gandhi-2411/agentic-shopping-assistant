#!/usr/bin/env python
"""One-off: live-prove the deployed router-vocabulary fix against the real
Cloud Run URL, per DEPLOY.md's QA rule (proof must come from the live URL,
not localhost). Not a permanent script.
"""
from __future__ import annotations

import json

import requests

BASE = "https://asa-stylist-api-rm7rz66wza-el.a.run.app"

QUERIES = [
    "navy blue kurta for men",
    "maroon saree",
    "black jacket for women",
    "gold lehenga",
]


def main() -> None:
    session_resp = requests.post(f"{BASE}/demo/session", timeout=30)
    session_resp.raise_for_status()
    token = session_resp.json()["session_token"]
    print(f"session token acquired: {token[:15]}...\n")

    for q in QUERIES:
        print(f"=== QUERY: {q!r} ===")
        resp = requests.post(
            f"{BASE}/chat",
            json={"message": q, "conversation_id": None},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items") or []
        print(f"n_items: {len(items)}")
        for it in items[:5]:
            print(
                f"  - {it.get('prod_name')} | type={it.get('product_type')} "
                f"| colour={it.get('colour')} | gender={it.get('gender')}"
            )
        if not items:
            print("  RAW response:", json.dumps(data)[:800])
        print()


if __name__ == "__main__":
    main()
