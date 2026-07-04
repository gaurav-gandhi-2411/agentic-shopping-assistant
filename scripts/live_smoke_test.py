"""Live smoke test for unified asa-stylist-api (B2C wave).

Run after every deploy:
    python scripts/live_smoke_test.py

Tests: demo session, dress query (garment type), context retention, save/retrieve look.
"""
from __future__ import annotations

import sys

import requests

BASE = "https://asa-stylist-api-657468372797.asia-south1.run.app"
FAIL = False


def _check(label: str, ok: bool, detail: str = "") -> None:
    global FAIL
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL = True


def _h(r: requests.Response, step: str) -> dict:
    if r.status_code not in (200, 201):
        print(f"  FAIL  {step}: HTTP {r.status_code} — {r.text[:300]}")
        sys.exit(1)
    return r.json()


print(f"=== Live smoke: {BASE} ===\n")

# ── 1. Demo session ──────────────────────────────────────────────────────────
s = requests.post(f"{BASE}/demo/session", json={"brand": "unified"}, timeout=30)
d = _h(s, "demo/session")
token = d["session_token"]
headers = {"Authorization": f"Bearer {token}"}
_check("demo session", bool(token), f"token={token[:16]}...")

# ── 2. Chat: "black dress for women" → dresses (not kurtis) ─────────────────
r = requests.post(f"{BASE}/chat", json={"message": "black dress for women"}, headers=headers, timeout=90)
d = _h(r, "chat dress")
items = d.get("items", [])
cid = d.get("conversation_id", "")
_check("conv_id present", bool(cid))
_check("items returned", len(items) > 0, f"{len(items)} items")
nan_names = [i.get("display_name") for i in items if "nan" in str(i.get("display_name", "")).lower()]
_check("no nan in names", len(nan_names) == 0, str(nan_names))
images_ok = all(i.get("image_url") for i in items)
_check("all items have image_url", images_ok)
non_dress = [i.get("product_type") for i in items if i.get("product_type", "").lower() not in ("dress", "")]
_check("results are dresses (not kurtis)", len(non_dress) == 0, f"non-dress types: {non_dress}")
stores = {i.get("store") for i in items}
print(f"   stores={stores}  types={list({i.get('product_type') for i in items})}")

# ── 3. Suggestion chips from initial search ──────────────────────────────────
chips = d.get("suggestion_chips")
_check("suggestion_chips returned", chips is not None and len(chips) > 0,
       f"chips={chips}")

# ── 4. Follow-up turn — colour refinement returns CARDS not prose ────────────
r2 = requests.post(
    f"{BASE}/chat",
    json={"message": "in blue please", "conversation_id": cid},
    headers=headers, timeout=90
)
d2 = _h(r2, "follow-up blue")
items2 = d2.get("items", [])
resp2 = d2.get("response", "").lower()
context_lost = "start from scratch" in resp2 or ("which product" in resp2 and not items2)
_check("follow-up retains context (no start-from-scratch)", not context_lost,
       f"items={len(items2)}")
_check("refinement returns cards (not prose-only)", len(items2) > 0,
       f"items_on_refinement={len(items2)}, response_snippet={resp2[:80]!r}")
non_dress_r2 = [it.get("product_type") for it in items2
                if it.get("product_type", "").lower() not in ("dress", "")]
_check("refinement keeps garment type (dress)", len(non_dress_r2) == 0,
       f"non-dress on refinement: {non_dress_r2}")

# ── 5. Save look → retrieve ──────────────────────────────────────────────────
snap = {
    "items": [{"article_id": i.get("article_id", "x"), "prod_name": i.get("prod_name", "x"),
               "image_url": i.get("image_url"), "price_inr": i.get("price_inr"),
               "pdp_url": i.get("pdp_url")} for i in items[:2]],
    "rationale": "Smoke test look"
}
r3 = requests.post(f"{BASE}/looks", json={
    "session_id": cid, "brand": "unified",
    "snapshot": snap, "look_gender": "women", "occasion": "casual"
}, headers=headers, timeout=30)
d3 = _h(r3, "save look")
look_id = d3.get("id", "")
_check("save look returns id", bool(look_id), f"id={look_id[:12]}...")

r4 = requests.get(f"{BASE}/looks/{look_id}", timeout=30)
d4 = _h(r4, "get look")
_check("get look round-trip", d4.get("id") == look_id)

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if FAIL:
    print("SMOKE FAILED — see FAIL lines above")
    sys.exit(1)
else:
    print("All smoke checks PASS")
