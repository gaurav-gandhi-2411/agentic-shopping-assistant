"""Build a handle → default-variant map for each Shopify brand.

For each Shopify brand (snitch, powerlook, fashor, virgio), paginates
/products.json and writes one artifact per brand:

    data/processed/shopify_variants/{brand}.json

Map schema per handle:
    {
        "<handle>": {
            "default_variant_id": "<str>",
            "size": "<label>",
            "product_id": "<str>"
        }
    }

Default-size policy (in priority order):
  1. Variant whose size option/title contains " / M" or is exactly "M" (case-insensitive).
  2. First available variant (available_for_sale=true) if no M found.
  3. First variant overall.

Usage:
    python scripts/build_shopify_variant_map.py
    python scripts/build_shopify_variant_map.py --brands snitch powerlook

The script is polite: 0.5 s sleep between pages, 3 retries per page,
15 s socket timeout. Skips a brand gracefully on persistent error.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHOPIFY_BRANDS: dict[str, str] = {
    "snitch": "snitch.co.in",
    "powerlook": "powerlook.in",
    "fashor": "fashor.com",
    "virgio": "virgio.com",
}

PAGE_SIZE = 250
MAX_PAGES = 60
PAGE_SLEEP_S = 0.5
TIMEOUT_S = 15
MAX_RETRIES = 3
RETRY_SLEEP_S = 2.0

UA = "Mozilla/5.0 (compatible; demo-pull/1.0; +https://github.com/gaurav-gandhi-2411)"

OUT_DIR = Path("data/processed/shopify_variants")


# ---------------------------------------------------------------------------
# Size-matching helpers
# ---------------------------------------------------------------------------

def _size_is_m(title: str) -> bool:
    """Return True if the variant title indicates size M.

    Matches patterns like: "M", "White / M", "Blue / M / Regular",
    " / M", "M / 30" — all case-insensitive.
    Rejects "XM", "XMM", "Slim", etc.
    """
    t = title.strip()
    # Exact "M"
    if t.upper() == "M":
        return True
    # A segment that is exactly "M" (surrounded by " / " or at boundaries)
    parts = [p.strip() for p in t.split("/")]
    return any(p.upper() == "M" for p in parts)


def _pick_default_variant(variants: list[dict]) -> dict | None:
    """Pick the best default variant according to size policy."""
    if not variants:
        return None

    # Priority 1: size "M"
    for v in variants:
        title = v.get("title") or ""
        if _size_is_m(title):
            return v

    # Priority 2: first available
    for v in variants:
        if v.get("available") is True:
            return v

    # Priority 3: first variant
    return variants[0]


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _get_with_retry(
    session: requests.Session,
    url: str,
) -> requests.Response | None:
    """GET url with up to MAX_RETRIES retries; returns None on permanent failure."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=TIMEOUT_S)
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_S)
    print(f"    [WARN] Failed after {MAX_RETRIES} attempts: {last_exc}")
    return None


def fetch_variant_map(brand: str, domain: str, session: requests.Session) -> dict[str, dict]:
    """Paginate /products.json for domain and build handle → variant info map.

    Returns empty dict on any unrecoverable error (brand degrades to per-item links).
    """
    handle_map: dict[str, dict] = {}
    pages_fetched = 0
    products_seen: set[int] = set()

    for page in range(1, MAX_PAGES + 1):
        url = f"https://{domain}/products.json?limit={PAGE_SIZE}&page={page}"
        resp = _get_with_retry(session, url)

        if resp is None:
            print(f"  [{brand.upper()}] page {page}: no response — stopping pagination.")
            break

        if resp.status_code == 404:
            print(f"  [{brand.upper()}] page {page}: 404 — not a public Shopify store.")
            break

        if resp.status_code != 200:
            print(f"  [{brand.upper()}] page {page}: HTTP {resp.status_code} — stopping.")
            break

        try:
            products: list[dict] = resp.json().get("products", [])
        except Exception as exc:
            print(f"  [{brand.upper()}] page {page}: JSON parse error ({exc}) — stopping.")
            break

        if not products:
            break  # empty page = done

        # Detect non-moving pagination (store ignores ?page=N)
        first_id = products[0].get("id")
        if first_id in products_seen:
            print(f"  [{brand.upper()}] page {page}: same first product as before — pagination done.")
            break

        for p in products:
            pid = p.get("id")
            if pid:
                products_seen.add(pid)

            handle = (p.get("handle") or "").strip()
            if not handle:
                continue

            variants: list[dict] = p.get("variants") or []
            chosen = _pick_default_variant(variants)
            if chosen is None:
                continue

            variant_id = chosen.get("id")
            if not variant_id:
                continue

            # Extract clean size label from variant title
            title = chosen.get("title") or ""
            parts = [part.strip() for part in title.split("/")]
            # Size label: the part matching "M" or the last part when multi-segment
            size_label = title
            for part in parts:
                if re.match(r"^(XS|S|M|L|XL|XXL|XXXL|\d+)$", part.strip(), re.IGNORECASE):
                    size_label = part.strip()
                    break

            handle_map[handle] = {
                "default_variant_id": str(variant_id),
                "size": size_label,
                "product_id": str(p.get("id") or ""),
            }

        pages_fetched += 1
        print(f"  [{brand.upper()}] page {page}: +{len(products)} products  (handles mapped so far: {len(handle_map)})")

        if len(products) < PAGE_SIZE:
            break  # last partial page

        time.sleep(PAGE_SLEEP_S)

    return handle_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Shopify variant maps for all Shopify brands.")
    parser.add_argument(
        "--brands",
        nargs="*",
        default=list(SHOPIFY_BRANDS.keys()),
        choices=list(SHOPIFY_BRANDS.keys()),
        help="Brands to crawl (default: all Shopify brands).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = UA

    summary: dict[str, dict] = {}

    for brand in args.brands:
        domain = SHOPIFY_BRANDS[brand]
        print(f"\n=== {brand.upper()} ({domain}) ===")

        # Quick probe — does /products.json respond at all?
        probe_url = f"https://{domain}/products.json?limit=1"
        probe = _get_with_retry(session, probe_url)
        if probe is None or probe.status_code != 200:
            status = probe.status_code if probe is not None else "no response"
            print(f"  [{brand.upper()}] probe failed (HTTP {status}) — skipping, will degrade to per-item links.")
            summary[brand] = {"handles": 0, "pages": 0, "status": f"FAILED ({status})"}
            continue

        try:
            probe_json = probe.json()
            if "products" not in probe_json:
                raise ValueError("no 'products' key in response")
        except Exception as exc:
            print(f"  [{brand.upper()}] probe response is not valid Shopify JSON ({exc}) — skipping.")
            summary[brand] = {"handles": 0, "pages": 0, "status": "FAILED (not shopify)"}
            continue

        handle_map = fetch_variant_map(brand, domain, session)

        # Write artifact — pretty JSON, sorted keys, deterministic
        out_path = OUT_DIR / f"{brand}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(handle_map, fh, indent=2, sort_keys=True, ensure_ascii=False)

        print(f"  [{brand.upper()}] Done. {len(handle_map)} handles mapped -> {out_path}")
        summary[brand] = {"handles": len(handle_map), "status": "OK"}

    # Summary table
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for brand, info in summary.items():
        status = info.get("status", "")
        handles = info.get("handles", 0)
        print(f"  {brand:<12} {handles:>5} handles   {status}")
    print("=" * 50)


if __name__ == "__main__":
    main()
