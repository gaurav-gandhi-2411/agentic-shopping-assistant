"""
Download a Shopify store's public product catalogue via /products.json.

Usage:
    python scripts/download_shopify.py --domain snitch.co.in
    python scripts/download_shopify.py --domain powerlook.in
    python scripts/download_shopify.py --domain fashor.com
    python scripts/download_shopify.py --domain virgio.com

Saves to: data/raw/shopify/<slug>/products.csv
where <slug> is the first segment of the domain (e.g. "snitch" from "snitch.co.in").

Checks robots.txt before fetching. Skips if /products.json returns non-Shopify content.
Pauses 0.5 s between pages. Max 60 pages (15 000 products).

Cloudflare TLS-fingerprint fallback (2026-07-19): several Indian D2C Shopify stores sit
behind Cloudflare bot-management that fingerprints Python's requests/OpenSSL TLS
ClientHello and returns HTTP 429 "Verifying your connection..." — while curl.exe
(Windows' native Schannel TLS stack) gets a normal 200 on the identical URL. This is a
TLS-fingerprint-level block, not a true rate limit (retrying with different headers/UA
does not help). When a request returns a Cloudflare-challenge-shaped response (HTTP 429,
or 200/503 body containing a Cloudflare challenge marker), _get() transparently retries
that single request via `curl.exe` (subprocess) and wraps the stdout in a requests.Response
look-alike so all downstream code (pagination, JSON parsing, robots check) is unchanged.
requests remains the default/primary path — curl is only invoked on a detected block, so
the common (unblocked) case pays no subprocess overhead.
"""
from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
import time
import urllib.robotparser
from pathlib import Path

import pandas as pd
import requests

MAX_PAGES = 60
PAGE_SIZE = 250
PAUSE_S = 0.5
UA = "Mozilla/5.0 (compatible; demo-pull/1.0; +https://github.com/gaurav-gandhi-2411)"

# Substrings seen in Cloudflare's bot-management challenge/interstitial page —
# case-sensitive markers Cloudflare actually emits, checked case-insensitively below.
_CLOUDFLARE_MARKERS: tuple[str, ...] = (
    "verifying your connection",
    "checking your browser",
    "cf-browser-verification",
    "cloudflare",
    "attention required",
)


def _looks_like_cloudflare_challenge(resp: requests.Response) -> bool:
    """Return True if *resp* looks like a Cloudflare bot-management challenge.

    HTTP 429 from these stores is not a real rate limit — it's Cloudflare's TLS
    fingerprint check rejecting python-requests' OpenSSL ClientHello. A 200/503
    whose body contains a known Cloudflare challenge marker is treated the same way.
    """
    if resp.status_code == 429:
        return True
    if resp.status_code in (200, 503):
        body_lower = resp.text[:2000].lower()
        return any(marker in body_lower for marker in _CLOUDFLARE_MARKERS)
    return False


class _CurlResponse:
    """Minimal requests.Response look-alike wrapping curl.exe's stdout.

    Only the surface used by this script (status_code, text, .json()) is
    implemented — just enough for fetch_all_products()/probe logic to treat it
    identically to a real requests.Response.
    """

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def json(self) -> dict:
        import json

        return json.loads(self.text)


def _curl_fallback(url: str, timeout: int) -> _CurlResponse:
    """Fetch *url* via curl.exe (Windows' native Schannel TLS stack).

    Used only when a request via `requests` looks like a Cloudflare TLS-fingerprint
    block (see _looks_like_cloudflare_challenge) — curl's TLS ClientHello is not
    fingerprinted the same way python-requests'/OpenSSL's is, so it reaches the
    real origin response instead of the challenge page.

    Raises subprocess.CalledProcessError if curl.exe itself fails (network error,
    non-2xx handled by caller via the returned body/status).
    """
    result = subprocess.run(
        ["curl.exe", "-s", "-A", UA, "--max-time", str(timeout), url],
        capture_output=True,
        text=True,
        check=True,
    )
    return _CurlResponse(result.stdout, status_code=200)


def _get(session: requests.Session, url: str, timeout: int) -> requests.Response | _CurlResponse:
    """GET *url* via requests; fall back to curl.exe on a Cloudflare-shaped block.

    requests is always tried first (common case, no subprocess overhead). Only when
    the response looks like a Cloudflare challenge do we retry the SAME request via
    curl.exe and return that instead.
    """
    resp = session.get(url, timeout=timeout)
    if _looks_like_cloudflare_challenge(resp):
        print(f"  [INFO] Cloudflare-shaped block on {url} (HTTP {resp.status_code}) — retrying via curl.exe.")
        try:
            return _curl_fallback(url, timeout)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  [WARN] curl.exe fallback failed: {exc}")
            return resp
    return resp


def _domain_slug(domain: str) -> str:
    """'snitch.co.in' -> 'snitch', 'fashor.com' -> 'fashor'."""
    return domain.split(".")[0]


def _check_robots(domain: str, session: requests.Session) -> bool:
    """Return True if /products.json is allowed by robots.txt."""
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"https://{domain}/robots.txt")
    try:
        resp = session.get(f"https://{domain}/robots.txt", timeout=8)
        rp.parse(resp.text.splitlines())
    except Exception:
        return True  # if robots.txt is unreachable, assume allowed
    allowed = rp.can_fetch(UA, f"https://{domain}/products.json")
    if not allowed:
        print(f"[SKIP] {domain}/robots.txt disallows /products.json for our agent.")
    return allowed


def _strip_html(raw: str) -> str:
    """Strip HTML tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_all_products(domain: str, session: requests.Session) -> list[dict]:
    """Paginate /products.json until empty. Returns flat list of Shopify product dicts."""
    products: list[dict] = []
    seen_ids: set[int] = set()

    for page in range(1, MAX_PAGES + 1):
        url = f"https://{domain}/products.json?limit={PAGE_SIZE}&page={page}"
        try:
            resp = _get(session, url, timeout=15)
        except requests.RequestException as exc:
            print(f"  [WARN] page {page} failed: {exc}")
            break

        if resp.status_code == 404:
            print(f"  [SKIP] {domain} returned 404 on page {page}.")
            break
        if resp.status_code != 200:
            print(f"  [WARN] HTTP {resp.status_code} on page {page}, stopping.")
            break

        try:
            batch = resp.json().get("products", [])
        except Exception:
            print(f"  [WARN] Non-JSON response on page {page}, stopping.")
            break

        if not batch:
            break

        # Detect if pagination is not moving (some stores ignore ?page=N)
        first_id = batch[0].get("id")
        if first_id in seen_ids:
            print(
                f"  [INFO] Page {page} returned same products as page 1 "
                "— pagination exhausted."
            )
            break

        for p in batch:
            seen_ids.add(p.get("id"))
        products.extend(batch)
        print(f"  Page {page}: +{len(batch)} products  (total {len(products)})")

        if len(batch) < PAGE_SIZE:
            break  # last page

        time.sleep(PAUSE_S)

    return products


def normalize(products: list[dict], domain: str) -> pd.DataFrame:
    """Flatten Shopify product dicts into the generic adapter's column schema.

    Drops products with ZERO purchasable variants (2026-07-13, launch-critical:
    a live search was surfacing sold-out items because this catalogue is a
    static snapshot with no other stock signal). Shopify's public /products.json
    exposes per-variant "available": true/false — a product is kept only if at
    least one variant is available (a size being sold out while others remain
    in stock is not itself a reason to drop the product; ALL variants sold out
    is). n_out_of_stock is logged so re-syncs report exactly how many were cut.
    """
    rows = []
    n_out_of_stock = 0
    for p in products:
        variants = p.get("variants") or []
        if variants and not any(v.get("available") for v in variants):
            n_out_of_stock += 1
            continue
        price_str = variants[0].get("price", "0") if variants else "0"
        try:
            price_inr = float(price_str)
        except (ValueError, TypeError):
            price_inr = None
        if not price_inr or price_inr <= 0:
            continue

        images = p.get("images") or []
        image_url = images[0].get("src") if images else None

        rows.append(
            {
                "id": str(p.get("id", "")),
                "title": str(p.get("title", "")).strip(),
                "type": str(p.get("product_type", "")).strip() or "Fashion",
                "brand": str(p.get("vendor", "")).strip(),
                "description": _strip_html(p.get("body_html") or ""),
                "price_inr": price_inr,
                "image_url": image_url,
                "handle": str(p.get("handle", "")).strip(),
            }
        )

    if n_out_of_stock:
        print(f"  Dropped {n_out_of_stock} fully out-of-stock product(s) (zero available variants).")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Shopify store's product catalogue."
    )
    parser.add_argument("--domain", required=True, help="e.g. snitch.co.in")
    args = parser.parse_args()

    domain = args.domain.lower().strip()
    slug = _domain_slug(domain)
    out_dir = Path(f"data/raw/shopify/{slug}")
    out_csv = out_dir / "products.csv"

    session = requests.Session()
    session.headers["User-Agent"] = UA

    print(f"Target: https://{domain}")

    # robots.txt check
    if not _check_robots(domain, session):
        sys.exit(1)

    # Probe
    probe_url = f"https://{domain}/products.json?limit=1"
    try:
        probe = _get(session, probe_url, timeout=10)
    except requests.RequestException as exc:
        print(f"[ERROR] Cannot reach {domain}: {exc}", file=sys.stderr)
        sys.exit(1)

    if probe.status_code != 200:
        print(
            f"[SKIP] {domain} returned HTTP {probe.status_code} "
            "— not a public Shopify store."
        )
        sys.exit(0)

    try:
        probe_json = probe.json()
        if "products" not in probe_json or "handle" not in str(probe_json):
            raise ValueError("not shopify")
    except Exception:
        print(f"[SKIP] {domain} /products.json does not return valid Shopify JSON.")
        sys.exit(0)

    print("Probe OK. Paginating ...")
    products = fetch_all_products(domain, session)
    if not products:
        print(f"[SKIP] No products found at {domain}.")
        sys.exit(0)

    df = normalize(products, domain)
    print(f"\n{len(df):,} products with valid prices.")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved to {out_csv}")
    print("\nTop product types:")
    for pt, cnt in df["type"].value_counts().head(10).items():
        print(f"  {pt:<40} {cnt:>4}")
    print(f"\nPrice range: Rs.{df['price_inr'].min():.0f} - Rs.{df['price_inr'].max():.0f}")


if __name__ == "__main__":
    main()
