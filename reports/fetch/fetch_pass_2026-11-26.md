# Feed fetch pass, 2026-11-26 (from this machine, polite, one pass): BLOCKED BY HTTP 429

Fetcher: `scripts/ops/refresh_feed.py` (limiter + robots + 429/5xx policy) driven by
`scripts/ops/fetch_pass.py` at commit f31d5a4 (fetcher code unchanged since 2ad5eb3). Registry
scope: the 38 refreshable stores (`refreshable_stores()`); the other registry entries are not
fetchable by design (myntra, flipkart, hm: static dumps; bhasinbrothers feed 404, priyaasi
unverified, berrylush password-walled: all `refresh_enabled=False`). Nothing was written into the
catalogue or index; no item payload is committed.

## Result: no usable counts

Every `/products.json` request from this machine was answered `429` with `Retry-After: 60` and a
Shopify JSON body (`"code": "TOO_MANY_REQUESTS"`, served via Cloudflare). `robots.txt` answered 200
on every store. The fetcher applied the production policy exactly (wait the 60 s once, retry once,
stop the store on the second 429): all 38 stores ended `rate_limited`.

| quantity | value | source |
|---|---|---|
| stores attempted / `rate_limited` | 38 / 38 | `summary.json` of the pass (scratch, not committed) |
| stores with any items | 1 (kalpraag: page 1 = 250 items, then 429 twice on page 2) | same |
| `capped` stores | 0 | same |
| HTTP 403 / challenge events | 0 / 0 | same |
| HTTP 429 events | 76 | same (`http_status_counts`: 200 x 39, 429 x 76) |
| requests in the pass | 115 (38 robots + 1 feed page = 39 x 200; 76 x 429) | request log |
| elapsed | 2,347.1 s (37 stores x ~61.7 s: 2 requests, one 60 s wait; kalpraag 63.1 s) | `summary.json` |
| max observed request rate per domain | 1.0 req/s: smallest gap between consecutive request starts to the same domain over all 115 requests = 1.0000989 s (per-domain minimum gaps 1.000 to 1.277 s) | request log (`t` = monotonic clock at request start) |

Previously truncated stores (libas, snitch, theamethyststore): NOT MEASURED. Each got robots 200,
then 429, wait 60 s, 429, stopped. Their new totals, pages fetched and `capped` flags are unknown;
the cap/pagination behaviour is covered only by the mocked tests.

## What was tried, and why it stops here

* Single polite probes before the pass (shopkop, kraftojodhpur, vhaan, 5-elements, vhaan
  `/collections/all/products.json`): all 429, Retry-After 60.
* 12 probes one per 3 minutes, 07:09 to 07:42 local, each a single request: all 429
  (`waitgo.log`, scratch). The pass itself then ran 07:44 to 08:23: all 429 except the kalpraag
  page-1 request at ~08:07. A last single probe at 08:25 (voylla): 429.
* Diagnostic single requests outside the logged pass number roughly 26 (not individually logged:
  NOT MEASURED exactly), spread over ~2 hours across 5 domains.
* Not tried, deliberately: a different User-Agent, `curl.exe`/another TLS stack
  (`download_shopify.py`'s fallback), a proxy or a GitHub runner. The task rules say a store that
  pushes back is stopped and recorded, never evaded; a `429` with `Retry-After` is exactly that.

Cause NOT determined. Observed facts: the same machine got HTTP 200 on 38 of 40 feed probes on
2026-11-14 (V3a, `reports/wave1b/v3a-catalogue-inventory.md`, "Main technical risk" paragraph); the GitHub-runner
rehearsal (#132, run 43920759269, 2026-11-16) got 599 x 200 and no 429 at 0.5 req/s global; today
all stores, including stores not contacted earlier in the session, answer 429 at the very first
feed request. That pattern is more consistent with a per-IP (edge-level) throttle on this residential
address than with a per-store limit, but whether it is the IP, the User-Agent string or the TLS
fingerprint was not tested (testing the last two would be evasion).

## Re-running when the throttle lifts

```
python scripts/ops/fetch_pass.py --out summary.json --md-out table.md --log-out request_log.csv \
    --raw-dir <scratch dir outside git>
python scripts/ops/feed_delta_funnel.py --slug voylla --raw-dir <same dir> \
    --snapshot-csv <primary checkout>/data/raw/shopify/voylla/products.csv
```

Stop condition recorded for the next attempt: if the first feed request of a store is 429 again,
do not loop; one probe per hour is enough to find out when it clears.
