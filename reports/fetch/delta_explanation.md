# Feed-vs-index deltas (R3 item 6a): what is explained, what is not

**Status: partly explained from code and older data; the filter-by-filter counts the task asked
for could NOT be produced today**, because every `/products.json` request from this machine was
answered HTTP 429 (see `fetch_pass_2026-11-26.md`). Whatever is marked "consistent with" or
"undetermined" below is exactly that. The tool that produces the missing numbers is committed
(`scripts/ops/feed_delta_funnel.py`, tested) and needs only the fetched pages.

## Inputs (all re-derived today)

| name | what | source |
|---|---|---|
| feed | items per store, #132 run 43920759269 (2026-11-16, GitHub runner, 0.5 req/s). Counts EVERY product in `/products.json`, sold out or not | artifact `rehearsal-summary` -> `rehearsal_summary.json` (downloaded read-only) |
| snapshot | rows of the raw CSV the live index was built from (`brands/<slug>.yaml` `catalogue_path`, primary checkout, read-only); date = file mtime | `pd.read_csv(...)` row count |
| index | rows per `store` in the live `data/processed/unified/catalogue.parquet` (112,425 rows, 42 stores, built 2026-07-24) | `pd.read_parquet` value_counts |
| V3a sample | n=20 per store, 2026-11-14: share of indexed items whose page is now 404 (delisted) or whose live `.js` says sold out | `reports/wave1b/v3a/buylinks_summary.csv` (primary checkout) |

Which stores are below the index: exactly four have feed < index (#132): irasoles, kalpraag,
mortantra, tjori. All other 34 of the 38 refreshable stores have feed >= index.

## Mechanisms established from the code (not from fetched data)

1. **The index only holds in-stock, priced products; the feed count holds everything.**
   `scripts/download_shopify.py:238` drops a product when none of its variants is `available`;
   `:246` drops price missing or <= 0. So feed minus index is NOT "items we are missing": it
   contains every sold-out product by construction. #132 counted raw products, which is why the
   gap is large for stores with many sold-out items.
2. **The old download hard-capped at 15,000 products**: `scripts/download_shopify.py:43`
   `MAX_PAGES = 60` x 250. Stores whose feed is larger are truncated both in the snapshot and in
   #132 (libas, snitch, theamethyststore all report exactly 15,000 in #132). For a capped store
   the index lacks everything beyond item 15,000 and the feed count is a floor, not a size.
3. **The build adds only small filters after that**: `scripts/01_build_retrieval.py:165` religious
   decor rows, `:169` rows without a description; `scripts/build_unified_index.py:238-262` fabric
   bolts, hygiene and `allowed_categories` (the live index predates the hygiene step: it was built
   2026-07-24, R1 #130 hygiene only applies on the next rebuild). There is **no per-store row cap
   and no public-store filter at build time** (grep for `per_store|store_cap|max_per|head(|.sample(`
   in `build_unified_index.py` returns nothing; `public` and store diversity are applied at query
   time: `src/config/store_registry.py`, `src/retrieval/hybrid_search.py:95`), so those two
   hypotheses are ruled out as causes of a build-time gap.

## Per-store table (n = rows or items; "-" = not measurable today)

| store | feed (#132) | snapshot rows (date) | index | feed - index | snapshot - index | V3a n=20: gone / now sold out | verdict |
|---|--:|--:|--:|--:|--:|---|---|
| voylla | 12,172 | 2,871 (2026-07-19) | 2,835 | +9,337 | +36 | 0 / 7 | undetermined split: sold-out (mechanism 1) vs new products |
| snitch | 15,000 (capped, floor) | 7,040 (2026-07-12) | 7,040 | +7,960 | 0 | 4 / 2 | consistent with mechanism 1 + 2: see note A |
| sukkhi | 8,964 | 1,082 (2026-07-19) | 1,073 | +7,891 | +9 | 0 / 4 | undetermined split, as voylla |
| irasoles | 249 | 254 (2026-07-19) | 254 | -5 | 0 | 0 / 0 | store delisted a few items; consistent, see note B |
| kalpraag | 541 | 637 (2026-07-19) | 637 | -96 | 0 | 12 / 0 | delisting + new products: see note C |
| mortantra | 4,710 | 5,213 (2026-07-19) | 5,213 | -503 | 0 | 5 / 0 | consistent with delisting (25% sample gone) |
| tjori | 2,472 | 2,739 (2026-07-19) | 2,714 | -242 | +25 | 0 / 2 | consistent with delisting, not established: see note B |

Snapshot minus index (0 to 36 rows) is the small build-filter loss of mechanism 3; it is zero for
five of the seven stores and cannot be attributed row by row without the build's drop reports
(the live index has none).

### Notes

* **A (snitch).** Index (7,040) equals the snapshot, and #132 saw 15,000 = the cap. If the July
  download also scanned 15,000 products (it must have, if the feed was already >= 15,000 then),
  exactly 15,000 - 7,040 = 7,960 were removed by the in-stock / price filter, which equals the
  reported delta to the unit. That is arithmetic consistency, not a measurement: the July feed
  size is unknown, and the true current feed is larger than 15,000. Same shape for
  theamethyststore: 15,000 scanned, 10,132 kept in the snapshot (4,868 removed), 10,085 in the
  index (47 more removed by decor / description filters, as designed).
* **B (irasoles, tjori).** The feed still contains sold-out products, yet is smaller than the
  index, so the stores removed more products than they added since July. V3a found 0 of 20 sampled
  pages gone for both; a 0/20 sample has a Wilson 95% upper bound of 16.1%, which covers the
  observed deficits (irasoles 5 of 254 = 2.0%, tjori 242 of 2,714 = 8.9%). Consistent, but the
  sample cannot confirm it.
* **C (kalpraag).** The only store with any fetched data today (page 1, n=250 of ~541; the
  funnel tool, run on the saved page): 250 distinct ids -> 250 in stock and priced -> 248 after
  the 01_build filters (2 without description, 0 decor) -> 248 after cleaning and hygiene.
  Against the 637-row snapshot, 104 of those 250 are in the snapshot and 146 were created after
  it (created_at 2026-07-25 to 2026-11-12). V3a found 12 of 20 indexed pages gone (404). So the
  catalogue churns heavily in both directions, which is enough to turn 637 into 541; the split
  for the full feed was not measured (only page 1 was fetched).
* **voylla and sukkhi: cause not determined.** Mechanism 1 certainly contributes (V3a: 7 of 20
  voylla and 4 of 20 sukkhi items that were in stock in July are sold out today, so sold-out
  churn is high), but nothing available today bounds how much of +9,337 / +7,891 is sold-out
  stock versus products added since July. Voylla (12,172) and sukkhi (8,964) are below the
  15,000 cap, so truncation does not explain them. Hypotheses NOT excluded: large sold-out share,
  large catalogue growth, products without a price. Items counted as products because of
  variants are excluded (one row per product in `_adapt_generic`, `src/catalogue/adapter.py:319`).

## To finish 6a (needs the throttle to lift)

```
python scripts/ops/fetch_pass.py --only voylla,snitch,sukkhi,irasoles,kalpraag,mortantra,tjori \
    --out s.json --raw-dir <scratch outside git>
python scripts/ops/feed_delta_funnel.py --slug voylla --raw-dir <scratch> --snapshot-csv <primary>/data/raw/shopify/voylla/products.csv
```

The funnel prints, per store: distinct ids, all-variants-sold-out, after normalize (in stock,
price > 0), religious-decor and no-description drops, rows after the 01_build and Phase A /
hygiene filters, and the snapshot overlap (still in stock / now sold out / not in feed / new and
created after the snapshot). Those eight numbers per store settle the split above.
