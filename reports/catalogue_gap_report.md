# Catalogue Gap Report — what we can't serve yet

**Generated:** 2026-07-11 · **SHA:** `3da79e4` (original), follow-up patch below ·
**Source:** `scripts/catalogue_gap_audit.py` against
`data/processed/unified/catalogue.parquet` (61,883 rows, 34,495 women / 25,336 men) ·
**Raw output:** `reports/catalogue_gap_audit_raw.txt`

## Follow-up (2026-07-11, later same day) — faceted what was cheap, corrected two findings

Per request: give the sherwani/jewellery/footwear items a filterable facet where cheap.
`scripts/patch_thin_category_facets.py` (2-row surgical patch, no bulk reclassification,
no index rebuild — the retrieval hard-filter reads `product_type_name`/`facets` live off
the dataframe every query) + `intent_parser._GARMENT_RULES`/`graph._PRODUCT_TYPE_KEYWORDS`
now recognize "sherwani"/"bandhgala" as a garment type.

- **Sherwani: now faceted, worth it.** The single real sherwani had product_type_name
  `"Kurtas, Ethnic Sets and Bottoms"` — same generic bucket as 117 unrelated items — so a
  hard type filter could never isolate it; "sherwani for X" relied on embedding-similarity
  noise (live-proven: returned a crop top and a bucket hat). Patched to its own
  `product_type_name="sherwani"`. Verified: a hard filter now returns exactly the 1 correct
  item, not noise. Strict eval confirms: `gold_024` went from 1/5 (4 data-ceiling misses) to
  the data-ceiling row disappearing from the taxonomy entirely (0 remaining data-capped
  misses catalogue-wide). Still only 1 item — this fixes *findability*, not *inventory
  depth*; the "need dozens across colours" ask below is unchanged.
- **Women's footwear: corrected from 1 to 0 — the "1" was a mislabel, not inventory.**
  The single row previously counted (`Wine High Waisted Boot Cut Cotton Blend Lower`,
  `product_type_name="footwear"`) is a **bottom garment**, not footwear — actively wrong
  data, not thin data. Patched to `product_type_name="trousers"`. There is no facet worth
  adding here because there is no real item behind it; true women's footwear count is
  **zero**, unchanged as a genuine inventory gap.
- **Jewellery: not worth faceting — investigated further, inventory isn't what it looks
  like.** Broadening past the original keyword count: `product_type_name="Accessories"`
  is **100% men's** (100 items) — there is no women's-accessories bucket at all to facet.
  A precise word-boundary search for necklace/earring/bangle/bracelet/maang
  tikka/jhumka/nose pin/anklet/pendant/choker/mangalsutra against women's rows returns
  **1 false positive** (a top with a "choker neck" cut, not jewellery). The 31 free-text
  matches reported originally are exclusively **men's bracelets**. A coarse "jewellery"
  facet is cheap to add (31 items, one existing bucket) but wouldn't serve the actual
  ask — a wedding-stylist "jewellery" query means bridal/women's pieces, and none exist.
  Not implemented; **this is a pure inventory gap, not a facet gap** — faceting men's
  bracelets under "jewellery" would be building a filter for the wrong 31 items.
- **Found but not fixed (out of the three named categories, flagged for a future
  follow-up if wanted):** of the 214 men's-ethnic-bottom name-matches, 89 are catalogued
  `nightwear`. Sampling them shows a clean, verifiable discriminator — bare
  `"Men Pyjama (Pack of N)"` listings are genuine sleepwear (correctly classified, leave
  alone), while `"Men Kurta and Pyjama Set — Dupion Silk/Jacquard/Pure Cotton"` and
  `"Men Ethnic Jacket and Pyjama Set"` listings are festive ethnic sets miscatalogued as
  bedtime wear. Not patched this session (a ~60-80 item reclassification wasn't part of
  the three named categories and deserved its own explicit go-ahead rather than riding
  along) — noted here as a well-scoped, evidenced next step.

Strict gold eval impact (pipeline mode, hand-audited, zero unlabeled, commit-pinned):
occasion category P@5 0.778 → **0.824**; data-ceiling miss reason: 4 → **0** catalogue-wide.
See `reports/strict_eval_FACETFIX_pipeline.txt`.

---

## Original report (2026-07-11)

**Methodology:** every number below is a direct pandas count/filter against the live
catalogue — no LLM, no retrieval, no ranking involved. This report answers one question
only: *does a matching product exist in inventory at all?* It is deliberately separate
from ranking quality (covered by `reports/strict_eval_POSTFIX_pipeline.txt`) — a query
can fail for either reason, and conflating them would misdirect either an engineering fix
or a buying decision. Every row here was confirmed to be a **true zero/thin count**, not
a ranking miss — see the "confirmed NOT a gap" section for the one case that looked like
inventory but wasn't.

## Zero or near-zero inventory (hard floor — no code fix can close these)

| Category | Count | Detail |
|---|---|---|
| **Sherwanis (men)** | **1** | Single item, catalogued under a generic "Kurtas, Ethnic Sets and Bottoms" type — no dedicated `sherwani` facet exists at all, so it can't even be reliably *filtered* on, let alone offered in colour choices. Zero sherwanis in any colour other than whatever this one item is (colour attribute is blank). |
| **Bandhgalas (men)** | **0** | None in catalogue. |
| **Women's footwear** | **1** | Single item, one store (globalrepublic). Every women's outfit board ships shoe-less with an honest suppression note as a direct result. |
| **Ethnic footwear, either gender** (jutti/mojari/kolhapuri) | **0 / 0** | Zero for women, zero for men — every festive/wedding look defaults to Western footwear or none at all. |
| **Jewellery** (dedicated facet) | **0** | No `jewellery` product-type facet exists in the schema. 31 items match jewellery keywords in free text (necklace/earring/bangle/etc.), but with no facet they can't be reliably filtered, boosted, or offered as a distinct accessory category — for practical purposes this is a zero. |

## Thin inventory (technically non-zero, functionally unusable)

| Category | Count | Why it's still a gap |
|---|---|---|
| **Men's blazers** | **7 total** (5 under ₹5,000) | 3 colours only: Blue (2), Black (1), Off White (1) — the other 3 have no colour attribute at all. Any query naming a colour outside blue/black/white (navy is a blue-family match, but red/green/grey/brown blazers: **0**) returns nothing genuine. Concentrated in 2 stores (snitch: 4, flipkart: 2, globalrepublic: 1). |
| **Men's ethnic bottoms** (churidar/pyjama/dhoti by name) | **214 by name, but only 7 usably classified** | Of 214 name-matches, 118 are dumped into the same generic "Kurtas, Ethnic Sets and Bottoms" bucket as the sherwani (no fine-grained facet), and **89 are catalogued as `nightwear`** (bedtime pyjamas) — not festive-wear, and would be actively wrong to offer for a wedding look. Only 7 rows carry a clean, usable `kurta`-adjacent facet. |

## For contrast — a healthy category

**Men's kurtas: 272 items across 21 distinct colours**, cleanly faceted. This is what
"not a data ceiling" looks like — the retrieval/ranking work from this session targets
categories shaped like this one; none of it can help the categories above.

## Confirmed NOT a data ceiling (ranking-fixable, kept out of this report on purpose)

**"Red embellished lehenga for sangeet" (strict eval `multi_003`, scored 0/5 in every
mode tested)** looked like it could belong on this list but does not: direct query
confirms **4 genuine red-embellished adult lehengas exist** in catalogue (e.g. "Anouk Red
& Pink Embellished Ready to Wear Lehenga & Choli"). They simply never rank into the
top-5 for that exact phrasing — a colour-aware reranking gap, not missing inventory.
Already labeled `colour-family` (code-fixable) in `eval/fixtures/strict_gold_labels.yaml`
and left for a future ranking pass, not a buying decision.

## What this means, plainly

The three defensible line items for a data-partnership or catalogue-expansion
conversation, in priority order by how often they're asked for in the wedding-stylist
use case:
1. **Men's occasionwear depth** — sherwanis (1→need dozens across colours, with a real
   `sherwani` facet), bandhgalas (0), a usable men's ethnic-bottom facet (currently
   conflated with nightwear).
2. **Footwear, both genders** — women's is a single item; ethnic footwear (jutti/mojari)
   is zero for everyone.
3. **Jewellery as a real facet** — 31 items exist in free text with no way to reliably
   surface, filter, or recommend them as accessories.

Men's blazers (7, 3 usable colours) is a smaller, cheaper add if a quick win is wanted
before a full partnership — even 20–30 more blazers across navy/grey/maroon would close
most of the remaining occasion-register misses in that category.
