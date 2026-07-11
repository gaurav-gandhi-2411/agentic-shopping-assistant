# Catalogue Gap Report — what we can't serve yet

**Generated:** 2026-07-11 · **SHA:** `3da79e4` (current HEAD) · **Source:**
`scripts/catalogue_gap_audit.py` against `data/processed/unified/catalogue.parquet`
(61,883 rows, 34,495 women / 25,336 men) · **Raw output:**
`reports/catalogue_gap_audit_raw.txt`

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
