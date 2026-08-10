# fabric_score_delta promotion — office-register hard-gate diagnosis (2026-08-10)

**Question:** would promoting `fabric_score_delta`'s existing
`HALDI_LIGHTWEIGHT_KEYWORDS` vocabulary (`cotton`, `floral`, `tie-dye`,
`georgette`, `chiffon`, `printed`, `casual`, `lightweight`, `summer`,
`marigold`, `yellow`, `daisy` — `src/agents/outfit/slots.py`) from its
current soft +/-0.1 rerank nudge into a **hard reject gate for the office
occasion** safely catch the office-register misses this catalogue has
(the athleisure-track-jacket-with-a-formal-name and casual-co-ord-set
misses found in the compose-wave shrug-fix follow-up)?

**Method:** `python -m eval.diagnose_fabric_score_delta_promotion`. Every
item already hand-labeled (true or false) across the 8 office-context
queries in `strict_gold_labels.yaml` (`gold_015`, `multi_001`,
`multi_005`, `multi_012`, `multi_022`, `occ_adv_003`, `occ_adv_008`,
`occ_office_001` — 87 labeled query-item slots) was checked against the
real catalogue row's `prod_name` + `detail_desc` for any
`HALDI_LIGHTWEIGHT_KEYWORDS` hit — the same "measure against currently-
labeled candidates" methodology `reports/compose_wave_final_20260807.md`'s
Cluster A used for the embellishment-vocabulary attempt.

## Result

| Basis | Would-be-wrongly-rejected (currently relevant) | Would-be-correctly-rejected (currently irrelevant) | FP rate |
|---|---|---|---|
| Raw, per query-item slot | 31 | 14 | **68.9%** (31/45) |
| Deduplicated by distinct article_id | 26 | 8 | **76.5%** (26/34) |

**Worse than both prior attempts** (60% embellishment-vocabulary, 25%
desc-marker broad-scan — both `reports/compose_wave_final_20260807.md`).

Sample false positives — genuinely office-appropriate items that would be
hard-rejected purely for containing an occasion-neutral fabric/pattern
word:
- "Self Design Single Breasted Formal Men Full Sleeve Blazer" — matched
  `casual`/`printed` (a stray "casual" mention elsewhere in the desc; the
  item is a formal blazer).
- "Regular Fit Men White/Grey/Beige/Brown Cotton Blend Trousers" ×4 —
  matched `cotton`. Cotton is an entirely occasion-neutral fabric for
  men's formal trousers.
- "Linen Blend Regular/Relaxed Fit Trousers" ×2, "Beige Linen Blend
  Trousers" — matched `cotton`/`summer`/`casual`. Linen trousers are a
  genuine business-casual/office-appropriate category, not casual
  register, in this catalogue (same finding as the separate linen-blazer
  false positive found for the desc-marker athleisure check in the same
  wave).
- "Women Floral Print White Dress", "Women White Multi Floralary Floral
  Dress" — matched `floral`. A floral print alone says nothing about
  office-appropriateness; this catalogue's own gold labels already
  accept plain floral dresses for office as "business-casual plausible."
- "Women Zebra/Collared Viscose Printed Shirt" ×2 — matched `printed`.
  Printed dress shirts are a standard, common office-shirt category.

The small number of genuine true positives (novelty tees, the athleisure
jacket, the leisure co-ord set, the "Occasion: Casual" dress) are **already
correctly caught by other, more specific, already-shipped mechanisms**
(`_CASUAL_MARKER_RE`'s `tee`/`t-shirt`/`leisure` markers, and hand-labeling
grounded in the item's own explicit spec/marketing text) — this hard gate
would be almost entirely redundant with what already exists while adding
massive, unacceptable collateral damage to genuinely formal cotton/linen/
printed/floral items.

## Conclusion

**Rejected. Not built.** Confirms the standing pattern: a text-formality
signal derived from product marketing copy/fabric-descriptor vocabulary is
not a reliable formality signal in this catalogue, regardless of which
specific vocabulary or promotion mechanism is tried. See
`eval/README.md`'s "Text-formality signals from marketing copy have failed
repeatedly" standing rule.
