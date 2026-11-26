# Live 5-chip check, cap 3 (2026-11-26, local backend, real index, gpt-oss-20b only)

Command: `python scripts/plain_search_live_gate.py --runs 1 --max-calls 10 --max-tokens 15000 --gap-s 5`
(code at the commit that adds this directory minus this folder). Raw: `live.json`, `console.txt`.

Totals: 9 live LLM calls, 3,058 tokens (prompt + completion), models seen: `openai/gpt-oss-20b` only
(no 120b / qwen path: groq_fallback_models empty, reranker fallback off, foreign-model guard).
Preconditions: canary run 45750231450 (2026-11-26T03:38Z) completed; prod logs of the 15 minutes
before the run (and after it) had no llm_fallback_served / gpt-oss-120b / qwen / 429 line.

Counts come from the graph's `plain_search_postprocess` event for the turn: `removed` = rejections
on the last selection pass (duplicate / store_cap), `gate_dropped` = cards the later result gates
dropped and the selection refilled (summed over passes). No event logged = all zero.

| Chip | Route | Shown | Dedupe drops | Store-cap drops | Gate-dropped / refill | Genders shown |
|---|---|---|---|---|---|---|
| Sangeet outfit under 3,000 | outfit composer (not plain search) | 3 | n/a | n/a | n/a | women |
| Haldi kurta set | search | 5 | 0 | 0 | 0 | men 4, women 1 |
| Wedding guest saree | search | 5 | 0 | 7 | 0 | women |
| Office wear under 1,500 | search | 4 | 0 | 0 | 6 gate-dropped, shown 4 of 5 | men only |
| Men's kurta for a wedding | search | 5 | 0 | 0 | 0 | men |

## Top-5 titles (store | gender | price INR | title)

Haldi kurta set
- kisah | men | 3299 | Yellow Geometric Printed Kurta Set
- globalrepublic | women | 2299 | Mustard Knit Regular Fit Kurta Set
- kisah | men | 3299 | Yellow Printed Front Open Kurta Set
- kisah | men | 2499 | White Cotton Kurta Set with Mustard Ikat-Inspired Prints and Cream Trousers - Festive & Wedding Ensemble
- jadeblue | men | 3667 | Yellow Self Design Regular Fit Kurta Set | Azania

Wedding guest saree
- libas | women | 5399 | Pink Embellished Satin Saree
- 5-elements | women | 41900 | Dityaa Gold (Saree)
- libas | women | 6299 | Burgundy Silk Blend Saree
- libas | women | 2279 | Navy Georgette Saree With Floral Embellished Pallu
- 5-elements | women | 41900 | Dityaa Silver (Saree)

Office wear under 1,500 (budget 1500; 4 shown)
- jadeblue | men | 1078 | Light Blue Checked Slim Fit Formal Shirt | Metal
- jadeblue | men | 1078 | Peacock Blue Solid Slim Fit Formal Shirt | Metal
- jadeblue | men | 890 | White Solid Slim Fit Formal Shirt | Metal
- powerlook | men | 999 | Green Button Down Checks Shirt

Men's kurta for a wedding
- vastramay | men | 2199 | Men Black Silk Embroidered Kurta
- jadeblue | men | 1312 | Maroon with Circular Bandhani Print Cotton Blend Men's Kurta
- globalrepublic | men | 1259 | Mens Black Round Nack Solid Kurta
- vastramay | men | 1339 | Plus Men's Black Cotton Linen Kurta
- globalrepublic | men | 1469 | Mens Off White Round Nack Printed Kurta

Sangeet outfit under 3,000 (routes to the outfit composer, reported as such)
- tjori | women | 1069 | Set of 2: Snow White Cotton Anarkali Kurti with Flared Pants
- vastramay | women | 489 | Women's Red Handcrafted Potli Clutch
- vhaan | women | 950 | Ruby Ladies Kolhapuri Chappal Embroidered With Authentic Handicraft

## Criteria (informational; misses are misses)

- Duplicates: 0 in every turn. Store cap (<= 3): met in every turn (max per store: 1, 3, 3, 3, 2).
- Budget: 0 over-budget cards (Office wear).
- Gender mix: Haldi mixed (4 men, 1 woman). Office wear is gender-neutral with 4 men and 0 women, although
  the selection reported interleaved=true and the later result gates dropped 6 cards. Not diagnosed
  further (cause unverified: a women's item may have been gate-dropped). A miss on that criterion.
  The Women / Men refine control is offered on that turn (refine=yes).
- Shown < 5 on Office wear (4) after the gates dropped 6 cards: pool narrow for the budget.
- Wedding guest saree: two 41,900 INR "Dityaa" sarees in the top 5 are catalogue prices, not touched by
  this change; flagged for the owner, not a post-processing effect.
- One run per chip (n=1 per chip): this is an existence check of the behaviour, not a statistic.
