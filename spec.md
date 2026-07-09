# Spec: StyleMitra — Wedge Focus (Women's · Couples · Weddings · Body-Type, India)

## The wedge (what we are now the best in the world at)

StyleMitra focuses on the thing our catalogue and engine genuinely support and that no one does
well: **AI styling for the Indian occasion/wedding market — women's-led, with couple
coordination, wedding-attire combinations, and body-type-aware guidance.** We stop trying to be
a general everything-store; we become the go-to stylist for Indian weddings and occasions.

Four pillars:
1. **Women's styling + couple coordination** — "style my saree" and "what should my husband wear
   with this?" → a coordinated his-and-hers pair. (Partner styling already exists from Phase B —
   deepen it.)
2. **Wedding attire combinations** — sangeet / mehendi / haldi / reception / wedding-guest looks,
   for her, for him, and as a couple, using current trend knowledge to feel fresh.
3. **Body-type-aware guidance** — recommend silhouettes/cuts that flatter a stated body type;
   explain *why*. Pure styling knowledge, free to build, strongly differentiating.
4. **India-first** — occasions, sizing, price bands, aesthetics all Indian by default.

## Current state (build on, don't rebuild)
- StyleMitra live (asa-stylist.vercel.app/demo). Phases A/B/C done: clean deep index (~8.7k
  sarees, colour fixed, ~14k enriched with season/occasion/style), gender-consistent looks,
  honest slot suppression, partner styling, budget/occasion persistence, history-aware rationale.
- Backend asa-stylist-api on Cloud Run; image=REST /style/from-image, chat=WS /chat/stream.
- Data reality (honor it): women's ethnic/occasion depth is strong; women's footwear=1,
  jewellery=0, men's occasionwear/footwear thin. Looks suppress absent slots honestly.

## Scope

### P1 — Wedding-occasion styling as the hero flow
- First-class Indian wedding occasions: sangeet, mehendi, haldi, reception, wedding-guest,
  engagement, roka. Each maps to formality + palette + register (ethnic/indo-western) + typical
  garments (lehenga, saree, anarkali, sharara, kurta sets for her; kurta/sherwani/bandhgala/
  indo-western for him).
- "Wedding look for a sangeet under ₹8000" → a complete, occasion-correct, coherent look, women's
  by default, with honest slot suppression where the catalogue can't fill (e.g. jewellery).
- Occasion-aware retrieval + palette (haldi = bright/daytime, sangeet = embellished/evening,
  reception = glam) — reuse and extend Phase B/C occasion logic.

### P2 — Couple / his-and-hers coordination (deepen the existing feature)
- "What should my husband/wife/partner wear with this?" → a coordinated OPPOSITE-gender companion
  look: harmonized palette (complementary, not identical), matched formality + occasion, shown as
  a separate labeled board; anchor person's look unchanged. (Already built — extend for weddings:
  e.g. her lehenga ↔ his sherwani, palette-matched.)
- Also support "style us as a couple for <occasion>" → both looks generated together.
- Honor the data ceiling: the men's companion look may be thinner (shirt/kurta + trousers, often
  no footwear) — suppress honestly; never cross-gender-fill.

### P3 — Body-type-aware guidance (new, free, differentiating)
- Let the user state a body type (e.g. pear / apple / hourglass / rectangle / petite / plus) via
  chat ("I'm pear-shaped, what suits me?") or an optional chip.
- Encode a styling-knowledge ruleset: which silhouettes/necklines/cuts/drapes flatter each body
  type (e.g. A-line/anarkali for pear; empire/wrap for apple; saree drape styles by type), and
  which to avoid — as GENERAL styling guidance, framed positively and inclusively.
- Apply it two ways: (a) bias retrieval/ranking toward flattering silhouettes for the stated
  type; (b) explain the *why* in the rationale ("an A-line anarkali balances a pear silhouette…").
- Fable 5 RESEARCHES current, reputable body-type styling guidance (esp. for Indian wear —
  saree/lehenga/anarkali silhouettes) to build the ruleset. Keep it body-positive and optional;
  never make the user feel judged. This is guidance, not gatekeeping.

### P4 — Trend-fresh wedding results (Pinterest-style, feasible mechanism)
- Goal: results feel current/trendy like a Pinterest wedding board.
- FEASIBILITY (build this way): Pinterest has no free API and scraping violates its terms — do
  NOT scrape Pinterest. Instead, Fable 5 RESEARCHES current Indian wedding-wear trends via the
  web (trending colours, silhouettes, styles for this season) and uses that knowledge to (a) drive
  better trend-aware queries, (b) curate "Trending this wedding season" collections FROM OUR
  CATALOGUE, and (c) inform the rationale ("pastel lehengas are trending this season…").
- Refresh the trend knowledge periodically (documented), not live-scraped.

### P5 — India-first polish
- Indian occasion/sizing/price-band defaults; ₹ everywhere; Indian aesthetic in copy and curation.
- Suggestion chips reoriented to the wedge (occasions, "style my partner", "what suits my body
  type", "trending sangeet looks").

### Out of scope (unchanged)
- Live Pinterest/marketplace scraping; paid data/APIs; in-app payment (deep-link buy only);
  fixing the inventory ceiling (women's footwear/jewellery/men's occasionwear depth — that's
  data/partnerships, not code). Honest suppression stays.

## Tech / rules
- Free-tier only ($0); self-hosted models + free local Ollama for any enrichment; Groq LLM.
- Deep-link buy only; secrets in Secret Manager; never commit secrets/.env/indices; GCS uploads
  OVERWRITE; never set ANTHROPIC_API_KEY.
- Real-browser content-asserting proof for every feature (automation-only green has hidden bugs
  repeatedly); prove on LIVE URL; stop for the user's browser retest; no self-declared done.
- Body-type guidance must be inclusive, optional, body-positive.

## Verification (per feature, real browser)
- Wedding: "sangeet look under ₹8000 for a woman" → occasion-correct ethnic look, budget honored.
- Couple: her look → "what should my husband wear" → coordinated men's board, palette-harmonized.
- Body-type: "I'm pear-shaped, sangeet look" → flattering silhouettes + a why-note; body-positive framing.
- Trend: "trending sangeet looks" → a curated current-trend collection from catalogue + trend rationale.
- Regression: Phase A/B/C behaviors still green; gender consistency + honest suppression intact.

## Build order (each: plan → build autonomously → real-browser proof → stop for user retest)
1. P1 wedding-occasion styling (hero flow) + P5 chips/India polish.
2. P3 body-type guidance (research ruleset → apply to ranking + rationale).
3. P2 deepen couple coordination for weddings (her↔his palette-matched pairs; "style us as a couple").
4. P4 trend-fresh curation (research trends → catalogue collections + rationale).
Present a short plan per phase; prove each in a real browser; stop for the user's retest.

## Note on the goal
This wedge — Indian wedding/occasion styling, couples, body-type-aware — is the fundable
positioning: differentiated, catalogue-supported, and something incumbents do poorly. The
remaining ceiling (inventory breadth: footwear/jewellery/men's occasionwear) is what
partnerships/funding unlock, not code — so these features make the wedge excellent within the
data we have, and define exactly what deeper inventory would complete.
```
