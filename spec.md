# Project Spec: agentic-shopping-assistant — "Complete the Look, India Edition"

## Goal

Build the best **India-specialized fashion-styling engine** in existence: any item or
occasion request → a coherent, shoppable **outfit**, with **Indian ethnic & occasion styling
as the hero capability**, a **conversion-data flywheel** as the long-term moat, and a **brand
dashboard** that proves the ROI (basket-size lift).

Strategic frame (do not lose this): the Indian marketplace giants (Flipkart/Myntra/Shopsy,
Walmart-backed; Amazon) are already shipping generic agentic styling and are on the global
rails. We do NOT compete with them on breadth or checkout. We win on the thing they're
mediocre at and won't prioritize: **deep, culturally-correct Indian occasion/ethnic styling
for the long tail of brands they don't serve**, plus a **proprietary taste→conversion dataset**
that compounds. This is both a niche product (for D2C brands and smaller marketplaces) and
the specialized IP/capability that makes the project valuable to — or acquirable by — the winners.

The make-or-break remains **outfit coherence**. For India that means occasion-correctness and
ethnic-set logic must be excellent. A wrong sangeet look is worse than none.

## Current state (read first)

- Live styling app (LangGraph + hybrid retrieval + reranker), deployed Cloud Run + Vercel,
  8 brands, demo mode with Postgres-backed cost/rate guards, 👍/👎 feedback signal already wired.
- **A coherence ruleset has already been PROPOSED by the orchestrator** (slot taxonomy:
  top/bottom/one_piece/footwear/outerwear/accessory; colour-harmony tiers; a 1–5 formality
  matrix; slot-fill-per-anchor logic; a 25-anchor coherence eval). **Extend that work — do not
  redo it.** This spec adds the India layer, the flywheel, and the dashboard on top of it.
- Two table-stakes defects still open: image rendering for all brands (Myntra returned
  text-only); dead PDP links (Snitch links hit a "we moved" page on a stale scrape).

**Load-bearing — do not break:** grounding layer, LLM Protocol, demo/auth/cost guards, the
8-brand config pattern, the already-proposed coherence engine's structure.

## Scope

### In scope (this iteration)

**H — Hero: Indian occasion & ethnic styling (the differentiator)**
- Extend the occasion taxonomy beyond western (casual/office/party) to India-first occasions:
  `casual`, `smart_casual`, `office`, `party_evening`, `festive_puja`, `wedding_guest`,
  `sangeet_mehendi_haldi`, `traditional_ethnic`. Each maps to a formality band + an
  ethnic/western/either lean.
- Extend slot logic for **ethnic sets**: an ethnic top (kurta/kurti/kameez) anchor fills
  `bottom` (palazzo/churidar/leggings/sharara) + `accessory` (dupatta) + optional `footwear`
  (juttis/heels); lehenga/saree (`one_piece`) fills `accessory` (dupatta/jewellery/clutch) +
  `footwear`, never a top/bottom. Ethnic items must not be paired into western looks for
  formal/festive occasions and vice-versa (formality + ethnic-lean coherence filter).
- **Occasion-driven entry** (the killer demo): "build me a sangeet look under ₹5000",
  "festive kurta set", "wedding-guest outfit" → ethnic-leaning composition with correct
  occasion formality. This, not generic search, is the headline.
- Indian colour/occasion sensibility in the colour rules (festive = richer/jewel tones allowed
  to co-star, not forced neutral; this needs GG sign-off — see escalations).

**F — Conversion-data flywheel (the moat — first-class, instrument from day one)**
- **Event logging:** capture `look_shown`, `item_view`, `add_single`, `add_the_look`,
  `swap_slot`, `thumbs_up/down` — each tagged with anchor item, filled slots+items, occasion,
  price band, and brand. Store in Postgres (already a dependency; new append-only events table).
- **Aggregation:** compute pairing-level signal — for an (anchor_category, fill_category,
  occasion) tuple, a positive-signal rate from add-the-look / thumbs. Materialize as a stats table.
- **Feedback into ranking:** blend a data-derived boost into candidate scoring — cold-start uses
  the rules; as data accrues, learned pairings that convert get ranked up. Keep it honest and
  simple (a transparent boost term, NOT a black-box recommender). Document the blend.
- This loop is the compounding first-mover advantage; treat it as a feature, not telemetry.

**D — Brand dashboard (the ROI proof — what closes a sale)**
- A `/dashboard` view (basic, real data from the events table) showing: looks shown,
  add-the-look rate, **estimated AOV uplift** (mean look-total vs mean single-item price —
  defined honestly, see escalations), top-converting pairings, and a breakdown by occasion and
  brand. This is the artifact you show a brand/marketplace to justify paying.

**T — Table-stakes fixes (foundational, do first)**
- Fix image rendering for all 8 brands (no outfit ships without images).
- Dead-PDP-link validation at index build; prefer live-link items for demo brands.

**Carried from prior spec (still in):** visual `OutfitBoard` with "Buy the look — ₹total";
"Style this" anchor-from-item entry; conversational refinement (swap/budget/occasion/"more
ethnic"/"more formal"); the structured outfit output; the coherence eval (extended for India).

### Out of scope (explicitly defer/avoid)
- Cross-**website**/cross-marketplace aggregation and price-comparison — strategically wrong
  (walks the client's traffic to competitors); never build.
- Checkout / payment / ordering-on-behalf — link-out only; the agentic rails (UCP/AP2) are the
  giants' to build, not ours.
- Full ML personalization / collaborative-filtering recommender — the flywheel v1 is
  instrument + aggregate + transparent re-rank only.
- Embeddable PDP JS widget/SDK — next phase, once the engine + dashboard are undeniable.
- Virtual try-on / image generation; fit/size engine.

## Tech stack
- Existing stack only. Extend the proposed `outfit/` composition modules. Add occasion/ethnic
  config to `BrandConfig`. New Postgres tables for events + pairing stats (append-only). New
  `/dashboard` route + a simple frontend dashboard page. No new heavy deps without escalation.

## Architecture (additions to the already-proposed structure)
```
src/agents/outfit/
  slots.py        # (proposed) + ethnic-set fill logic
  composer.py     # (proposed) + occasion-driven composition + data-boost blend
  coherence.py    # (proposed) + India occasion/ethnic-lean rules
  occasions.py    # NEW: occasion taxonomy → formality band + ethnic/western lean
src/flywheel/
  events.py       # NEW: event logging API + schema
  stats.py        # NEW: pairing aggregation + the ranking boost term
api/routes/
  events.py       # NEW: POST styling events
  dashboard.py    # NEW: GET dashboard metrics
alembic/          # NEW migration: styling_events + pairing_stats tables (append-only, no ALTER existing)
frontend/
  components/OutfitBoard.tsx, StyleThisButton.tsx
  app/dashboard/  # NEW: basic ROI dashboard page
brands/<brand>.yaml  # + occasion defaults / ethnic-lean hints
```

## Verification commands
```yaml
- name: tests
  cmd: pytest -m "not requires_ollama" -v
  required: true
- name: lint
  cmd: ruff check .
  required: true
- name: frontend-build
  cmd: cd frontend && npm run build
  required: true
- name: coherence-eval
  cmd: python -m eval.outfit_coherence
  required: true   # extended with India occasion/ethnic anchors
```

## Subagent rules
- `executor` codes, `verifier` verifies. Orchestrator delegates, never codes.

## Escalation rules (ask before doing)
- **BEFORE building H:** present the revised India occasion taxonomy (the 8 occasions →
  formality band + ethnic/western lean), the ethnic-set fill rules, and the festive colour
  handling, for GG sign-off. This is culturally-specific product judgment — do not guess it.
- **BEFORE building F:** present the event schema and the data→ranking blend approach for sign-off.
- **BEFORE building D:** present the exact metric definitions — especially how "AOV uplift" is
  computed — so we don't ship a misleading number. Honesty here protects the sales pitch.
- Extend the existing 25-anchor coherence eval with India anchors (sangeet/festive/kurta-set/
  lehenga); present the additions for sign-off before it gates.
- Ask before any dependency, any LangGraph routing change, any new Alembic migration (show plan),
  or any cloud redeploy.

## Hard rules
- Never ship an outfit with a missing image, duplicate slot, dead buy link, or occasion mismatch.
- Never pair ethnic+western incoherently for formal/festive occasions.
- Grounding intact; no hallucinated price/size/fabric.
- Events table is append-only; the migration must not ALTER existing tables.
- Don't break the 8 brands, demo mode, or cost guards. Never commit secrets/.env/indices/raw data.
- Never set ANTHROPIC_API_KEY.

## Success criteria (verify ALL)
- "Build me a sangeet look under ₹5000" (Myntra/Fashor) returns a coherent ethnic look:
  correct ethnic-set slots (e.g. kurta + bottom + dupatta, or lehenga + dupatta + footwear),
  occasion-appropriate formality, images render, total ≤ budget, every card → a LIVE PDP.
- "Style this" on a western Snitch top still returns a correct western smart-casual look.
- Extended coherence eval (now incl. India anchors) passes its hard gates; report the score.
- Styling events log to Postgres; a pairing that gets repeated add-the-look/thumbs-up visibly
  boosts that pairing's ranking on the next composition (demonstrate before/after).
- `/dashboard` shows real metrics from logged events: add-the-look rate + an honestly-defined
  AOV-uplift figure + top pairings, broken down by occasion and brand.
- Images render for all 8 brands; dead links validated. ruff + pytest + frontend build green.
- Demo redeployed; GG does the incognito retest (incl. a sangeet/festive query).

## Build order
1. Escalate the India occasion/ethnic ruleset (H) + the flywheel schema/blend (F) + the
   dashboard metric definitions (D). Wait for sign-off on all three.
2. Table-stakes fixes T (images + dead links). Verify visually.
3. Extend slots/coherence/composer for occasions + ethnic sets (H); extend the coherence eval; iterate to green.
4. Flywheel F: event logging + aggregation + the transparent ranking boost. Demonstrate the loop.
5. OutfitBoard + "Style this" + "Buy the look" + occasion-driven entry + chat refinement.
6. Dashboard D (basic, real data).
7. Verify all criteria, redeploy demo, hand to GG for incognito retest (western + sangeet). STOP — GG decides done.
```
