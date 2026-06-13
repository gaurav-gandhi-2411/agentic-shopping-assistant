# Project Spec: agentic-shopping-assistant — Stylist-Grade Experience + Productization (wave)

## Goal

Elevate the India styling engine from "coherent grid" to **stylist-grade and deployable**:
(1) explain *why* a look works + offer variants, (2) make "add the look" actually fill the
brand's cart, (3) accept an image/inspiration photo as input, and (4) package the whole thing
as an embeddable widget a brand drops onto their store. Plus pin a stable public URL so the
demo link stops changing every deploy.

Thesis unchanged: best India-specialized styling (the moat) + users who love it + a basket-lift
story brands pay for. Every phase serves "would a shopper use this daily, and would a brand
deploy it on their PDP?"

## Current state (read first)
- Live on Cloud Run (asa-snitch/asa-myntra/asa-flipkart, asia-south1, project
  iconic-reactor-496423-m4) + Vercel frontend. Anonymous DEMO_MODE, Postgres guards, flywheel
  (styling_events/pairing_stats), /dashboard, 9-occasion gender-conditioned ethnic styling.
- Grounding layer scrubs hallucinated price/size/fabric — reuse it for the new rationale.
- Multimodal/CLIP capability exists in the codebase history (used in earlier image work) — reuse for image input.
- Shopify brands (live products.json with variant IDs): snitch, powerlook, fashor, virgio.
  Non-Shopify: myntra, flipkart (link-out only).
- **Vercel URL changes per deploy** (per-deployment hash URLs) — no stable alias pinned yet.

**Load-bearing — do not break:** grounding, LLM Protocol, the outfit engine + gender gates,
demo mode + cost guards, all 8 brands, the flywheel schema.

## Scope

### Phase 0 — Stable public URL + housekeeping (do first)
- Determine and PIN the canonical Vercel **production** URL (stable across deploys), via a
  production alias/domain — not a per-deployment hash. Report the exact user-facing link
  (`<stable-domain>/demo`).
- Lock CORS on all 3 Cloud Run services to that stable origin, and ensure future deploys keep
  the same origin (so CORS doesn't break each redeploy).
- (Optional, flag only) note the path to add a cheap custom domain later.

### Phase 1 — Stylist rationale + look variants (the taste layer)
- For each composed look, generate a **grounded styling rationale** ("the mustard dupatta picks
  up the gold in the kurta; keep the bottom neutral so the embroidery stays the hero"). It may
  reference ONLY true item attributes (colour, type, occasion, slot) — run it through the
  grounding layer; no invented fabric/price/brand claims.
- Return **2–3 look variants** per request (e.g. different colour stories or a dressier vs
  lighter lean) the user can switch between.
- UI: rationale shown on the OutfitBoard; a variant switcher (tabs/chips). Log which variant the user engages.

### Phase 2 — Add-the-look to cart + save/share (the conversion mechanic)
- **Shopify brands** (snitch/powerlook/fashor/virgio): build a cart permalink that adds ALL the
  look's items (variant IDs from products.json) in one action, opening the brand's cart
  pre-filled. This is the literal basket-lift mechanism.
- **Non-Shopify** (myntra/flipkart): fall back to per-item buy links + an "open all" action.
- **Save a look**: persist a look (anonymous-session-scoped) so it can be revisited.
- **Share a look**: a shareable URL that renders a read-only look board (the viral loop).
- Ensure the cart action fires the existing `add_the_look` flywheel event.

### Phase 3 — Image / inspiration input
- Upload an image (a garment photo or an inspiration look) → embed via CLIP → find the closest
  catalogue **anchor** item in the active brand → compose a look around it (reuse the engine).
- UI: an upload control on the demo; "find this + style it".
- Privacy: process in-session only; do not persist user-uploaded images beyond the request.
  Reasonable size/type limits; reject non-images.

### Phase 4 — Embeddable "Complete the Look" widget (the product)
- A drop-in widget (JS snippet or iframe) a brand embeds on a product page: a "Complete the
  Look" trigger → opens the styling experience (phases 1–3) scoped + themed to that brand via
  BrandConfig, calling the existing API.
- Configurable by brand key; respects the same demo/anonymous guards and CORS.
- Ship an **example "mock PDP" page** that embeds the widget, so a brand can see exactly how it
  looks on their store (this doubles as a sales artifact).

### Out of scope (this wave)
- Payment/checkout — the cart permalink hands off to the brand's own checkout; we never touch payment.
- The A/B AOV-proof harness — separate later wave (needs live pilot traffic to be meaningful).
- Cross-website/marketplace aggregation, price comparison — never.
- Full ML personalization — flywheel stays transparent-boost.
- Men's-ethnic catalogue expansion — queued separately (data caveat below).

## Tech stack
- Existing stack only. CLIP reuse for Phase 3. New shareable-look storage uses existing
  Postgres (append-only or a simple looks table — escalate the migration). No new heavy deps
  without escalation.

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
  required: true
```

## Subagent rules
- executor codes, verifier verifies, orchestrator delegates and never codes.

## Escalation rules (ask before doing)
- Phase 0: confirm the chosen stable production URL before locking CORS to it.
- Phase 1: rationale must be grounded — if grounding can't verify a claim, drop it; confirm the approach.
- Phase 2: present the save/share look data model + any Alembic migration plan before building (append-only; no ALTER existing).
- Phase 3: confirm the image-input privacy approach (no persistence) and size/type limits.
- Phase 4: confirm the widget embedding approach (iframe vs JS snippet) and its CORS/security model before building.
- Ask before any dependency, LangGraph routing change, or cloud redeploy.

## Hard rules
- DEMO_MODE=false behaviour unchanged. Never ship a look with a missing image, dead link,
  duplicate slot, gender conflict, or occasion mismatch.
- Rationale text passes grounding — no hallucinated attributes.
- Don't persist user-uploaded images. Append-only flywheel; migrations don't ALTER existing tables.
- Keep all 8 brands, demo mode, cost guards working. Never commit secrets/.env/indices.
- Never set ANTHROPIC_API_KEY. No cloud redeploy without GG approval. Don't touch aetherart-497918.
  All 3 services stay min-instances=0.

## Success criteria (verify; GG owns final visual retest)
- Phase 0: a STABLE user URL is reported and survives a redeploy; CORS works from it.
- Phase 1: every look shows a grounded rationale (no invented attributes) + 2–3 switchable variants.
- Phase 2: on a Shopify brand, "Add the look" opens the brand cart with ALL items pre-filled;
  non-Shopify falls back gracefully; a saved look is revisitable; a shared link renders a read-only board.
- Phase 3: uploading a garment/inspiration image returns a coherent look built around the closest
  catalogue anchor; no image persisted.
- Phase 4: the mock-PDP page embeds the widget; clicking "Complete the Look" opens the themed,
  brand-scoped styling experience; guards + CORS intact.
- coherence eval green; ruff + pytest + frontend build green throughout.

## Build order
0. Phase 0 (stable URL + CORS) — verify, report the user link.
1. Phase 1 (rationale + variants) — checkpoint report, continue.
2. Phase 2 (cart + save/share) — checkpoint report, continue.
3. Phase 3 (image input) — checkpoint report, continue.
4. Phase 4 (widget + mock PDP) — checkpoint report.
5. Full verify, commit (secrets/indices out), then STOP for GG's redeploy approval + visual retest.
Report at each phase boundary; escalate on the flagged decisions; do not redeploy to cloud until approved.

## Data caveat (queue, not this wave)
Men's ethnic catalogue is thin (Snitch mostly western, Myntra women-only). A credible men's
sangeet/sherwani demo needs a men's-ethnic source (e.g. a Manyavar-type brand). Note for a later data wave.
```
