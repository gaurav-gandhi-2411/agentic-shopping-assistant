# Master Spec: agentic-shopping-assistant — Market-Ready Build (Fable 5, 3 waves)

## How to run this (read first)

Orchestrator model: **claude-fable-5** (set via `/model`). Fable 5 may **research the web** to
find the best approach, and should — especially for Wave 2 (styling visualization / try-on) and
Wave 3 (platform search + affiliate feasibility). Present findings honestly, including what is
NOT feasible for a solo/free-tier build, and propose the best BUILDABLE version.

**Autonomy model — autonomous within a wave, gated between phases:**
- For EACH wave: (1) research + present a PLAN (and feasibility findings) → **STOP for GG
  approval**; (2) after "go", build autonomously — plan sub-steps, delegate to executor/verifier,
  reproduce-red-then-fix-green, self-correct, no stopping for mechanics; (3) **prove every result
  on the LIVE Cloud Run URL via the WS /chat/stream path with raw output** (not internal tests,
  not POST /chat, not localhost); (4) **STOP for GG's browser retest**; (5) only after GG
  confirms in the browser → next wave.
- Fable 5 does NOT self-declare "done." Done = GG confirms in the browser. Every prior "all
  green" that self-certified turned out broken in the browser — that is the failure mode this
  gating exists to prevent.

## Project context
- B2C cross-store fashion app. Backend asa-stylist-api on Cloud Run (project
  iconic-reactor-496423-m4, asia-south1); frontend https://asa-stylist.vercel.app/demo.
- Browser chat uses **WS /chat/stream**. Deterministic search core (LLM extracts structured
  intent, does not route/rewrite/pick). Unified index ~68,663 items, 9 stores (Myntra, Flipkart,
  Snitch, Fashor, Powerlook, Virgio, Berrylush, Globalrepublic, Libas), normalized garment_type
  + gender + colour, on GCS. CLIP image index. Demo-mode anonymous guards (rate limits currently
  raised for testing — revert before public). Deploy via scripts/deploy_unified.sh (Step 1 = GCS
  index OVERWRITE upload).
- Catalogue identity (from data audit): women's + India-strong depth (dresses, kurtas, sarees,
  tops, lehengas); menswear = shirts/jeans. Known limits: semantic-richness gap ("summer dress",
  "boho" underperform — scraped titles lack season/vibe attributes); no live stock/price data.

## Global hard rules (all waves)
- Prove on the LIVE Cloud Run URL via WS /chat/stream, raw output. POST /chat or localhost proof
  is invalid. Browser retest by GG gates every wave.
- Free-tier only; NO paid service/API/scraping without escalation. Never set ANTHROPIC_API_KEY.
- Secrets in Secret Manager, never plaintext/committed. Never commit .env/indices/raw data.
- GCS index uploads OVERWRITE (never --no-clobber). Rebuild → upload → redeploy → prove live.
- Keep the app in a shippable state after every wave (demo URL always works; no half-features).
- Respect robots.txt / platform ToS. Keep old services as rollback until GG confirms.

---

## WAVE 1 — Fix the 4 outstanding bugs (foundation; do first)

**Goal:** the app renders correct results in the browser. Nothing else is built on a broken base.

Bugs (reproduce each as a failing WS-path check FIRST — show GG the red baseline — then fix):
1. **Wrong buy links** — new stores (berrylush/globalrepublic/libas) weren't registered in
   STORE_CONFIG (stores.py) + _STORES (intent_parser.py), so build_pdp_url → None → frontend
   falls back to the H&M template. Register them; badge must match the real PDP domain. Curl 3
   PDP links per store → HTTP 200 to the correct domain.
2. **Cards not rendering in the browser** — WS returns items but the browser shows prose only
   (null pdp_url/store_display likely crashes the card render). Confirm the exception in the
   browser console; fix so cards ALWAYS display when items exist, including new-store items.
3. **Refinement drops context** — "white shirt men" → "in blue now" returned women's items.
   A colour-only refinement must keep prior garment_type + gender from session context. Prove
   the exact multi-turn sequence on the live site → blue MEN'S SHIRTS.
4. **Image search hangs / text dropped** — "Finding your match…" then nothing (CLIP index not on
   GCS for the live revision). Ensure CLIP index uploaded to GCS + loaded on the live revision;
   image search returns cards. Also: image upload must carry the user's typed text ("buy
   similar") and echo BOTH image + text in the chat.

**Wave 1 done =** all 4 proven on the live URL (raw output + browser-console confirm for #2) AND
GG confirms in the browser.

---

## WAVE 2 — Styling-visualization USP (the differentiator)

**Goal (GG's USP):** a user uploads (or picks) an item → the app shows how to STYLE it — a
complete look with complementary pieces, a colour palette that resonates with their item, and
buy links for the suggested pieces across stores. Ideally a visual of the styled look; at
minimum a polished visual outfit board. This is the reason people come to the platform.

**Fable 5 must RESEARCH and present feasibility before building:**
- Virtual try-on / "model wearing the user's item styled with others": research open/free models
  (e.g. VTON-family, outfit/lookbook generation) and be honest about cost/latency/GPU — Cloud Run
  min-0 + free-tier is the constraint. If true on-model generation isn't free/feasible, propose
  the best buildable MVP: a styled outfit BOARD (their item + complementary pieces as product
  images) + colour palette + grounded styling rationale + buy links — with on-model generation as
  a staged upgrade if a free path exists.
- Present options with tradeoffs; recommend the best one; GG picks.

**In scope (MVP, adjust per research):** colour-palette suggestions that resonate with the
uploaded item; complementary-item styling across stores (reuse the outfit engine — it's strong
for women's/India); a visual look presentation; buy links (deep-link) for each suggested piece;
"style this / more like this" working correctly (fix the ones GG flagged as not working well).

**Out of scope unless research proves free+feasible:** heavy GPU generation, paid try-on APIs.

**Wave 2 done =** proven on live URL AND GG confirms in the browser that uploading an item
produces a coherent styled look with palette + working buy links.

---

## WAVE 3 — Real-time cross-platform search + affiliate buy (the ambitious backend)

**Goal (GG's vision):** move from "search only our stored catalogue" toward real-time search
across platforms, re-rank, and present unified results the user can buy — via **affiliate
deep-link** (user pays on the retailer; we earn commission). This is the largest, riskiest wave —
last, after the app is solid and differentiated.

**Fable 5 must RESEARCH and present an architecture + feasibility plan before building:**
- Which platforms allow programmatic real-time search for a solo/free-tier dev, and under what
  ToS: Shopify /products.json + search, affiliate product APIs (Amazon PA-API, Flipkart, Myntra
  via networks), Google Shopping / product feeds, others. Report what's free, what needs
  approval, what's ToS-prohibited.
- Design the realistic architecture: live-query permitted sources per request, brief cache,
  MERGE with the existing indexed catalogue, re-rank with the engine, present unified. NOT
  "scrape every platform live on every query" (latency, ToS, bans) — Fable 5 designs the
  feasible hybrid.
- Buy mechanism = **affiliate deep-link** (confirmed by GG). NOT bot-account checkout (violates
  platform ToS, gets accounts banned, incurs payment/merchant-of-record liability). Structure so
  affiliate tags swap in per platform once programs are approved; plain deep-links until then.
- Present the plan honestly (what's live-searchable now, what's deferred to program approval /
  paid tier); GG approves the architecture before build.

**Wave 3 done =** the designed hybrid live+indexed search proven on the live URL returning
unified re-ranked results with working deep-links AND GG confirms in the browser.

---

## Market-ready checklist (before GG calls the whole thing shippable)
- Rate limits reverted to safe public values (currently raised for testing).
- Secrets in Secret Manager; no plaintext/committed secrets.
- Demo URL stable and working; frontend re-aliased on every deploy (deploy.sh Step 8).
- Errors handled gracefully (no hangs, no blank screens; honest "no matches" states).
- Known limitations documented (semantic richness, stock/price, menswear depth).
- Every "Buy" link resolves to the correct retailer (HTTP 200).

## Verification (every wave)
```yaml
- name: tests            {cmd: pytest -m "not requires_ollama" -v, required: true}
- name: qa-ws-matrix     {cmd: python -m qa.user_flow_matrix, required: true}   # MUST hit WS /chat/stream
- name: category-precision {cmd: python -m eval.category_normalization, required: true}
- name: lint             {cmd: ruff check ., required: true}
- name: frontend-build   {cmd: cd frontend && npm run build, required: true}
```

## Build order (strict)
1. WAVE 1 (bugs) → live proof → GG browser retest → GG confirms.
2. WAVE 2 (styling USP: research → GG approves plan → build → live proof → GG browser retest).
3. WAVE 3 (real-time search: research → GG approves architecture → build → live proof → GG browser retest).
Do not start a wave until GG confirms the previous one in the browser.
```
