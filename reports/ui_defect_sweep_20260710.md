# Live-URL UI/Relevance Defect Sweep — 2026-07-10

Method: automated Playwright crawl of every customer surface on https://stylemaitri.vercel.app
at desktop (1440×900) and mobile (390×844), real query flows executed, content extracted,
28 surface/viewport entries, 39 screenshots. Zero console errors, zero failed requests
(>=400), zero horizontal overflow on mobile — all defects below are content, correctness,
or visual-design issues, not crashes. Screenshots + sweep_report.json in session scratchpad
(`scratchpad/sweep/`); sweep script at `scratchpad/ui_sweep.py` (promotable to scripts/ if wanted).

Claims verified against ground truth where marked. **No fixes applied — triage list only.**

## P0 — functional / relevance breaks (launch blockers)

1. **Colour-family refinement misrouted to out-of-domain refusal.** In a sangeet-lehenga
   conversation, "show me pastel colours instead" → *"I don't carry food and drink products —
   this catalogue is clothing only."* Reproduced desktop + mobile. The OOC classifier reads
   "pastel" as food. The UI's own chips invite colour refinements. MODEL/CODE.
   Evidence: `desktop_chat-search-refine_q2-refine-pastel.png`.

2. **Every men's outfit board suppresses the Bottom slot with a false inventory claim.**
   "Bottom — No men's bottoms in our partner stores yet" — verified false: unified
   catalogue has 4,748 men's bottoms (2,786 trousers + 1,882 jeans). Composer bottom-slot
   category mapping bug, NOT a data ceiling. Knock-on: reception board = sherwani +
   waistcoat + oxfords with no trousers. The eval's `suppression_honest=100%` gate only
   checks a reason was *reported*, not that it is *true* — harness cannot catch this class.
   MODEL/CODE. Evidence: `desktop_chat-outfit-board_board.png`.

3. **B2B widget chain dead-ends: `/embed/snitch` → "Brand not found".** `brands/snitch.yaml`
   exists; root cause is `NEXT_PUBLIC_SNITCH_BACKEND_URL` (and MYNTRA/FLIPKART vars) unset in
   the Vercel deploy (likely casualty of the 3→1 service collapse). The pdp-demo "Complete
   the Look" button opens this broken iframe, so the entire B2B demo flow fails. CODE/CONFIG.
   Evidence: `desktop_embed-snitch_load.png`.

4. **Gibberish input returns a confident product recommendation.** "asdfgh qwerty zxcvb" →
   "It seems like you're looking for something to add a stylish touch… Zivame Women Black
   Longline Shrug" at 1% match, blank image. Electronics/food OOD refusals work; nonsense
   does not trigger clarification. MODEL/CODE. Evidence: `desktop_chat-edge-cases_gibberish.png`.

## P1 — trust and polish wounds on core surfaces

5. **"1% match" / "2% match" badges on nearly every card.** `ItemCard.tsx:19` renders
   `item.score * 100` assuming a 0–1 probability, but backend hybrid/RRF scores are ~0.01–0.03
   by construction. Unit mismatch displaying near-zero confidence on good results. CODE.

6. **Stylist's note is visibly templated and often wrong.** Raw store category strings
   stuffed into prose ("the blue blazers, waistcoats and suits and black footwear keeps the
   focus…"), wrong colours ("purple kurta" for a Wine kurta; "grey and blue complement" beside
   a red waistcoat), broken grammar. Appears on every board and persists into saved/shared
   looks. CODE.

7. **Saved-look page leaks internals** (`/look/[id]`): item names suffixed with raw category
   strings incl. stray space "( Kurtas, Ethnic Sets and Bottoms)"; a red waistcoat labeled
   "(Blue Blazers, Waistcoats…)"; internal "Base" variant tag; footer "Styled with **Unified**"
   (internal index name). CODE. Evidence: `desktop_saved-look_look.png`.

8. **Share/OG gaps.** `/look/[id]` og:image points at the OLD domain
   `asa-stylist.vercel.app` (renders today; breaks if the old project is removed; off-brand) —
   the old-URL sweep (fd02e73) missed metadataBase. `/demo` has NO og:image/og:title at all —
   WhatsApp/social shares of the demo link render bare. CODE/CONFIG.

## P2 — visual / layout / branding

9. **Desktop result layout wastes ~60% of the viewport.** Cards keep a fixed ~165px image
   with a large empty field beside it; search results render as one giant column. Reads as a
   stretched phone UI. DESIGN. Evidence: `desktop_chat-search-refine_q1-sangeet-lehenga.png`.

10. **Root URL (`/`) redirects to an unbranded sign-in card** — no logo, no product name, no
    demo link; a first-time visitor dead-ends. `/pdp-demo` also generic/unbranded (B2B page,
    rebrand phase 1 scoped to core chat — flagged for phase 2). DESIGN/BRAND.

11. **Intermittent blank pink product images** (desktop captures: puffer jacket, belt,
    camisole; same belt loaded fine on mobile) — lazy-load/CDN timing with no skeleton or
    retry fallback. CODE.

12. **Mobile composer placeholder clips mid-word** ("…style, outfits" second line cut). CSS nit.

13. **Snitch demo PDP "product image" is a man's headshot**, not a blazer. Demo-asset nit.

## Relevance observations from real flows (not all defects)

- Sangeet lehenga under ₹15,000: 5/5 lehengas, all within budget, occasion-plausible. Good.
- Partner "his look" for sangeet: casual grey kurta hero + ₹759 combat boots — occasion
  register miss; property gates pass, stylistic fit does not. MODEL (composition ranking).
- Pear body-type request surfaced an explicitly "Plus Size" item — body-shape ≠ size
  conflation risk in guidance context. MODEL.
- Image upload (t-shirt.webp): correct visual match, sensible casual board; puffer jacket
  suggested in July (no seasonality signal). Works.
- Body-type chip flow copy is good; photo path present with on-device privacy note.

## DATA-CEILING (inventory, not code)

- Women's footwear: 0 in partner stores → every women's look ships shoe-less with an honest
  note. Real gap; needs ingestion/partner fix, not code.
- Jewellery/accessories depth thin (boards fall back to dupattas as the only accessory).
