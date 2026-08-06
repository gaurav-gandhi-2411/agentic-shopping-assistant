# Deep Diagnosis — Style Maitri Live Site (2026-07-12)

**Method:** 6 independent subagents drove the live site (`https://stylemaitri.vercel.app/demo`, backend
`asa-stylist-api` on Cloud Run) in a real browser via chrome-devtools MCP, playing a skeptical
first-time user (28yo Indian woman, wedding shopper, phone or laptop). 3 agents did a visual/UX
sweep (mobile 390px, desktop 1440px, secondary flows) with actual screenshot inspection. 3 agents
judged 32 real, messy wedding-shopper queries as a human, not against the eval gold set. The live
per-IP demo rate limit (default 35/hr, `api/demo/guards.py:23`) was temporarily raised to 1000/hr
for the duration of this run (Cloud Run revision `asa-stylist-api-00074-scr`) and **reverted to
default immediately after** (`asa-stylist-api-00075-6wr`, confirmed via `gcloud run services
describe`). No code was changed. No console JS errors or failed/slow network requests were found
in any session beyond one recurring a11y warning — backend infra is stable; every finding below is
a UX/relevance/trust quality issue, not a reliability issue.

Findings are ordered most-severe first. Each is tagged with root cause: **(a)** ranking/relevance
bug, **(b)** routing/intent bug, **(c)** UI presentation problem, **(d)** inventory gap (not
code-fixable). **[FIX]** = code-fixable this wave. **[INVENTORY]** = catalogue-capped, needs data
not code.

---

## Blockers — trust-destroying, fix first

### 1. Body-shape photo upload fabricates a specific physical claim from a photo with no person in it — [FIX] (b)
Uploaded a plain product photo of a t-shirt lying flat (no human visible) to the "body-shape
suggestion" feature. The system confidently responded *"You might have a ~inverted triangle
silhouette — does that sound right?"*, then used that fabricated trait to justify styling advice
("flared lehenga... balances a broader shoulder line") after the user confirmed it. Reproduced
identically at both 390px and 1440px. Runs client-side via MediaPipe pose-landmarker with **no
"no person detected" gate** — it happily runs pose inference on an image with no pose to find and
reports a confident result anyway. This is a body-image feature aimed at a self-conscious use case;
silently fabricating a physical trait with total confidence is the single most damaging bug found.
*Fix direction: gate on pose-landmark confidence/presence before surfacing a body-shape guess;
fall back to the existing (and good) manual chip picker (Pear/Apple/Hourglass/Rectangle/Inverted
triangle) when no person is detected.*

### 2. Nonsense/garbage queries never get an honest "no results" — the assistant hallucinates a confident match instead — [FIX] (b), cross-verified 2×
Both the mobile and desktop sweeps independently sent gibberish ("asdkfjhqwoiuerlkj purple flying
shoes" / "purple flying unicorn shoes qwxyz"). Neither ever got a "no results" or clarifying
response. Both times the assistant seized on one real word ("purple") and confidently recommended
a **saree** with a straight-faced style pitch, with zero acknowledgment that most of the query was
gibberish or that "shoes" was never matched. For a system whose entire value prop is "I understand
what you're asking for," silently substituting a plausible wrong answer is worse than a blank
state — it's the fastest way to lose a skeptical user's trust once they poke at the edges.
*Fix direction: a relevance-confidence floor on the retrieval/generation path that triggers an
honest "I couldn't quite match that — try describing the occasion or garment" instead of always
generating a confident pitch.*

### 3. Children's/kids' items rank above or alongside adult items in bridal/wedding queries — [FIX] (a)/(b), cross-verified 2×
- `"red lehenga bridal"` (batch 1): top 2 results were **girls'/children's lehengas**
  ("Bitiya by Bhama Girls...", "Cutiekins Girls..." — second product photo literally shows two
  children) — ranked *above* genuinely good adult bridal options that exist in the same result set
  (EthnoVogue red/yellow ₹6,409, mirror-work ₹6,341).
- `"gold jewellery to go with red lehenga"` (batch 3): among the substituted lehenga results was
  a children's item ("BownBee Girls... South Indian Pavda Pattu Lehenga").

No adult-vs-children segmentation is applied to bridal/wedding-attire queries. This is a
correctness bug with real embarrassment/safety-adjacent risk (an adult shopper being served kids'
clothing for her own wedding lehenga search), not a cosmetic one.
*Fix direction: age/audience facet on the index, hard-excluded (or heavily demoted) for adult
occasion queries like "bridal", "wedding guest", "sangeet" unless "kids"/"girls"/"daughter" is
explicit in the query.*

### 4. "kurta pajama" and "minimalist wedding dress" queries return literal sleepwear/nightgowns — [FIX] (b)
- `"kurta pajama for father in law"` — all 5 results were "[Color] Cotton Slim Fit **Night Suit
  Set**" (sleep t-shirt + track pants, brand "Global Republic · Nightwear"). A shopper asking for
  wedding attire for her father-in-law was shown pajamas to sleep in.
- `"minimalist wedding guest dress"` — the single result returned was a "Green Geometric Printed
  Cotton Kaftan **Night Dress**" (₹929, Nightwear, shown on a barefoot model on a couch), with the
  assistant's own text calling it *"perfect for a wedding celebration."*

Literal keyword collision ("pajama", generic "dress"/"minimalist") is routing straight into the
Nightwear category with no category-boundary check for an explicitly wedding-context query. This
is the kind of miss a real user notices instantly and never forgets.
*Fix direction: exclude Nightwear/Loungewear category from occasion-tagged wedding queries unless
explicitly requested; treat "kurta pajama" as a single ethnic-wear compound term, not two
independently-matched tokens.*

### 5. Two total query failures — misrouted into the wrong flow, zero products returned — [FIX] (b)
- `"outfit for engagement under 5k"` → bot replied *"To build an outfit, tell me the occasion and
  your budget — e.g. 'sangeet look under ₹5000'"* despite the user having supplied exactly that.
  "Engagement" wasn't recognized as an occasion and/or "5k" wasn't parsed as ₹5,000.
- `"groom's sister outfit ideas"` → misrouted entirely into the "style my partner" flow (*"Tell me
  or show me what you're wearing first, then I'll style your partner to match"*) — "groom's" was
  read as a partner reference instead of a common first-person wedding-guest persona.

Both are complete dead ends for an otherwise well-formed, realistic query.

### 6. Every tap target on every product card is roughly half the minimum usable size — [FIX] (c), measured not eyeballed
Measured via `getBoundingClientRect()` at true 390px viewport: "Style this" 65×26px, "More like
this" 87×26px, "Buy at X" 104×28px on every card; outfit-tab and refinement chips are 25px tall.
The 44px minimum is missed by roughly half, on every actionable control, on a chat app whose whole
premise is thumb-driven mobile browsing.

### 7. Product photos render mostly blank background instead of the garment — [FIX] (c), reproduced 2×
Multiple cards (e.g. "navyasa Pink & Blue Floral Saree," a Chhabra 555 lehenga) use a centered
`object-fit: cover` crop on source photos where the garment occupies only ~25% of the frame on one
side — the card ends up showing 70-80% dead grey/beige studio backdrop and a thin sliver of actual
product. Verified via `naturalWidth/Height` + computed style and reproduced identically across two
independent sends of the same query. On a fashion app, broken-looking product photography reads as
"this catalogue isn't real."

---

## Major — real damage, not yet trust-destroying

### 8. Soft/qualitative constraints are consistently dropped, not applied as filters — [FIX] (b), systemic across ~8 queries
"comfortable" (sangeet dancing → still surfaced a heavy made-to-measure bridal lehenga choli
first), "cheap" (→ AI's own text falsely claimed *"I don't have pricing information"* despite
prices shown; cheapest item ranked 3rd of 5; a ₹28,900 item — 11× the median — included in a
"cheap" set), "not too flashy" (→ 3/5 results literally named "Sequin **Embellished** Anarkali"),
"minimalist," "pastel" (best pastel/daytime option ranked last of 5), "plus size" (constraint never
addressed anywhere in filtering or copy, though the rest of the outfit board was coherent). This is
the single highest-leverage, most systemic pattern across the whole search-quality pass — the
retrieval path anchors on the occasion/category noun and silently discards adjectival modifiers.

### 9. Gender filtering is inconsistently applied, leaking wrong-gender items into results — [FIX] (b)
`"trendy indowestern"` returned a **men's sherwani modeled by a man** as the top result in what's
clearly a female-shopper context, plus unrelated traditional sarees; only 2/5 results actually fit.
`"show me sherwanis"` returned 3/4 wrong-gender items (women's crop top, harem pant, kurta set)
padding out a catalogue that only has ~1 real sherwani — no honest "we only have one" disclosure.
`"outfit ideas"` (maximally vague) defaulted to random Western party dresses and leaked a men's
polo t-shirt into a presumed women's flow.

### 10. Inconsistent honesty about inventory gaps — hallucinates in one code path, discloses honestly in another — [FIX] (b), the highest-leverage single fix
Jewellery and footwear **do not exist in the catalogue at all** (0 items confirmed across multiple
queries). The **outfit-board** path handles this correctly and honestly: *"No women's footwear
that match this look in our partner stores yet."* The **plain-search** path does not: `"footwear
for lehenga"` returned zero footwear but the assistant's text confidently described specific
non-existent items — *"pairs well with delicate ankle-strap sandals," "would look stunning with
statement block heels"* — implying real matches exist when none do. `"gold jewellery to go with a
red lehenga"` silently substituted more lehengas instead of admitting no jewellery exists.
`"jacket style lehenga"` (no true jacket-style items exist) got a rationalized non-match instead of
an honest gap admission (*"the embroidered details... give it a sophisticated, almost jacket-like
feel"*). `"what's trending for wedding season 2026"` asserted unqualified trend authority
(*"For wedding season 2026, I think..."*) the system has no way to actually back. Since the honest
disclosure pattern **already exists and works** in the outfit-board path, wiring the same pattern
into plain search is the single most fixable, highest-leverage change from this whole audit.

### 11. Typo handling degrades hard category filters — [FIX] (b)
`"lehnga for haldi"` (typo) got the color/occasion mapping right (yellow) but only 1/5 results was
actually a lehenga (rest were kurtas/kurtis) — visibly worse than the correctly-spelled
`"designer lehenga under 15000"`, which returned 5/5 lehengas. The misspelling weakened the
garment-type filter specifically, not the broader semantic match.

### 12. Near-duplicate/same-template crowding recurs across multiple queries — [FIX] (a)
Recolored variants of the same base product fill 3 of 5 slots in at least 3 separate queries
("sangeet dancing," "not too flashy," "cream saree" pair in query 1 of batch 3) — reduces perceived
variety and wastes result slots that could show genuinely different options.

### 13. Outfit-board stylist copy is duplicated verbatim within the same message — [FIX] (c), reproduced 3×
The rationale sentence and the footwear-gap caveat are each printed twice in the same
response — once in an "Outfit suggestion" intro box, once again word-for-word in a "Stylist's
note" box. Confirmed on all 3 outfit-board responses tested. Reads as a templating bug.

### 14. Partner styling: content order and structure are inconsistent between "Her" and "His" look — [FIX] (c)
Both rationale paragraphs stack together before any product images appear; matching cards for each
person appear far down the page, well after their own text has scrolled out of view. "His look"
gets a real `<h3>` heading in the DOM; "Her look" does not (confirmed via accessibility snapshot) —
an asymmetric experience for screen-reader users specifically.

### 15. The AI's named "top pick" often isn't the first card shown, and rationale isn't stable across identical repeat queries — [FIX] (a)/(c)
Sent `"wedding-guest saree under 5000"` twice. Both times the assistant's prose named a specific
"best" item that was never the actual first card in the grid (2nd-4th position instead), and the
rationale text itself differed between the two identical runs. Breaks the implicit contract that
the words and the visual order agree.

### 16. Reload silently wipes the entire session with no warning — [FIX] (c)
Confirmed via both hard reload and direct URL navigation: the full multi-turn conversation, all
results, and any unsaved "Save look" progress vanish instantly with zero recovery path or
confirmation. A real risk given accidental mobile pull-to-refresh, especially since saved-look
confirmations (with their one-time share link) also vanish this way (see #20).

### 17. Desktop layout was designed for mobile and stretched, not art-directed for the width — [FIX] (c), multiple sub-findings
- Landing/empty-chat state: content is a small centered column with the rest of a 1440px canvas
  pure empty cream space, top and bottom.
- Results grids that don't fill a row (e.g. 5 items in a 4-column grid) leave the wrapped row's
  remaining columns as large dead space instead of reflowing — happened on every multi-item result
  set tested.
- The "More like this" expandable panel doesn't adapt to desktop width at all — renders as a
  cramped, truncated single-column list squeezed into the ~340px card column while ~1000px sits
  empty beside it. The clearest single "built for mobile, stretched onto desktop" moment found.
- Assistant response text has no max-width constraint and runs the full ~1100px content width
  (150+ characters/line) — well past comfortable reading measure, hurting readability rather than
  using the space well.
- Outfit-board "coordinated look" photos are a collage of unrelated stock images, not one cohesive
  shot — switching between "Style 1" and "Colour Palette" tabs swaps in a photo of a visibly
  different outfit on a different model, which is disorienting rather than looking like a curated
  look.

### 18. Loading/"assistant is typing" indicator looks like a rendering glitch, not a loading state — [FIX] (c), consistent across mobile and desktop
Mobile: a small ~44×44 pink pill containing a single static-looking vertical bar. Desktop: a tiny
unstyled black vertical bar floating at the far-left edge of the screen with no bubble or
background. Both read as "the page is broken," not "the AI is working," on first glance.

### 19. Photo-upload (garment) follow-up text fabricates a description that doesn't match the actual uploaded photo — [FIX] (b)
Uploaded a real photo of a plain solid-white v-neck t-shirt; it was correctly recognized initially,
but the "Where can I buy one like this?" follow-up called it "a bit unique" and recommended shirts
with "bold abstract strokes" / "ombre" patterns — a fabricated visual description with no relation
to the actual uploaded image.

### 20. Saved looks have no durable, in-app retrieval surface — [FIX] (c)
"Save look" works and produces a real, shareable link (`POST /looks` → 201) — but there's no
"My saved looks" page anywhere in the app. The copied link at the moment of saving is the *only*
way to ever see that look again; combined with #16 (reload wipes the session), a user who saves
several looks while browsing and doesn't copy every link immediately loses the rest permanently.
Undermines the core "compare a few options" use case a wedding shopper actually has.

### 21. Share page (`/look/{uuid}`) is visually broken on desktop and under-built for how it's actually shared — [FIX] (c)
At 1440×900 the whole card is pinned top-left in a ~220px column with ~85% of the viewport blank —
looks like a failed page load. No visible "Buy" button/affordance on the product card, unlike every
other card in the app. Open Graph/Twitter meta uses generic app branding ("Style Maitri" / generic
logo card) instead of the actual saved item, even though the page's own `<title>`/meta description
do have the specific item text — so a link shared over WhatsApp (the dominant real-world sharing
channel for Indian wedding shopping) previews as generic app promo, not "look what I found,"
undercutting the entire point of the share feature.

### 22. Widget/embed is completely undiscoverable from the live site — [FIX] (c) / possibly (d) if truly unshipped
No nav, footer, or metadata on any live page mentions "embed," "widget," "API," or "developers."
`/embed` returns a clean 404. Per project memory an embeddable widget was built in an earlier wave;
if so, it's not linked anywhere a real user or evaluator would find it — functionally equivalent to
not existing for anyone outside the team.

---

## Minor / nits — polish, not blocking

- Redundant "Style Maitri" branding/logo appears twice within ~100-400px of the header on first
  load (mobile and desktop). **[FIX] (c)**
- Long product names hard-truncate to a single line, cutting off fabric/style detail with no way to
  see the full name short of leaving the app. **[FIX] (c)**
- Store icon (house/shop glyph) prefixing retailer names reads semantically as "home," not "store."
  **[FIX] (c)**
- Possible ghost/skeleton badge artifact behind an "Accessory" pill on one card — seen once, not
  reproduced. **[FIX] (c), unconfirmed**
- No hover feedback at all on product card images on desktop (only the pill buttons respond) —
  feels static/unfinished on a mouse-driven surface. **[FIX] (c)**
- Card heights and button rows misalign across a grid row when one title wraps to 2 lines and its
  neighbors don't. **[FIX] (c)**
- Color filter chips below the results grid have no heading/label and unclear purpose or
  affordance. **[FIX] (c)**
- Photo-upload icons (garment photo, body-shape photo) are unlabeled icon buttons with no visible
  text — label only surfaces via hover title, which mobile users never see. Both features work once
  found, but are barely discoverable. **[FIX] (c)**
- "kurta for mom" — 3/4 returned items are black, limiting perceived variety even though items are
  individually fine. **[FIX] (a)**
- Budget cutoffs aren't strictly enforced (one mehndi-outfit item ranked 3rd of 5 landed ₹49 over a
  stated ₹3,000 ceiling). **[FIX] (a)**
- "saree for wedding reception, budget ~8000" — every result landed well under budget; didn't use
  available budget headroom to surface the catalogue's premium end for the most formal occasion in
  the set. **[FIX] (a), low confidence — can't rule out (d) without a full catalogue scan**
- Recurring console warning across every session: the chat textarea has no `id`/`name` attribute
  (autofill/label-association a11y gap). **[FIX] (c)**

---

## Inventory-capped — no code fix changes this

- **Footwear: 0 items in the entire catalogue.** Confirmed via the outfit-board path's own honest
  disclosure and via zero footwear results across every relevant query. The fix for #10 (routing
  honesty) still applies, but no ranking change surfaces shoes that don't exist. **[INVENTORY] (d)**
- **Jewellery: 0 items in the entire catalogue.** Same — confirmed via zero results across
  jewellery-adjacent queries. **[INVENTORY] (d)**
- **Sherwanis: ~1 real item in the catalogue.** "Show me sherwanis" can honestly show exactly one
  item and nothing else; the current bug is that it pads with 3 wrong-gender items instead of
  disclosing the count — the routing fix (#9) still matters, but the underlying shelf is thin.
  **[INVENTORY] (d)**
- **True jacket-style-cut lehengas: none identified.** The 4 results returned for that query were
  all standard-cut; the honest answer may be "the catalogue doesn't have this silhouette."
  **[INVENTORY] (d), moderate confidence — only checked the 4 returned + did not exhaustively scan**

---

## Summary

- **32 real-shopper queries judged** across 3 batches: 9 genuinely impressive, 8 adequate-but-
  unimpressive, 15 disappointing. The dominant failure mode by a wide margin is **(b) routing/
  intent** — soft constraints dropped, gender/age filtering inconsistent, category-boundary misses
  (nightwear leaking into wedding-attire queries), and an honesty inconsistency between the
  outfit-board path (discloses gaps) and the plain-search path (hallucinates around them). Plain,
  well-formed, concrete queries ("designer lehenga under 15000," "maroon anarkali," "dupatta only")
  perform well — the problems concentrate specifically in natural, messy, qualifier-heavy phrasing,
  which is exactly how a real shopper actually types.
- **Visual/UX sweep across mobile, desktop, and 5 secondary flows**: zero console errors and zero
  failed/slow network requests anywhere — the app is functionally stable. But the design was
  evidently built and QA'd on a narrow mobile viewport only: every dynamic content block breaks or
  cramps on desktop, tap targets are roughly half the usable minimum on mobile, and product
  photography frequently center-crops the garment out of frame.
- **Two findings rise above UX polish into trust/safety territory** and should be prioritized
  first regardless of engineering effort: the body-shape feature fabricating a physical claim from
  a photo of a t-shirt, and children's clothing surfacing in adult bridal searches.
- Most of what's broken is genuinely **[FIX]**-able (routing logic, honesty-of-disclosure
  consistency, responsive layout, tap targets, image cropping). Only 3-4 findings are
  **[INVENTORY]**-capped (footwear, jewellery, sherwani depth, jacket-style lehengas) — those need
  catalogue expansion, not code.
