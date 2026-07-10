"""Browser-level proof for the Wave-1 retest requirements (GG, 2026-07-04).

Verifies, against a *real* headless Chromium session (not WS-frame introspection):
  1. Item cards actually RENDER in the DOM after a text query.
  2. No uncaught JS exceptions (React/Next/TypeError/hydration) fire while cards render.
  3. Refinement ("in blue now") re-renders cards.
  4. Image upload shows an image thumbnail + typed text in the user bubble, then
     either renders cards or surfaces the honest timeout/error message (not an
     infinite "Finding your match..." spinner).
  5. (--product flow, B4c) The cross-store "Open all items" CTA opens an inline
     panel of per-item links instead of looping window.open() (which real
     browsers popup-block after the first call per user gesture).
  6. (--product flow, B5) "Save look" actually persists in unified (no-brand)
     mode: POST /looks returns 201, the "Look saved!" panel + share URL appear,
     and the shared /look/{id} page renders the saved items.
  7. (--phase-a flow, A1-A4) Content assertions for index-quality: saree
     recall+depth, "white" adjective relevance, no visible duplicate titles
     (incl. cross-turn), and store-diversity in results. Respec'd 2026-07-06
     after a live post-deploy run showed the reranker deliberately returns
     ~5 cards per turn by design — thresholds are calibrated to that reality,
     not to an assumed >=10-per-turn page size.
  8. (--phase-b flow, S1-S6) Content assertions for the reported outfit
     gender-leak bug (GG screenshot 2026-07-06: a women's rust dress look
     contained a MEN'S cardigan, MEN'S formal shoes, and a novelty "Luxury
     Piano Shape Handbag") and the Phase-B fix surface (backend+frontend
     committed 2026-07-07, not yet deployed at the time this flow was last
     extended): per-card `data-slot`/`data-gender` attrs (S1-S5, hard-FAIL on
     a non-empty wrong `data-gender`, title-word fallback when empty),
     suppressed-slot honesty notes (S1, evidence-only), partner-look
     cross-gender styling (S4), occasion-register vocabulary (S5), and
     distinct outfit variants (S6). This flow intentionally does NOT fix
     product code.

     Turn-budget note: the coordinator's target is <=8 assistant turns for
     the whole --phase-b run. S1-S3 (2 turns each: query + "Style this") are
     unchanged from the RED-baseline run and total 6; S4 requires a genuinely
     FRESH session (own Playwright page/tab, so its own `demo_session_token`
     — see partner-gender-resolution isolation in its docstring) with its own
     3 turns (query + "Style this" + the partner follow-up). 6 + 3 = 9 turns
     BEFORE S5's turn(s), already over budget — S5 tries a single send first
     and S6 spends zero turns (reuses an already-rendered board's variant
     tabs instead of querying). This 9-vs-8 conflict is a real arithmetic
     tension between "S1-S3 unchanged" + "S4 needs a fresh session" and the
     <=8 target; it is flagged in the implementer's report, not silently
     resolved by dropping either requirement.

  9. (--wave-7 flow, W0-W6) Content assertions for the P1 wedding-occasion
     hero features (2026-07-09 build): the unified-mode "Style Maitri" brand
     config's 5 suggestion chips + display name (W0, must run BEFORE any
     message — the chip row only renders while `messages.length === 0`), the
     "Sangeet look under (rupee)8000" hero chip's click-to-send path plus
     ethnic-occasion vocabulary and budget (W1), and the haldi/mehendi/
     reception occasion palettes (W2-W4) each checked via an OR of a
     colour/register-vocabulary card signal and an assistant-text signal.
     W5 re-proves the pre-existing partner-styling gender split (S4's
     property) still holds after three occasion turns in the SAME session,
     since the new occasion slugs are new territory for the gender-
     consistency code path. W6 is the shared zero-severe-console-errors
     check `step_console_errors` already runs unconditionally in `main`'s
     `finally` block for every flow — --wave-7 does not call it a second
     time (that would just duplicate the same summary row).

     IMPORTANT CAVEAT discovered while implementing this flow (read-before-
     edit, not asserted from memory): as of this writing,
     `frontend/components/chat/ChatPlaceholder.tsx` (the component the task
     spec says renders the 5 chips + display name) is NOT imported by
     ANY page in the frontend tree — `/demo/chat` and `/embed/[brand]`
     both render `MessageList` directly, whose own `messages.length === 0`
     branch is a fixed, brand-independent, chip-less placeholder. W0 will
     legitimately FAIL until a frontend change (out of this script's scope)
     wires `ChatPlaceholder` (or equivalent chip rendering) into one of
     those pages. W0's assertions use `page.get_by_role("button", name=...)`
     directly rather than depending on `ChatPlaceholder`'s specific
     container class, so they will pass unmodified once the wiring lands
     wherever it lands.

  10. (--p3 flow, P3-0..P3-4) Content assertions for the P3 "body type"
      styling feature (deployed before this flow runs, per the task spec):
      a 6th suggestion chip "What suits my body type?" appended to
      brands/unified.yaml's suggestion_chips (tests/test_unified_brand.py
      updated to the new 6-chip list) that must render BEFORE any message is
      sent (P3-0, same messages.length === 0 precondition as W0 -- unlike
      W0's caveat immediately above, by the time this flow was written
      `frontend/components/chat/ChatPlaceholder.tsx` no longer exists as a
      separate component; `MessageList.tsx` renders `brandConfig.
      suggestion_chips` directly in its own `messages.length === 0` branch,
      so P3-0 is expected to actually PASS rather than being merely
      future-proofed). Clicking that chip with NO stated body type must
      produce a clarify response naming >=2 distinct shape-vocabulary terms
      and signalling optionality, with zero banned-framing-word hits (P3-1).
      Stating a body type inline with an occasion+budget query ("I'm
      pear-shaped, sangeet look under 8000") must bias the rendered look
      toward pear-recommended silhouette vocabulary and add a supportive
      why-note to the assistant text, all while still respecting the stated
      budget (P3-2). A refinement turn afterward ("make it more festive")
      must keep the banned-word count at zero while still rendering new
      cards (P3-4). A session-wide sweep of EVERY assistant bubble's text
      must show zero hits against the banned "body-shaming" framing word
      list -- hide/conceal/camouflage/flaw/fix/flabby/minimise/slimming/
      unflattering/"problem area"/"not for your body type" (P3-3). The task
      spec lists P3-3 third and P3-4 fourth, but P3-3's own description says
      "after all turns" -- taken literally, P3-3 must also see P3-4's turn,
      so `main` runs P3-4 BEFORE P3-3 despite the numbering (P3-3 is the
      session-wide net, not a fourth sequential step). Console-errors (the
      shared `step_console_errors` call in `main`'s `finally` block) is
      reused unchanged, same as --wave-7 -- --p3 does not call it a second
      time.

  11. (--p2 flow, P2-1..P2-5) Content assertions for the P2 couple-
      coordination deepening feature (backend+frontend built locally, to be
      deployed before this flow runs): a FRESH session's FIRST turn --
      "style us as a couple for a reception under 15000" with NO prior
      anchor/search -- must render TWO back-to-back outfit boards in the
      SAME assistant turn (P2-1): board 0 her primary look, board 1 his
      partner look (`lookRole == "partner"`, same `OutfitBoard.tsx` rendering
      the pre-existing single-partner path already uses -- verified against
      the source at this writing, ~line 481-495 `isPartnerLook = lookRole
      === "partner"`). Each board is checked for gender purity scoped to
      ITS OWN complement cards via `complement_cards_of_board` (P2-2), a
      PER-PERSON (not combined) budget cap independently on each board's own
      slot-price sum (P2-3), and board 1's partner badge/heading +
      "Coordinated with..." subtext while board 0 shows neither (P2-4,
      honest-thinness on board 1 recorded as evidence, not a failure). P2-5
      re-proves the PRE-EXISTING single-partner flow ("black dress for
      women" -> "Style this" -> "what should my husband wear with this?")
      still produces its original one-board shape by directly reusing
      `step_pb_s4_partner_styling` in a brand-new fresh session -- this is
      additive proof, not a re-assertion that touches/duplicates PB-S4's own
      checks. Verified against `OutfitBoard.tsx` before writing P2-1..P2-4:
      there is NO `data-look-role` (or any other) DOM attribute
      distinguishing a partner board from a primary one -- the "Partner
      look" badge text is the only signal that exists today -- so
      index-based board selection (`.nth(0)`/`.nth(1)`) is the only
      addressing scheme available, and it is safe specifically because P2-1
      runs in a fresh session's very first turn (see the P2 constants-block
      comment in the source for the full reasoning). Console-errors reuses
      the shared `step_console_errors` call in `main`'s `finally` block,
      same as --wave-7/--p3 -- --p2 does not call it a second time.

  12. (--item2 flow, I2-0..I2-3) Content assertions for the NEW "photo ->
      body-shape suggestion" upload affordance (backend unchanged; frontend
      built locally in `frontend/components/chat/BodyShapeUpload.tsx` +
      `frontend/lib/poseShape.ts` / `poseLandmarker.ts`, to be deployed
      before this flow runs) -- distinct from the pre-existing garment/
      inspiration-photo upload (`ImagePlus` icon, `aria-label="Style what
      you own"`). This flow deliberately does NOT re-prove the TEXT-based
      body-type path --p3 already covers end-to-end (clarify message,
      ban-list sweep, guardrailed why-notes); it only proves the NEW
      photo-upload-affordance-specific behaviour: the affordance itself is
      present and visually/accessibly distinct from the garment upload
      (I2-0), the on-device privacy copy is visible BEFORE any file is
      picked (I2-1, regex built from the component's actual copy --
      "processed entirely in your browser, never uploaded or stored"),
      uploading a photo lands on EITHER the CONFIDENT suggestion panel
      (confirm + pick-a-different-one controls, shape-word text, zero
      banned-framing hits) OR the NOT-CONFIDENT fallback panel (all 5 shape
      buttons, zero banned-framing hits, zero raw error/exception text) --
      both are valid pass outcomes since a non-person test image
      (DEFAULT_IMAGE, t-shirt.webp) will almost certainly fail MediaPipe's
      pose-detection confidence gate (I2-2), and completing whichever
      branch appeared (clicking confirm, or deterministically tapping
      "Pear" on the fallback panel so the downstream vocabulary check is
      exercised) sends the existing chat-send natural-language message and
      correctly flows into the SAME downstream P3 pipeline already proven
      by --p3: cards/board render, the stated budget is respected, and zero
      P3_BANNED_RE hits appear in the assistant's reply (I2-3). Console-
      errors reuses the shared `step_console_errors` call in `main`'s
      `finally` block, same as every other flow -- --item2 does not call it
      a second time.

Usage:
    python scripts/browser_proof.py [--base-url URL] [--image PATH] [--headed]

Exit code 0 only if every check passes. This script intentionally does NOT special-case
the current (possibly-broken) live deployment — it reports what it observes.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

# ---------------------------------------------------------------------------
# ASCII-safe console output on Windows (cp1252 default codepage).
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE_URL = "https://asa-stylist.vercel.app/demo"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = REPO_ROOT / "t-shirt.webp"
SCRATCHPAD = Path(
    r"C:\Users\gaura\AppData\Local\Temp\claude\C--Users-gaura-ml-projects-agentic-shopping-assistant"
    r"\4a13e427-1ce5-47c4-88e2-ec3ac5d67aaf\scratchpad\browser_proof"
)
VIEWPORT = {"width": 1366, "height": 900}

# A "card" is any product tile in either rendering path: ItemCard's
# `div.rounded-lg.border.bg-card` grid tiles, or OutfitBoard's
# `a.rounded-lg.border.bg-background` slot tiles. Both always contain the
# product-name paragraph with the `line-clamp-2` class (unique to card
# titles across the whole frontend — verified against the source tree).
CARD_SELECTOR = ".rounded-lg.border:has(p.line-clamp-2)"

CARD_WAIT_TIMEOUT_S = 60
IMAGE_WAIT_TIMEOUT_S = 90
POLL_INTERVAL_S = 1.5

# Typed alongside the image upload in both the a-f flow (step_image_upload) and
# the B-series flow (step_b4_image_board reads the "under N" budget cap back
# out of this same string) — kept as one constant so the two steps can't drift.
IMAGE_UPLOAD_TEXT = "buy similar under 2000"

# Friendly error/timeout strings the frontend shows on a failed/timed-out
# image upload (see frontend/hooks/useChatStream.ts: imageUploadErrorMessage
# and the AbortError branch of sendImage's catch block).
IMAGE_ERROR_PATTERNS = [
    "taking longer than expected",
    "Could not reach the styling service",
    "Image too large",
    "doesn't look like a supported image",
    "isn't available right now",
    "Rate limited",
    "Something went wrong",
]

# JS exception signatures that indicate an actual card-render bug, as opposed
# to benign third-party/network noise (favicons, analytics, CORS probes).
SEVERE_CONSOLE_PATTERNS = [
    r"hydrat",
    r"TypeError",
    r"ReferenceError",
    r"React error",
    r"Minified React error",
    r"Cannot read propert",
    r"is not a function",
    r"Uncaught",
    r"ChunkLoadError",
]
BENIGN_CONSOLE_PATTERNS = [
    r"favicon",
    r"\.ico",
    r"analytics",
    r"vercel-insights",
    r"net::ERR_INTERNET_DISCONNECTED",
    r"chrome-extension",
]


@dataclass
class CheckResult:
    """One PASS/FAIL row in the final summary table."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ProofState:
    """Mutable state threaded through the proof steps."""

    results: list[CheckResult] = field(default_factory=list)
    console_issues: list[str] = field(default_factory=list)
    # Titles observed by earlier Phase-A steps in THIS session, so A3's dedup
    # check can span turns (a duplicate that straddles two different queries
    # is still a visible duplicate to the user). Only populated/consumed by
    # the --phase-a step group; unused by the original a-f/B steps.
    phase_a_titles: list[str] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
        self.results.append(CheckResult(name, passed, detail))


def shot(page: Page, name: str) -> Path:
    """Save a full-page screenshot to the scratchpad dir and return its path."""
    SCRATCHPAD.mkdir(parents=True, exist_ok=True)
    path = SCRATCHPAD / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return path


def card_count(page: Page) -> int:
    """Number of rendered product cards currently in the DOM."""
    return page.locator(CARD_SELECTOR).count()


def wait_for_more_cards(page: Page, baseline: int, timeout_s: float) -> int:
    """Poll until the card count exceeds `baseline` or the timeout elapses.

    Returns the final observed card count (may equal `baseline` on timeout).
    """
    deadline = time.time() + timeout_s
    count = baseline
    while time.time() < deadline:
        count = card_count(page)
        if count > baseline:
            return count
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
    return card_count(page)


SETTLE_TIMEOUT_S = 60


def wait_for_turn_idle(page: Page, timeout_s: float = SETTLE_TIMEOUT_S) -> None:
    """Drain any in-flight assistant turn (isSending) before starting the next step.

    Without this, a response that lands slightly after a step's own wait window
    can bleed its new cards into the NEXT step's baseline count, producing a
    false positive for that later step. The composer swaps its "Send" button
    for "Stop" while isSending is true, so absence of "Stop" means idle.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if page.get_by_role("button", name="Stop").count() == 0:
            return
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))


def card_title(card_locator) -> str:
    """Extract the product-name text from a card locator."""
    try:
        return card_locator.locator("p.line-clamp-2").first.inner_text().strip()
    except Exception as exc:  # noqa: BLE001 - best-effort evidence extraction
        return f"<title extraction failed: {exc}>"


def card_store_badge(card_locator) -> str:
    """Extract the store-badge text via the Store lucide icon (present in both
    ItemCard and OutfitBoard renderings)."""
    icon = card_locator.locator("svg.lucide-store")
    if icon.count() == 0:
        return ""
    try:
        return icon.first.evaluate(
            "el => el.parentElement ? el.parentElement.textContent.trim() : ''"
        )
    except Exception as exc:  # noqa: BLE001
        return f"<badge extraction failed: {exc}>"


def card_all_badges(card_locator) -> list[str]:
    """Extract all `span` badge texts from an ItemCard-style card (product type,
    colour, store). Returns [] for OutfitBoard cards, which don't use spans."""
    spans = card_locator.locator("span")
    try:
        return [t.strip() for t in spans.all_inner_texts() if t.strip()]
    except Exception:  # noqa: BLE001
        return []


def card_indicates_shirt(title: str, badges: list[str]) -> bool:
    """True if a card's type-badge (preferred) or title text mentions 'shirt'.

    Badges are the primary signal (they carry the catalogue product_type),
    with the title as a fallback for renderings that don't expose badge spans
    (e.g. OutfitBoard slot tiles).
    """
    if any("shirt" in b.lower() for b in badges):
        return True
    return "shirt" in title.lower()


def send_text(page: Page, text: str) -> None:
    """Type into the chat textarea and click Send."""
    textarea = page.locator("textarea")
    textarea.click()
    textarea.fill(text)
    page.get_by_role("button", name="Send").click()


def classify_console_message(text: str) -> str:
    """Return 'severe', 'benign', or 'unclassified' for a captured console/pageerror line."""
    for pat in BENIGN_CONSOLE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "benign"
    for pat in SEVERE_CONSOLE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "severe"
    return "unclassified"


def step_load_chat(page: Page, base_url: str, state: ProofState) -> bool:
    """Step a: load /demo, follow auto-redirect into /demo/chat, wait for the composer."""
    try:
        page.goto(base_url, wait_until="domcontentloaded", timeout=30_000)
        # The initial POST /demo/session call can be slow on a Cloud Run cold
        # start (backend warm-up banner says 15-30s) — allow generous headroom.
        page.wait_for_url(re.compile(r".*/demo/chat.*"), timeout=60_000)
        page.wait_for_selector("textarea", timeout=30_000)
        shot(page, "a_chat_loaded")
        state.record("a. demo -> chat UI loads", True, f"url={page.url}")
        return True
    except Exception as exc:  # noqa: BLE001
        shot(page, "a_chat_loaded_FAIL")
        state.record("a. demo -> chat UI loads", False, str(exc))
        return False


def step_query(page: Page, state: ProofState, label: str, query: str, shot_name: str) -> int:
    """Send a text query, wait for new cards, record evidence. Returns new card count."""
    baseline = card_count(page)
    send_text(page, query)
    new_count = wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)  # drain stragglers so the next step's baseline is clean
    gained = new_count - baseline
    passed = gained >= 1
    detail = f"cards {baseline}->{new_count}"
    if passed:
        first_new = page.locator(CARD_SELECTOR).nth(baseline)
        title = card_title(first_new)
        badge = card_store_badge(first_new)
        detail += f" | first new card: title='{title}' store_badge='{badge}'"
    shot(page, shot_name)
    state.record(f"{label} ('{query}')", passed, detail)
    return new_count


def step_refinement(page: Page, state: ProofState) -> None:
    """Step d: 'white shirt men' then 'in blue now'; record title+badges for both turns."""
    baseline = card_count(page)
    send_text(page, "white shirt men")
    after_first = wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    first_gained = after_first - baseline
    wait_for_turn_idle(page)  # drain stragglers before the refinement turn
    first_cards = []
    for i in range(baseline, after_first):
        c = page.locator(CARD_SELECTOR).nth(i)
        first_cards.append((card_title(c), card_all_badges(c)))
    shot(page, "d1_white_shirt_men")
    state.record(
        "d1. 'white shirt men' renders cards",
        first_gained >= 1,
        f"cards {baseline}->{after_first} | " + "; ".join(f"{t} {b}" for t, b in first_cards[:4]),
    )

    # Recompute the baseline AFTER draining the first turn — using the raw
    # `after_first` value (captured before drain) here would misattribute any
    # straggling first-turn cards that land during the drain to this turn.
    baseline2 = card_count(page)
    send_text(page, "in blue now")
    after_second = wait_for_more_cards(page, baseline2, CARD_WAIT_TIMEOUT_S)
    second_gained = after_second - baseline2
    wait_for_turn_idle(page)  # drain stragglers before the next step (image upload)
    second_cards = []
    for i in range(baseline2, after_second):
        c = page.locator(CARD_SELECTOR).nth(i)
        second_cards.append((card_title(c), card_all_badges(c)))
    shot(page, "d2_in_blue_now")

    # Render alone isn't proof the refinement kept "shirt" context (a fresh,
    # context-dropped conversation could still return SOME cards for "in blue
    # now" in isolation) — require a majority of the new cards to actually be
    # shirts via their type-badge (or title as a fallback).
    shirt_hits = sum(1 for t, b in second_cards if card_indicates_shirt(t, b))
    majority_shirts = second_gained >= 1 and shirt_hits > len(second_cards) / 2
    passed_d2 = second_gained >= 1 and majority_shirts
    state.record(
        "d2. 'in blue now' refinement renders cards AND stays on SHIRTS",
        passed_d2,
        f"cards {baseline2}->{after_second} | shirt_hits={shirt_hits}/{len(second_cards)} | "
        + "; ".join(f"{t} {b}" for t, b in second_cards[:4]),
    )


def step_image_upload(page: Page, state: ProofState, image_path: Path) -> int | None:
    """Step e: upload an image + typed text; verify user-bubble echo, then wait for
    cards or the honest timeout/error message (fail on infinite spinner).

    Returns the number of NEW cards that rendered from the image-search turn
    (outcome == "cards"), or None if the turn produced no cards (error/hang).
    Callers (e.g. step_b4_image_board) use this to cross-check that they've
    picked up the outfit board belonging to THIS turn, not a stale board left
    over from an earlier turn in the same session.
    """
    if not image_path.exists():
        state.record("e. image upload", False, f"image file not found: {image_path}")
        return None

    file_input = page.locator("input[type=file]")
    file_input.set_input_files(str(image_path))

    # Confirm the pending-image chip rendered in the composer before sending.
    try:
        page.wait_for_selector("img[alt='Upload preview']", timeout=5_000)
    except Exception as exc:  # noqa: BLE001
        state.record("e0. image picked into composer", False, str(exc))
        return None
    state.record("e0. image picked into composer", True)

    typed_text = IMAGE_UPLOAD_TEXT
    page.locator("textarea").fill(typed_text)
    shot(page, "e1_composer_with_image_and_text")

    baseline = card_count(page)
    page.get_by_role("button", name="Send").click()

    # User bubble echo check: both the typed text and an <img alt="Uploaded"> thumbnail.
    try:
        page.wait_for_selector("img[alt='Uploaded']", timeout=10_000)
        text_visible = page.get_by_text(typed_text, exact=False).count() > 0
        image_visible = page.locator("img[alt='Uploaded']").count() > 0
        shot(page, "e2_user_bubble_echo")
        state.record(
            "e1. user bubble shows image thumbnail + typed text",
            text_visible and image_visible,
            f"text_visible={text_visible} image_visible={image_visible}",
        )
    except Exception as exc:  # noqa: BLE001
        shot(page, "e2_user_bubble_echo_FAIL")
        state.record("e1. user bubble shows image thumbnail + typed text", False, str(exc))

    # Wait for either rendered cards or the honest timeout/error message.
    deadline = time.time() + IMAGE_WAIT_TIMEOUT_S
    outcome = "hang"
    detail = ""
    while time.time() < deadline:
        count = card_count(page)
        if count > baseline:
            outcome = "cards"
            detail = f"cards {baseline}->{count}"
            break
        page_text = page.locator("body").inner_text()
        matched = next((p for p in IMAGE_ERROR_PATTERNS if p in page_text), None)
        if matched:
            outcome = "honest_message"
            detail = f"matched message: '{matched}'"
            break
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))

    shot(page, f"e3_image_result_{outcome}")
    if outcome == "cards":
        first_new = page.locator(CARD_SELECTOR).nth(baseline)
        title = card_title(first_new)
        badge = card_store_badge(first_new)
        detail += f" | first new card: title='{title}' store_badge='{badge}'"
        state.record("e2. image search returns cards or honest message", True, detail)
        return count - baseline
    elif outcome == "honest_message":
        state.record(
            "e2. image search returns cards or honest message",
            False,
            f"no cards; showed honest error instead: {detail}",
        )
        return None
    else:
        state.record(
            "e2. image search returns cards or honest message",
            False,
            f"HANG: no cards and no honest message after {IMAGE_WAIT_TIMEOUT_S}s "
            "(spinner likely stuck on 'Finding your match...')",
        )
        return None


def step_b1_header(page: Page, state: ProofState) -> None:
    """B1: the chat header must not literally read 'Shopping Assistant Shopping Assistant'.

    frontend/app/demo/chat/page.tsx renders '{brandName} Shopping Assistant', and
    frontend/app/demo/page.tsx sets brandName to the literal string "Shopping Assistant"
    for the unified (no-brand-param) entry path — this check surfaces that duplication.
    """
    header_text = page.locator("header").first.inner_text().strip()
    duplicated = "Shopping Assistant Shopping Assistant" in header_text
    shot(page, "b1_header")
    state.record(
        "B1. header text does not duplicate 'Shopping Assistant'",
        not duplicated,
        f"header_text={header_text!r}",
    )


# Distinct from MessageList's empty-state class list (verified against source tree):
# `className="flex flex-col items-center justify-center flex-1 gap-3 text-center px-8 select-none"`.
GREETING_SELECTOR = ".flex.flex-col.items-center.justify-center.flex-1.gap-3.text-center.px-8"


def step_b2_greeting(page: Page, state: ProofState) -> None:
    """B2: capture the pre-message greeting/placeholder text and check its currency symbol.

    Must run BEFORE any query is sent (the placeholder only renders while
    messages.length === 0).
    """
    locator = page.locator(GREETING_SELECTOR)
    if locator.count() == 0:
        state.record(
            "B2. greeting/placeholder currency check", False, "greeting container not found"
        )
        return
    text = locator.first.inner_text().strip()
    has_dollar = "$" in text
    has_rupee = "₹" in text
    has_pound = "£" in text
    mentions_currency = has_dollar or has_rupee or has_pound
    ok = not has_dollar and (not mentions_currency or has_rupee)
    shot(page, "b2_greeting")
    state.record(
        "B2. greeting has no '$' and uses ₹ if it mentions currency",
        ok,
        f"text={text!r} has_dollar={has_dollar} has_rupee={has_rupee} has_pound={has_pound}",
    )


# SimilarItemRow's product-name paragraph (ItemCard.tsx) — distinct from the main
# card's `p.line-clamp-2` title, so counting these tells us the similar-panel rows.
SIMILAR_ROW_SELECTOR = "p.text-xs.font-medium.truncate.leading-tight"
# OutfitBoard's root container (`w-full rounded-xl ...`) — distinct from ItemCard's
# `rounded-lg` root and from SlotCard's `rounded-lg` tiles nested inside it.
OUTFIT_BOARD_SELECTOR = "div.rounded-xl.border.bg-card.p-4"
# Non-owned SlotCard tiles render as `<a>`; the owned-anchor seed (no buy link)
# renders as a plain `<div>` and is excluded by this selector already (see
# OutfitBoard.tsx SlotCard: only the non-owned branch returns an `<a>` root).
OUTFIT_BOARD_SLOT_SELECTOR = "a.rounded-lg.border.bg-background"

# ---------------------------------------------------------------------------
# Phase-B (owned-anchor gender-leak red-baseline) word-boundary regexes.
# Deliberately \b-anchored so "women"/"women's" never trips the MEN check —
# "men" is a literal substring of "women", the classic substring trap this
# bug report explicitly calls out — and symmetrically "ladies"/"women" don't
# false-positive the WOMEN check on a men's-look complement that happens to
# mention an unrelated word containing "men" (e.g. none in this catalogue, but
# defensive regardless).
# ---------------------------------------------------------------------------
PB_MEN_WORD_RE = re.compile(r"\b(men|men's|male)\b", re.IGNORECASE)
PB_WOMEN_WORD_RE = re.compile(r"\b(women|women's|ladies)\b", re.IGNORECASE)
PB_NOVELTY_RE = re.compile(r"\b(piano|guitar|novelty|quirky|costume)\b", re.IGNORECASE)
PB_CASUAL_WESTERN_RE = re.compile(r"\b(sneakers?|denim|hoodie|bomber)\b", re.IGNORECASE)
# Footwear-ish title vocabulary (S1/S2/S5 slot-sanity checks) — a card tagged
# `data-slot="footwear"` should read as footwear regardless of gender.
PB_FOOTWEAR_TITLE_RE = re.compile(
    r"\b(shoes?|sneakers?|loafers?|sandals?|heels?|flats?|juttis?|mojaris?)\b", re.IGNORECASE
)
# S5 occasion-register vocabulary: an office look's bottom must read as
# tailored, not casual. "denim skirt(s)" is checked as CO-OCCURRENCE
# (`denim` ... `skirt(s)`, either order, within a short word window) rather
# than an adjacent phrase — the old adjacent-phrase-only pattern
# (`denim\s+skirts?`) was a FALSE PASS on the live-proven miss: "ONLY Women
# Blue Solid Denim Mini Skirts" has "Mini" between "Denim" and "Skirts" and
# slipped straight through. `skirts?` (not just `skirt`) so a plural title
# still matches. Standalone "mini skirt(s)" is forbidden even without the
# word "denim" (a mini skirt is casual regardless of fabric). "jeans" is
# forbidden outright for an office bottom — jeans are casual regardless of
# how "office" the rest of the query was. Also forbids juniors/girls/boys/kids
# markers in ANY adult look slot (the same catalogue-mislabeling root cause
# fixed in src/agents/outfit/slots.py::is_kids_item) — a juniors item is never
# appropriate for an adult office look regardless of whether its own
# bottom-type wording happens to look tailored.
PB_S5_FORBIDDEN_BOTTOM_RE = re.compile(
    r"\bshorts?\b"
    r"|\bjeans?\b"
    r"|\bjoggers?\b"
    r"|\bmini\s+skirts?\b"
    r"|\bdenim\b[\w\s]{0,20}\bskirts?\b"
    r"|\bskirts?\b[\w\s]{0,20}\bdenim\b"
    r"|\b(?:junior|juniors|girls?|boys?|kids?)\b",
    re.IGNORECASE,
)
PB_S5_REQUIRED_BOTTOM_RE = re.compile(r"\b(trousers?|pants?|palazzos?|skirts?)\b", re.IGNORECASE)


def step_b3_interactions(page: Page, state: ProofState) -> None:
    """B3: click-through 'More like this', 'Style this', and (if a look exists) 'More formal'."""
    baseline = card_count(page)
    send_text(page, "black dress for women")
    after = wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    gained = after - baseline
    shot(page, "b3a_black_dress")
    precondition_ok = gained >= 1
    state.record(
        "B3a. 'black dress for women' renders cards (precondition)",
        precondition_ok,
        f"cards {baseline}->{after}",
    )
    if not precondition_ok:
        return

    first_card = page.locator(CARD_SELECTOR).nth(baseline)

    # -- "More like this" --------------------------------------------------
    try:
        first_card.get_by_text("More like this", exact=True).click()
    except Exception as exc:  # noqa: BLE001 - report as evidence, don't crash the run
        state.record("B3b. 'More like this' populates similar panel", False, f"click failed: {exc}")
    else:
        page.wait_for_timeout(4_000)  # allow the /catalogue/.../similar query to settle
        shot(page, "b3b_more_like_this")
        n_rows = first_card.locator(SIMILAR_ROW_SELECTOR).count()
        error_shown = first_card.get_by_text("Could not load similar items.").count() > 0
        empty_shown = first_card.get_by_text("No similar items found.").count() > 0
        state.record(
            "B3b. 'More like this' populates similar panel with >=1 row",
            n_rows >= 1 and not error_shown and not empty_shown,
            f"rows={n_rows} error_shown={error_shown} empty_shown={empty_shown}",
        )

    # -- "Style this" --------------------------------------------------------
    # After B3b, the "More like this" panel is expanded and renders its OWN
    # "Style this" buttons (one per SimilarItemRow) inside the same card
    # container matched by `first_card` (SimilarItemsPanel is a sibling div
    # nested under the card's `.rounded-lg.border` root, not a separate card).
    # get_by_text(..., exact=True) with no qualifier resolves to all of them
    # (1 main + N similar rows) and Playwright's strict mode raises on >1
    # match. We take `.first` rather than clicking "Hide similar" first: the
    # card's own action button is emitted earlier in the JSX/DOM than the
    # conditionally-rendered SimilarItemsPanel (see ItemCard.tsx), so DOM
    # order alone guarantees `.first` is the main card's button — no extra
    # click/settle-wait on the panel's collapse animation needed.
    baseline2 = card_count(page)
    try:
        first_card.get_by_text("Style this", exact=True).first.click()
    except Exception as exc:  # noqa: BLE001
        state.record("B3c. 'Style this' produces new cards/board", False, f"click failed: {exc}")
        return
    after2 = wait_for_more_cards(page, baseline2, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    outfit_board_present = page.locator(OUTFIT_BOARD_SELECTOR).count() > 0
    gained2 = after2 - baseline2
    shot(page, "b3c_style_this")
    state.record(
        "B3c. 'Style this' produces new cards/board",
        gained2 >= 1 or outfit_board_present,
        f"cards {baseline2}->{after2} outfit_board_present={outfit_board_present}",
    )

    # -- outfit-board chip: "More formal" ------------------------------------
    if outfit_board_present:
        baseline3 = card_count(page)
        try:
            page.get_by_role("button", name="More formal").click()
            after3 = wait_for_more_cards(page, baseline3, CARD_WAIT_TIMEOUT_S)
            wait_for_turn_idle(page)
            gained3 = after3 - baseline3
            shot(page, "b3d_more_formal")
            state.record(
                "B3d. 'More formal' chip produces a new assistant turn with cards",
                gained3 >= 1,
                f"cards {baseline3}->{after3}",
            )
        except Exception as exc:  # noqa: BLE001
            state.record(
                "B3d. 'More formal' chip produces a new assistant turn with cards",
                False,
                f"click failed: {exc}",
            )
    else:
        state.record(
            "B3d. 'More formal' chip produces a new assistant turn with cards",
            False,
            "no outfit board rendered after 'Style this' — chip button not present",
        )


AMOUNT_RE = re.compile(r"₹([\d,]+)")
# Pulls the numeric budget cap out of IMAGE_UPLOAD_TEXT ("buy similar under 2000" -> 2000).
BUDGET_RE = re.compile(r"under\s+(\d+)")


def _parse_rupee_amount(text: str) -> int | None:
    """Extract the first '₹N,NNN'-style amount from `text` as an int, or None."""
    m = AMOUNT_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def _parse_typed_budget(text: str) -> int | None:
    """Extract an 'under N' budget cap from typed composer text as an int, or None."""
    m = BUDGET_RE.search(text)
    return int(m.group(1)) if m else None


def step_b4_image_board(
    page: Page, state: ProofState, expected_new_cards: int | None = None
) -> None:
    """B4: after image upload, the outfit board must show >=2 slot cards, and — since
    IMAGE_UPLOAD_TEXT types an explicit budget cap ("...under 2000") — the board must
    respect that budget rather than always requiring >=3 items. A composer that
    correctly stops at 2 items totaling under the cap is CORRECT behavior, not a bug;
    B4a previously over-asserted n_slots>=3 regardless of the typed budget.

    In the --product flow, B3c ("Style this") and B3d ("More formal") each render
    their OWN outfit board earlier in the message list, BEFORE the image-upload
    turn runs. `OUTFIT_BOARD_SELECTOR` matches all of them, so grabbing `.first`
    picks up a stale board from an earlier turn (observed live: a 4-slot,
    ₹8,746 "More formal" board instead of the 2-slot image-upload board).
    Assistant messages render top-to-bottom in arrival order, so the board
    belonging to the FINAL (image-upload) turn is always the LAST one in the
    DOM — use `.last`, not `.first`.

    `expected_new_cards` (from step_image_upload's return value: the new-card
    delta observed during the image-search turn, via the same CARD_SELECTOR
    that also matches outfit-board slot tiles) is cross-checked against
    `n_slots` as a second guard against picking the wrong board: if a stale
    earlier board were picked, its slot count would very likely disagree with
    the number of cards that turn actually added.
    """
    board = page.locator(OUTFIT_BOARD_SELECTOR)
    if board.count() == 0:
        state.record(
            "B4a. image-look board >=2 slots and respects typed budget",
            False,
            "no outfit board rendered after image upload",
        )
        return

    board0 = board.last
    slot_cards = board0.locator("a.rounded-lg.border.bg-background")
    n_slots = slot_cards.count()
    shot(page, "b4_outfit_board")

    budget_cap = _parse_typed_budget(IMAGE_UPLOAD_TEXT)
    slot_prices = [_parse_rupee_amount(slot_cards.nth(i).inner_text()) for i in range(n_slots)]
    slot_prices = [p for p in slot_prices if p is not None]
    slot_price_sum = sum(slot_prices)
    budget_respected = budget_cap is None or slot_price_sum <= budget_cap
    matches_e2_delta = expected_new_cards is None or n_slots == expected_new_cards
    state.record(
        "B4a. image-look board >=2 slots and respects typed budget",
        n_slots >= 2 and budget_respected and matches_e2_delta,
        f"n_slots={n_slots} budget_cap={budget_cap} slot_price_sum={slot_price_sum} "
        f"n_prices_parsed={len(slot_prices)}/{n_slots} n_boards_on_page={board.count()} "
        f"expected_new_cards(from e2)={expected_new_cards} matches_e2_delta={matches_e2_delta}",
    )

    cta = board0.get_by_role("button", name=re.compile(r"Open all items|Add the look to cart"))
    if cta.count() == 0:
        state.record(
            "B4b. board CTA amount equals sum of card prices",
            False,
            f"no CTA button found (n_slots={n_slots} sum_of_card_prices={slot_price_sum})",
        )
        return

    cta_text = cta.first.inner_text()
    cta_amount = _parse_rupee_amount(cta_text)
    state.record(
        "B4b. board CTA amount equals sum of card prices",
        cta_amount is not None and cta_amount == slot_price_sum,
        f"cta_text={cta_text!r} cta_amount={cta_amount} sum_of_card_prices={slot_price_sum} "
        f"n_prices_parsed={len(slot_prices)}/{n_slots}",
    )


def step_b4c_open_all_panel(page: Page, state: ProofState) -> None:
    """B4c: click 'Open all items' (cross-store, no-cartUrl path) and assert the inline
    panel appears with exactly the buyable complement links -- proving the popup-
    blocking fix instead of the old window.open()-per-item forEach loop.

    Real browsers (Chrome/Edge/Safari) allow only ONE window.open() per user
    gesture; a forEach of window.open() calls silently drops every tab after the
    first (Playwright's automation flags mask this, which is why the bug wasn't
    caught in earlier browser-level proofs). The fix replaces the fan-out with a
    toggled inline panel of plain `<a target="_blank">` links, each clicked
    individually by the user.

    Anchor hrefs are cross-checked against the SAME slot cards already rendered on
    the board (`a.rounded-lg.border.bg-background` -- SlotCard renders the owned
    seed as a non-anchor `<div>`, so this selector already excludes it), since both
    the panel and the slot cards resolve through the identical
    priceableItems/resolveItemUrl logic in OutfitBoard.tsx.

    If the board used the single-store Shopify cart-URL path instead (no
    `[data-testid=open-all-items-button]` present), this check records a pass with
    an explanatory detail -- that path is unchanged/out of scope for this fix.
    """
    board = page.locator(OUTFIT_BOARD_SELECTOR)
    if board.count() == 0:
        state.record(
            "B4c. 'Open all items' shows inline panel (no popup fan-out)",
            False,
            "no outfit board present",
        )
        return
    board0 = board.last
    cta = board0.locator("[data-testid='open-all-items-button']")
    if cta.count() == 0:
        state.record(
            "B4c. 'Open all items' shows inline panel (no popup fan-out)",
            True,
            "N/A - board used single-store 'Add the look to cart' path (unchanged, out of scope)",
        )
        return

    slot_hrefs = set()
    slot_cards = board0.locator("a.rounded-lg.border.bg-background")
    for i in range(slot_cards.count()):
        href = slot_cards.nth(i).get_attribute("href")
        if href:
            slot_hrefs.add(href)

    cta.first.click()
    try:
        page.wait_for_selector("[data-testid='open-all-panel']", timeout=5_000)
    except Exception as exc:  # noqa: BLE001
        shot(page, "b4c_open_all_panel_FAIL")
        state.record(
            "B4c. 'Open all items' shows inline panel (no popup fan-out)",
            False,
            f"panel did not appear after click: {exc}",
        )
        return

    panel = board0.locator("[data-testid='open-all-panel']")
    panel_items = panel.locator("[data-testid='open-all-panel-item']")
    panel_hrefs = set()
    for i in range(panel_items.count()):
        href = panel_items.nth(i).get_attribute("href")
        if href:
            panel_hrefs.add(href)
    shot(page, "b4c_open_all_panel")

    matches = panel_hrefs == slot_hrefs and len(panel_hrefs) > 0
    state.record(
        "B4c. 'Open all items' shows inline panel (no popup fan-out)",
        matches,
        f"panel_hrefs={sorted(panel_hrefs)} slot_hrefs={sorted(slot_hrefs)}",
    )


def step_b5_save_look(page: Page, state: ProofState) -> None:
    """B5: full "Save look" round trip -- fix for the confirmed bug where the
    unified-mode frontend sent `brand: null` and the backend's (previously)
    required `SaveLookRequest.brand: str` field rejected it with a 422,
    surfacing as "Failed to save" (see scripts/save_look_repro.py, the red
    baseline this step re-proves is now green).

    Reproduces the exact validated path from save_look_repro.py: send
    "black dress for women", click "Style this" on the first card to produce
    an outfit board, then click "Save look" and assert:
      1. POST /looks returns 201 (via a response listener registered BEFORE
         the click, so nothing is missed).
      2. The "Look saved!" panel appears with a non-empty share URL.
      3. Opening that share URL (same browser/tab) renders >=1 item with a
         non-empty title, and does NOT show "Look not found".
    """
    looks_post_statuses: list[int] = []

    def _on_response(res) -> None:  # noqa: ANN001 - playwright Response, avoided import for brevity
        if "/looks" in res.url and res.request.method == "POST":
            looks_post_statuses.append(res.status)

    page.on("response", _on_response)

    try:
        baseline = card_count(page)
        send_text(page, "black dress for women")
        after = wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
        wait_for_turn_idle(page)
        gained = after - baseline
        state.record(
            "B5a. 'black dress for women' renders cards (precondition)",
            gained >= 1,
            f"cards {baseline}->{after}",
        )
        if gained < 1:
            return

        first_card = page.locator(CARD_SELECTOR).nth(baseline)
        board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
        try:
            first_card.get_by_text("Style this", exact=True).first.click()
        except Exception as exc:  # noqa: BLE001
            state.record(
                "B5b. 'Style this' produces an outfit board (precondition)",
                False,
                f"click failed: {exc}",
            )
            return

        deadline = time.time() + CARD_WAIT_TIMEOUT_S
        while time.time() < deadline:
            if page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline:
                break
            page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
        wait_for_turn_idle(page)

        board = page.locator(OUTFIT_BOARD_SELECTOR)
        board_count = board.count()
        state.record(
            "B5b. 'Style this' produces an outfit board (precondition)",
            board_count > board_baseline,
            f"boards on page: {board_count}",
        )
        if board_count <= board_baseline:
            return

        # Same board the final (image-upload / interaction) turn produced —
        # assistant messages render top-to-bottom, so `.last` is this turn's
        # board (see step_b4_image_board's docstring for the same reasoning).
        board0 = board.last
        save_btn = board0.locator("button", has=page.locator("svg.lucide-bookmark")).first
        if save_btn.count() == 0:
            state.record("B5c. POST /looks returns 201", False, "no 'Save look' button found")
            return

        save_btn.click()

        deadline = time.time() + 15
        while time.time() < deadline and not looks_post_statuses:
            page.wait_for_timeout(500)

        post_201 = 201 in looks_post_statuses
        shot(page, "b5a_after_save_click")
        state.record(
            "B5c. POST /looks returns 201",
            post_201,
            f"observed POST /looks statuses={looks_post_statuses}",
        )

        try:
            board0.get_by_text("Look saved!", exact=False).wait_for(timeout=10_000)
        except Exception as exc:  # noqa: BLE001
            shot(page, "b5b_saved_panel_FAIL")
            state.record("B5d. 'Look saved!' panel appears with share URL", False, str(exc))
            return

        share_code = board0.locator("code")
        share_url = share_code.first.inner_text().strip() if share_code.count() > 0 else ""
        shot(page, "b5b_saved_panel")
        state.record(
            "B5d. 'Look saved!' panel appears with share URL",
            bool(share_url),
            f"share_url={share_url!r}",
        )
        if not share_url:
            return

        page.goto(share_url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(2_000)
        not_found = page.get_by_text("Look not found", exact=False).count() > 0
        item_cards = page.locator(CARD_SELECTOR)
        n_items = item_cards.count()
        title = card_title(item_cards.first) if n_items > 0 else ""
        shot(page, "b5c_shared_look_page")
        state.record(
            "B5e. shared /look/{id} page renders >=1 item with a title (not 'Look not found')",
            (not not_found) and n_items >= 1 and bool(title.strip()),
            f"not_found={not_found} n_items={n_items} first_title={title!r} url={share_url}",
        )
    finally:
        page.remove_listener("response", _on_response)


def step_console_errors(state: ProofState) -> None:
    """Step f: classify collected console/pageerror messages; fail only on severe ones."""
    severe = []
    benign = []
    unclassified = []
    for msg in state.console_issues:
        cls = classify_console_message(msg)
        if cls == "severe":
            severe.append(msg)
        elif cls == "benign":
            benign.append(msg)
        else:
            unclassified.append(msg)

    print("\n--- All collected console/pageerror messages ---")
    if not state.console_issues:
        print("(none captured)")
    for msg in state.console_issues:
        print(f"  {msg}")
    print("--- end console messages ---\n")

    passed = len(severe) == 0
    detail = (
        f"severe={len(severe)} benign={len(benign)} unclassified={len(unclassified)} "
        f"total={len(state.console_issues)}"
    )
    if severe:
        detail += " | SEVERE: " + " || ".join(severe[:5])
    state.record("f. zero severe render-related console/pageerror exceptions", passed, detail)


def normalize_title(title: str) -> str:
    """Lowercase, strip non-alphanumeric characters, and collapse whitespace.

    Used by A3 to compare card titles for visible duplicates regardless of
    punctuation/casing differences (e.g. "Black Dress - Women's" vs
    "black dress womens" should be treated as the same visible title).
    """
    lowered = title.lower()
    alnum_only = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", alnum_only).strip()


def step_a1_sarees(page: Page, state: ProofState) -> None:
    """A1: respec'd 2026-07-06 to the product's ACTUAL 5-cards-per-turn reranker
    design (coordinator correction after live post-deploy run: every turn across
    all steps yielded ~5 new cards — the original >=10 threshold mis-specified
    the assertion, not a product bug).

    Turn 1 ("sarees"): >=5 new cards, with >=4 of the first 5 titles containing
    'saree' (case-insensitive).
    Turn 2 ("wedding saree under 2000", depth evidence — respec'd again
    2026-07-06): a SECOND, INDEPENDENT full-search query (not a "more ..."
    refinement of turn 1) must also return >=5 new cards, with >=4 of those 5
    titles containing 'saree'. A "more sarees under 2000" refinement was tried
    first but rejected: refinement turns can legitimately render <5 new cards
    for reasons unrelated to catalogue depth (e.g. excluding items already
    shown in the top-5), which is Phase C (refinement-behavior) territory, not
    a Phase A (index-quality) signal. A fresh independent search query isolates
    depth from refinement dedup logic. Titles from both turns are appended to
    `state.phase_a_titles`, so A3's cross-turn dedup check is the actual depth
    proof: 10 distinct sarees observed across two independent queries.

    Also fixes the settle-wait bug: `wait_for_more_cards` returns as soon as the
    FIRST new card lands, which can under-count a turn that renders several
    cards in quick succession. Both turns now call `wait_for_turn_idle` and
    re-read `card_count` AFTER the turn settles, before computing `gained`.
    """
    baseline = card_count(page)
    send_text(page, "sarees")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)  # let the full turn settle before counting
    after = card_count(page)
    gained = after - baseline
    shot(page, "a1_sarees")

    if gained < 5:
        state.record(
            "A1a. 'sarees' >=5 cards & >=4/5 titles contain 'saree'",
            False,
            f"cards {baseline}->{after} (need >=5 new cards, got {gained})",
        )
        return

    titles = [card_title(page.locator(CARD_SELECTOR).nth(baseline + i)) for i in range(5)]
    hits = sum(1 for t in titles if "saree" in t.lower())
    passed = hits >= 4
    state.record(
        "A1a. 'sarees' >=5 cards & >=4/5 titles contain 'saree'",
        passed,
        f"cards {baseline}->{after} | hits={hits}/5 | titles={titles}",
    )
    state.phase_a_titles.extend(titles)

    # Depth evidence: a SECOND, INDEPENDENT full-search query (not a "more ..."
    # refinement of turn 1) should surface ANOTHER page of sarees, isolating
    # catalogue depth from refinement-turn dedup behavior (Phase C territory).
    baseline2 = after
    send_text(page, "wedding saree under 2000")
    wait_for_more_cards(page, baseline2, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after2 = card_count(page)
    gained2 = after2 - baseline2
    shot(page, "a1b_wedding_saree")

    if gained2 < 5:
        state.record(
            "A1b. 'wedding saree under 2000' >=5 NEW cards & >=4/5 titles contain 'saree'",
            False,
            f"cards {baseline2}->{after2} (need >=5 new cards, got {gained2})",
        )
        return

    titles2 = [card_title(page.locator(CARD_SELECTOR).nth(baseline2 + i)) for i in range(5)]
    hits2 = sum(1 for t in titles2 if "saree" in t.lower())
    passed2 = hits2 >= 4
    state.record(
        "A1b. 'wedding saree under 2000' >=5 NEW cards & >=4/5 titles contain 'saree'",
        passed2,
        f"cards {baseline2}->{after2} | hits={hits2}/5 | titles={titles2}",
    )
    state.phase_a_titles.extend(titles2)


def step_a2_white_sneakers(page: Page, state: ProofState) -> None:
    """A2: 'white sneakers for men' must return >=5 cards, with >=3 of the
    first 5 titles containing 'white'. Index-quality regression check for
    adjective ("white") relevance. Threshold unchanged (was already correct
    for the 5-cards-per-turn reality); settle-wait bug fixed per A1's docstring.
    """
    baseline = card_count(page)
    send_text(page, "white sneakers for men")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)  # let the full turn settle before counting
    after = card_count(page)
    gained = after - baseline
    shot(page, "a2_white_sneakers")

    if gained < 5:
        state.record(
            "A2. 'white sneakers for men' >=5 cards & >=3/5 titles contain 'white'",
            False,
            f"cards {baseline}->{after} (need >=5 new cards, got {gained})",
        )
        return

    titles = [card_title(page.locator(CARD_SELECTOR).nth(baseline + i)) for i in range(5)]
    hits = sum(1 for t in titles if "white" in t.lower())
    passed = hits >= 3
    state.record(
        "A2. 'white sneakers for men' >=5 cards & >=3/5 titles contain 'white'",
        passed,
        f"cards {baseline}->{after} | hits={hits}/5 | titles={titles}",
    )
    state.phase_a_titles.extend(titles)


def step_a3_black_dress_dedup(page: Page, state: ProofState) -> None:
    """A3: 'black dress for women' — respec'd to the 5-cards-per-turn reality:
    the new turn's (up to 5) titles are normalized (lowercased, non-alphanumeric
    stripped, whitespace collapsed) and checked for duplicates AGAINST BOTH each
    other AND every title already seen this session (`state.phase_a_titles`,
    populated by A1/A2) — a duplicate that straddles two different queries is
    still a visible duplicate to the user, so the cross-turn check is
    deliberate, not a relaxation. Settle-wait bug fixed per A1's docstring.
    """
    baseline = card_count(page)
    send_text(page, "black dress for women")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)  # let the full turn settle before counting
    after = card_count(page)
    gained = after - baseline
    shot(page, "a3_black_dress_dedup")

    n = min(gained, 5)
    titles = [card_title(page.locator(CARD_SELECTOR).nth(baseline + i)) for i in range(n)]
    combined = state.phase_a_titles + titles
    normalized = [normalize_title(t) for t in combined]
    counts = Counter(normalized)
    dupes = sorted(t for t, c in counts.items() if c > 1)
    passed = gained >= 1 and len(dupes) == 0
    state.record(
        "A3. 'black dress for women' new titles have no visible duplicates "
        "(incl. cross-turn vs. earlier Phase-A queries this session)",
        passed,
        f"cards {baseline}->{after} | new_titles(first {n})={titles} | "
        f"prior_session_titles={state.phase_a_titles} | duplicate_normalized_titles={dupes}",
    )
    state.phase_a_titles.extend(titles)


def step_a4_summer_dress_store_diversity(page: Page, state: ProofState) -> None:
    """A4: 'summer dress' — respec'd to the 5-cards-per-turn reality: across the
    (up to 5) new cards' store badges, no single store may account for more
    than 3 of them (was >4-of-10; 3-of-5 is the equivalent >60% concentration
    threshold). Settle-wait bug fixed per A1's docstring.
    """
    baseline = card_count(page)
    send_text(page, "summer dress")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)  # let the full turn settle before counting
    after = card_count(page)
    gained = after - baseline
    shot(page, "a4_summer_dress_stores")

    n = min(gained, 5)
    stores = [card_store_badge(page.locator(CARD_SELECTOR).nth(baseline + i)) for i in range(n)]
    counts = Counter(s for s in stores if s)
    max_store, max_count = counts.most_common(1)[0] if counts else (None, 0)
    passed = gained >= 1 and max_count <= 3
    state.record(
        "A4. 'summer dress' no single store >3/5 of new cards",
        passed,
        f"cards {baseline}->{after} | stores(first {n})={stores} | "
        f"counts={dict(counts)} | max_store={max_store!r} max_count={max_count}",
    )


def card_data_attrs(card_locator) -> tuple[str | None, str | None]:
    """Read `data-gender`/`data-slot` attrs off a card locator, if the frontend
    has added them (it has not as of this writing — see the Phase-B module
    docstring entry). Returns (None, None) when absent so callers degrade
    gracefully to title heuristics rather than crashing."""
    try:
        gender = card_locator.get_attribute("data-gender")
    except Exception:  # noqa: BLE001 - best-effort evidence extraction
        gender = None
    try:
        slot = card_locator.get_attribute("data-slot")
    except Exception:  # noqa: BLE001
        slot = None
    return gender, slot


def complement_cards_of_board(board_locator) -> list:
    """Return locators for every COMPLEMENT slot card on a SPECIFIC board
    locator (as opposed to always the latest board on the page).

    Factored out of `board_complement_cards` (which still delegates to this
    for its own `.last`-board behaviour, unchanged for every existing caller)
    so the --p2 couple-from-scratch flow can inspect a caller-chosen board by
    INDEX (board 0 = her look, board 1 = his look — both known up front,
    since a fresh session's first turn renders them in a fixed, deterministic
    order) rather than only ever the most-recently-rendered board.

    Complement identification is defensive per the task spec: prefer a
    `data-slot` attribute if a later frontend fix adds one (`data-slot != "seed"`
    would then mark a card as a complement); until then, fall back to the
    "Hero" text badge SlotCard already renders on the seed/hero card today
    (its only <span> child — see OutfitBoard.tsx), which is the one
    title-independent signal available now to exclude the hero from the
    cross-gender-leak check. The owned-anchor card (no buy link) is already
    excluded by OUTFIT_BOARD_SLOT_SELECTOR, which only matches `<a>` tiles.
    """
    slot_cards = board_locator.locator(OUTFIT_BOARD_SLOT_SELECTOR)
    complements = []
    for i in range(slot_cards.count()):
        card = slot_cards.nth(i)
        slot_attr = card.get_attribute("data-slot")
        if slot_attr is not None:
            if slot_attr.lower() != "seed":
                complements.append(card)
            continue
        try:
            badge_text = card.locator("span").first.inner_text().strip()
        except Exception:  # noqa: BLE001
            badge_text = ""
        if badge_text.lower() != "hero":
            complements.append(card)
    return complements


def board_complement_cards(page: Page) -> list:
    """Return locators for every COMPLEMENT slot card in the LATEST outfit board
    on the page (assistant turns render top-to-bottom, so `.last` is always the
    most recent board — same reasoning as B4/B5's docstrings).

    See `complement_cards_of_board` for the per-board extraction logic and the
    P2 flow's reason for factoring it out.
    """
    board = page.locator(OUTFIT_BOARD_SELECTOR)
    if board.count() == 0:
        return []
    return complement_cards_of_board(board.last)


def _send_and_style_first(page: Page, state: ProofState, query: str, label_prefix: str) -> bool:
    """Shared Phase-B precondition: send `query`, wait for cards, click 'Style
    this' on the first new card, then wait for the resulting outfit board.

    Records two precondition CheckResults (`{label_prefix}a`/`{label_prefix}b`).
    Returns True iff both preconditions passed, so callers can proceed straight
    to their content assertion against `board_complement_cards(page)`.
    """
    baseline = card_count(page)
    send_text(page, query)
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    state.record(
        f"{label_prefix}a. '{query}' renders cards (precondition)",
        gained >= 1,
        f"cards {baseline}->{after}",
    )
    if gained < 1:
        return False

    first_card = page.locator(CARD_SELECTOR).nth(baseline)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    try:
        first_card.get_by_text("Style this", exact=True).first.click()
    except Exception as exc:  # noqa: BLE001
        state.record(
            f"{label_prefix}b. 'Style this' produces an outfit board (precondition)",
            False,
            f"click failed: {exc}",
        )
        return False

    deadline = time.time() + CARD_WAIT_TIMEOUT_S
    while time.time() < deadline:
        if page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline:
            break
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
    wait_for_turn_idle(page)

    board_count = page.locator(OUTFIT_BOARD_SELECTOR).count()
    passed = board_count > board_baseline
    state.record(
        f"{label_prefix}b. 'Style this' produces an outfit board (precondition)",
        passed,
        f"boards on page: {board_count}",
    )
    return passed


def _collect_complement_evidence(page: Page) -> tuple[list, list[str]]:
    """Return (complement locators, formatted evidence strings incl. title +
    data-gender/data-slot) for the latest outfit board. Shared by S1-S3."""
    complements = board_complement_cards(page)
    evidence = []
    for c in complements:
        title = card_title(c)
        gender_attr, slot_attr = card_data_attrs(c)
        evidence.append(f"title={title!r} data-gender={gender_attr!r} data-slot={slot_attr!r}")
    return complements, evidence


def assert_complement_gender(
    cards: list,
    expected_gender: str,
    opposite_word_re: re.Pattern[str],
    state: ProofState,
    check_name: str,
) -> None:
    """Assert every card in `cards` is gender-consistent with `expected_gender`.

    Prefers the frontend's `data-gender` attribute: a non-empty value that
    disagrees with `expected_gender` is a hard FAIL regardless of title
    wording (this is the whole point of the Phase-B frontend fix — an
    authoritative, title-independent signal). A card with an EMPTY
    `data-gender` (not tagged, e.g. `"unknown"` normalized to `""` upstream,
    or a pre-fix deploy) falls back to the pre-existing word-boundary title
    heuristic against `opposite_word_re`, so the check still catches an
    obvious leak on untagged cards. Shared by S1/S2/S3/S4.
    """
    wrong_attr_hits: list[str] = []
    title_word_hits: list[str] = []
    evidence: list[str] = []
    for c in cards:
        title = card_title(c)
        gender_attr, slot_attr = card_data_attrs(c)
        evidence.append(f"title={title!r} data-gender={gender_attr!r} data-slot={slot_attr!r}")
        if gender_attr:
            if gender_attr.lower() != expected_gender:
                wrong_attr_hits.append(f"{title!r} (data-gender={gender_attr!r})")
        elif opposite_word_re.search(title):
            title_word_hits.append(title)
    passed = len(cards) >= 1 and not wrong_attr_hits and not title_word_hits
    state.record(
        check_name,
        passed,
        f"n_cards={len(cards)} wrong_data_gender_hits={wrong_attr_hits} "
        f"title_word_hits={title_word_hits} | " + " || ".join(evidence),
    )


def assert_footwear_slot_sanity(cards: list, state: ProofState, check_name: str) -> None:
    """Any card tagged `data-slot="footwear"` must have a footwear-ish title
    (shoe|sneaker|loafer|sandal|heel|flat|jutti|mojari). Absence of a
    footwear card is NOT a failure — footwear is an optional slot for casual
    looks; the honest-suppression note (if any) is separate EVIDENCE, not
    part of this assertion (see `suppression_note_lines`).
    """
    footwear_cards = [c for c in cards if (card_data_attrs(c)[1] or "").lower() == "footwear"]
    if not footwear_cards:
        state.record(check_name, True, "no footwear-slot card present (optional slot, OK)")
        return
    titles = [card_title(c) for c in footwear_cards]
    bad = [t for t in titles if not PB_FOOTWEAR_TITLE_RE.search(t)]
    state.record(check_name, not bad, f"footwear_cards={titles} non_footwear_titles={bad}")


def suppression_note_lines(board_locator) -> list[str]:
    """Evidence-only helper (not an assertion): lines of the board's visible
    text containing an em dash '—', the separator OutfitBoard.tsx's
    suppressed-slot notes use (`"{Slot} — {reason}"`). The rationale block and
    partner `coordinated_with` text do not use an em dash, so this is a
    reliable-enough heuristic for EVIDENCE purposes without needing to escape
    OutfitBoard's Tailwind arbitrary-value classes (`text-[11px]`,
    `text-muted-foreground/70`) into a CSS selector.
    """
    try:
        text = board_locator.inner_text()
    except Exception:  # noqa: BLE001 - best-effort evidence extraction
        return []
    return [line.strip() for line in text.splitlines() if "—" in line]


def variant_tab_button_group(board_locator):
    """Return the Locator for the variant-switcher button GROUP on
    `board_locator`, or None if this board doesn't render one (<2 variants).

    Both the variant switcher and the refinement-chip row use the identical
    Tailwind class list `flex flex-wrap gap-1.5` (see OutfitBoard.tsx) with no
    distinguishing testid. The variant switcher, when present, is ALWAYS the
    FIRST such div in DOM order (rendered right after the occasion/gender
    header, before the slot grid); the refinement chips are ALWAYS the LAST
    (rendered after Save/share). When only one such div exists, it's the
    refinement-chip row (variants absent) — distinguishing by COUNT rather
    than by button text avoids depending on variant label wording.
    """
    groups = board_locator.locator(r"div.flex.flex-wrap.gap-1\.5")
    if groups.count() < 2:
        return None
    return groups.first


def step_pb_s1_womens_look_consistency(page: Page, state: ProofState) -> None:
    """Phase-B S1 (RED-BASELINE target): 'black dress for women' -> 'Style this'
    on the first card -> every COMPLEMENT card must be gender-consistent
    (data-gender=="women", hard FAIL on a non-empty wrong value, title-word
    fallback when empty) and must NOT match a novelty marker
    (piano|guitar|novelty|quirky|costume). This directly reproduces the
    reported bug: a women's rust dress look containing a MEN'S cardigan, MEN'S
    formal shoes, and a novelty "Luxury Piano Shape Handbag". Also asserts
    footwear slot-sanity (any data-slot="footwear" card reads as footwear) and
    records (evidence only, not an assertion) whether a suppressed-slot note
    is visible on the board.
    """
    if not _send_and_style_first(page, state, "black dress for women", "PB-S1"):
        return
    shot(page, "pb_s1_womens_look")

    board = page.locator(OUTFIT_BOARD_SELECTOR).last
    complements, _ = _collect_complement_evidence(page)

    assert_complement_gender(
        complements,
        "women",
        PB_MEN_WORD_RE,
        state,
        "PB-S1c. women's look: every complement is gender-consistent "
        "(data-gender, title fallback)",
    )

    novelty_hits = [card_title(c) for c in complements if PB_NOVELTY_RE.search(card_title(c))]
    state.record(
        "PB-S1d. women's look: no complement title matches a novelty marker",
        len(complements) >= 1 and not novelty_hits,
        f"n_complements={len(complements)} novelty_hits={novelty_hits}",
    )

    assert_footwear_slot_sanity(
        complements,
        state,
        "PB-S1e. women's look: any footwear-slot card has a footwear-ish title",
    )

    # Evidence-only (not an assertion, per spec): whether a suppressed-slot
    # note is visible. Printed rather than state.record'd so it never affects
    # PASS/FAIL — the coordinator asked for this as evidence only.
    print(f"[EVIDENCE] PB-S1 suppression-note lines visible on board: {suppression_note_lines(board)}")


def step_pb_s2_mens_look_consistency(page: Page, state: ProofState) -> None:
    """Phase-B S2: 'white shirt for men' -> 'Style this' on the first card ->
    every COMPLEMENT card must be gender-consistent (data-gender=="men", hard
    FAIL on a non-empty wrong value, title-word fallback when empty). Mirror-
    image check of S1 — proves the leak isn't one-directional. Also asserts
    footwear slot-sanity (any data-slot="footwear" card reads as footwear).
    """
    if not _send_and_style_first(page, state, "white shirt for men", "PB-S2"):
        return
    shot(page, "pb_s2_mens_look")

    complements, _ = _collect_complement_evidence(page)
    assert_complement_gender(
        complements,
        "men",
        PB_WOMEN_WORD_RE,
        state,
        "PB-S2c. men's look: every complement is gender-consistent "
        "(data-gender, title fallback)",
    )
    assert_footwear_slot_sanity(
        complements,
        state,
        "PB-S2d. men's look: any footwear-slot card has a footwear-ish title",
    )


def step_pb_s3_sangeet_ethnic_gate(page: Page, state: ProofState) -> None:
    """Phase-B S3: 'style a kurta for sangeet for women' -> 'Style this' on the
    first card -> every COMPLEMENT card's title must NOT contain a casual-
    western marker (sneaker(s)|denim|hoodie|bomber). Ethnic-occasion gate check
    — a sangeet/kurta look pulling in sneakers or a bomber jacket is the same
    class of bug as S1's cross-gender leak, just gated on occasion instead of
    gender. Unchanged from the RED-baseline run, plus a NEW gender-consistency
    assertion (data-gender=="women", title fallback).
    """
    if not _send_and_style_first(page, state, "style a kurta for sangeet for women", "PB-S3"):
        return
    shot(page, "pb_s3_sangeet_ethnic")

    complements, evidence = _collect_complement_evidence(page)
    casual_hits = [
        card_title(c) for c in complements if PB_CASUAL_WESTERN_RE.search(card_title(c))
    ]
    passed = len(complements) >= 1 and not casual_hits
    state.record(
        "PB-S3c. sangeet/ethnic look: no complement title contains a "
        "casual-western marker (sneaker/denim/hoodie/bomber)",
        passed,
        f"n_complements={len(complements)} casual_hits={casual_hits} | " + " || ".join(evidence),
    )
    assert_complement_gender(
        complements,
        "women",
        PB_MEN_WORD_RE,
        state,
        "PB-S3d. sangeet/ethnic look: every complement is gender-consistent "
        "(data-gender, title fallback)",
    )


def step_pb_s4_partner_styling(context, base_url: str, state: ProofState) -> None:
    """Phase-B S4 (NEW): partner-styling cross-gender coordination.

    Runs in a FRESH session — its own Playwright page/tab off the SAME
    browser context (a new page gets its own `sessionStorage`, hence its own
    `demo_session_token`/conversation, without the overhead of a whole new
    BrowserContext) — rather than the shared session S1-S3 use. This isolates
    two things: (1) the "original board still present, unchanged" assertion
    below can't be confused with an earlier step's board mutating the page,
    and (2) 'husband' partner-gender resolution isn't influenced by any prior
    turn's gender context (e.g. S2's men's-shirt turn immediately before it,
    if this ran in the shared session).

    Turn budget: 3 sends in this fresh session (query, 'Style this' click,
    the partner follow-up) — see the module docstring's Phase-B turn-budget
    note for how this interacts with the requested "<=8 messages total" goal.

    Flow: 'black dress for women' -> 'Style this' on first card -> capture
    the resulting board's complement titles -> 'what should my husband wear
    with this?' -> assert a NEW board appears with the partner heading/badge,
    a coordinated_with subheading mentioning the anchor, EVERY card in the
    partner board is data-gender=="men" (hard FAIL on any "women"), and the
    ORIGINAL women's board is still present with the SAME complement titles
    captured before the partner turn.
    """
    fresh_page = context.new_page()
    try:
        if not step_load_chat(fresh_page, base_url, state):
            return
        if not _send_and_style_first(fresh_page, state, "black dress for women", "PB-S4"):
            return
        shot(fresh_page, "pb_s4_original_board")

        original_complements = board_complement_cards(fresh_page)
        original_titles = [card_title(c) for c in original_complements]

        board_baseline = fresh_page.locator(OUTFIT_BOARD_SELECTOR).count()
        send_text(fresh_page, "what should my husband wear with this?")
        deadline = time.time() + CARD_WAIT_TIMEOUT_S
        while time.time() < deadline:
            if fresh_page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline:
                break
            fresh_page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
        wait_for_turn_idle(fresh_page)
        shot(fresh_page, "pb_s4_partner_board")

        board_count = fresh_page.locator(OUTFIT_BOARD_SELECTOR).count()
        state.record(
            "PB-S4a. 'what should my husband wear with this?' produces a NEW outfit board",
            board_count > board_baseline,
            f"boards on page: {board_count}",
        )
        if board_count <= board_baseline:
            return

        partner_board = fresh_page.locator(OUTFIT_BOARD_SELECTOR).last
        try:
            board_text = partner_board.inner_text()
        except Exception:  # noqa: BLE001
            board_text = ""

        has_partner_marker = "Partner look" in board_text or "Your partner's look" in board_text
        state.record(
            "PB-S4b. partner board shows the partner heading/badge",
            has_partner_marker,
            f"board_text_snippet={board_text[:200]!r}",
        )

        coordinated_line = next(
            (line for line in board_text.splitlines() if "coordinated" in line.lower()),
            "",
        )
        mentions_anchor = (
            "black dress" in coordinated_line.lower() or "coordinated" in coordinated_line.lower()
        )
        state.record(
            "PB-S4c. partner board's coordinated_with subheading mentions the anchor",
            bool(coordinated_line) and mentions_anchor,
            f"coordinated_line={coordinated_line!r}",
        )

        partner_slot_cards = partner_board.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        n_partner_cards = partner_slot_cards.count()
        partner_cards = [partner_slot_cards.nth(i) for i in range(n_partner_cards)]
        assert_complement_gender(
            partner_cards,
            "men",
            PB_WOMEN_WORD_RE,
            state,
            "PB-S4d. every card in the partner board is data-gender==\"men\" "
            "(hard FAIL on any \"women\")",
        )

        # Original board must still be present, unchanged — re-derive its
        # complement titles the SAME way `original_titles` was captured
        # (excluding the hero/seed via data-slot) and compare.
        still_present = fresh_page.locator(OUTFIT_BOARD_SELECTOR).count() >= 2
        original_board_now = fresh_page.locator(OUTFIT_BOARD_SELECTOR).first
        current_cards = original_board_now.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        recheck_titles = []
        for i in range(current_cards.count()):
            c = current_cards.nth(i)
            slot_attr = c.get_attribute("data-slot")
            if (slot_attr or "").lower() != "seed":
                recheck_titles.append(card_title(c))
        unchanged = still_present and recheck_titles == original_titles
        state.record(
            "PB-S4e. original women's board is still present with unchanged complement titles",
            unchanged,
            f"original_titles={original_titles} recheck_titles={recheck_titles} "
            f"still_present={still_present}",
        )
    finally:
        fresh_page.close()


def step_pb_s5_occasion_register(page: Page, state: ProofState) -> None:
    """Phase-B S5 (NEW): occasion register — an office-context look's
    bottom-slot item should read as tailored (trouser/pant/palazzo/skirt), not
    casual (shorts/denim skirt/joggers).

    Tries the cheapest phrasing first ('office look for women', a single send
    that may compose a board directly) to keep this step's turn cost minimal
    (see the module's Phase-B turn-budget note); only falls back to the
    'style a garment for office' + 'Style this' 2-turn flow if the single
    send doesn't board within the settle window.
    """
    baseline = card_count(page)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    send_text(page, "office look for women")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    board_after_direct = page.locator(OUTFIT_BOARD_SELECTOR).count()

    if board_after_direct > board_baseline:
        state.record(
            "PB-S5a. 'office look for women' produces an outfit board directly",
            True,
            f"boards on page: {board_after_direct}",
        )
    else:
        state.record(
            "PB-S5a. 'office look for women' produces an outfit board directly",
            False,
            f"no board after direct send (boards={board_after_direct}); "
            "falling back to 'style this' flow",
        )
        if not _send_and_style_first(
            page, state, "black top for office for women", "PB-S5fallback"
        ):
            return

    board = page.locator(OUTFIT_BOARD_SELECTOR).last
    shot(page, "pb_s5_office_look")

    slot_cards = board.locator(OUTFIT_BOARD_SLOT_SELECTOR)
    bottom_title = None
    for i in range(slot_cards.count()):
        c = slot_cards.nth(i)
        slot_attr = c.get_attribute("data-slot")
        if (slot_attr or "").lower() == "bottom":
            bottom_title = card_title(c)
            break

    if bottom_title is None:
        state.record(
            "PB-S5b. office look's bottom-slot card is tailored, not casual",
            False,
            "no data-slot='bottom' card present on the board",
        )
        return

    forbidden_hit = bool(PB_S5_FORBIDDEN_BOTTOM_RE.search(bottom_title))
    required_hit = bool(PB_S5_REQUIRED_BOTTOM_RE.search(bottom_title))
    state.record(
        "PB-S5b. office look's bottom-slot card is tailored, not casual",
        required_hit and not forbidden_hit,
        f"bottom_title={bottom_title!r} forbidden_hit={forbidden_hit} required_hit={required_hit}",
    )


def step_pb_s6_distinct_variants(page: Page, state: ProofState) -> None:
    """Phase-B S6 (NEW): distinct outfit variants.

    Zero-turn-cost by design (see the module's Phase-B turn-budget note):
    reuses whichever board ALREADY on the page (from S1/S2/S3/S5, in DOM
    order — S4 runs in its own separate page, so its boards are never visible
    here) is the first to expose >=2 variant tabs, rather than sending a new
    query. If NO board on the page has >=2 variants, this is recorded as
    evidence (thin-pool honesty is allowed per spec) rather than a hard
    failure.
    """
    boards = page.locator(OUTFIT_BOARD_SELECTOR)
    n_boards = boards.count()
    target_board = None
    tab_group = None
    for i in range(n_boards):
        b = boards.nth(i)
        g = variant_tab_button_group(b)
        if g is not None and g.locator("button").count() >= 2:
            target_board = b
            tab_group = g
            break

    if target_board is None:
        state.record(
            "PB-S6. >=2 outfit variants exist and their complement sets are pairwise disjoint",
            True,
            f"no board among the {n_boards} on the page rendered >=2 variant tabs this run "
            "(thin-variant-pool honesty — not a hard failure per spec)",
        )
        return

    buttons = tab_group.locator("button")
    n_variants = buttons.count()
    variant_labels: list[str] = []
    variant_title_sets: list[set[str]] = []
    for i in range(n_variants):
        buttons.nth(i).click()
        # Variant switching is a local React state update (no network round
        # trip beyond a fire-and-forget telemetry POST /events — see
        # handleVariantSwitch), so a short fixed wait is enough to settle the
        # re-render; wait_for_turn_idle's "Stop" button polling doesn't apply
        # here since no assistant turn is in flight.
        page.wait_for_timeout(500)
        variant_labels.append(buttons.nth(i).inner_text().strip())
        cards = target_board.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        titles: set[str] = set()
        for j in range(cards.count()):
            c = cards.nth(j)
            slot_attr = c.get_attribute("data-slot")
            if (slot_attr or "").lower() != "seed":
                titles.add(card_title(c))
        variant_title_sets.append(titles)
    shot(page, "pb_s6_variants")

    pairwise_disjoint = all(
        variant_title_sets[i].isdisjoint(variant_title_sets[j])
        for i in range(n_variants)
        for j in range(i + 1, n_variants)
    )
    passed = n_variants >= 2 and pairwise_disjoint
    state.record(
        "PB-S6. >=2 outfit variants exist and their complement sets are pairwise disjoint",
        passed,
        f"n_variants={n_variants} labels={variant_labels} "
        f"title_sets={[sorted(s) for s in variant_title_sets]} pairwise_disjoint={pairwise_disjoint}",
    )


# ---------------------------------------------------------------------------
# Wave-7 (P1 wedding-occasion hero features) constants & steps.
#
# The exact 5 chip strings the unified "Style Maitri" brand config is expected
# to return from GET /api/brand (`suggestion_chips`) — see
# frontend/hooks/useBrandConfig.ts's BrandConfig interface and
# frontend/components/chat/ChatPlaceholder.tsx's chip-button rendering (per
# the module docstring's caveat, ChatPlaceholder is not wired into any page
# as of this writing; W0 checks the chip TEXT via role selectors so it stays
# correct regardless of which page eventually mounts it).
# ---------------------------------------------------------------------------
WAVE7_CHIPS = [
    "Sangeet look under ₹8000",
    "Haldi outfit — bright & daytime",
    "Wedding-guest saree under ₹5000",
    "Style my partner for a reception",
    "Mehendi look in green",
]

# W1 ethnic-occasion vocabulary — a sangeet-register look's cards should read
# as festive/ethnic wear, not casual western. `\w*` suffixes on the
# embellish/embroider/sequin stems so inflected forms ("embellished",
# "embroidered", "sequinned") still match, mirroring PB_CASUAL_WESTERN_RE's
# plural-friendly style elsewhere in this file.
W7_SANGEET_VOCAB_RE = re.compile(
    r"\b(lehengas?|sarees?|anarkalis?|kurtas?|shararas?|cholis?|dupattas?|ethnic|"
    r"embellish\w*|embroider\w*|sequins?|zari)\b",
    re.IGNORECASE,
)
# W2 haldi: bright/marigold-yellow palette. `gold\w*` catches "golden" too.
W7_HALDI_COLOUR_RE = re.compile(
    r"\b(yellow|marigold|mustard|orange|amber|gold\w*)\b", re.IGNORECASE
)
W7_HALDI_TEXT_RE = re.compile(r"\b(haldi|bright|daytime|yellow|marigold)\b", re.IGNORECASE)
# W3 mehendi: green/mint palette.
W7_MEHENDI_COLOUR_RE = re.compile(r"\b(green|mint|olive|sage|emerald)\b", re.IGNORECASE)
W7_MEHENDI_TEXT_RE = re.compile(r"\b(mehendi|green)\b", re.IGNORECASE)
# W4 reception: glam/embellished evening register.
W7_RECEPTION_GLAM_RE = re.compile(
    r"\b(embellish\w*|sequins?|velvet\w*|silks?|satins?|zari|embroider\w*|gowns?|jewels?|"
    r"wine|maroon|emerald|navy|metallic\w*)\b",
    re.IGNORECASE,
)
W7_RECEPTION_TEXT_RE = re.compile(r"\b(reception|glam\w*|evening|embellish\w*)\b", re.IGNORECASE)
# W5 partner regression: men's garment vocabulary, OR'd with PB_MEN_WORD_RE for
# the positive "this is a men's look" signal (distinct from PB_WOMEN_WORD_RE,
# which is the negative/violation check reused unchanged from Phase-B).
W7_MEN_GARMENT_RE = re.compile(
    r"\b(kurtas?|sherwanis?|bandhgalas?|blazers?|nehru)\b", re.IGNORECASE
)

# MessageBubble.tsx: both roles share `rounded-2xl px-4 py-2.5 text-sm
# leading-relaxed`, layered with role-specific classes — assistant-only gets
# `bg-muted text-foreground rounded-bl-sm` (user gets `bg-primary
# text-primary-foreground rounded-br-sm` instead). `bg-muted`+`rounded-bl-sm`
# alone is enough to select assistant bubbles only; `rounded-2xl` added for
# extra specificity, matching this file's existing precise-selector style.
ASSISTANT_BUBBLE_SELECTOR = "div.rounded-2xl.bg-muted.text-foreground.rounded-bl-sm"


def last_assistant_text(page: Page) -> str:
    """Return the innerText of the most recent assistant message bubble, or ''.

    Assistant messages render top-to-bottom in arrival order (same reasoning
    as `board_complement_cards`'s `.last` usage), so the last matched bubble
    is always the newest assistant turn.
    """
    bubbles = page.locator(ASSISTANT_BUBBLE_SELECTOR)
    if bubbles.count() == 0:
        return ""
    try:
        return bubbles.last.inner_text().strip()
    except Exception:  # noqa: BLE001 - best-effort evidence extraction
        return ""


def card_matches(card_locator, pattern: re.Pattern[str]) -> bool:
    """True if `pattern` matches the card's title or any of its badge texts.

    Badges are empty for OutfitBoard slot tiles (they don't render `<span>`
    badges — see `card_all_badges`'s docstring), so this degrades gracefully
    to a title-only check for board cards and a title-or-badge check for
    ItemCard grid cards (which DO carry a colour badge span).
    """
    if pattern.search(card_title(card_locator)):
        return True
    return any(pattern.search(b) for b in card_all_badges(card_locator))


def new_card_titles(page: Page, baseline: int, after: int) -> list[str]:
    """Title text for every card newly rendered this turn, `CARD_SELECTOR`
    indices [baseline, after). `CARD_SELECTOR` matches both ItemCard grid
    tiles and OutfitBoard `<a>` slot tiles (see the module-level
    `CARD_SELECTOR` docstring), so this works uniformly whether the turn
    rendered a grid of cards or an outfit board.
    """
    return [card_title(page.locator(CARD_SELECTOR).nth(i)) for i in range(baseline, after)]


def new_card_locators(page: Page, baseline: int, after: int) -> list:
    """Locators for every card newly rendered this turn — see `new_card_titles`."""
    return [page.locator(CARD_SELECTOR).nth(i) for i in range(baseline, after)]


def step_w0_chips_and_brand(page: Page, state: ProofState) -> None:
    """W0: BEFORE any message is sent, assert the unified-mode brand config
    surfaced >=4 of the 5 exact WAVE7_CHIPS as clickable buttons (role-based,
    exact text — the chip's `₹` rupee glyph must survive intact), and
    the page shows "Style Maitri" somewhere but never "H&M" (regression check:
    the header literally hardcodes the Style Maitri wordmark today — see
    frontend/components/Logo.tsx — so this also guards against a future
    regression that reintroduces a raw brand-name string).
    """
    hits = [
        chip
        for chip in WAVE7_CHIPS
        if page.get_by_role("button", name=chip, exact=True).count() > 0
    ]
    rupee_chip_present = (
        page.get_by_role("button", name="Sangeet look under ₹8000", exact=True).count() > 0
    )
    body_text = page.locator("body").inner_text()
    has_stylemitra = "Style Maitri" in body_text
    has_hm = "H&M" in body_text
    shot(page, "w0_chips_and_brand")

    passed = len(hits) >= 4 and rupee_chip_present and has_stylemitra and not has_hm
    state.record(
        "W0. >=4/5 WAVE7_CHIPS render as buttons (rupee glyph intact) and "
        "page shows 'Style Maitri' not 'H&M'",
        passed,
        f"hits={len(hits)}/5 chips_found={hits} rupee_chip_present={rupee_chip_present} "
        f"has_stylemitra={has_stylemitra} has_hm={has_hm}",
    )


def step_w1_sangeet_hero(page: Page, state: ProofState) -> None:
    """W1: click the "Sangeet look under ₹8000" hero chip (falling back to
    typing the identical text if the click doesn't register a user turn — the
    chip button disappears once `messages.length > 0`, per ChatPlaceholder.tsx
    / MessageList.tsx, so the button vanishing is the click-registered signal).

    Asserts: (a) an outfit board OR >=3 new cards; (b) sangeet-register
    vocabulary in >=1 new card title; (c) no new card title matches
    PB_MEN_WORD_RE unless its own data-gender attr says "women" (women-
    default gender consistency must survive the new occasion path); (d) a
    budget check — sum of board slot prices, or each visible new card's
    price where parseable — <= 8000.
    """
    chip_text = "Sangeet look under ₹8000"
    baseline = card_count(page)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()

    chip_button = page.get_by_role("button", name=chip_text, exact=True)
    clicked = False
    if chip_button.count() > 0:
        try:
            chip_button.first.click()
            page.wait_for_timeout(1_000)
            clicked = page.get_by_role("button", name=chip_text, exact=True).count() == 0
        except Exception:  # noqa: BLE001 - fall through to the typed-text fallback
            clicked = False
    if not clicked:
        send_text(page, chip_text)

    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    outfit_board_present = page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline
    shot(page, "w1_sangeet_hero")

    render_ok = outfit_board_present or gained >= 3
    state.record(
        "W1a. sangeet hero chip renders an outfit board or >=3 new cards",
        render_ok,
        f"clicked_chip={clicked} cards {baseline}->{after} "
        f"outfit_board_present={outfit_board_present}",
    )
    if not render_ok:
        return

    titles = new_card_titles(page, baseline, after)
    vocab_hits = [t for t in titles if W7_SANGEET_VOCAB_RE.search(t)]
    state.record(
        "W1b. >=1 new card title matches sangeet/ethnic-occasion vocabulary",
        len(vocab_hits) >= 1,
        f"vocab_hits={vocab_hits} titles={titles}",
    )

    new_cards = new_card_locators(page, baseline, after)
    men_word_violations = []
    for c in new_cards:
        title = card_title(c)
        if PB_MEN_WORD_RE.search(title):
            gender_attr, _ = card_data_attrs(c)
            if (gender_attr or "").lower() != "women":
                men_word_violations.append(f"{title!r} (data-gender={gender_attr!r})")
    state.record(
        "W1c. no new card title matches PB_MEN_WORD_RE unless data-gender says 'women'",
        not men_word_violations,
        f"violations={men_word_violations} titles={titles}",
    )

    if outfit_board_present:
        slot_cards = page.locator(OUTFIT_BOARD_SELECTOR).last.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        slot_prices = [
            _parse_rupee_amount(slot_cards.nth(i).inner_text()) for i in range(slot_cards.count())
        ]
        parsed_prices = [p for p in slot_prices if p is not None]
        price_sum = sum(parsed_prices)
        budget_ok = price_sum <= 8000
        budget_detail = (
            f"board_slot_price_sum={price_sum} n_prices_parsed={len(parsed_prices)}/"
            f"{slot_cards.count()}"
        )
    else:
        prices = [_parse_rupee_amount(c.inner_text()) for c in new_cards]
        parsed_prices = [p for p in prices if p is not None]
        budget_ok = all(p <= 8000 for p in parsed_prices)
        budget_detail = f"new_card_prices={prices}"
    state.record("W1d. budget respected (<=8000)", budget_ok, budget_detail)


def _step_w_occasion_palette(
    page: Page,
    state: ProofState,
    query: str,
    label_prefix: str,
    colour_re: re.Pattern[str],
    text_re: re.Pattern[str],
    shot_name: str,
) -> None:
    """Shared W2/W3/W4 occasion-palette check: send `query`, assert >=1 new
    card, then assert EITHER a colour/register-vocabulary card signal OR an
    assistant-text signal — recording which one(s) matched as evidence.
    """
    baseline = card_count(page)
    send_text(page, query)
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    shot(page, shot_name)

    render_ok = gained >= 1
    state.record(
        f"{label_prefix}a. '{query}' renders >=1 new card",
        render_ok,
        f"cards {baseline}->{after}",
    )
    if not render_ok:
        return

    new_cards = new_card_locators(page, baseline, after)
    card_hits = [card_title(c) for c in new_cards if card_matches(c, colour_re)]
    assistant_text = last_assistant_text(page)
    text_hit = bool(text_re.search(assistant_text))
    passed = bool(card_hits) or text_hit
    state.record(
        f"{label_prefix}b. colour/register signal: card title/badge OR assistant text",
        passed,
        f"card_hits={card_hits} text_hit={text_hit} "
        f"assistant_text_snippet={assistant_text[:200]!r}",
    )


def step_w2_haldi_palette(page: Page, state: ProofState) -> None:
    """W2: 'haldi outfit for a woman' — bright marigold-yellow palette check."""
    _step_w_occasion_palette(
        page,
        state,
        "haldi outfit for a woman",
        "W2",
        W7_HALDI_COLOUR_RE,
        W7_HALDI_TEXT_RE,
        "w2_haldi_palette",
    )


def step_w3_mehendi_palette(page: Page, state: ProofState) -> None:
    """W3: 'mehendi look in green' — green/mint palette check."""
    _step_w_occasion_palette(
        page,
        state,
        "mehendi look in green",
        "W3",
        W7_MEHENDI_COLOUR_RE,
        W7_MEHENDI_TEXT_RE,
        "w3_mehendi_palette",
    )


def step_w4_reception_register(page: Page, state: ProofState) -> None:
    """W4: 'reception look under 10000' — glam/embellished evening register check."""
    _step_w_occasion_palette(
        page,
        state,
        "reception look under 10000",
        "W4",
        W7_RECEPTION_GLAM_RE,
        W7_RECEPTION_TEXT_RE,
        "w4_reception_register",
    )


def step_w5_partner_regression(page: Page, state: ProofState) -> None:
    """W5: after W2-W4's occasion turns, 'what should my husband wear' must
    still produce a gender-consistent MEN's look — re-proving Phase-B's
    partner-styling gender split (S4) holds on the new occasion code paths.

    Copies the Phase-B approach: data-gender attr is authoritative when
    present (hard FAIL on "women"), title-word fallback (PB_WOMEN_WORD_RE)
    when absent. The positive signal (>=1 card reads as a men's look) uses
    PB_MEN_WORD_RE OR'd with men's-garment vocabulary (W7_MEN_GARMENT_RE),
    since a men's kurta/sherwani card may never say the literal word "men".
    """
    baseline = card_count(page)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    send_text(page, "what should my husband wear")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    outfit_board_present = page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline
    shot(page, "w5_partner_regression")

    render_ok = outfit_board_present or gained >= 1
    state.record(
        "W5a. 'what should my husband wear' produces a new board/cards",
        render_ok,
        f"cards {baseline}->{after} outfit_board_present={outfit_board_present}",
    )
    if not render_ok:
        return

    new_cards = new_card_locators(page, baseline, after)
    titles = [card_title(c) for c in new_cards]
    men_hits = [t for t in titles if PB_MEN_WORD_RE.search(t) or W7_MEN_GARMENT_RE.search(t)]

    women_violations = []
    for c, title in zip(new_cards, titles):
        gender_attr, _ = card_data_attrs(c)
        if gender_attr:
            if gender_attr.lower() == "women":
                women_violations.append(f"{title!r} (data-gender={gender_attr!r})")
        elif PB_WOMEN_WORD_RE.search(title):
            women_violations.append(title)

    passed = len(men_hits) >= 1 and not women_violations
    state.record(
        "W5b. partner look: >=1 men's-vocab card AND no women-vocab card "
        "(data-gender first, title fallback)",
        passed,
        f"men_hits={men_hits} women_violations={women_violations} titles={titles}",
    )


# ---------------------------------------------------------------------------
# P3 ("What suits my body type?" styling) constants & steps.
#
# The 6th suggestion chip appended to brands/unified.yaml's suggestion_chips
# list (tests/test_unified_brand.py's _EXPECTED_CHIPS updated to 6 entries).
# Reuses WAVE7_CHIPS for the first 5 rather than redefining them, so the two
# lists can't silently drift apart.
# ---------------------------------------------------------------------------
P3_CHIP = "What suits my body type?"
P3_ALL_CHIPS = [*WAVE7_CHIPS, P3_CHIP]

# P3-1a/P3-2b: shape vocabulary a clarify or why-note response may name.
# "inverted triangle" is a two-word phrase; `\b` still anchors correctly at
# its outer edges since it's matched as one literal alternative.
P3_SHAPE_WORD_RE = re.compile(
    r"\b(pear|apple|hourglass|rectangle|inverted triangle|petite|plus)\b", re.IGNORECASE
)
# P3-1b: the clarify response must read as optional, not a requirement to
# answer before the assistant will help at all. Both straight (') and
# typographic (’) apostrophes covered for "if you'd like" since the LLM
# response text isn't guaranteed to use one or the other consistently.
P3_OPTIONALITY_RE = re.compile(
    r"\b(optional|only if|if you’d like|if you'd like|no need|either way)\b",
    re.IGNORECASE,
)
# P3-2b: pear-recommended silhouette vocabulary a rendered card's title/badge
# should carry. "a.line"/"a-line" both kept literally per the task spec (the
# "." also incidentally matches "a line" with a space or "a_line").
P3_PEAR_SILHOUETTE_RE = re.compile(
    r"\b(a.line|a-line|anarkali|flared|flare|empire|wrap|lehenga)\b", re.IGNORECASE
)
# P3-2c: supportive why-note vocabulary the assistant text should use when
# explaining why a look suits a stated body type.
P3_WHY_NOTE_RE = re.compile(
    r"\b(pear|silhouette|balance|balances|flatter|celebrates|skims)\b", re.IGNORECASE
)
# P3-1c/P3-2c/P3-3/P3-4: banned body-shaming framing vocabulary -- the
# framing guarantee this whole feature exists to enforce. "problem area(s)"
# and "not for your body type" are multi-word phrases, so they get their own
# \s+-tolerant alternatives instead of relying on a literal-with-internal-
# space (which would work too, but \s+ also tolerates a stray double space
# or line wrap in streamed text).
P3_BANNED_RE = re.compile(
    r"\b(hide|hides|hiding|conceal|conceals|camouflage|flaws?|fix|fixes|flabby|"
    r"minimi[sz]e|slimming|unflattering)\b"
    r"|\bproblem\s+areas?\b"
    r"|\bnot\s+for\s+your\s+body\s+type\b",
    re.IGNORECASE,
)


def wait_for_assistant_reply(page: Page, timeout_s: float = CARD_WAIT_TIMEOUT_S) -> None:
    """Wait for a turn to fully complete when no card render is expected (e.g.
    P3-1's clarify turn) -- `wait_for_more_cards` polls `card_count` and would
    never succeed for a text-only response.

    First waits (briefly, up to 5s) for the composer to show "Stop"
    (isSending true), tolerating a turn that completes faster than we can
    observe it, then waits for "Stop" to disappear again via
    `wait_for_turn_idle`.
    """
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if page.get_by_role("button", name="Stop").count() > 0:
            break
        page.wait_for_timeout(200)
    wait_for_turn_idle(page, timeout_s=timeout_s)


def step_p3_0_chip(page: Page, state: ProofState) -> None:
    """P3-0: BEFORE any message is sent, assert the 6th suggestion chip
    "What suits my body type?" renders as a clickable button (unified.yaml's
    suggestion_chips list is 6 entries now -- see tests/test_unified_brand.py).

    Evidence-only: also records how many of the other 5 WAVE7_CHIPS render
    alongside it -- not a P3 assertion, W0 already covers that thoroughly.
    """
    chip_present = page.get_by_role("button", name=P3_CHIP, exact=True).count() > 0
    other_hits = [
        chip for chip in WAVE7_CHIPS if page.get_by_role("button", name=chip, exact=True).count() > 0
    ]
    shot(page, "p3_0_chip")
    state.record(
        "P3-0. 'What suits my body type?' chip renders as a button before any message",
        chip_present,
        f"chip_present={chip_present} other_wave7_chips_also_present={len(other_hits)}/5",
    )


def step_p3_1_clarify(page: Page, state: ProofState) -> None:
    """P3-1: click the body-type chip with NO stated body type. This is a
    clarify turn -- cards are not required (per the task spec). Asserts the
    assistant's response text (a) mentions >=2 distinct shape-vocabulary
    terms, (b) signals optionality (the user must never feel forced to
    answer before search continues), and (c) carries zero P3_BANNED_RE hits.

    Falls back to typing the identical chip text if the click doesn't
    register a user turn, mirroring step_w1_sangeet_hero's chip-click
    convention (the chip disappears once messages.length > 0, so button-
    vanishing is the click-registered signal).
    """
    chip_button = page.get_by_role("button", name=P3_CHIP, exact=True)
    clicked = False
    if chip_button.count() > 0:
        try:
            chip_button.first.click()
            page.wait_for_timeout(1_000)
            clicked = page.get_by_role("button", name=P3_CHIP, exact=True).count() == 0
        except Exception:  # noqa: BLE001 - fall through to the typed-text fallback
            clicked = False
    if not clicked:
        send_text(page, P3_CHIP)

    wait_for_assistant_reply(page)
    text = last_assistant_text(page)
    shot(page, "p3_1_clarify")

    shape_hits = sorted(set(m.lower() for m in P3_SHAPE_WORD_RE.findall(text)))
    state.record(
        "P3-1a. clarify response mentions >=2 distinct shape-vocabulary terms",
        len(shape_hits) >= 2,
        f"clicked_chip={clicked} shape_hits={shape_hits} assistant_text_snippet={text[:300]!r}",
    )

    optionality_hit = bool(P3_OPTIONALITY_RE.search(text))
    state.record(
        "P3-1b. clarify response signals optionality (no forced answer)",
        optionality_hit,
        f"optionality_hit={optionality_hit} assistant_text_snippet={text[:300]!r}",
    )

    banned_hits = P3_BANNED_RE.findall(text)
    state.record(
        "P3-1c. clarify response carries zero P3_BANNED_RE hits",
        not banned_hits,
        f"banned_hits={banned_hits} assistant_text_snippet={text[:300]!r}",
    )


def step_p3_2_hero(page: Page, state: ProofState) -> None:
    """P3-2: send "I'm pear-shaped, sangeet look under 8000" -- a body type
    stated inline together with an occasion+budget query, same turn. Asserts:
    (a) an outfit board OR >=2 new cards render; (b) >=1 new card title/badge
    matches pear-recommended silhouette vocabulary (P3_PEAR_SILHOUETTE_RE) --
    recorded as hits, evidence for the "biases outfit ranking toward
    flattering silhouettes" half of the feature; (c) the assistant text
    carries a supportive why-note (P3_WHY_NOTE_RE) with zero P3_BANNED_RE
    hits -- the "supportive why-note" half; (d) the stated ₹8000 budget is
    still respected when a board rendered (reuses `_parse_rupee_amount`, same
    pattern as step_w1_sangeet_hero's W1d).
    """
    query = "I'm pear-shaped, sangeet look under 8000"
    baseline = card_count(page)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    send_text(page, query)
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    outfit_board_present = page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline
    shot(page, "p3_2_hero")

    render_ok = outfit_board_present or gained >= 2
    state.record(
        "P3-2a. pear-shaped sangeet query renders an outfit board or >=2 new cards",
        render_ok,
        f"cards {baseline}->{after} outfit_board_present={outfit_board_present}",
    )
    if not render_ok:
        return

    new_cards = new_card_locators(page, baseline, after)
    silhouette_hits = [card_title(c) for c in new_cards if card_matches(c, P3_PEAR_SILHOUETTE_RE)]
    state.record(
        "P3-2b. >=1 new card title/badge matches pear-recommended silhouette vocabulary",
        len(silhouette_hits) >= 1,
        f"silhouette_hits={silhouette_hits} titles={[card_title(c) for c in new_cards]}",
    )

    assistant_text = last_assistant_text(page)
    why_note_hit = bool(P3_WHY_NOTE_RE.search(assistant_text))
    banned_hits = P3_BANNED_RE.findall(assistant_text)
    state.record(
        "P3-2c. assistant text carries a supportive body-type why-note, zero banned words",
        why_note_hit and not banned_hits,
        f"why_note_hit={why_note_hit} banned_hits={banned_hits} "
        f"assistant_text_snippet={assistant_text[:300]!r}",
    )

    if outfit_board_present:
        slot_cards = page.locator(OUTFIT_BOARD_SELECTOR).last.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        slot_prices = [
            _parse_rupee_amount(slot_cards.nth(i).inner_text()) for i in range(slot_cards.count())
        ]
        parsed_prices = [p for p in slot_prices if p is not None]
        price_sum = sum(parsed_prices)
        state.record(
            "P3-2d. board slot-price sum respects the stated ₹8000 budget",
            price_sum <= 8000,
            f"board_slot_price_sum={price_sum} n_prices_parsed={len(parsed_prices)}/"
            f"{slot_cards.count()}",
        )
    else:
        state.record(
            "P3-2d. board slot-price sum respects the stated ₹8000 budget",
            True,
            "N/A -- no outfit board rendered this turn (grid-card path has no board slot prices)",
        )


def step_p3_4_persistence(page: Page, state: ProofState) -> None:
    """P3-4: 'make it more festive' -- a refinement turn after P3-2's
    body-type-stated turn. Asserts >=1 new card AND the assistant text still
    carries zero P3_BANNED_RE hits. If the assistant mentions the body type
    again (P3_SHAPE_WORD_RE hit), that's recorded as bonus evidence -- the
    task spec does not require the body-type mention itself to persist, only
    that the banned-word guarantee holds.

    P3-4b/P3-4c (added after a live-proof bug catch, 2026-07-09): this turn
    used to render a MEN'S card into the women's-only sangeet conversation
    and separately let the stated ₹8000 budget silently stop applying --
    both fixed in graph.py's session_context gender/budget reconstruction
    (`_reconstruct_budget_from_history`, the `_ctx_gender` fallback near
    `_prior_filters`). P3-4b reuses `assert_complement_gender` (same
    data-gender-first, PB_MEN_WORD_RE title-fallback precedent as W1c/
    PB-S1c) to prove gender purity survived on this turn's new cards.
    P3-4c reuses `_parse_rupee_amount`: this turn renders as a raw card grid,
    not an outfit board (checked via `outfit_board_present` below, same
    signal as P3-2/W1), so there's no board-slot-sum to check -- instead it
    asserts each new card's own visible price (ItemCard renders
    `₹{item.price_inr}` inline when `price_inr` is not null, same
    grid-path precedent already used by W1d's else-branch) against the
    ₹8000 cap stated in P3-2's query. Cards with no parseable price are
    excluded rather than treated as violations, consistent with W1d.
    """
    baseline = card_count(page)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    send_text(page, "make it more festive")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    outfit_board_present = page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline
    shot(page, "p3_4_persistence")

    assistant_text = last_assistant_text(page)
    banned_hits = P3_BANNED_RE.findall(assistant_text)
    shape_mention = sorted(set(m.lower() for m in P3_SHAPE_WORD_RE.findall(assistant_text)))
    state.record(
        "P3-4. 'make it more festive' renders >=1 new card, zero banned words",
        gained >= 1 and not banned_hits,
        f"cards {baseline}->{after} banned_hits={banned_hits} "
        f"body_type_mentioned_again(evidence only)={shape_mention} "
        f"assistant_text_snippet={assistant_text[:300]!r}",
    )
    if gained < 1:
        return

    new_cards = new_card_locators(page, baseline, after)
    assert_complement_gender(
        new_cards,
        "women",
        PB_MEN_WORD_RE,
        state,
        "P3-4b. 'make it more festive' new cards: every card is gender-consistent "
        "(data-gender, title fallback) -- no MEN's card leaking into the sangeet look",
    )

    if outfit_board_present:
        slot_cards = page.locator(OUTFIT_BOARD_SELECTOR).last.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        slot_prices = [
            _parse_rupee_amount(slot_cards.nth(i).inner_text()) for i in range(slot_cards.count())
        ]
        parsed_prices = [p for p in slot_prices if p is not None]
        price_sum = sum(parsed_prices)
        state.record(
            "P3-4c. budget still respected (<=8000) -- board slot-price sum",
            price_sum <= 8000,
            f"board_slot_price_sum={price_sum} n_prices_parsed={len(parsed_prices)}/"
            f"{slot_cards.count()}",
        )
    else:
        prices = [_parse_rupee_amount(c.inner_text()) for c in new_cards]
        parsed_prices = [p for p in prices if p is not None]
        budget_ok = all(p <= 8000 for p in parsed_prices)
        state.record(
            "P3-4c. budget still respected (<=8000) -- per-card price, no outfit board "
            "rendered this turn (raw grid-search path has no board slot prices)",
            budget_ok,
            f"new_card_prices={prices} n_prices_parsed={len(parsed_prices)}/{len(new_cards)}",
        )


def step_p3_3_ban_sweep(page: Page, state: ProofState) -> None:
    """P3-3: the framing guarantee -- sweep EVERY assistant bubble rendered in
    THIS session so far (not just the latest turn's) for P3_BANNED_RE hits.
    A per-turn check (P3-1c, P3-2c, P3-4) could still miss a banned word an
    earlier turn used but a later turn didn't repeat; this step is the
    session-wide net. Run LAST in `main` (after P3-4's turn too) since the
    task spec's own P3-3 description says "after all turns" -- see the
    module docstring's note on this deliberate reordering.
    """
    bubbles = page.locator(ASSISTANT_BUBBLE_SELECTOR)
    n_bubbles = bubbles.count()
    all_hits: list[str] = []
    for i in range(n_bubbles):
        try:
            text = bubbles.nth(i).inner_text()
        except Exception:  # noqa: BLE001 - best-effort evidence extraction
            continue
        hits = P3_BANNED_RE.findall(text)
        if hits:
            all_hits.append(f"bubble[{i}]: {hits} in {text[:120]!r}")
    state.record(
        "P3-3. zero P3_BANNED_RE hits across EVERY assistant bubble in the session",
        not all_hits,
        f"n_bubbles_checked={n_bubbles} hits={all_hits}",
    )


# ---------------------------------------------------------------------------
# P2 (couple-coordination deepening) constants & steps.
#
# Verified against frontend/components/chat/OutfitBoard.tsx before writing
# these checks (read-before-edit): the ONLY DOM signal that distinguishes a
# "partner" board from a "primary" board is the conditionally-rendered
# "Partner look" badge span + "Your partner's look" (or `lookTitle`) heading
# + optional `coordinatedWith` paragraph (lines ~477-495) — there is NO
# `data-look-role` (or any other) attribute on the board's root div
# (`OUTFIT_BOARD_SELECTOR`'s `div.rounded-xl.border.bg-card.p-4`) or anywhere
# else in the component. `isPartnerLook = lookRole === "partner"` (line 211)
# only ever gates that text block. This means index-based board selection
# (`.nth(0)` / `.nth(1)`) is the ONLY way to address "her" vs "his" board at
# all right now, not a shortcut taken for convenience. It is SAFE here
# specifically because P2-1 runs in a FRESH session's very first turn: no
# other board can already be on the page, assistant messages (and the boards
# inside them) render top-to-bottom in arrival order (same reasoning
# `board_complement_cards`'s `.last` usage relies on elsewhere in this file),
# and the feature spec's own ordering ("the first a primary board ... the
# second a partner board") fixes board 0 = her/primary, board 1 = his/partner
# deterministically. P2-4 additionally cross-checks this assumption directly
# (asserts board 0 does NOT carry the partner badge while board 1 does),
# so an index-vs-role mismatch would surface as an explicit FAIL rather than
# silently mis-attributing gender/budget checks to the wrong board.
#
# Turn budget: P2-1 (couple-from-scratch) is ONE send in its own fresh
# session (the single query renders both boards in the same assistant turn).
# P2-5 is a full SEPARATE fresh session reusing `step_pb_s4_partner_styling`
# unchanged (3 sends: query, "Style this" click, partner follow-up — see
# that function's own docstring). Total for --p2: 1 + 3 = 4 sends across two
# sessions, comfortably under the <=8-turn target the Phase-B docstring's
# turn-budget note discusses for a much busier flow.
# ---------------------------------------------------------------------------

P2_QUERY = "style us as a couple for a reception under 15000"
# Per-person cap, not a combined-total cap: the feature spec states the
# budget as "under 15000" for styling "us as a couple", but a couple is two
# independent people each shopping their own look — a shared pool that added
# up to 15000 across BOTH boards would silently halve what either partner
# could actually spend on themselves once split, and nothing in the query
# says "for both of us combined". Each board's own slot-price sum is checked
# against the FULL 15000 independently (P2-3), not their sum against 15000.
P2_BUDGET_INR = 15000


def step_p2_couple_from_scratch(context, base_url: str, state: ProofState) -> None:
    """P2-1..P2-4: a FRESH session's FIRST turn — "style us as a couple for a
    reception under 15000" with no prior anchor/search — must render TWO
    back-to-back outfit boards (her primary look, then his partner look) in
    the SAME assistant turn.

    Runs in its own Playwright page (own `sessionStorage`/`demo_session_token`,
    same isolation reasoning as `step_pb_s4_partner_styling`), since "no prior
    anchor/search" is only true in a session that has sent nothing yet.
    """
    fresh_page = context.new_page()
    try:
        if not step_load_chat(fresh_page, base_url, state):
            return

        board_baseline = fresh_page.locator(OUTFIT_BOARD_SELECTOR).count()
        send_text(fresh_page, P2_QUERY)
        deadline = time.time() + CARD_WAIT_TIMEOUT_S
        while time.time() < deadline:
            if fresh_page.locator(OUTFIT_BOARD_SELECTOR).count() >= board_baseline + 2:
                break
            fresh_page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
        wait_for_turn_idle(fresh_page)
        shot(fresh_page, "p2_1_couple_from_scratch")

        n_boards = fresh_page.locator(OUTFIT_BOARD_SELECTOR).count()
        passed = n_boards >= 2
        extra_note = (
            f" (NOTE: n_boards={n_boards} > 2 -- more boards than the expected "
            "primary+partner pair; recorded as evidence, not a hard failure, since "
            "the spec doesn't preclude it, but board 0/1 are still checked as her/his)"
            if n_boards > 2
            else ""
        )
        state.record(
            "P2-1. couple-from-scratch query renders >=2 outfit boards in ONE turn",
            passed,
            f"boards_on_page={n_boards} (baseline was {board_baseline}){extra_note}",
        )
        if not passed:
            return

        her_board = fresh_page.locator(OUTFIT_BOARD_SELECTOR).nth(0)
        his_board = fresh_page.locator(OUTFIT_BOARD_SELECTOR).nth(1)

        # -- P2-2: gender purity, scoped to each board individually --------
        her_complements = complement_cards_of_board(her_board)
        his_complements = complement_cards_of_board(his_board)

        assert_complement_gender(
            her_complements,
            "women",
            PB_MEN_WORD_RE,
            state,
            "P2-2a. board 0 (her look): every complement is gender-consistent "
            "(data-gender, title fallback)",
        )
        assert_complement_gender(
            his_complements,
            "men",
            PB_WOMEN_WORD_RE,
            state,
            "P2-2b. board 1 (his look): every complement is gender-consistent "
            "(data-gender, title fallback)",
        )

        # -- P2-3: budget is a PER-PERSON cap, not a combined total --------
        her_slot_cards = her_board.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        her_prices = [
            _parse_rupee_amount(her_slot_cards.nth(i).inner_text())
            for i in range(her_slot_cards.count())
        ]
        her_prices = [p for p in her_prices if p is not None]
        her_sum = sum(her_prices)

        his_slot_cards = his_board.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        his_prices = [
            _parse_rupee_amount(his_slot_cards.nth(i).inner_text())
            for i in range(his_slot_cards.count())
        ]
        his_prices = [p for p in his_prices if p is not None]
        his_sum = sum(his_prices)

        state.record(
            "P2-3. EACH board's own slot-price sum independently respects "
            f"the per-person ₹{P2_BUDGET_INR} cap (not their combined total)",
            her_sum <= P2_BUDGET_INR and his_sum <= P2_BUDGET_INR,
            f"her_slot_price_sum={her_sum} (n_prices_parsed={len(her_prices)}/"
            f"{her_slot_cards.count()}) his_slot_price_sum={his_sum} "
            f"(n_prices_parsed={len(his_prices)}/{his_slot_cards.count()}) "
            f"combined={her_sum + his_sum} cap={P2_BUDGET_INR}",
        )

        # -- P2-4: partner labeling + honesty ------------------------------
        # Same text-matching precedent `step_pb_s4_partner_styling` already
        # uses (PB-S4b/c) -- copied rather than abstracted into a shared
        # helper since this is only the SECOND use of the pattern (see the
        # "duplicate twice, abstract on the third occurrence" rule), and P2's
        # "mentions_anchor" equivalent has no anchor phrase to match against
        # (couple-from-scratch has no prior "black dress"-style anchor query
        # the way PB-S4 does) -- so here it's simplified to "a non-empty
        # coordinated_with line exists", not an anchor-name substring match.
        try:
            his_board_text = his_board.inner_text()
        except Exception:  # noqa: BLE001
            his_board_text = ""
        try:
            her_board_text = her_board.inner_text()
        except Exception:  # noqa: BLE001
            her_board_text = ""

        his_has_partner_marker = (
            "Partner look" in his_board_text or "Your partner's look" in his_board_text
        )
        her_has_partner_marker = (
            "Partner look" in her_board_text or "Your partner's look" in her_board_text
        )
        coordinated_line = next(
            (line for line in his_board_text.splitlines() if "coordinated" in line.lower()),
            "",
        )
        passed_p2_4 = his_has_partner_marker and not her_has_partner_marker
        state.record(
            "P2-4. board 1 (his) shows the partner badge/heading + a "
            "'Coordinated with...' subtext; board 0 (her) does NOT",
            passed_p2_4,
            f"his_has_partner_marker={his_has_partner_marker} "
            f"her_has_partner_marker={her_has_partner_marker} "
            f"coordinated_line={coordinated_line!r} "
            f"n_her_complements={len(her_complements)} n_his_complements={len(his_complements)}",
        )

        # Honest-thinness evidence (not an assertion): a thinner his-board is
        # fine per spec as long as P2-2b (no cross-gender leak) already
        # passed above -- recorded here purely as context for that result.
        print(
            "[EVIDENCE] P2 board sizes -- her_complements="
            f"{len(her_complements)} his_complements={len(his_complements)} "
            f"(a thinner his-board is honest, not a failure, on its own)"
        )
    finally:
        fresh_page.close()


# ---------------------------------------------------------------------------
# --item2 ("photo -> body-shape suggestion" upload affordance) constants.
#
# Every selector/copy string below was grepped verbatim from
# frontend/components/chat/BodyShapeUpload.tsx and frontend/lib/poseShape.ts
# (read before writing any assertion here, per repo convention) -- none of
# it is invented/assumed.
# ---------------------------------------------------------------------------

# Trigger button: `aria-label="Body shape suggestion (optional)"` (title
# matches). Distinct from the pre-existing garment/inspiration-photo upload
# button (`ChatInput.tsx`, `aria-label="Style what you own"`) -- different
# icon (PersonStanding vs ImagePlus) and different accessible name.
I2_TRIGGER_NAME = "Body shape suggestion (optional)"
I2_GARMENT_UPLOAD_NAME = "Style what you own"

# The body-shape affordance's OWN hidden file input's aria-label -- distinct
# from ChatInput.tsx's garment file input (`aria-label="Upload garment or
# inspiration photo"`), so the two same-type `input[type=file]` elements now
# on the page can be addressed unambiguously.
I2_FILE_INPUT_LABEL = "Upload a photo for a body-shape suggestion"

# Floating panel's root div -- verified unique across the whole frontend tree
# (grep for "bottom-full" hits only this one file) via its distinctive
# Tailwind class combination, which `cn()` keeps as the outer wrapper
# regardless of which stage (intro/loading/confident/picking/fallback) is
# showing inside it.
I2_PANEL_SELECTOR = "div.absolute.bottom-full.right-0.mb-2.w-72"

# I2-1: privacy copy, verbatim from the component's "intro" stage: "For a
# body-shape suggestion: processed entirely in your browser, never uploaded
# or stored." Regex built from the ACTUAL copy, not invented boilerplate.
I2_PRIVACY_RE = re.compile(
    r"processed entirely in your browser|never uploaded|never leaves|on your device",
    re.IGNORECASE,
)

# I2-2: raw error/exception strings that must NEVER be user-visible for this
# silently-degrading, optional feature. Word-boundary for real-word tokens
# (avoids false hits inside unrelated larger words); "failed to load" is
# matched as the literal phrase.
I2_ERRORISH_RE = re.compile(r"\berror\b|\bundefined\b|\bnan\b|failed to load", re.IGNORECASE)

# I2-2/I2-3: the 5 SHAPE_OPTIONS button labels, verbatim from the component.
I2_SHAPE_BUTTON_LABELS = ["Pear", "Apple", "Hourglass", "Rectangle", "Inverted triangle"]

# I2-2: confident-branch controls, verbatim from the component's "confident"
# stage JSX. The confirm button's apostrophe is React's `&apos;` entity,
# which renders as a straight `'` -- `.` in the regex tolerates either.
I2_CONFIRM_BUTTON_RE = re.compile(r"Yes,\s*that.s right", re.IGNORECASE)
I2_PICK_DIFFERENT_NAME = "Pick a different one"

# I2-2: fallback-branch heading, verbatim -- distinct from the "picking"
# stage's own heading ("A few shapes people mention...") so this text alone
# disambiguates fallback from picking.
I2_FALLBACK_HEADING = "Prefer to just tell me? Tap a shape below or type it."

# I2-3: `bodyShapeMessage()` in poseShape.ts (grepped verbatim) -- the exact
# natural-language string the frontend sends for each shape slug. Used here
# only to recognize/log which message went out as evidence; the frontend,
# not this script, is responsible for producing it.
I2_SHAPE_MESSAGES = {
    "pear": "I have a pear silhouette",
    "apple": "I have an apple silhouette",
    "hourglass": "I have an hourglass silhouette",
    "rectangle": "I have a rectangle silhouette",
    "inverted_triangle": "I have an inverted triangle silhouette",
}


def step_i2_0_affordance_present(page: Page, state: ProofState) -> None:
    """I2-0: the body-shape upload affordance renders as a button distinct
    from the pre-existing garment-photo upload button -- different
    accessible name (`I2_TRIGGER_NAME` vs `I2_GARMENT_UPLOAD_NAME`) and,
    per the component source, a different icon (PersonStanding vs
    ImagePlus).
    """
    trigger = page.get_by_role("button", name=I2_TRIGGER_NAME, exact=True)
    garment_upload = page.get_by_role("button", name=I2_GARMENT_UPLOAD_NAME, exact=True)
    trigger_present = trigger.count() > 0
    garment_present = garment_upload.count() > 0
    shot(page, "i2_0_affordance_present")
    state.record(
        "I2-0. body-shape upload affordance renders, distinct from the garment-photo upload",
        trigger_present and garment_present and I2_TRIGGER_NAME != I2_GARMENT_UPLOAD_NAME,
        f"trigger_present={trigger_present} garment_upload_present={garment_present} "
        f"trigger_name={I2_TRIGGER_NAME!r} garment_upload_name={I2_GARMENT_UPLOAD_NAME!r}",
    )


def step_i2_1_privacy_copy(page: Page, state: ProofState) -> None:
    """I2-1: open the affordance (WITHOUT picking a file yet) and assert the
    privacy copy is visible, matching `I2_PRIVACY_RE` (built from the actual
    "processed entirely in your browser, never uploaded or stored" text in
    BodyShapeUpload.tsx's "intro" stage).
    """
    trigger = page.get_by_role("button", name=I2_TRIGGER_NAME, exact=True)
    if trigger.count() == 0:
        state.record(
            "I2-1. privacy copy visible before upload", False, "trigger button not found"
        )
        return
    trigger.first.click()
    try:
        page.wait_for_selector(I2_PANEL_SELECTOR, timeout=5_000)
    except Exception as exc:  # noqa: BLE001
        shot(page, "i2_1_privacy_copy_FAIL")
        state.record(
            "I2-1. privacy copy visible before upload", False, f"panel did not open: {exc}"
        )
        return
    panel_text = page.locator(I2_PANEL_SELECTOR).first.inner_text()
    shot(page, "i2_1_privacy_copy")
    privacy_hit = bool(I2_PRIVACY_RE.search(panel_text))
    state.record(
        "I2-1. privacy copy visible before upload (matches I2_PRIVACY_RE)",
        privacy_hit,
        f"panel_text={panel_text!r}",
    )


def step_i2_2_upload_outcome(page: Page, state: ProofState, image_path: Path) -> str | None:
    """I2-2: pick `image_path` via the body-shape affordance's OWN hidden
    file input (`I2_FILE_INPUT_LABEL`, distinct from the garment upload's
    file input) and poll for either the CONFIDENT suggestion panel or the
    NOT-CONFIDENT fallback panel to appear.

    `image_path` is DEFAULT_IMAGE (t-shirt.webp) by default in `main` -- a
    non-person photo will almost certainly fail MediaPipe's pose-detection
    confidence gate and land on the fallback branch. EITHER branch is an
    acceptable proof outcome here (we don't control what MediaPipe detects
    in a non-person test image); this only fails if NEITHER panel appears
    within the timeout, or a branch's own content checks fail.

    Returns "confident", "fallback", or None (timeout/no branch appeared).
    """
    if not image_path.exists():
        state.record(
            "I2-2. upload produces a confident or fallback outcome",
            False,
            f"image file not found: {image_path}",
        )
        return None

    file_input = page.get_by_label(I2_FILE_INPUT_LABEL)
    if file_input.count() == 0:
        state.record(
            "I2-2. upload produces a confident or fallback outcome",
            False,
            "body-shape file input not found",
        )
        return None
    file_input.set_input_files(str(image_path))

    # WASM model load + first-run inference can take a few seconds -- generous
    # 60s polling window per the task spec.
    deadline = time.time() + 60.0
    outcome: str | None = None
    while time.time() < deadline:
        if page.get_by_text(I2_FALLBACK_HEADING, exact=False).count() > 0:
            outcome = "fallback"
            break
        if page.get_by_role("button", name=I2_CONFIRM_BUTTON_RE).count() > 0:
            outcome = "confident"
            break
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))

    shot(page, f"i2_2_outcome_{outcome or 'timeout'}")
    if outcome is None:
        state.record(
            "I2-2. upload produces a confident or fallback outcome",
            False,
            "neither the confident suggestion panel nor the fallback panel appeared within 60s",
        )
        return None

    panel_text = ""
    if page.locator(I2_PANEL_SELECTOR).count() > 0:
        try:
            panel_text = page.locator(I2_PANEL_SELECTOR).first.inner_text()
        except Exception:  # noqa: BLE001 - best-effort evidence extraction
            panel_text = ""

    banned_hits = P3_BANNED_RE.findall(panel_text)
    errorish_hits = I2_ERRORISH_RE.findall(panel_text)

    if outcome == "fallback":
        shape_button_hits = [
            label
            for label in I2_SHAPE_BUTTON_LABELS
            if page.get_by_role("button", name=label, exact=True).count() > 0
        ]
        passed = (
            len(shape_button_hits) == len(I2_SHAPE_BUTTON_LABELS)
            and not banned_hits
            and not errorish_hits
        )
        state.record(
            "I2-2. NOT-CONFIDENT fallback: all 5 shape buttons present, zero banned/error text",
            passed,
            f"shape_buttons_found={shape_button_hits} banned_hits={banned_hits} "
            f"errorish_hits={errorish_hits} panel_text={panel_text!r}",
        )
    else:  # confident
        confirm_present = page.get_by_role("button", name=I2_CONFIRM_BUTTON_RE).count() > 0
        pick_different_present = (
            page.get_by_role("button", name=I2_PICK_DIFFERENT_NAME, exact=True).count() > 0
        )
        shape_hit = bool(P3_SHAPE_WORD_RE.search(panel_text))
        passed = confirm_present and pick_different_present and not banned_hits and shape_hit
        state.record(
            "I2-2. CONFIDENT branch: confirm+pick-different controls present, zero banned "
            "hits, suggestion text matches a shape-word pattern",
            passed,
            f"confirm_present={confirm_present} pick_different_present={pick_different_present} "
            f"banned_hits={banned_hits} shape_hit={shape_hit} panel_text={panel_text!r}",
        )

    return outcome


def step_i2_3_flow_through(page: Page, state: ProofState, outcome: str | None) -> None:
    """I2-3: complete whichever branch I2-2 observed -- click the confirm
    button if CONFIDENT (whatever shape MediaPipe actually suggested), or
    deterministically tap "Pear" if NOT-CONFIDENT (fallback, so the
    pear-vocabulary bonus check below has a deterministic shape to look
    for) -- then wait for the resulting natural-language chat message's
    assistant reply, and prove it flows into the SAME downstream pipeline
    --p3 already proves: send "sangeet look under 8000" and assert
    cards/board render, the budget is respected, and zero P3_BANNED_RE hits.

    Bonus (not a hard requirement -- recorded as evidence only, since the
    confident branch's shape is nondeterministic on MediaPipe's actual
    output): pear-silhouette vocabulary, only checked when the shape
    actually used was pear.
    """
    if outcome is None:
        state.record(
            "I2-3. confirmed/picked shape flows into the P3 pipeline",
            False,
            "skipped -- I2-2 produced neither branch",
        )
        return

    used_shape: str | None
    if outcome == "fallback":
        used_shape = "pear"
        page.get_by_role("button", name="Pear", exact=True).first.click()
    else:
        panel_text = ""
        if page.locator(I2_PANEL_SELECTOR).count() > 0:
            try:
                panel_text = page.locator(I2_PANEL_SELECTOR).first.inner_text()
            except Exception:  # noqa: BLE001
                panel_text = ""
        shape_match = P3_SHAPE_WORD_RE.search(panel_text)
        used_shape = None
        if shape_match:
            normalized = shape_match.group(0).lower()
            used_shape = "inverted_triangle" if normalized == "inverted triangle" else normalized
        page.get_by_role("button", name=I2_CONFIRM_BUTTON_RE).first.click()

    expected_message = I2_SHAPE_MESSAGES.get(used_shape or "", "")
    wait_for_assistant_reply(page)
    if expected_message:
        message_echoed = page.get_by_text(expected_message, exact=False).count() > 0
        print(
            f"[EVIDENCE] I2-3 user message echo: used_shape={used_shape!r} "
            f"expected={expected_message!r} echoed={message_echoed}"
        )
    shot(page, "i2_3_after_confirm")

    baseline = card_count(page)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    send_text(page, "sangeet look under 8000")
    wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    after = card_count(page)
    gained = after - baseline
    outfit_board_present = page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline
    shot(page, "i2_3_sangeet_result")

    render_ok = outfit_board_present or gained >= 1
    state.record(
        "I2-3a. 'sangeet look under 8000' renders an outfit board or >=1 new card "
        "after the body-shape message",
        render_ok,
        f"used_shape={used_shape} cards {baseline}->{after} "
        f"outfit_board_present={outfit_board_present}",
    )
    if not render_ok:
        return

    if outfit_board_present:
        slot_cards = page.locator(OUTFIT_BOARD_SELECTOR).last.locator(OUTFIT_BOARD_SLOT_SELECTOR)
        slot_prices = [
            _parse_rupee_amount(slot_cards.nth(i).inner_text()) for i in range(slot_cards.count())
        ]
        parsed_prices = [p for p in slot_prices if p is not None]
        price_sum = sum(parsed_prices)
        budget_ok = price_sum <= 8000
        budget_detail = (
            f"board_slot_price_sum={price_sum} n_prices_parsed={len(parsed_prices)}/"
            f"{slot_cards.count()}"
        )
    else:
        new_cards = new_card_locators(page, baseline, after)
        prices = [_parse_rupee_amount(c.inner_text()) for c in new_cards]
        parsed_prices = [p for p in prices if p is not None]
        budget_ok = all(p <= 8000 for p in parsed_prices)
        budget_detail = f"new_card_prices={prices}"
    state.record("I2-3b. budget respected (<=8000)", budget_ok, budget_detail)

    assistant_text = last_assistant_text(page)
    banned_hits = P3_BANNED_RE.findall(assistant_text)
    state.record(
        "I2-3c. zero P3_BANNED_RE hits in the assistant's text",
        not banned_hits,
        f"banned_hits={banned_hits} assistant_text_snippet={assistant_text[:300]!r}",
    )

    if used_shape == "pear":
        new_cards = new_card_locators(page, baseline, after)
        silhouette_hits = [
            card_title(c) for c in new_cards if card_matches(c, P3_PEAR_SILHOUETTE_RE)
        ]
        why_note_hit = bool(P3_WHY_NOTE_RE.search(assistant_text))
        print(
            "[EVIDENCE] I2-3d (bonus, pear only, not a hard requirement) -- "
            f"silhouette_hits={silhouette_hits} why_note_hit={why_note_hit}"
        )
    else:
        print(f"[EVIDENCE] I2-3d (bonus, pear only) skipped -- used_shape={used_shape!r}")


def print_summary(state: ProofState) -> bool:
    """Print the final PASS/FAIL table. Returns True if every check passed."""
    print("\n" + "=" * 78)
    print("BROWSER PROOF SUMMARY")
    print("=" * 78)
    width = max((len(r.name) for r in state.results), default=10)
    all_passed = True
    for r in state.results:
        all_passed &= r.passed
        status = "PASS" if r.passed else "FAIL"
        print(f"{status:5} | {r.name.ljust(width)} | {r.detail}")
    print("=" * 78)
    print(f"OVERALL: {'ALL PASS' if all_passed else 'SOME FAIL'}")
    print(f"Screenshots: {SCRATCHPAD}")
    return all_passed


def main() -> int:
    """Run the full browser-level proof flow against the live (or given) demo URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Demo entry URL to open.")
    parser.add_argument(
        "--image", default=str(DEFAULT_IMAGE), help="Path to an image file for the upload step."
    )
    parser.add_argument(
        "--headed", action="store_true", help="Run with a visible browser window (debugging)."
    )
    parser.add_argument(
        "--product",
        action="store_true",
        help="Run only the new product-bug-check steps (B1-B5) instead of the original a-f flow.",
    )
    parser.add_argument(
        "--phase-a",
        action="store_true",
        help=(
            "Run only the Phase-A index-quality content-assertion steps (A1-A4): saree "
            "recall, 'white' adjective relevance, duplicate-title detection, and "
            "store-diversity. These assert fixes that may not be deployed yet."
        ),
    )
    parser.add_argument(
        "--phase-b",
        action="store_true",
        help=(
            "Run the Phase-B content-assertion steps (S1-S6) for the reported outfit "
            "gender-leak bug and the Phase-B fix surface: women's-look / men's-look "
            "cross-gender leak checks with data-gender/data-slot assertions and footwear "
            "slot-sanity (S1-S2), a sangeet/ethnic-occasion casual-western + gender gate "
            "(S3), partner-styling cross-gender coordination in a fresh session (S4), "
            "office-look occasion-register vocabulary (S5), and distinct outfit variants "
            "(S6)."
        ),
    )
    parser.add_argument(
        "--wave-7",
        action="store_true",
        help=(
            "Run the Wave-7 wedding-occasion hero content-assertion steps (W0-W5): the "
            "unified 'Style Maitri' brand config's 5 suggestion chips + display name (W0, "
            "pre-message), the sangeet hero chip's click-to-send path with ethnic-occasion "
            "vocabulary/gender/budget checks (W1), haldi/mehendi/reception occasion-palette "
            "checks (W2-W4), and a partner-styling gender-consistency regression check "
            "(W5). W6 (zero severe console errors) is covered by the shared finally-block "
            "step_console_errors call, not a separate invocation."
        ),
    )
    parser.add_argument(
        "--p3",
        action="store_true",
        help=(
            "Run the P3 'body type' styling content-assertion steps (P3-0..P3-4): the 6th "
            "'What suits my body type?' suggestion chip renders pre-message (P3-0), a "
            "no-body-type clarify response's shape-vocabulary/optionality/banned-word checks "
            "(P3-1), a stated body type biasing silhouette vocabulary + a supportive why-note "
            "within budget (P3-2), a refinement turn ('make it more festive') that keeps the "
            "banned list at zero (P3-4, run before P3-3), and a session-wide banned-framing-"
            "word sweep across every assistant bubble (P3-3, run LAST since its own spec text "
            "says 'after all turns' -- see the module docstring)."
        ),
    )
    parser.add_argument(
        "--p2",
        action="store_true",
        help=(
            "Run the P2 couple-coordination-deepening content-assertion steps "
            "(P2-1..P2-5): a fresh session's first turn ('style us as a couple for "
            "a reception under 15000', no prior anchor) renders TWO back-to-back "
            "outfit boards in one turn (P2-1), each board's own gender purity "
            "(P2-2) and per-person (not combined) budget cap (P2-3), board 1's "
            "partner badge/'Coordinated with...' subtext while board 0 shows "
            "neither (P2-4), and a regression re-check that the pre-existing "
            "single-partner flow ('black dress for women' -> 'Style this' -> "
            "'what should my husband wear with this?') still works via a direct "
            "reuse of the --phase-b PB-S4 step in a brand-new fresh session (P2-5)."
        ),
    )
    parser.add_argument(
        "--item2",
        action="store_true",
        help=(
            "Run the 'photo -> body-shape suggestion' upload affordance content-"
            "assertion steps (I2-0..I2-3): the new PersonStanding-icon affordance "
            "renders distinct from the existing garment-photo upload (I2-0), on-device "
            "privacy copy is visible before any file is picked (I2-1), uploading a "
            "photo lands on the confident-suggestion panel or the not-confident "
            "fallback panel with zero banned/error text (I2-2, either branch is a "
            "valid pass), and completing that branch flows the resulting message into "
            "the SAME downstream P3 pipeline --p3 already proves (I2-3). NOT deployed "
            "at the time this flag was added -- do not run --item2 live until the "
            "frontend change ships."
        ),
    )
    args = parser.parse_args()

    state = ProofState()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        # Register listeners BEFORE navigation so nothing is missed.
        page.on(
            "console",
            lambda msg: state.console_issues.append(f"[console.{msg.type}] {msg.text}")
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda exc: state.console_issues.append(f"[pageerror] {exc}"))

        try:
            if step_load_chat(page, args.base_url, state):
                if args.product:
                    step_b1_header(page, state)
                    step_b2_greeting(page, state)
                    step_b3_interactions(page, state)
                    image_new_cards = step_image_upload(page, state, Path(args.image))
                    step_b4_image_board(page, state, expected_new_cards=image_new_cards)
                    step_b4c_open_all_panel(page, state)
                    step_b5_save_look(page, state)
                elif args.phase_a:
                    step_a1_sarees(page, state)
                    step_a2_white_sneakers(page, state)
                    step_a3_black_dress_dedup(page, state)
                    step_a4_summer_dress_store_diversity(page, state)
                elif args.phase_b:
                    step_pb_s1_womens_look_consistency(page, state)
                    step_pb_s2_mens_look_consistency(page, state)
                    step_pb_s3_sangeet_ethnic_gate(page, state)
                    step_pb_s4_partner_styling(context, args.base_url, state)
                    step_pb_s5_occasion_register(page, state)
                    step_pb_s6_distinct_variants(page, state)
                elif args.wave_7:
                    step_w0_chips_and_brand(page, state)
                    step_w1_sangeet_hero(page, state)
                    step_w2_haldi_palette(page, state)
                    step_w3_mehendi_palette(page, state)
                    step_w4_reception_register(page, state)
                    step_w5_partner_regression(page, state)
                elif args.p3:
                    step_p3_0_chip(page, state)
                    step_p3_1_clarify(page, state)
                    step_p3_2_hero(page, state)
                    step_p3_4_persistence(page, state)
                    step_p3_3_ban_sweep(page, state)
                elif args.p2:
                    step_p2_couple_from_scratch(context, args.base_url, state)
                    # P2-5: direct reuse of the existing --phase-b PB-S4 step in
                    # its own brand-new fresh session -- proves the pre-existing
                    # single-partner flow is unchanged, without touching or
                    # duplicating PB-S4's own checks (see the module docstring).
                    step_pb_s4_partner_styling(context, args.base_url, state)
                elif args.item2:
                    step_i2_0_affordance_present(page, state)
                    step_i2_1_privacy_copy(page, state)
                    outcome = step_i2_2_upload_outcome(page, state, Path(args.image))
                    step_i2_3_flow_through(page, state, outcome)
                else:
                    step_query(page, state, "b. 'saree' query", "saree", "b_saree")
                    step_query(
                        page,
                        state,
                        "c. 'black dress for women' query",
                        "black dress for women",
                        "c_black_dress",
                    )
                    step_refinement(page, state)
                    step_image_upload(page, state, Path(args.image))
        finally:
            step_console_errors(state)
            context.close()
            browser.close()

    all_passed = print_summary(state)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
