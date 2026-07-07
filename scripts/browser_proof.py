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
# tailored, not casual. "denim skirt" is deliberately checked as a phrase (a
# plain "skirt" is fine; "denim skirt" is not) per spec.
PB_S5_FORBIDDEN_BOTTOM_RE = re.compile(r"\b(shorts?|denim\s+skirt|joggers?)\b", re.IGNORECASE)
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


def board_complement_cards(page: Page) -> list:
    """Return locators for every COMPLEMENT slot card in the LATEST outfit board
    on the page (assistant turns render top-to-bottom, so `.last` is always the
    most recent board — same reasoning as B4/B5's docstrings).

    Complement identification is defensive per the task spec: prefer a
    `data-slot` attribute if a later frontend fix adds one (`data-slot != "seed"`
    would then mark a card as a complement); until then, fall back to the
    "Hero" text badge SlotCard already renders on the seed/hero card today
    (its only <span> child — see OutfitBoard.tsx), which is the one
    title-independent signal available now to exclude the hero from the
    cross-gender-leak check. The owned-anchor card (no buy link) is already
    excluded by OUTFIT_BOARD_SLOT_SELECTOR, which only matches `<a>` tiles.
    """
    board = page.locator(OUTFIT_BOARD_SELECTOR)
    if board.count() == 0:
        return []
    board0 = board.last
    slot_cards = board0.locator(OUTFIT_BOARD_SLOT_SELECTOR)
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
