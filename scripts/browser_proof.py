"""Browser-level proof for the Wave-1 retest requirements (GG, 2026-07-04).

Verifies, against a *real* headless Chromium session (not WS-frame introspection):
  1. Item cards actually RENDER in the DOM after a text query.
  2. No uncaught JS exceptions (React/Next/TypeError/hydration) fire while cards render.
  3. Refinement ("in blue now") re-renders cards.
  4. Image upload shows an image thumbnail + typed text in the user bubble, then
     either renders cards or surfaces the honest timeout/error message (not an
     infinite "Finding your match..." spinner).

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


def step_image_upload(page: Page, state: ProofState, image_path: Path) -> None:
    """Step e: upload an image + typed text; verify user-bubble echo, then wait for
    cards or the honest timeout/error message (fail on infinite spinner)."""
    if not image_path.exists():
        state.record("e. image upload", False, f"image file not found: {image_path}")
        return

    file_input = page.locator("input[type=file]")
    file_input.set_input_files(str(image_path))

    # Confirm the pending-image chip rendered in the composer before sending.
    try:
        page.wait_for_selector("img[alt='Upload preview']", timeout=5_000)
    except Exception as exc:  # noqa: BLE001
        state.record("e0. image picked into composer", False, str(exc))
        return
    state.record("e0. image picked into composer", True)

    typed_text = "buy similar under 2000"
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
    elif outcome == "honest_message":
        state.record(
            "e2. image search returns cards or honest message",
            False,
            f"no cards; showed honest error instead: {detail}",
        )
    else:
        state.record(
            "e2. image search returns cards or honest message",
            False,
            f"HANG: no cards and no honest message after {IMAGE_WAIT_TIMEOUT_S}s "
            "(spinner likely stuck on 'Finding your match...')",
        )


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
    baseline2 = card_count(page)
    try:
        first_card.get_by_text("Style this", exact=True).click()
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


def _parse_rupee_amount(text: str) -> int | None:
    """Extract the first '₹N,NNN'-style amount from `text` as an int, or None."""
    m = AMOUNT_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def step_b4_image_board(page: Page, state: ProofState) -> None:
    """B4: after image upload, the outfit board must show >=3 slot cards whose
    prices sum to the 'Open all items' / 'Add the look to cart' CTA amount."""
    board = page.locator(OUTFIT_BOARD_SELECTOR)
    if board.count() == 0:
        state.record(
            "B4a. image-look outfit board shows >=3 slot cards",
            False,
            "no outfit board rendered after image upload",
        )
        return

    board0 = board.first
    slot_cards = board0.locator("a.rounded-lg.border.bg-background")
    n_slots = slot_cards.count()
    shot(page, "b4_outfit_board")
    state.record(
        "B4a. image-look outfit board shows >=3 slot cards",
        n_slots >= 3,
        f"n_slots={n_slots}",
    )

    prices = [_parse_rupee_amount(slot_cards.nth(i).inner_text()) for i in range(n_slots)]
    prices = [p for p in prices if p is not None]
    price_sum = sum(prices)

    cta = board0.get_by_role("button", name=re.compile(r"Open all items|Add the look to cart"))
    if cta.count() == 0:
        state.record(
            "B4b. board CTA amount equals sum of card prices",
            False,
            f"no CTA button found (n_slots={n_slots} sum_of_card_prices={price_sum})",
        )
        return

    cta_text = cta.first.inner_text()
    cta_amount = _parse_rupee_amount(cta_text)
    state.record(
        "B4b. board CTA amount equals sum of card prices",
        cta_amount is not None and cta_amount == price_sum,
        f"cta_text={cta_text!r} cta_amount={cta_amount} sum_of_card_prices={price_sum} "
        f"n_prices_parsed={len(prices)}/{n_slots}",
    )


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
        help="Run only the new product-bug-check steps (B1-B4) instead of the original a-f flow.",
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
                    step_image_upload(page, state, Path(args.image))
                    step_b4_image_board(page, state)
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
