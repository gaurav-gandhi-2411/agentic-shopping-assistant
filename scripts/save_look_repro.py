"""Red-baseline repro for the "Save look" failure on the LIVE unified demo (GG, 2026-07-06).

Hypothesis (from static read of api/schemas.py + frontend/components/chat/OutfitBoard.tsx):
in unified (no-brand-param) mode `brand` is `undefined` on the OutfitBoard component, so
`handleSaveLook`'s POST /looks body sends `"brand": null`. The backend's
`SaveLookRequest.brand: str = Field(...)` is a REQUIRED, non-Optional string, so FastAPI/
Pydantic should reject the request with 422 before any handler code runs, and the frontend's
`if (!res.ok) throw ...` catch block then flips the button to "Failed to save — tap to retry".

This script does NOT fix anything. It reproduces the failure against the real, live site in a
real Chromium session and prints the exact network evidence (request payload, response status +
body) plus the resulting button label and any console errors, then exits non-zero if the save
did not succeed -- i.e. a "PASS" here would mean the bug is NOT reproducible, and the script's
own exit code / RED BASELINE banner make that unambiguous either way.

Usage:
    python scripts/save_look_repro.py [--base-url URL] [--headed]

Exit code:
    0  -- save succeeded (bug NOT reproduced; unexpected -- investigate before trusting this).
    1  -- save failed (RED BASELINE reproduced, as hypothesized).
    2  -- could not even get far enough to click "Save look" (setup/environment failure).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page, Request, Response, sync_playwright

# ---------------------------------------------------------------------------
# ASCII-safe console output on Windows (cp1252 default codepage).
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE_URL = "https://asa-stylist.vercel.app/demo"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCHPAD = Path(
    r"C:\Users\gaura\AppData\Local\Temp\claude\C--Users-gaura-ml-projects-agentic-shopping-assistant"
    r"\21a65b53-fb88-4c7e-a1d7-9ef9fde6fb18\scratchpad\save_look_repro"
)
VIEWPORT = {"width": 1366, "height": 900}

CARD_SELECTOR = ".rounded-lg.border:has(p.line-clamp-2)"
OUTFIT_BOARD_SELECTOR = "div.rounded-xl.border.bg-card.p-4"

CARD_WAIT_TIMEOUT_S = 60
POLL_INTERVAL_S = 1.5
POST_SAVE_WAIT_S = 5
# handleSaveLook's catch block resets saveState "error" -> "idle" after exactly 3s
# (see OutfitBoard.tsx: setTimeout(() => setSaveState("idle"), 3000)), so we must
# sample the button label BEFORE that reset fires to actually see the
# "Failed to save — tap to retry" text, in addition to the final POST_SAVE_WAIT_S
# sample (which shows the post-reset/steady-state label).
IMMEDIATE_LABEL_WAIT_S = 1.5

# Query chosen to reliably render >=1 product card (reused verbatim from
# browser_proof.py's step_b3_interactions, which already validated this phrasing
# live), so this script only needs ONE typed chat message plus one button click
# ("Style this") to reach an outfit board -- staying well within the requested
# <=3-message budget.
STYLING_QUERY = "black dress for women"


@dataclass
class NetworkEvent:
    """One captured request or response touching the /looks endpoint."""

    kind: str  # "request" or "response"
    method: str
    url: str
    status: int | None = None
    body: str | None = None


@dataclass
class ReproState:
    """Mutable state threaded through the repro steps."""

    console_issues: list[str] = field(default_factory=list)
    looks_events: list[NetworkEvent] = field(default_factory=list)


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
    """Poll until the card count exceeds `baseline` or the timeout elapses."""
    deadline = time.time() + timeout_s
    count = baseline
    while time.time() < deadline:
        count = card_count(page)
        if count > baseline:
            return count
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
    return card_count(page)


def wait_for_turn_idle(page: Page, timeout_s: float = 60) -> None:
    """Drain any in-flight assistant turn (isSending) before proceeding."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if page.get_by_role("button", name="Stop").count() == 0:
            return
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))


def send_text(page: Page, text: str) -> None:
    """Type into the chat textarea and click Send."""
    textarea = page.locator("textarea")
    textarea.click()
    textarea.fill(text)
    page.get_by_role("button", name="Send").click()


def on_request(state: ReproState, req: Request) -> None:
    """Capture request payloads for any call touching /looks."""
    if "/looks" not in req.url:
        return
    post_data = None
    try:
        post_data = req.post_data
    except Exception as exc:  # noqa: BLE001 - best-effort evidence capture
        post_data = f"<post_data extraction failed: {exc}>"
    state.looks_events.append(
        NetworkEvent(kind="request", method=req.method, url=req.url, body=post_data)
    )


def on_response(state: ReproState, res: Response) -> None:
    """Capture response status + body for any call touching /looks."""
    if "/looks" not in res.url:
        return
    body_text: str | None
    try:
        body_text = res.text()
    except Exception as exc:  # noqa: BLE001
        body_text = f"<response body extraction failed: {exc}>"
    state.looks_events.append(
        NetworkEvent(
            kind="response",
            method=res.request.method,
            url=res.url,
            status=res.status,
            body=body_text,
        )
    )


def step_load_chat(page: Page, base_url: str) -> bool:
    """Load /demo, follow auto-redirect into /demo/chat, wait for the composer."""
    try:
        page.goto(base_url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_url(re.compile(r".*/demo/chat.*"), timeout=60_000)
        page.wait_for_selector("textarea", timeout=30_000)
        shot(page, "1_chat_loaded")
        print(f"[setup] demo -> chat UI loaded ok (url={page.url})")
        return True
    except Exception as exc:  # noqa: BLE001
        shot(page, "1_chat_loaded_FAIL")
        print(f"[setup] FAILED to load chat UI: {exc}")
        return False


def step_reach_outfit_board(page: Page) -> bool:
    """Send STYLING_QUERY, then click 'Style this' on the first card to get an outfit board.

    Returns True once an OUTFIT_BOARD_SELECTOR element with a 'Save look' button is present.
    """
    baseline = card_count(page)
    send_text(page, STYLING_QUERY)
    after = wait_for_more_cards(page, baseline, CARD_WAIT_TIMEOUT_S)
    wait_for_turn_idle(page)
    gained = after - baseline
    shot(page, "2_styling_query")
    print(f"[setup] '{STYLING_QUERY}' -> cards {baseline}->{after} (gained={gained})")
    if gained < 1:
        print("[setup] FAILED: styling query produced no cards; cannot reach outfit board")
        return False

    first_card = page.locator(CARD_SELECTOR).nth(baseline)
    board_baseline = page.locator(OUTFIT_BOARD_SELECTOR).count()
    try:
        first_card.get_by_text("Style this", exact=True).first.click()
    except Exception as exc:  # noqa: BLE001
        print(f"[setup] FAILED: could not click 'Style this': {exc}")
        return False

    deadline = time.time() + CARD_WAIT_TIMEOUT_S
    while time.time() < deadline:
        if page.locator(OUTFIT_BOARD_SELECTOR).count() > board_baseline:
            break
        page.wait_for_timeout(int(POLL_INTERVAL_S * 1000))
    wait_for_turn_idle(page)
    shot(page, "3_outfit_board")

    board_count = page.locator(OUTFIT_BOARD_SELECTOR).count()
    if board_count <= board_baseline:
        print("[setup] FAILED: 'Style this' did not produce a new outfit board")
        return False
    print(f"[setup] outfit board rendered (boards on page: {board_count})")
    return True


def find_save_button(page: Page):
    """Locate the last-rendered board's 'Save look' button (or its retry-labeled state)."""
    board = page.locator(OUTFIT_BOARD_SELECTOR).last
    # OutfitBoard's save button starts as "Save look"; on error it relabels to
    # "Failed to save — tap to retry" but remains the same <button> element,
    # so match on the Bookmark-icon button rather than exact text.
    btn = board.locator("button", has=page.locator("svg.lucide-bookmark"))
    return btn.first


def main() -> int:
    """Run the Save-look red-baseline repro against the live (or given) demo URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Demo entry URL to open.")
    parser.add_argument(
        "--headed", action="store_true", help="Run with a visible browser window (debugging)."
    )
    args = parser.parse_args()

    state = ReproState()

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
        page.on("request", lambda req: on_request(state, req))
        page.on("response", lambda res: on_response(state, res))

        setup_ok = False
        button_text_immediate = "<not clicked>"
        button_text_after = "<not clicked>"
        save_succeeded = False

        try:
            if step_load_chat(page, args.base_url) and step_reach_outfit_board(page):
                setup_ok = True
                save_btn = find_save_button(page)
                if save_btn.count() == 0:
                    print("[repro] FAILED: no 'Save look' button found on the rendered board")
                else:
                    button_text_before = save_btn.inner_text().strip()
                    print(f"[repro] Save button text BEFORE click: {button_text_before!r}")
                    save_btn.click()

                    # Sample the label BEFORE the 3s error-reset timer fires.
                    page.wait_for_timeout(IMMEDIATE_LABEL_WAIT_S * 1000)
                    shot(page, "4a_immediately_after_save_click")
                    save_btn_immediate = find_save_button(page)
                    button_text_immediate = (
                        save_btn_immediate.inner_text().strip()
                        if save_btn_immediate.count() > 0
                        else "<button gone already>"
                    )
                    print(
                        f"[repro] Save button text {IMMEDIATE_LABEL_WAIT_S}s after click "
                        f"(pre error-reset): {button_text_immediate!r}"
                    )

                    # Sample again after the full wait window (post error-reset).
                    page.wait_for_timeout((POST_SAVE_WAIT_S - IMMEDIATE_LABEL_WAIT_S) * 1000)
                    shot(page, "4b_after_save_click_settled")
                    save_btn_after = find_save_button(page)
                    if save_btn_after.count() > 0:
                        button_text_after = save_btn_after.inner_text().strip()
                    else:
                        # Button disappears entirely on success (replaced by the
                        # "Look saved!" panel) -- treat that as success evidence.
                        saved_panel = page.get_by_text("Look saved!", exact=False)
                        button_text_after = (
                            "<button gone; 'Look saved!' panel present>"
                            if saved_panel.count() > 0
                            else "<button gone; no 'Look saved!' panel found>"
                        )
                    save_succeeded = "saved" in button_text_after.lower()
        finally:
            context.close()
            browser.close()

    print("\n" + "=" * 78)
    print("NETWORK EVIDENCE: /looks")
    print("=" * 78)
    if not state.looks_events:
        print("(no request/response touching /looks was captured)")
    for ev in state.looks_events:
        if ev.kind == "request":
            print(f"--> REQUEST  {ev.method} {ev.url}")
            print(f"    post_data: {ev.body}")
        else:
            print(f"<-- RESPONSE {ev.method} {ev.url} status={ev.status}")
            print(f"    body: {ev.body}")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("CONSOLE / PAGEERROR MESSAGES")
    print("=" * 78)
    if not state.console_issues:
        print("(none captured)")
    for msg in state.console_issues:
        print(f"  {msg}")
    print("=" * 78)

    print(f"\nSave button text {IMMEDIATE_LABEL_WAIT_S}s after click (pre error-reset): "
          f"{button_text_immediate!r}")
    print(f"Save button text {POST_SAVE_WAIT_S}s after click (settled): {button_text_after!r}")
    print(f"Screenshots: {SCRATCHPAD}")

    print("\n" + "=" * 78)
    if not setup_ok:
        print("RED BASELINE: INCONCLUSIVE -- could not reach an outfit board to test Save look")
        print("=" * 78)
        return 2
    if save_succeeded:
        print("RED BASELINE: NOT REPRODUCED -- Save look succeeded (bug may be fixed already)")
        print("=" * 78)
        return 0
    print("RED BASELINE: REPRODUCED -- Save look FAILED as hypothesized")
    print("=" * 78)
    return 1


if __name__ == "__main__":
    sys.exit(main())
