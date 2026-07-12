"""S4a/422 fix — every POST /events call site in OutfitBoard.tsx must send
anchor_item_id + anchor_category.

Live-proven bug (phase-b browser retest, 2026-07-07): clicking a variant tab
fired POST /events {event_type: "variant_selected", ...} WITHOUT
anchor_item_id/anchor_category, which the backend's EventRequest schema
(api/routes/events.py) requires on every event type — the frontend has no
test runner configured (no package.json "test" script, no jest/vitest), so
this is a lightweight, dependency-free source-text regression guard: every
`postEvent(backendUrl, token, { ... })` call block in OutfitBoard.tsx must
contain both `anchor_item_id` and `anchor_category` before the closing `})`.

Pure Python file-parsing — no new frontend tooling required, consistent with
the existing source-text pinning pattern in
tests/test_owned_anchor.py::TestBuySimilarRegexMatchesFrontendChip.
"""
from __future__ import annotations

import re
from pathlib import Path

OUTFIT_BOARD_TSX = (
    Path(__file__).resolve().parent.parent
    / "frontend" / "components" / "chat" / "OutfitBoard.tsx"
)

# Matches each `postEvent(backendUrl, token, { ... })` call block, non-greedy so
# each match stops at its own closing `})` rather than spanning to a later call.
_POST_EVENT_CALL_RE = re.compile(r"postEvent\(backendUrl,\s*token,\s*\{.*?\}\s*\)", re.DOTALL)
_EVENT_TYPE_RE = re.compile(r'event_type:\s*"([^"]+)"')


def _read_source() -> str:
    return OUTFIT_BOARD_TSX.read_text(encoding="utf-8")


def test_outfit_board_file_exists() -> None:
    assert OUTFIT_BOARD_TSX.exists(), f"expected file at {OUTFIT_BOARD_TSX}"


def test_at_least_one_post_event_call_found() -> None:
    """Guards against the regex silently matching zero calls (e.g. after an
    unrelated refactor of postEvent's call signature) and this test file
    passing vacuously."""
    calls = _POST_EVENT_CALL_RE.findall(_read_source())
    assert len(calls) >= 4, f"expected >=4 postEvent call sites, found {len(calls)}"


def test_every_post_event_call_includes_anchor_fields() -> None:
    """Every postEvent(...) call block must include BOTH anchor_item_id and
    anchor_category — the 422-causing gap was "variant_selected" omitting both.
    """
    source = _read_source()
    calls = _POST_EVENT_CALL_RE.findall(source)
    assert calls, "no postEvent(...) call sites found — regex may need updating"

    missing: list[str] = []
    for call in calls:
        event_type_match = _EVENT_TYPE_RE.search(call)
        event_type = event_type_match.group(1) if event_type_match else "<unknown>"
        has_item_id = "anchor_item_id" in call
        has_category = "anchor_category" in call
        if not (has_item_id and has_category):
            missing.append(
                f"event_type={event_type!r} anchor_item_id={has_item_id} "
                f"anchor_category={has_category}"
            )

    assert not missing, (
        "every postEvent(...) call must send both anchor_item_id and "
        "anchor_category (backend EventRequest requires both on every event "
        f"type) — offending calls: {missing}"
    )


def test_variant_selected_call_specifically_includes_anchor_fields() -> None:
    """Pins the specific live-proven regression: the "variant_selected" call
    site (handleVariantSwitch) must include both fields.
    """
    source = _read_source()
    calls = _POST_EVENT_CALL_RE.findall(source)
    variant_calls = [c for c in calls if '"variant_selected"' in c]
    assert variant_calls, "no 'variant_selected' postEvent call found"
    for call in variant_calls:
        assert "anchor_item_id" in call
        assert "anchor_category" in call
