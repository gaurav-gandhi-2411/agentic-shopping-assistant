#!/usr/bin/env python3
"""Wave-6 content-asserting live-proof smoke test.

Run against the deployed Cloud Run service (default: production URL below):

    python scripts/wave6_smoke.py
    python scripts/wave6_smoke.py --url https://asa-stylist-api-rm7rz66wza-el.a.run.app -v

Every check asserts on response CONTENT (article ids, slot roles, gender,
pdp_url shapes, prose length/keywords) over the REAL HTTP + WebSocket wire
protocol — no internal imports, no mocks:

  1  Owned-anchor image upload: exactly one "seed" item (owned, unbuyable,
     no pdp_url), >=2 buyable "complement" items, seed excluded from
     cart_url/item_links.
  2  Western footwear slot present on a men's casual image-anchored outfit.
  3  "Buy one like this" secondary turn returns buyable lookalikes.
  4  Cross-turn conversation memory + the historical gender-regression check
     ("in blue now" must stay men's shirts).
  5  "Swap the bottom in this look" changes exactly the bottom slot.
  6  "Make this look more ethnic" shifts the look toward ethnic garments.
  7  suggestion_chips present on a plain search turn.
  8  Stylist reply depth (>=2 sentences) + grounding guardrail (no
     price/size/stock/fabric leakage).

Request budget
--------------
Only turns that hit /chat, /chat/stream, or /style/from-image consume the
demo per-IP rate cap (api/demo/guards.py::check_ip_rate_limit, called once
per WS connection in api/routes/chat.py::ws_chat and once per call in
api/routes/image_style.py::post_style_from_image). POST /demo/session
(api/routes/demo.py:34-69) and POST /auth/ws-ticket (api/routes/auth.py:16-25)
are NOT rate-limited — minting a fresh ticket per WS reconnect is free.

This script performs exactly 10 rate-limited turns total:
  1 REST call   (/style/from-image, covers checks 1 & 2)
  9 WS turns    (1 for check 3, 3 for check 4, 2 for check 5, 2 for check 6,
                 1 for checks 7 & 8 combined)

Response-schema assumptions (with file:line the script was based on):
  - ItemSummary fields (article_id, slot_role, is_owned, pdp_url,
    outfit_slot, product_type, prod_name, display_name):
    api/schemas.py:53-106
  - POST /style/from-image payload shape (items, cart_url, item_links,
    conversation_id): api/routes/image_style.py:445-458
  - WS done frame final_state keys (filters, suggestion_chips, response):
    api/routes/chat.py:668-693
  - Gender reconstruction from filters (gender key, else index_group_name):
    src/agents/graph.py:763-769 (_resolve_session_gender / router carry-forward)
  - Deterministic outfit-trigger phrasing ("build me a casual outfit for X"):
    src/agents/graph.py:36-41 (_OUTFIT_INTENT_RE), :42-46 (_OUTFIT_OCCASION_RE)
  - Deterministic swap/ethnic-refinement phrasing ("swap the {slot} in this
    look", "make this look more {word}"): src/agents/graph.py:64-78
    (_LOOK_REFINEMENT_RE, _ETHNIC_REFINEMENT_WORDS, _SWAP_SLOT_WORD_MAP)
  - Buy-similar trigger phrasing ("like this"): src/agents/graph.py:834-846
    (_BUY_SIMILAR_RE)
  - Outfit slot names (bottom/top/footwear/outerwear/accessory) and which
    anchor classes require a "bottom" complement: src/agents/outfit/slots.py
  - Ethnic garment keywords: src/agents/outfit/slots.py:7-16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests
import websockets

_REPO_ROOT = Path(__file__).parent.parent
_TSHIRT_FIXTURE = _REPO_ROOT / "t-shirt.webp"

_DEFAULT_URL = "https://asa-stylist-api-rm7rz66wza-el.a.run.app"
_TIMEOUT = 60.0

# Ethnic garment keywords — mirrors src/agents/outfit/slots.py's
# ETHNIC_TOP_KEYWORDS / ETHNIC_ONE_PIECE_KEYWORDS / ETHNIC_BOTTOM_KEYWORDS
# plus men's ethnic formalwear, so check 6 asserts on the same vocabulary the
# composer itself uses to classify a garment as ethnic.
_ETHNIC_KEYWORDS: frozenset[str] = frozenset(
    {
        "kurta",
        "kurti",
        "kameez",
        "tunic",
        "kaftan",
        "lehenga",
        "saree",
        "anarkali",
        "suit-set",
        "suit set",
        "sharara",
        "salwar",
        "palazzo",
        "churidar",
        "pyjama",
        "dhoti",
        "sherwani",
        "bandhgala",
        "ethnic",
    }
)

_GROUNDING_FORBIDDEN_WORDS: tuple[str, ...] = ("price", "size", "stock", "fabric")

Status = Literal["PASS", "FAIL"]


@dataclass
class CheckResult:
    """One recorded PASS/FAIL outcome, printed inline and in the final summary."""

    check_id: str
    name: str
    status: Status
    detail: str = ""


RESULTS: list[CheckResult] = []


def _record(check_id: str, name: str, status: Status, detail: str = "") -> None:
    """Append a CheckResult and print it immediately (PASS/FAIL + raw evidence)."""
    RESULTS.append(CheckResult(check_id, name, status, detail))
    print(f"[{check_id}] {status} — {name}")
    if detail:
        print(f"      {detail}")


def _sentence_terminators(text: str) -> int:
    """Count '.', '!', '?' characters in text — a cheap proxy for sentence count."""
    return sum(text.count(c) for c in ".!?")


def _gender_from_filters(filters: dict[str, Any]) -> str | None:
    """Reconstruct 'men'/'women' from a WS done-frame final_state.filters dict.

    Mirrors src/agents/graph.py:763-769's own carry-forward precedence: prefer
    the explicit "gender" key, fall back to "index_group_name" (Menswear/
    Ladieswear). Returns None when neither is present or resolvable.
    """
    gender = filters.get("gender")
    if gender in ("men", "women"):
        return gender
    ign = str(filters.get("index_group_name") or "").lower()
    if "ladieswear" in ign:
        return "women"
    if "menswear" in ign:
        return "men"
    return None


# ---------------------------------------------------------------------------
# HTTP / WS transport helpers
# ---------------------------------------------------------------------------


def _auth_header(token: str) -> dict[str, str]:
    """Return an Authorization: Bearer header dict for the given demo token."""
    return {"Authorization": f"Bearer {token}"}


def _mint_demo_session(url: str) -> dict[str, Any]:
    """POST /demo/session — mint an anonymous demo bearer token + first WS ticket.

    Not rate-limited (api/routes/demo.py:34-69 only checks the daily brand cap).
    """
    r = requests.post(f"{url}/demo/session", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _mint_ws_ticket(url: str, token: str) -> str:
    """POST /auth/ws-ticket — mint a fresh 60s single-use WS ticket. Not rate-limited."""
    r = requests.post(f"{url}/auth/ws-ticket", headers=_auth_header(token), timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()["ticket"]


def _post_style_from_image(url: str, token: str, image_path: Path, message: str) -> dict[str, Any]:
    """POST /style/from-image with the t-shirt fixture + a styling message.

    Consumes one rate-limited demo turn (api/routes/image_style.py:230-254).
    """
    with image_path.open("rb") as fh:
        files = {"file": (image_path.name, fh, "image/webp")}
        data = {"message": message}
        r = requests.post(
            f"{url}/style/from-image",
            headers=_auth_header(token),
            files=files,
            data=data,
            timeout=_TIMEOUT,
        )
    r.raise_for_status()
    return r.json()


async def _ws_turn_async(
    ws_base: str, ticket: str, message: str, conversation_id: str | None, verbose: bool
) -> dict[str, Any]:
    """Send one user_message over a fresh /chat/stream connection and collect the turn.

    One WS connection = one rate-limited turn (api/routes/chat.py:424-454, ws_chat's
    per-connection check_ip_rate_limit/record_request for anonymous demo users).
    Returns conversation_id, items (from the "items" frame), streamed text
    (concatenated "token" frames, falling back to final_state.response), and
    final_state (from the "done" frame; api/routes/chat.py:668-693).
    """
    url = f"{ws_base}/chat/stream?ticket={ticket}"
    payload: dict[str, Any] = {"type": "user_message", "message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id

    items: list[dict[str, Any]] = []
    text_parts: list[str] = []
    final_state: dict[str, Any] = {}
    cid = conversation_id

    async with websockets.connect(url, open_timeout=30, close_timeout=10) as ws:
        await ws.send(json.dumps(payload))
        async for raw in ws:
            frame = json.loads(raw)
            ftype = frame.get("type")
            if ftype == "session":
                cid = frame["conversation_id"]
            elif ftype == "items":
                items = frame.get("items", [])
            elif ftype == "token":
                text_parts.append(frame.get("text", ""))
            elif ftype == "done":
                final_state = frame.get("final_state", {})
                break
            elif ftype == "error":
                raise RuntimeError(f"WS error frame for {message!r}: {frame}")

    text = "".join(text_parts) or str(final_state.get("response") or "")
    if verbose:
        print(f"    >>> {message!r} (conv={cid})")
        print(f"    <<< items={len(items)} text={text[:120]!r}")
    return {"conversation_id": cid, "items": items, "text": text, "final_state": final_state}


def _ws_turn(
    ws_base: str, ticket: str, message: str, conversation_id: str | None, verbose: bool
) -> dict[str, Any]:
    """Sync wrapper around _ws_turn_async (one asyncio event loop per turn)."""
    return asyncio.run(_ws_turn_async(ws_base, ticket, message, conversation_id, verbose))


# ---------------------------------------------------------------------------
# Checks 1 & 2 — owned-anchor image upload + western footwear slot
# ---------------------------------------------------------------------------


def _check_1_owned_anchor(resp: dict[str, Any]) -> None:
    """Check 1a-1d: seed is owned/unbuyable, complements are buyable, no cart leak."""
    items: list[dict[str, Any]] = resp.get("items") or []
    seeds = [it for it in items if it.get("slot_role") == "seed"]
    complements = [it for it in items if it.get("slot_role") == "complement"]

    # 1a: exactly one seed; owned; no pdp_url.
    seed_ok = len(seeds) == 1
    detail_1a = f"seed_count={len(seeds)}"
    if seed_ok:
        seed = seeds[0]
        detail_1a += f" is_owned={seed.get('is_owned')!r} pdp_url={seed.get('pdp_url')!r}"
        seed_ok = seed.get("is_owned") is True and not seed.get("pdp_url")
    _record(
        "1a",
        "exactly one seed item; is_owned=True; pdp_url empty",
        "PASS" if seed_ok else "FAIL",
        detail_1a,
    )

    # 1b: every complement not-owned with a real https pdp_url.
    bad_complements = [
        (c.get("article_id"), c.get("is_owned"), c.get("pdp_url"))
        for c in complements
        if c.get("is_owned") or not str(c.get("pdp_url") or "").startswith("https")
    ]
    b_ok = len(complements) > 0 and not bad_complements
    _record(
        "1b",
        "every complement is not-owned with a non-empty https pdp_url",
        "PASS" if b_ok else "FAIL",
        f"complements={len(complements)} bad={bad_complements}",
    )

    # 1c: seed article_id never leaks into cart_url or item_links.
    seed_id = seeds[0].get("article_id") if seeds else None
    leaks: list[str] = []
    if seed_id:
        cart_url = str(resp.get("cart_url") or "")
        if seed_id in cart_url:
            leaks.append("cart_url")
        for lk in resp.get("item_links") or []:
            if lk.get("article_id") == seed_id:
                leaks.append("item_links")
    c_ok = seed_id is not None and not leaks
    _record(
        "1c",
        "seed article_id absent from cart_url/item_links",
        "PASS" if c_ok else "FAIL",
        f"seed_id={seed_id} leaks={leaks} cart_url={resp.get('cart_url')!r}",
    )

    # 1d: at least 2 complements.
    d_ok = len(complements) >= 2
    _record(
        "1d",
        "at least 2 complements returned",
        "PASS" if d_ok else "FAIL",
        f"complements={len(complements)}",
    )


def _check_2_western_footwear(resp: dict[str, Any]) -> None:
    """Check 2: a footwear-slotted item is present on the men's casual image anchor."""
    items: list[dict[str, Any]] = resp.get("items") or []
    slots = [it.get("outfit_slot") for it in items]
    has_footwear = "footwear" in slots
    _record(
        "2",
        "western footwear slot present (men's casual image-anchored outfit)",
        "PASS" if has_footwear else "FAIL",
        f"outfit_slot values={slots}",
    )


# ---------------------------------------------------------------------------
# Check 3 — buy-similar secondary turn
# ---------------------------------------------------------------------------


def _check_3_buy_similar(turn: dict[str, Any]) -> None:
    """Check 3: 'buy one like this' returns >=1 buyable lookalike + non-empty text."""
    items: list[dict[str, Any]] = turn["items"]
    text: str = turn["text"]
    missing_pdp = [it.get("article_id") for it in items if not str(it.get("pdp_url") or "").strip()]
    ok = len(items) >= 1 and not missing_pdp and bool(text.strip())
    _record(
        "3",
        "buy-similar secondary turn returns buyable lookalikes",
        "PASS" if ok else "FAIL",
        f"items={len(items)} missing_pdp={missing_pdp} text_len={len(text)}",
    )


# ---------------------------------------------------------------------------
# Check 4 — conversation memory + gender-regression check
# ---------------------------------------------------------------------------


def _check_4a(t1: dict[str, Any]) -> None:
    """Check 4a: turn1 'white shirts for men' returns items (+ gender if available)."""
    items: list[dict[str, Any]] = t1["items"]
    gender = _gender_from_filters(t1["final_state"].get("filters", {}))
    ok = len(items) >= 1 and (gender is None or gender == "men")
    _record(
        "4a",
        "turn1 'white shirts for men' returns items (gender consistent if available)",
        "PASS" if ok else "FAIL",
        f"items={len(items)} gender={gender!r}",
    )


def _check_4b(t2: dict[str, Any]) -> None:
    """Check 4b: turn2 date-night follow-up is non-empty, multi-sentence, contextual."""
    text: str = t2["text"]
    terms = _sentence_terminators(text)
    lowered = text.lower()
    context_words = ("shirt", "white", "these", "date")
    has_context = any(w in lowered for w in context_words)
    ok = bool(text.strip()) and terms >= 2 and has_context
    _record(
        "4b",
        "turn2 date-night reply: non-empty, >=2 sentences, references context",
        "PASS" if ok else "FAIL",
        f"terminators={terms} has_context={has_context} text={text[:160]!r}",
    )


def _check_4c(t3: dict[str, Any]) -> None:
    """Check 4c (historical gender regression): turn3 'in blue now' stays men's shirts."""
    items: list[dict[str, Any]] = t3["items"]
    types = [str(it.get("product_type") or "").lower() for it in items]
    shirts_ok = bool(types) and all("shirt" in tp for tp in types)
    gender = _gender_from_filters(t3["final_state"].get("filters", {}))
    ok = shirts_ok and gender == "men"
    _record(
        "4c",
        "turn3 'in blue now' stays shirts + men (gender-regression check)",
        "PASS" if ok else "FAIL",
        f"product_types={types} gender={gender!r}",
    )


# ---------------------------------------------------------------------------
# Check 5 — swap slot
# ---------------------------------------------------------------------------


def _slot_article_map(items: list[dict[str, Any]]) -> dict[str, str]:
    """Return {outfit_slot: article_id} for every complement item (seed has no slot)."""
    return {str(it["outfit_slot"]): str(it["article_id"]) for it in items if it.get("outfit_slot")}


def _seed_article_id(items: list[dict[str, Any]]) -> str | None:
    """Return the article_id of the item tagged slot_role=='seed', or None."""
    seed = next((it for it in items if it.get("slot_role") == "seed"), None)
    return seed.get("article_id") if seed else None


def _check_5_swap_slot(turn1: dict[str, Any], turn2: dict[str, Any]) -> None:
    """Check 5a/5b: 'swap the bottom in this look' changes only the bottom slot."""
    old_slots = _slot_article_map(turn1["items"])
    new_slots = _slot_article_map(turn2["items"])

    if "bottom" not in old_slots:
        _record(
            "5a",
            "bottom slot present in the composed outfit before swap",
            "FAIL",
            f"turn1 outfit_slots={sorted(old_slots)} (no 'bottom' — anchor class mismatch)",
        )
        _record("5b", "every other slot unchanged after swap", "FAIL", "skipped (5a failed)")
        return

    bottom_changed = "bottom" in new_slots and new_slots["bottom"] != old_slots["bottom"]
    _record(
        "5a",
        "bottom slot article_id changed after 'swap the bottom in this look'",
        "PASS" if bottom_changed else "FAIL",
        f"old_bottom={old_slots.get('bottom')} new_bottom={new_slots.get('bottom')}",
    )

    mismatches = [
        f"{slot}: {aid} -> {new_slots.get(slot)}"
        for slot, aid in old_slots.items()
        if slot != "bottom" and new_slots.get(slot) != aid
    ]
    old_seed, new_seed = _seed_article_id(turn1["items"]), _seed_article_id(turn2["items"])
    if old_seed != new_seed:
        mismatches.append(f"seed: {old_seed} -> {new_seed}")
    unchanged_ok = not mismatches
    _record(
        "5b",
        "every other slot (incl. seed) unchanged after swap",
        "PASS" if unchanged_ok else "FAIL",
        "unchanged" if unchanged_ok else "; ".join(mismatches),
    )


# ---------------------------------------------------------------------------
# Check 6 — more ethnic shift
# ---------------------------------------------------------------------------


def _check_6_more_ethnic(turn2: dict[str, Any]) -> None:
    """Check 6: 'make this look more ethnic' actually shifts >=1 item toward ethnic wear."""
    items: list[dict[str, Any]] = turn2["items"]
    hits = []
    for it in items:
        haystack = " ".join(
            str(it.get(k) or "") for k in ("product_type", "prod_name", "display_name")
        ).lower()
        if any(kw in haystack for kw in _ETHNIC_KEYWORDS):
            hits.append(it.get("article_id"))
    ok = bool(hits)
    _record(
        "6",
        "'make this look more ethnic' shifts >=1 item toward ethnic garments",
        "PASS" if ok else "FAIL",
        f"ethnic_hits={hits} of {len(items)} items; "
        f"types={[it.get('product_type') for it in items]}",
    )


# ---------------------------------------------------------------------------
# Checks 7 & 8 — suggestion chips + stylist reply depth/grounding
# ---------------------------------------------------------------------------


def _check_7_suggestion_chips(turn: dict[str, Any]) -> None:
    """Check 7: suggestion_chips is a non-empty list of strings on a plain search turn."""
    chips = turn["final_state"].get("suggestion_chips")
    ok = isinstance(chips, list) and len(chips) > 0 and all(isinstance(c, str) for c in chips)
    _record(
        "7",
        "suggestion_chips present in WS done.final_state after plain search",
        "PASS" if ok else "FAIL",
        f"chips={chips}",
    )


def _check_8_reply_depth(turn: dict[str, Any]) -> None:
    """Check 8: reply has >=2 sentences and never leaks price/size/stock/fabric talk."""
    text: str = turn["text"]
    terms = _sentence_terminators(text)
    lowered = text.lower()
    leaked = [w for w in _GROUNDING_FORBIDDEN_WORDS if w in lowered]
    ok = terms >= 2 and not leaked
    _record(
        "8",
        "stylist reply depth (>=2 sentences) + grounding guardrail held",
        "PASS" if ok else "FAIL",
        f"terminators={terms} leaked_words={leaked} text={text[:160]!r}",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(url: str, verbose: bool) -> int:
    """Run all wave-6 checks against the deployed service; return process exit code."""
    ws_base = url.replace("https://", "wss://").replace("http://", "ws://")

    session = _mint_demo_session(url)
    token = session["session_token"]
    print(f"Target        : {url}")
    print(f"Demo brand    : {session.get('brand')}")
    print(f"Session token : len={len(token)}\n")

    # ---- Checks 1 & 2: owned anchor + western footwear (REST) -------------
    image_conv_id: str | None = None
    try:
        img = _post_style_from_image(
            url,
            token,
            _TSHIRT_FIXTURE,
            "style this for a casual day out for men",
        )
        _check_1_owned_anchor(img)
        _check_2_western_footwear(img)
        image_conv_id = img.get("conversation_id")
    except Exception as exc:  # noqa: BLE001 — smoke test: report, don't crash the run
        for cid in ("1a", "1b", "1c", "1d", "2"):
            _record(cid, "owned anchor / footwear (upstream failure)", "FAIL", str(exc))

    # ---- Check 3: buy-similar secondary turn (same conversation) ----------
    try:
        ticket = _mint_ws_ticket(url, token)
        turn3 = _ws_turn(ws_base, ticket, "Where can I buy one like this?", image_conv_id, verbose)
        _check_3_buy_similar(turn3)
    except Exception as exc:  # noqa: BLE001
        _record("3", "buy-similar secondary turn", "FAIL", str(exc))

    # ---- Check 4: conversation memory (fresh conversation, 3 turns) -------
    try:
        ticket = _mint_ws_ticket(url, token)
        t1 = _ws_turn(ws_base, ticket, "show me white shirts for men", None, verbose)
        conv4 = t1["conversation_id"]
        _check_4a(t1)

        ticket = _mint_ws_ticket(url, token)
        t2 = _ws_turn(
            ws_base, ticket, "which of these would work for a date night?", conv4, verbose
        )
        _check_4b(t2)

        ticket = _mint_ws_ticket(url, token)
        t3 = _ws_turn(ws_base, ticket, "in blue now", conv4, verbose)
        _check_4c(t3)
    except Exception as exc:  # noqa: BLE001
        for cid in ("4a", "4b", "4c"):
            _record(cid, "conversation memory (upstream failure)", "FAIL", str(exc))

    # ---- Check 5: swap slot (fresh conversation, 2 turns) ------------------
    try:
        ticket = _mint_ws_ticket(url, token)
        s1 = _ws_turn(ws_base, ticket, "build me a casual outfit for men", None, verbose)
        conv5 = s1["conversation_id"]

        ticket = _mint_ws_ticket(url, token)
        s2 = _ws_turn(ws_base, ticket, "swap the bottom in this look", conv5, verbose)
        _check_5_swap_slot(s1, s2)
    except Exception as exc:  # noqa: BLE001
        for cid in ("5a", "5b"):
            _record(cid, "swap slot (upstream failure)", "FAIL", str(exc))

    # ---- Check 6: more ethnic (fresh conversation, 2 turns) -----------------
    try:
        ticket = _mint_ws_ticket(url, token)
        e1 = _ws_turn(ws_base, ticket, "build me a casual outfit for women", None, verbose)
        conv6 = e1["conversation_id"]

        ticket = _mint_ws_ticket(url, token)
        e2 = _ws_turn(ws_base, ticket, "make this look more ethnic", conv6, verbose)
        _check_6_more_ethnic(e2)
    except Exception as exc:  # noqa: BLE001
        _record("6", "more ethnic shift", "FAIL", str(exc))

    # ---- Checks 7 & 8: suggestion chips + reply depth (1 turn) --------------
    try:
        ticket = _mint_ws_ticket(url, token)
        c1 = _ws_turn(ws_base, ticket, "red dresses for women", None, verbose)
        _check_7_suggestion_chips(c1)
        _check_8_reply_depth(c1)
    except Exception as exc:  # noqa: BLE001
        for cid in ("7", "8"):
            _record(cid, "chips / reply depth (upstream failure)", "FAIL", str(exc))

    return _print_summary()


def _print_summary() -> int:
    """Print the final PASS/FAIL table; return 1 if any check failed, else 0."""
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    width = max((len(r.check_id) for r in RESULTS), default=2)
    for r in RESULTS:
        print(f"  [{r.check_id.ljust(width)}] {r.status:<4} {r.name}")
    n_fail = sum(1 for r in RESULTS if r.status == "FAIL")
    n_pass = len(RESULTS) - n_fail
    print(f"\n{n_pass}/{len(RESULTS)} checks PASS")
    return 1 if n_fail else 0


def main() -> int:
    """CLI entry point: parse args, verify the fixture exists, run all checks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default=_DEFAULT_URL, help="Base HTTPS URL of the deployed API.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print each WS turn's raw items/text."
    )
    args = parser.parse_args()

    if not _TSHIRT_FIXTURE.exists():
        print(f"FATAL: fixture not found: {_TSHIRT_FIXTURE}")
        return 2

    return run(args.url, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
