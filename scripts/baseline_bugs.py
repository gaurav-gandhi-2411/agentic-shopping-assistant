#!/usr/bin/env python3
"""RED-baseline reproduction of 4 browser bugs against the LIVE Cloud Run service.

Run BEFORE any fixes land, to confirm the failing state described by the user's
browser reports.  Every check hits the SAME paths the browser uses:

  - WS /chat/stream  (via a ws_ticket minted by POST /demo/session)
  - POST /demo/session -> POST /style/from-image (multipart, Bearer session_token)

Do NOT fix any bugs here.  Do NOT touch backend/frontend source.  This script
only observes and reports.

Usage:
    python scripts/baseline_bugs.py [backend_url] [--only 1,3,4]
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosedError

# Windows consoles default to cp1252, which raises UnicodeEncodeError on any
# stray non-ASCII byte.  Belt-and-braces: force utf-8 with a replace fallback
# so a stray char never crashes the whole baseline run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE: str = "https://asa-stylist-api-657468372797.asia-south1.run.app"


def _parse_args(argv: list[str]) -> tuple[str, set[int] | None]:
    """Parse CLI args: an optional backend URL and an optional --only bug filter.

    The URL and filter may appear in either order. The filter accepts either
    ``--only=1,3,4`` or ``--only 1,3,4`` (space-separated next argv token).
    Returns (base_url, only_bug_numbers) where only_bug_numbers is None if
    ``--only`` was not passed, meaning "run every bug".
    """
    base = DEFAULT_BASE
    only: set[int] | None = None
    args = argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("http"):
            base = arg
        elif arg.startswith("--only"):
            if "=" in arg:
                val = arg.split("=", 1)[1]
            else:
                i += 1
                val = args[i] if i < len(args) else ""
            only = {int(x) for x in val.split(",") if x.strip()}
        i += 1
    return base, only


BASE, ONLY = _parse_args(sys.argv)
WS_BASE: str = BASE.replace("https://", "wss://").replace("http://", "ws://")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CATALOGUE_PATH: Path = REPO_ROOT / "data" / "processed" / "unified" / "catalogue.parquet"
TSHIRT_IMAGE_PATH: Path = REPO_ROOT / "t-shirt.webp"
USE_CHAT_STREAM_PATH: Path = REPO_ROOT / "frontend" / "hooks" / "useChatStream.ts"

BROWSER_UA: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

PASS_TXT: str = "PASS"
FAIL_TXT: str = "FAIL"
_USE_COLOUR: bool = sys.stdout.isatty()


def _tag(ok: bool) -> str:
    """Return the '[PASS]'/'[FAIL]' tag, ANSI-coloured only on a real TTY."""
    label = PASS_TXT if ok else FAIL_TXT
    if not _USE_COLOUR:
        return f"[{label}]"
    colour = "\033[32m" if ok else "\033[31m"
    return f"[{colour}{label}\033[0m]"


# -- result types -----------------------------------------------------------


@dataclasses.dataclass
class TurnResult:
    """Outcome of one WS /chat/stream turn, mirroring what the browser observes."""

    conv_id: str | None
    items: list[dict]
    prose: str
    got_done: bool
    died: bool
    latency_s: float
    error: str | None = None


@dataclasses.dataclass
class Tally:
    """Running pass/fail counter for one bug's set of checks."""

    passed: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        """Total checks recorded so far."""
        return self.passed + self.failed


# -- shared helpers -----------------------------------------------------------


async def get_ticket(http: httpx.AsyncClient) -> str:
    """Mint a ws_ticket the same way the browser does via POST /demo/session."""
    r = await http.post("/demo/session")
    r.raise_for_status()
    d = r.json()
    ticket = d.get("ws_ticket", "")
    if not ticket:
        token = d.get("session_token") or d.get("token", "")
        r2 = await http.post("/auth/ws-ticket", headers={"Authorization": f"Bearer {token}"})
        r2.raise_for_status()
        ticket = r2.json()["ticket"]
    return ticket


async def ws_turn(
    http: httpx.AsyncClient,
    user_msg: str,
    conv_id: str | None = None,
) -> TurnResult:
    """Run one WS /chat/stream turn, mirroring the browser's send/receive flow.

    Never raises out of this function: an abnormal close, a missing 'done'
    frame, a server 'error' frame, or any transport exception is captured in
    the returned TurnResult instead of propagating to the caller.
    """
    start = time.monotonic()
    items: list[dict] = []
    prose_parts: list[str] = []
    new_conv_id = conv_id
    got_done = False
    died = False
    error: str | None = None

    try:
        ticket = await get_ticket(http)
        url = f"{WS_BASE}/chat/stream?ticket={ticket}"
        async with websockets.connect(url, open_timeout=30, close_timeout=10) as ws:
            payload: dict = {"type": "user_message", "message": user_msg}
            if conv_id:
                payload["conversation_id"] = conv_id
            await ws.send(json.dumps(payload))

            async for raw in ws:
                frame = json.loads(raw)
                ftype = frame.get("type")
                if ftype == "session":
                    new_conv_id = frame["conversation_id"]
                elif ftype == "items":
                    items = frame.get("items", [])
                elif ftype == "token":
                    prose_parts.append(frame.get("text", ""))
                elif ftype == "done":
                    got_done = True
                    break
                elif ftype == "error":
                    error = f"WS server error frame: {frame.get('message', frame)}"
                    break

            if not got_done and error is None:
                # The iterator ended (server closed the socket) without ever
                # sending a 'done' frame -- treat as an abnormal death even
                # though no exception was raised.
                died = True
                error = "connection closed without a done frame"
    except ConnectionClosedError as exc:
        died = True
        error = error or f"connection closed abnormally: {exc!r}"
    except Exception as exc:  # noqa: BLE001 - deliberately broad: must never raise out of ws_turn
        died = True
        error = f"{type(exc).__name__}: {exc}"

    latency_s = time.monotonic() - start
    return TurnResult(
        conv_id=new_conv_id,
        items=items,
        prose="".join(prose_parts),
        got_done=got_done,
        died=died,
        latency_s=latency_s,
        error=error,
    )


def check(tally: Tally, name: str, condition: bool, detail: str = "") -> bool:
    """Print a PASS/FAIL line for one assertion and update the running tally."""
    print(f"  {_tag(condition)} {name}")
    if detail:
        print(f"         {detail}")
    if condition:
        tally.passed += 1
    else:
        tally.failed += 1
    return condition


# -- Bug #1: wrong buy links for new stores ----------------------------------

STORE_QUERIES: dict[str, str] = {
    "berrylush": "dress berrylush",
    "globalrepublic": "top globalrepublic",
    "libas": "kurta libas",
}

STORE_EXPECTED_DOMAINS: dict[str, str] = {
    "berrylush": "berrylush.com",
    "globalrepublic": "globalrepublic.in",
    "libas": "libas.in",
}


async def _get_pdp_with_retry(
    browser_http: httpx.AsyncClient, pdp_url: str
) -> tuple[httpx.Response, str | None]:
    """GET ``pdp_url``; on a 5xx, wait 10s and retry once.

    globalrepublic and libas throttle repeated bot hits from the same IP and
    have been observed returning transient 503s under this script's load.  A
    single retry after a 10s cool-down distinguishes a real broken link from
    throttling. Returns ``(response, throttle_detail)`` where ``throttle_detail``
    is set only when the 5xx persisted after the retry, so callers can report
    an honest FAIL detail instead of treating it as a broken link.
    """
    resp = await browser_http.get(pdp_url)
    if resp.status_code >= 500:
        await asyncio.sleep(10.0)
        retry_resp = await browser_http.get(pdp_url)
        if retry_resp.status_code >= 500:
            return retry_resp, f"{resp.status_code} twice (store throttling suspected)"
        return retry_resp, None
    return resp, None


async def test_bug1_buy_links(http: httpx.AsyncClient) -> Tally:
    """Buy-link checks for berrylush (hidden) / globalrepublic / libas.

    berrylush is intentionally hidden site-wide (STORE_CONFIG active=False --
    the store put up a Shopify password wall), so a berrylush-flavoured query
    must still return results overall but never surface any berrylush item.
    globalrepublic and libas must each carry their own live pdp_url, as before.
    """
    print("\n-- Bug 1: Wrong Buy Links (berrylush / globalrepublic / libas) --")
    tally = Tally()
    all_turn_items: list[dict] = []

    async with httpx.AsyncClient(
        follow_redirects=True, timeout=30, headers={"User-Agent": BROWSER_UA}
    ) as browser_http:
        for store_slug, expected_domain in STORE_EXPECTED_DOMAINS.items():
            query = STORE_QUERIES[store_slug]
            print(f"\n  store={store_slug}  query={query!r}")

            result = await ws_turn(http, query)
            if result.died:
                print(f"         first attempt died ({result.error}); retrying once")
                result = await ws_turn(http, query)

            if result.error and not result.items:
                check(tally, f"{store_slug}: WS turn succeeded", False, f"error={result.error}")
                continue

            all_turn_items.extend(result.items)

            store_items = [it for it in result.items if it.get("store") == store_slug]

            if store_slug == "berrylush":
                check(
                    tally,
                    "berrylush: hidden from results (store inactive)",
                    len(result.items) > 0 and len(store_items) == 0,
                    f"total_items={len(result.items)} berrylush_items={len(store_items)}",
                )
                continue

            check(
                tally,
                f"{store_slug}: >=1 item returned for this store",
                len(store_items) >= 1,
                f"total_items={len(result.items)} store_items={len(store_items)}",
            )
            if not store_items:
                continue

            # (b) every returned store item has a non-null pdp_url on the expected domain
            bad_domain: list[dict] = []
            for it in store_items:
                pdp_url = it.get("pdp_url")
                netloc = urlparse(pdp_url).netloc.lower() if pdp_url else ""
                if not pdp_url or expected_domain not in netloc:
                    bad_domain.append(
                        {"article_id": it.get("article_id"), "pdp_url": pdp_url}
                    )
            check(
                tally,
                f"{store_slug}: every item pdp_url on {expected_domain}",
                len(bad_domain) == 0,
                f"offenders={json.dumps(bad_domain)[:400]}" if bad_domain else "all ok",
            )

            # (c) store_display non-null
            no_display = [it.get("article_id") for it in store_items if not it.get("store_display")]
            check(
                tally,
                f"{store_slug}: every item has store_display",
                len(no_display) == 0,
                f"missing_on={no_display[:5]}" if no_display else "all ok",
            )

            # (d) HTTP GET up to 3 pdp_urls -> final status 200, final netloc still on domain.
            # A 2.5s gap between GETs plus a 10s-then-retry on 5xx avoids tripping
            # these stores' bot throttling, which has been observed to return 503s
            # after several same-IP GETs in quick succession.
            sample_urls = [it.get("pdp_url") for it in store_items[:3] if it.get("pdp_url")]
            for pdp_url in sample_urls:
                await asyncio.sleep(2.5)
                try:
                    resp, throttle_detail = await _get_pdp_with_retry(browser_http, pdp_url)
                    final_netloc = urlparse(str(resp.url)).netloc.lower()
                    ok = resp.status_code == 200 and expected_domain in final_netloc
                    if throttle_detail:
                        detail = f"url={pdp_url}  status={resp.status_code}  {throttle_detail}"
                    else:
                        detail = (
                            f"url={pdp_url}  status={resp.status_code}  final_url={resp.url}"
                        )
                except Exception as exc:
                    ok = False
                    detail = f"url={pdp_url}  request_error={exc}"
                check(tally, f"{store_slug}: pdp_url resolves live (200, same domain)", ok, detail)

        hidden_store_items = [
            it for it in all_turn_items if it.get("store") in ("berrylush", "hm")
        ]
        check(
            tally,
            "bug1: no item across all turns has store berrylush or hm",
            len(hidden_store_items) == 0,
            f"offenders={json.dumps(hidden_store_items)[:400]}"
            if hidden_store_items
            else "all ok",
        )

    return tally


# -- Bug #2: cards not rendering in browser (WS reliability + retry) ---------

BUG2_QUERIES: list[str] = [
    "saree",
    "black dress for women",
    "white shirt men",
    "red kurta",
    "jeans for men",
    "black dress for women",
]


async def test_bug2_cards_render(http: httpx.AsyncClient) -> Tally:
    """WS drop-rate (Part A) and browser resend-after-close semantics (Part B)."""
    print("\n-- Bug 2: Cards Not Rendering (WS reliability + retry semantics) --")
    tally = Tally()

    # -- Part A: drop rate across 6 fresh first-turn queries -----------------
    print("\n  Part A: drop-rate across fresh first-turn queries")
    results: list[TurnResult] = []
    for q in BUG2_QUERIES:
        r = await ws_turn(http, q)
        results.append(r)

    print(f"\n  {'query':<28} {'items':>6} {'done':>6} {'died':>6} {'latency_s':>10}")
    for q, r in zip(BUG2_QUERIES, results):
        print(
            f"  {q:<28} {len(r.items):>6} {str(r.got_done):>6} "
            f"{str(r.died):>6} {r.latency_s:>10.2f}"
        )

    died_count = sum(1 for r in results if r.died)
    check(
        tally,
        "Part A: zero died connections across all 6 turns",
        died_count == 0,
        f"died_count={died_count}  errors={[r.error for r in results if r.died]}",
    )

    bad_turns = [
        (q, r) for q, r in zip(BUG2_QUERIES, results) if not (len(r.items) > 0 and r.got_done)
    ]
    check(
        tally,
        "Part A: every turn has items > 0 and a done frame",
        len(bad_turns) == 0,
        f"offenders={[(q, len(r.items), r.got_done, r.error) for q, r in bad_turns]}",
    )

    missing_fields: list[dict] = []
    for q, r in zip(BUG2_QUERIES, results):
        for it in r.items:
            if not it.get("pdp_url") or not it.get("store_display") or not it.get("image_url"):
                missing_fields.append({"query": q, "item": it})
    check(
        tally,
        "Part A: every item has pdp_url + store_display + image_url",
        len(missing_fields) == 0,
        f"offenders={json.dumps(missing_fields)[:600]}" if missing_fields else "all ok",
    )

    # -- Part B: browser retry semantics after an abnormal close -------------
    print("\n  Part B: resend-same-message-on-same-conversation (browser retry path)")

    turn1 = await ws_turn(http, "white shirt men")
    print(
        f"    turn1 'white shirt men' (fresh conv): items={len(turn1.items)} "
        f"done={turn1.got_done} died={turn1.died} conv={turn1.conv_id}"
    )

    retry = await ws_turn(http, "white shirt men", conv_id=turn1.conv_id)
    print(
        f"    retry 'white shirt men' (same conv):  items={len(retry.items)} "
        f"done={retry.got_done} died={retry.died} prose={retry.prose[:80].strip()!r}"
    )
    check(
        tally,
        "Part B: resent message on same conversation delivers an items frame",
        len(retry.items) > 0,
        f"items={len(retry.items)} prose_only={len(retry.items) == 0 and bool(retry.prose)}",
    )

    refine = await ws_turn(http, "in blue now", conv_id=turn1.conv_id)
    print(
        f"    refine 'in blue now' (same conv):     items={len(refine.items)} "
        f"done={refine.got_done} died={refine.died}"
    )
    check(
        tally,
        "Part B: refinement-after-retry delivers an items frame",
        len(refine.items) > 0,
        f"items={len(refine.items)}",
    )

    return tally


# -- Bug #3: refinement drops context -----------------------------------------

SHIRT_LIKE_SUBSTR: tuple[str, ...] = ("shirt", "top", "polo")
BLUE_LIKE_SUBSTR: tuple[str, ...] = ("blue", "navy", "indigo")


def _load_catalogue() -> pd.DataFrame:
    """Load the unified catalogue and normalize article_id to str for joins."""
    df = pd.read_parquet(CATALOGUE_PATH)
    df["article_id"] = df["article_id"].astype(str)
    return df


async def test_bug3_refinement_context(http: httpx.AsyncClient) -> Tally:
    """'white shirt men' then 'in blue now' must stay within men's/unisex shirts in blue.

    The WS items payload has no 'gender' key (ItemSummary does not carry one),
    so this check joins the returned article_ids against the offline catalogue
    to verify gender / product_type / colour of what was actually returned.
    """
    print("\n-- Bug 3: Refinement Drops Context --")
    tally = Tally()

    turn1 = await ws_turn(http, "white shirt men")
    if turn1.died:
        print(f"    turn1 died ({turn1.error}); retrying once")
        turn1 = await ws_turn(http, "white shirt men")
    print(f"  Turn 1 'white shirt men': {len(turn1.items)} items, conv={turn1.conv_id}")

    turn2 = await ws_turn(http, "in blue now", conv_id=turn1.conv_id)
    if turn2.died:
        print(f"    turn2 died ({turn2.error}); retrying once")
        turn2 = await ws_turn(http, "in blue now", conv_id=turn1.conv_id)
    print(f"  Turn 2 'in blue now':     {len(turn2.items)} items")
    print(f"  prose: {turn2.prose[:120].strip()!r}")

    if not check(tally, "refinement: >=1 item returned", len(turn2.items) >= 1, ""):
        return tally

    catalogue = _load_catalogue()
    item_ids = [str(it.get("article_id")) for it in turn2.items if it.get("article_id")]
    joined = catalogue[catalogue["article_id"].isin(item_ids)].copy()

    joined_view = joined[
        ["article_id", "gender", "product_type_name", "colour_group_name", "store"]
    ]
    print("\n  Joined catalogue rows for turn-2 items:")
    print(joined_view.to_string(index=False))

    check(
        tally,
        "refinement: every returned article_id found in catalogue",
        len(joined) == len(set(item_ids)),
        f"items={len(item_ids)} matched_in_catalogue={len(joined)}",
    )

    genders = joined["gender"].fillna("").str.lower()
    non_men = joined_view[~genders.isin(["men", "unisex"])]
    check(
        tally,
        "refinement: all joined rows gender=men/unisex",
        len(non_men) == 0,
        f"genders={sorted(joined['gender'].unique())}"
        + (f" non_men_rows={non_men.to_dict('records')[:3]}" if len(non_men) else ""),
    )

    types_lower = joined["product_type_name"].fillna("").str.lower()
    is_shirt_like = types_lower.apply(lambda t: any(s in t for s in SHIRT_LIKE_SUBSTR))
    non_shirt = joined_view[~is_shirt_like]
    check(
        tally,
        "refinement: all joined rows are shirt-like product_type",
        len(non_shirt) == 0,
        f"types={sorted(joined['product_type_name'].unique())}"
        + (f" non_shirt_rows={non_shirt.to_dict('records')[:3]}" if len(non_shirt) else ""),
    )

    colours_lower = joined["colour_group_name"].fillna("").str.lower()
    is_blue_like = colours_lower.apply(lambda c: any(s in c for s in BLUE_LIKE_SUBSTR))
    blue_ratio = is_blue_like.mean() if len(is_blue_like) else 0.0
    seen_colours = sorted(joined["colour_group_name"].fillna("N/A").unique())
    check(
        tally,
        "refinement: majority of joined rows are blue-ish",
        blue_ratio > 0.5,
        f"blue_ratio={blue_ratio:.2f}  colours={seen_colours}",
    )

    return tally


# -- Bug #4: image search hangs / text dropped --------------------------------

IMAGE_POST_TIMEOUT_S: float = 90.0
TIMEOUT_DETAIL: str = (
    f"TIMEOUT after {IMAGE_POST_TIMEOUT_S:.0f}s -- this IS the 'Finding your match...' hang"
)


async def _mint_session_token(http: httpx.AsyncClient) -> str:
    """POST /demo/session and return the session_token, the same as the browser."""
    r = await http.post("/demo/session")
    r.raise_for_status()
    d = r.json()
    return str(d.get("session_token") or d.get("token") or "")


async def _post_image(
    http: httpx.AsyncClient,
    token: str,
    conversation_id: str,
    message: str | None = None,
) -> tuple[int | None, dict, float]:
    """POST /style/from-image the same way the browser's sendImage() does.

    Returns (status_code, response_json_or_error_dict, latency_seconds). On a
    client-side timeout, status_code is None and latency_seconds is the
    elapsed wait -- that IS the reported browser hang, so it is captured as
    evidence here rather than allowed to crash the script.
    """
    image_bytes = TSHIRT_IMAGE_PATH.read_bytes()
    files = {"file": ("t-shirt.webp", image_bytes, "image/webp")}
    data: dict[str, str] = {"conversation_id": conversation_id}
    if message is not None:
        data["message"] = message

    start = time.monotonic()
    try:
        resp = await http.post(
            "/style/from-image",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {token}"},
            timeout=IMAGE_POST_TIMEOUT_S,
        )
    except httpx.TimeoutException:
        latency_s = time.monotonic() - start
        return None, {"_timeout": True}, latency_s
    latency_s = time.monotonic() - start

    try:
        body = resp.json()
    except Exception:
        body = {"_raw_text": resp.text[:500]}
    return resp.status_code, body, latency_s


async def test_bug4_image_search(http: httpx.AsyncClient) -> Tally:
    """Browser-shaped POST /style/from-image: hang/latency + typed-text-dropped."""
    print("\n-- Bug 4: Image Search Hangs / Typed Text Dropped --")
    tally = Tally()

    token = await _mint_session_token(http)
    check(tally, "demo session minted for image upload", bool(token), f"token_len={len(token)}")
    if not token:
        return tally

    # -- baseline image-only upload (mirrors sendImage(file) with no typed text) --
    cid1 = str(uuid.uuid4())
    status1, body1, latency1 = await _post_image(http, token, cid1)
    print(f"\n  image-only POST: status={status1} latency={latency1:.2f}s conv={cid1}")

    if status1 is None:
        check(
            tally,
            "image upload returns HTTP 200 (not 404 = CLIP index missing)",
            False,
            TIMEOUT_DETAIL,
        )
        check(tally, "image upload returns items > 0", False, TIMEOUT_DETAIL)
        check(
            tally,
            "image upload latency < 30s (no 'Finding your match...' hang)",
            False,
            TIMEOUT_DETAIL,
        )
    else:
        check(
            tally,
            "image upload returns HTTP 200 (not 404 = CLIP index missing)",
            status1 == 200,
            f"status={status1}  body={json.dumps(body1)[:300]}",
        )

        n_items1 = len(body1.get("items", [])) if isinstance(body1, dict) else 0
        check(tally, "image upload returns items > 0", n_items1 > 0, f"items={n_items1}")

        check(
            tally,
            "image upload latency < 30s (no 'Finding your match...' hang)",
            latency1 < 30.0,
            f"latency_s={latency1:.2f}",
        )

    # -- typed-text-dropped check: same request + a typed message field ------
    cid2 = str(uuid.uuid4())
    typed_message = "buy something similar under 2000"
    status2, body2, latency2 = await _post_image(http, token, cid2, message=typed_message)
    print(f"\n  image+text POST: status={status2} latency={latency2:.2f}s conv={cid2}")

    if status2 is None:
        check(
            tally,
            "backend uses or acknowledges the typed text (expected RED today)",
            False,
            TIMEOUT_DETAIL,
        )
    else:
        whole_payload_json = json.dumps(body2)
        uses_text = (
            "2000" in whole_payload_json
            or "user_text" in body2
            or "message" in body2
        )
        check(
            tally,
            "backend uses or acknowledges the typed text (expected RED today)",
            uses_text,
            f"payload_snippet={whole_payload_json[:300]}",
        )

    # -- static frontend check: does sendImage() ever append the typed text? --
    if USE_CHAT_STREAM_PATH.exists():
        frontend_src = USE_CHAT_STREAM_PATH.read_text(encoding="utf-8", errors="replace")
        appends_message_field = 'body.append("message"' in frontend_src
        if appends_message_field:
            check(
                tally,
                "frontend sendImage() appends typed text to the multipart body",
                True,
                "found body.append(\"message\", ...) in useChatStream.ts",
            )
        else:
            print(
                f"  {_tag(False)} frontend drops typed text (never sent): "
                f"{USE_CHAT_STREAM_PATH} never calls body.append(\"message\", ...) "
                f"in sendImage(), so userText is only shown in the optimistic UI "
                f"bubble and never reaches the backend."
            )
            tally.failed += 1
    else:
        print(f"  {_tag(False)} frontend file not found: {USE_CHAT_STREAM_PATH}")
        tally.failed += 1

    return tally


# -- main ----------------------------------------------------------------------

BUG_REGISTRY: list[tuple[int, str, Callable[[httpx.AsyncClient], Awaitable[Tally]]]] = [
    (1, "Bug 1: Wrong Buy Links", test_bug1_buy_links),
    (2, "Bug 2: Cards Not Rendering", test_bug2_cards_render),
    (3, "Bug 3: Refinement Drops Context", test_bug3_refinement_context),
    (4, "Bug 4: Image Search Hangs / Text Dropped", test_bug4_image_search),
]


async def main() -> None:
    """Run the selected bug baselines against BASE and print a summary table.

    ONLY (parsed from --only) restricts which bugs run; None means "all".
    """
    print(f"Backend : {BASE}")
    print(f"WS base : {WS_BASE}")
    if ONLY is not None:
        print(f"Only    : bug(s) {sorted(ONLY)}")
    print("=" * 70)

    selected = [(name, fn) for num, name, fn in BUG_REGISTRY if ONLY is None or num in ONLY]

    async with httpx.AsyncClient(base_url=BASE, timeout=60) as http:
        tallies: list[tuple[str, Tally]] = []
        for name, fn in selected:
            tallies.append((name, await fn(http)))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'bug':<44} {'passed':>8} {'failed':>8}")
    total_failures = 0
    for name, tally in tallies:
        print(f"  {name:<44} {tally.passed:>8} {tally.failed:>8}")
        total_failures += tally.failed

    print("=" * 70)
    overall_ok = total_failures == 0
    print(f"{_tag(overall_ok)} {total_failures} check(s) failed across {len(tallies)} bugs")
    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
