#!/usr/bin/env python3
"""Content-asserting RED/GREEN checks for GG's 5 browser-reported product failures.

Runs against the LIVE Cloud Run backend and asserts on the actual JSON/WS
payload content (not just HTTP status), joining article_ids to the offline
catalogue for gender/product-type ground truth where the API's own response
does not carry it authoritatively.

This script only OBSERVES.  It does not fix product code.

Usage:
    python scripts/product_checks.py [backend_url] [--only 1,2a,2b,2c,3,5]

Budget: this script makes ~21 network operations (WS turn = 1 POST
/demo/session + 1 WS connect; each REST call = 1 request), well inside the
200/IP/hour rate limit and the ~30-request budget for this task.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import sys
import uuid
from pathlib import Path

import httpx
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosedError

# Windows consoles default to cp1252; force utf-8 with a replace fallback so a
# stray non-ASCII char never crashes the whole run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE: str = "https://asa-stylist-api-657468372797.asia-south1.run.app"

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
CATALOGUE_PATH: Path = REPO_ROOT / "data" / "processed" / "unified" / "catalogue.parquet"
TSHIRT_IMAGE_PATH: Path = REPO_ROOT / "t-shirt.webp"

MEN_LIKE_GENDERS: frozenset[str] = frozenset({"men", "unisex"})

PASS_TXT: str = "GREEN"
FAIL_TXT: str = "RED"
_USE_COLOUR: bool = sys.stdout.isatty()


def _tag(ok: bool) -> str:
    """Return the '[GREEN]'/'[RED]' tag, ANSI-coloured only on a real TTY."""
    label = PASS_TXT if ok else FAIL_TXT
    if not _USE_COLOUR:
        return f"[{label}]"
    colour = "\033[32m" if ok else "\033[31m"
    return f"[{colour}{label}\033[0m]"


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> tuple[str, set[str] | None]:
    """Parse an optional backend URL and an optional --only check-id filter."""
    base = DEFAULT_BASE
    only: set[str] | None = None
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
            only = {x.strip() for x in val.split(",") if x.strip()}
        i += 1
    return base, only


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Outcome:
    """One RED/GREEN assertion result."""

    check_id: str
    passed: bool
    evidence: str


OUTCOMES: list[Outcome] = []


def record(check_id: str, passed: bool, evidence: str = "") -> bool:
    """Print one RED/GREEN line, append to OUTCOMES, and return `passed`."""
    print(f"  {_tag(passed)} {check_id}: {evidence}")
    OUTCOMES.append(Outcome(check_id, passed, evidence))
    return passed


@dataclasses.dataclass
class WSTurnResult:
    """Outcome of one WS /chat/stream turn (extends baseline_bugs' TurnResult
    with the 'done' frame's final_state, needed for look_id / budget checks)."""

    conv_id: str | None
    items: list[dict]
    prose: str
    got_done: bool
    died: bool
    final_state: dict
    error: str | None = None


# ---------------------------------------------------------------------------
# Shared helpers (copied + extended from scripts/baseline_bugs.py)
# ---------------------------------------------------------------------------


async def mint_demo_session(http: httpx.AsyncClient) -> dict:
    """POST /demo/session and return the full JSON body (session_token, ws_ticket)."""
    r = await http.post("/demo/session")
    r.raise_for_status()
    return r.json()


async def ws_turn(
    http: httpx.AsyncClient,
    ws_base: str,
    user_msg: str,
    conv_id: str | None = None,
    ticket: str | None = None,
) -> WSTurnResult:
    """Run one WS /chat/stream turn, mirroring the browser's send/receive flow.

    Never raises: an abnormal close, a missing 'done' frame, a server 'error'
    frame, or any transport exception is captured in the returned WSTurnResult.
    A pre-minted `ticket` may be passed to avoid an extra POST /demo/session
    (used by CHECK 2a to reuse one session for both the WS turn and the
    follow-up REST call).
    """
    items: list[dict] = []
    prose_parts: list[str] = []
    new_conv_id = conv_id
    got_done = False
    died = False
    final_state: dict = {}
    error: str | None = None

    try:
        if ticket is None:
            session = await mint_demo_session(http)
            ticket = session["ws_ticket"]
        url = f"{ws_base}/chat/stream?ticket={ticket}"
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
                    final_state = frame.get("final_state", {})
                    break
                elif ftype == "error":
                    error = f"WS server error frame: {frame.get('message', frame)}"
                    break

            if not got_done and error is None:
                died = True
                error = "connection closed without a done frame"
    except ConnectionClosedError as exc:
        died = True
        error = error or f"connection closed abnormally: {exc!r}"
    except Exception as exc:  # noqa: BLE001 - deliberately broad: must never raise out of ws_turn
        died = True
        error = f"{type(exc).__name__}: {exc}"

    return WSTurnResult(
        conv_id=new_conv_id,
        items=items,
        prose="".join(prose_parts),
        got_done=got_done,
        died=died,
        final_state=final_state,
        error=error,
    )


async def ws_turn_retry(
    http: httpx.AsyncClient, ws_base: str, user_msg: str, conv_id: str | None = None
) -> WSTurnResult:
    """ws_turn with one retry on a died connection (transient Cloud Run cold start)."""
    result = await ws_turn(http, ws_base, user_msg, conv_id=conv_id)
    if result.died:
        result = await ws_turn(http, ws_base, user_msg, conv_id=conv_id)
    return result


_catalogue_cache: pd.DataFrame | None = None


def load_catalogue() -> pd.DataFrame:
    """Load + cache the unified catalogue with article_id normalised to str."""
    global _catalogue_cache
    if _catalogue_cache is None:
        df = pd.read_parquet(CATALOGUE_PATH)
        df["article_id"] = df["article_id"].astype(str)
        _catalogue_cache = df
    return _catalogue_cache


def join_catalogue(article_ids: list[str]) -> pd.DataFrame:
    """Return the catalogue rows for the given article_ids (ground truth join).

    article_id is unique in the unified catalogue (verified: zero duplicates),
    so a plain isin() filter + set_index is a safe 1:1 join.
    """
    df = load_catalogue()
    ids = {str(a) for a in article_ids}
    return df[df["article_id"].isin(ids)].set_index("article_id")


# ---------------------------------------------------------------------------
# CHECK 1 -- image styling quality (POST /style/from-image)
# ---------------------------------------------------------------------------

# Fixed garment-noun vocabulary for the grounding check (1f). Patterns allow
# singular/plural and an optional hyphen for "t-shirt".
_GARMENT_NOUN_PATTERNS: dict[str, str] = {
    "skirt": r"\bskirts?\b",
    "blazer": r"\bblazers?\b",
    "dress": r"\bdress(?:es)?\b",
    "saree": r"\bsarees?\b",
    "kurta": r"\bkurtas?\b",
    "jeans": r"\bjeans\b",
    "trousers": r"\btrousers?\b",
    "shirt": r"\bshirts?\b",
    "t-shirt": r"\bt-?shirts?\b",
    "tee": r"\btees?\b",
    "shoes": r"\bshoes?\b",
    "sneakers": r"\bsneakers?\b",
    "jacket": r"\bjackets?\b",
    "top": r"\btops?\b",
    "lehenga": r"\blehengas?\b",
    "heels": r"\bheels?\b",
}


def extract_garment_nouns(text: str) -> list[str]:
    """Return the fixed-vocabulary garment nouns mentioned in `text` (lowercased match)."""
    tl = text.lower()
    return [noun for noun, pat in _GARMENT_NOUN_PATTERNS.items() if re.search(pat, tl)]


def noun_is_grounded(noun: str, catalog_fields: list[str]) -> bool:
    """True if `noun` maps to some item's product_type/prod_name (case-insensitive,
    substring either direction, per the task's grounding-check spec)."""
    noun_l = noun.lower()
    return any(noun_l in f.lower() or f.lower() in noun_l for f in catalog_fields if f)


async def check_1_image_styling(http: httpx.AsyncClient) -> None:
    """CHECK 1: image styling quality via POST /style/from-image."""
    print("\n-- CHECK 1: image styling quality (POST /style/from-image) --")

    if not TSHIRT_IMAGE_PATH.exists():
        record("1", False, f"image file not found: {TSHIRT_IMAGE_PATH}")
        return

    session = await mint_demo_session(http)
    token = session["session_token"]

    image_bytes = TSHIRT_IMAGE_PATH.read_bytes()
    files = {"file": ("t-shirt.webp", image_bytes, "image/webp")}
    data = {"conversation_id": str(uuid.uuid4()), "message": "style this for a man"}
    resp = await http.post(
        "/style/from-image",
        files=files,
        data=data,
        headers={"Authorization": f"Bearer {token}"},
        timeout=90,
    )
    if resp.status_code != 200:
        record("1", False, f"POST /style/from-image returned {resp.status_code}: {resp.text[:300]}")
        return

    body = resp.json()
    items: list[dict] = body.get("items", [])
    rationale: str = body.get("outfit_rationale") or ""
    budget_total_inr = body.get("budget_total_inr")

    print(f"  n_items={len(items)}  budget_total_inr={budget_total_inr}")
    print(f"  outfit_rationale={rationale[:200]!r}")

    seeds = [it for it in items if it.get("slot_role") == "seed"]
    complements = [it for it in items if it.get("slot_role") == "complement"]

    # 1a: >=3 items total AND >=2 complements
    record(
        "1a",
        len(items) >= 3 and len(complements) >= 2,
        f"n_items={len(items)} n_complements={len(complements)}",
    )

    # Join to catalogue for ground-truth gender/product_type.
    article_ids = [it.get("article_id") for it in items if it.get("article_id")]
    joined = join_catalogue(article_ids)

    # 1b: no complement has the same catalogue product_type as the seed.
    if not seeds or joined.empty:
        record("1b", False, f"no seed item or empty catalogue join (n_joined={len(joined)})")
    else:
        seed_id = str(seeds[0]["article_id"])
        seed_type = (
            str(joined.loc[seed_id, "product_type_name"]) if seed_id in joined.index else None
        )
        dup_type_complements = [
            c.get("article_id")
            for c in complements
            if str(c.get("article_id")) in joined.index
            and str(joined.loc[str(c["article_id"]), "product_type_name"]) == seed_type
        ]
        record(
            "1b",
            bool(seed_type) and len(dup_type_complements) == 0,
            f"seed_type={seed_type!r} dup_type_complements={dup_type_complements}",
        )

    # 1c: ALL items (seed + complements) have catalogue gender in {men, unisex}.
    if joined.empty:
        record("1c", False, "empty catalogue join")
    else:
        genders = joined["gender"].fillna("").str.lower()
        offenders = joined[~genders.isin(MEN_LIKE_GENDERS)]
        record(
            "1c",
            len(offenders) == 0,
            f"genders_seen={sorted(joined['gender'].fillna('N/A').unique())}"
            + (f" offenders={offenders.index.tolist()}" if len(offenders) else ""),
        )

    # 1d: budget_total_inr == sum(price_inr of ALL returned items), within 1 rupee.
    prices = [it.get("price_inr") for it in items if it.get("price_inr") is not None]
    price_sum = sum(prices) if prices else None
    ok_1d = (
        budget_total_inr is not None
        and price_sum is not None
        and abs(float(budget_total_inr) - float(price_sum)) <= 1.0
    )
    record(
        "1d",
        ok_1d,
        f"budget_total_inr={budget_total_inr} sum(price_inr)={price_sum} n_priced_items={len(prices)}/{len(items)}",
    )

    # 1e: every item has slot_role set (seed exactly once) and non-null image_url + pdp_url.
    missing_slot_role = [it.get("article_id") for it in items if not it.get("slot_role")]
    missing_media = [
        it.get("article_id") for it in items if not it.get("image_url") or not it.get("pdp_url")
    ]
    record(
        "1e",
        len(seeds) == 1 and len(missing_slot_role) == 0 and len(missing_media) == 0,
        f"n_seeds={len(seeds)} missing_slot_role={missing_slot_role} missing_image_or_pdp={missing_media}",
    )

    # 1f: grounding -- every garment noun mentioned in the rationale maps to a returned item.
    if not rationale:
        record("1f", False, "outfit_rationale is empty")
    else:
        nouns = extract_garment_nouns(rationale)
        catalog_fields: list[str] = []
        for it in items:
            catalog_fields.append(str(it.get("product_type") or ""))
            catalog_fields.append(str(it.get("prod_name") or ""))
        if not joined.empty:
            catalog_fields.extend(joined["product_type_name"].fillna("").astype(str).tolist())
            catalog_fields.extend(joined["prod_name"].fillna("").astype(str).tolist())
        ungrounded = [n for n in nouns if not noun_is_grounded(n, catalog_fields)]
        record(
            "1f",
            len(ungrounded) == 0,
            f"nouns_mentioned={nouns} ungrounded={ungrounded}",
        )


# ---------------------------------------------------------------------------
# CHECK 2a -- "More like this" (GET /catalogue/{article_id}/similar)
# ---------------------------------------------------------------------------


async def check_2a_more_like_this(http: httpx.AsyncClient, ws_base: str) -> None:
    """CHECK 2a: the exact endpoint+auth the frontend's api.catalogue.similar() uses.

    frontend/lib/api/client.ts -> api.catalogue.similar(articleId) calls
    GET /catalogue/{article_id}/similar with `Authorization: Bearer <token>`,
    where in the demo flow <token> is the demo session_token from sessionStorage.
    """
    print("\n-- CHECK 2a: 'More like this' (GET /catalogue/{article_id}/similar) --")

    search = await ws_turn_retry(http, ws_base, "black dress for women")
    if not search.items:
        record("2a", False, f"seed WS search returned 0 items (error={search.error})")
        return
    article_id = search.items[0]["article_id"]
    print(f"  seed article_id={article_id}")

    # A fresh demo session's session_token is exactly what the browser sends as
    # the Bearer token here (frontend/lib/api/client.ts getToken() prefers
    # sessionStorage.demo_session_token over the Supabase JWT).
    session = await mint_demo_session(http)
    token = session["session_token"]

    resp = await http.get(
        f"/catalogue/{article_id}/similar",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        record(
            "2a",
            False,
            f"GET /catalogue/{article_id}/similar -> {resp.status_code} "
            f"(this route depends on get_current_user_id, which -- unlike "
            f"get_current_user_id_or_demo used by /style/from-image -- does not special-case "
            f"demo tokens; it always attempts RS256/JWKS verification, which appears to raise "
            f"an unhandled exception on the HS256 demo token rather than a clean 401): "
            f"{resp.text[:200]}",
        )
        return

    items = resp.json()
    with_media = [it for it in items if it.get("image_url") and it.get("pdp_url")]
    record(
        "2a",
        len(items) >= 1 and len(with_media) >= 1,
        f"status=200 n_items={len(items)} n_with_image_and_pdp={len(with_media)}",
    )


# ---------------------------------------------------------------------------
# CHECK 2b / 3 -- "Style this" (WS same-conversation turn)
# ---------------------------------------------------------------------------


async def check_2b_style_this(http: httpx.AsyncClient, ws_base: str) -> WSTurnResult | None:
    """CHECK 2b (+ CHECK 3's budget-sum assertion on this same look)."""
    print("\n-- CHECK 2b: 'Style this' (WS same-conversation turn) --")

    turn1 = await ws_turn_retry(http, ws_base, "white shirt men")
    if not turn1.items:
        record("2b", False, f"turn1 'white shirt men' returned 0 items (error={turn1.error})")
        return None
    prod_name = turn1.items[0]["prod_name"]
    print(f"  turn1 first item prod_name={prod_name!r} conv={turn1.conv_id}")

    turn2 = await ws_turn_retry(http, ws_base, f"Style this {prod_name}", conv_id=turn1.conv_id)
    look_id = turn2.final_state.get("look_id")
    complements = [it for it in turn2.items if it.get("slot_role") == "complement"]

    article_ids = [it.get("article_id") for it in turn2.items if it.get("article_id")]
    joined = join_catalogue(article_ids)
    majority_men = False
    if not joined.empty:
        genders = joined["gender"].fillna("").str.lower()
        majority_men = genders.isin(MEN_LIKE_GENDERS).mean() > 0.5

    ok = turn2.got_done and bool(look_id) and len(complements) >= 2 and majority_men
    record(
        "2b",
        ok,
        f"got_done={turn2.got_done} look_id={look_id} n_items={len(turn2.items)} "
        f"n_complements={len(complements)} genders={sorted(joined['gender'].fillna('N/A').unique()) if not joined.empty else []} "
        f"majority_men={majority_men}",
    )
    return turn2


def check_3_budget_sum(turn2: WSTurnResult | None) -> None:
    """CHECK 3: final_state.budget_total_inr == sum(price_inr) of seed+complement items."""
    print("\n-- CHECK 3: budget_total_inr == sum(price_inr) on the 2b look --")
    if turn2 is None:
        record("3", False, "no turn2 result available (2b did not complete)")
        return
    budget_total_inr = turn2.final_state.get("budget_total_inr")
    relevant = [it for it in turn2.items if it.get("slot_role") in ("seed", "complement")]
    prices = [it.get("price_inr") for it in relevant if it.get("price_inr") is not None]
    price_sum = sum(prices) if prices else None
    ok = (
        budget_total_inr is not None
        and price_sum is not None
        and abs(float(budget_total_inr) - float(price_sum)) <= 1.0
    )
    record(
        "3",
        ok,
        f"budget_total_inr={budget_total_inr} sum(price_inr)={price_sum} "
        f"n_priced={len(prices)}/{len(relevant)}",
    )


# ---------------------------------------------------------------------------
# CHECK 2c -- chips ("Make this look more formal")
# ---------------------------------------------------------------------------


async def check_2c_chips(http: httpx.AsyncClient, ws_base: str) -> None:
    """CHECK 2c: a refinement chip actually rebuilds/refines a look."""
    print("\n-- CHECK 2c: chips ('Make this look more formal') --")

    turn1 = await ws_turn_retry(http, ws_base, "put together a casual look for women")
    if not turn1.got_done:
        record("2c", False, f"turn1 casual-look request did not complete (error={turn1.error})")
        return
    print(f"  turn1 n_items={len(turn1.items)} look_id={turn1.final_state.get('look_id')}")

    turn2 = await ws_turn_retry(http, ws_base, "Make this look more formal", conv_id=turn1.conv_id)
    look_id = turn2.final_state.get("look_id")
    ok = turn2.got_done and len(turn2.items) >= 2 and bool(look_id)
    record(
        "2c",
        ok,
        f"got_done={turn2.got_done} n_items={len(turn2.items)} look_id={look_id}",
    )


# ---------------------------------------------------------------------------
# CHECK 5 -- conversational quality
# ---------------------------------------------------------------------------


async def check_5_conversational(http: httpx.AsyncClient, ws_base: str) -> None:
    """CHECK 5: three conversational turns return items + substantive prose."""
    print("\n-- CHECK 5: conversational quality --")

    turn_a = await ws_turn_retry(http, ws_base, "what goes with white jeans?")
    record(
        "5a",
        len(turn_a.items) >= 1 and len(turn_a.prose.strip()) > 40,
        f"n_items={len(turn_a.items)} prose_len={len(turn_a.prose.strip())} "
        f"got_done={turn_a.got_done} error={turn_a.error} "
        f"prose={turn_a.prose[:100].strip()!r}",
    )

    turn_b = await ws_turn_retry(http, ws_base, "something for a wedding")
    record(
        "5b",
        len(turn_b.items) >= 1 and len(turn_b.prose.strip()) > 40,
        f"n_items={len(turn_b.items)} prose_len={len(turn_b.prose.strip())} "
        f"got_done={turn_b.got_done} error={turn_b.error} "
        f"prose={turn_b.prose[:100].strip()!r}",
    )

    turn_c1 = await ws_turn_retry(http, ws_base, "black dress for women")
    if not turn_c1.items:
        record(
            "5c", False, f"turn_c1 'black dress for women' returned 0 items (error={turn_c1.error})"
        )
        return
    turn_c2 = await ws_turn_retry(http, ws_base, "cheaper options", conv_id=turn_c1.conv_id)

    prices_c1 = [it.get("price_inr") for it in turn_c1.items if it.get("price_inr") is not None]
    prices_c2 = [it.get("price_inr") for it in turn_c2.items if it.get("price_inr") is not None]
    max_c1 = max(prices_c1) if prices_c1 else None
    max_c2 = max(prices_c2) if prices_c2 else None
    ok = (
        len(turn_c2.items) >= 1
        and len(turn_c2.prose.strip()) > 40
        and max_c1 is not None
        and max_c2 is not None
        and max_c2 < max_c1
    )
    record(
        "5c",
        ok,
        f"n_items={len(turn_c2.items)} prose_len={len(turn_c2.prose.strip())} "
        f"max_price_turn1={max_c1} max_price_turn2={max_c2} "
        f"prose={turn_c2.prose[:100].strip()!r}",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

CHECK_IDS: list[str] = ["1", "2a", "2b", "3", "2c", "5"]


async def main() -> None:
    """Run all (or --only-selected) checks against BASE and print a summary table."""
    base, only = _parse_args(sys.argv)
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")

    print(f"Backend : {base}")
    print(f"WS base : {ws_base}")
    if only is not None:
        print(f"Only    : {sorted(only)}")
    print("=" * 78)

    def wanted(cid: str) -> bool:
        return only is None or cid in only

    async with httpx.AsyncClient(base_url=base, timeout=60) as http:
        if wanted("1"):
            await check_1_image_styling(http)
        if wanted("2a"):
            await check_2a_more_like_this(http, ws_base)
        turn2 = None
        if wanted("2b") or wanted("3"):
            turn2 = await check_2b_style_this(http, ws_base)
        if wanted("3"):
            check_3_budget_sum(turn2)
        if wanted("2c"):
            await check_2c_chips(http, ws_base)
        if wanted("5"):
            await check_5_conversational(http, ws_base)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'check_id':<10} {'result':<7} evidence")
    n_red = 0
    for o in OUTCOMES:
        if not o.passed:
            n_red += 1
        print(f"  {o.check_id:<10} {(PASS_TXT if o.passed else FAIL_TXT):<7} {o.evidence[:140]}")
    print("=" * 78)
    print(f"{n_red} RED / {len(OUTCOMES)} total checks")
    sys.exit(1 if n_red else 0)


if __name__ == "__main__":
    asyncio.run(main())
