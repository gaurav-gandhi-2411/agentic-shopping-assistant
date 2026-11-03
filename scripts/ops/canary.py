"""StyleMaitri production canary (Wave 1, W9 part A).

Drives 3 fixed queries end-to-end through the SAME public path the browser demo uses
(stdlib only, no third-party dependency):

    1. POST /demo/session            -> anon ``session_token``   (frontend/app/demo/page.tsx:88)
    2. POST /auth/ws-ticket (Bearer) -> single-use ``ticket``    (frontend/lib/api/client.ts:116)
    3. WS   /chat/stream?ticket=...  -> send ``user_message``, read frames until ``done``
                                        (frontend/hooks/useChatStream.ts:167-175, frames in
                                        api/schemas.py WS* models)

The frontend chats over a WebSocket, so the canary does too, using a minimal RFC 6455 client
written against ``socket``/``ssl`` (a pinned, hash-locked ``websockets`` install in CI was the
alternative; the stdlib client avoids any dependency for ~80 lines).

Assertions per query (all evaluated; every failure is recorded):
    * transport     - session/ticket HTTP 200 with valid JSON; WS handshake ok; a ``done`` frame
                      arrives (an ``error`` frame, early close or timeout fails).
    * answer        - the streamed ``token`` text is non-empty after strip().
    * cards         - >= 1 card. A card is an entry of the ``items`` frame (what the frontend
                      renders as ``pendingItems``); if no ``items`` frame arrived, the first
                      ``final_state.outfit_variants[*].items`` list is used instead.
    * price         - see PRICE INVARIANT.
    * latency       - ``user_message`` sent -> ``done`` received must be < LATENCY_BUDGET_MS
                      (15 000; exactly equal fails, the spec says "< 15 s"). Cloud Run cold start is absorbed by the (separately timed,
                      generous) /demo/session call, so it does not count against the budget.

PRICE INVARIANT ("every card price present in item data"). The API returns no separate item
database; the only item data is ``ItemSummary`` (api/schemas.py) as it appears in the ``items``
frame and again in ``done.final_state.outfit_variants[*].items`` / ``partner_look.items``. The
answer text is free LLM prose and is NOT parsed for prices. Each non-owned card must therefore have:
    (a) ``price_inr`` present, a real number (not bool/NaN/inf) and > 0;
    (b) a non-empty ``article_id``;
    (c) if the same ``article_id`` appears in another item list of the same response
        (``outfit_variants`` / ``partner_look`` / duplicate card), the prices are EQUAL
        (|diff| < 0.005).
(c) is the strongest cross-check available. LIMITATION, stated plainly: the other item lists
exist only on outfit turns. On a plain search turn (the ``items`` frame alone) nothing else in
the response carries prices, so the check degrades to "price present and > 0" with NO
cross-reference. Each query result records ``crosschecked`` and every run prints a SUMMARY line
with ``crosschecked_total`` so that degradation is visible rather than silently green; failing a
run on ``crosschecked == 0`` is deliberately NOT done (a stronger check against catalogue data
is left to W10/W11). Owned items (``is_owned``: the user's own garment) are not for sale and are
exempt from (a).

BUDGET. Hard cap 150 chat queries/day (UTC). 3 queries x 48 runs = 144 nominal. Before sending
anything, ``run`` counts today's runs of this workflow via ``gh api`` (every run except the
current one and ``skipped`` ones, x 3 queries - an over-count when a run died before querying,
which errs on the side of the cap) and SKIPS (exit 0, not a failure) if this run would exceed
150. A ``force_fail`` dispatch may exceed the cap by one run (allowed when spent + 3 <= 150 + 3,
i.e. 3 extra queries); if even that is exceeded it prints ``::error::`` and exits 1 so the
alert-path proof can never silently not happen.
Cap lookup: the listing must contain the current run id (empty/partial output is a lookup
failure, never spent=0; retried 3x). A failed lookup skips without querying (fail closed on
spend). The consecutive-failure count lives in ``--state-file`` persisted by actions/cache (no
secrets); the 3rd consecutive failure turns the run into ``fail`` (goes through the issue path),
so a canary that cannot check its budget is not silently green forever. If the cache is lost the
count restarts at 0 (alert delayed, never a false pass of the queries themselves). Each query costs
about 2 Groq LLM calls (measured for "black kurta set for haldi" in PR #79) -> roughly 290 of
the free tier's 1,000 requests/day. Note: anonymous demo sessions also consume the shared demo
daily cap (api/demo/guards.py), once per WS connection.

CAVEAT: GitHub's scheduled cron is unreliable for this repo (the */15 uptime cron's median gap
was 246 min), so */30 here does not guarantee 30-minute coverage.

Demo caps: an ``error`` frame with code ``demo_limit``/``rate_limited`` (api/routes/chat.py) or
HTTP 429 from /demo/session is not an outage: the query is ``capped``, the run is ``capped`` (a
``::warning::``, exit 0) - neither a pass nor a failure, and the issue step does nothing.
Each response read is bounded by one overall deadline (30 s) even if the server trickles bytes,
and frames over 8 MiB are rejected; any unexpected exception is reported as ``unexpected:
<TypeName>`` and a result file with status ``fail`` is ALWAYS written, so a crash still alerts.

Issue lifecycle (``issue`` subcommand, pure decision in ``decide_issue_action``): on failure
comment on the open ``prod-alert`` issue titled exactly ALERT_TITLE or create one; on pass close
those with a recovery comment. Other ``prod-alert`` issues (e.g. a future part-B alert) are
never touched. Issue text never contains tokens/headers: bearer tokens, JWTs and ticket/token
values are redacted by ``redact`` and response snippets are cut to 500 chars.

ERROR-log-count check (Cloud Run, via WIF) is part B and deliberately absent here.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import math
import os
import re
import secrets
import socket
import ssl
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://asa-stylist-api-ec3mwppesa-el.a.run.app"
QUERIES: tuple[str, ...] = (
    "sangeet outfit under ₹3000 for women",
    "black kurta set for haldi",
    "formal shirts for office under 1500",
)
LATENCY_BUDGET_MS = 15_000
WS_HARD_TIMEOUT_S = 30.0  # bound CI time/cost; the 15 s latency assertion fires well before
HTTP_TIMEOUT_S = 75.0  # generous: Cloud Run cold starts measured at 45-60 s+ (uptime-alert.yml)
DAILY_QUERY_CAP = 150
SNIPPET_CHARS = 500
ALERT_LABEL = "prod-alert"
ALERT_TITLE = "StyleMaitri prod canary is failing"
MAX_FRAME_BYTES = 8 * 1024 * 1024  # a hostile/garbled length must not allocate GBs
CAP_CODES = frozenset({"demo_limit", "rate_limited"})  # api/routes/chat.py WSErrorMessage codes
LOOKUP_FAIL_LIMIT = 3  # consecutive cap-lookup failures before the canary reports itself dead
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# --------------------------------------------------------------------------- redaction


_REDACTIONS = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"), "[jwt redacted]"),
    (re.compile(r"(?i)((?:ticket|token|session_token|authorization)[\"']?\s*[=:]\s*[\"']?)"
                r"[A-Za-z0-9._~+/=-]{6,}"), r"\1[redacted]"),
)


def redact(text: str) -> str:
    """Strip anything token-, ticket-, header- or URL-shaped from ``text``."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Redact then truncate to ``limit`` chars (redact first so a cut cannot split a token)."""
    return redact(text)[:limit]


# --------------------------------------------------------------------------- HTTP


class CanaryError(Exception):
    """A transport/protocol failure; the message is safe to publish (no URLs/tokens)."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


def http_post_json(
    url: str, headers: dict[str, str] | None = None, timeout: float = HTTP_TIMEOUT_S
) -> dict[str, Any]:
    """POST an empty body, require HTTP 200 and a JSON object. Raises CanaryError otherwise."""
    req = urllib.request.Request(  # noqa: S310 - scheme validated by caller (http/https)
        url, data=b"", method="POST", headers=headers or {}
    )
    path = urllib.parse.urlsplit(url).path
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            status, body = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        raise CanaryError(
            f"POST {path} returned HTTP {exc.code}: {snippet(_read(exc), 200)}", exc.code
        ) from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise CanaryError(f"POST {path} failed: {type(exc).__name__}") from None
    if status != 200:
        raise CanaryError(f"POST {path} returned HTTP {status}")
    try:
        data = json.loads(body)
    except ValueError:
        raise CanaryError(f"POST {path} returned malformed JSON: {snippet(body.decode('utf-8', 'replace'), 200)}") from None
    if not isinstance(data, dict):
        raise CanaryError(f"POST {path} returned JSON that is not an object")
    return data


def _read(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - best-effort body for the alert only
        return ""


# --------------------------------------------------------------------------- WebSocket


class WsClosed(Exception):
    """The peer closed the socket / sent a close frame before the caller was done."""


def _recv_exact(sock: socket.socket, n: int, deadline: float | None = None) -> bytes:
    """Read exactly n bytes; ``deadline`` (perf_counter) bounds the WHOLE read, not each recv."""
    buf = b""
    while len(buf) < n:
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError
            sock.settimeout(max(0.01, remaining))
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise WsClosed("connection closed")
        buf += chunk
    return buf


def ws_connect(url: str, timeout: float) -> socket.socket:
    """Open a ws:// or wss:// connection and complete the RFC 6455 opening handshake."""
    parts = urllib.parse.urlsplit(url)
    secure = parts.scheme == "wss"
    host = parts.hostname or ""
    port = parts.port or (443 if secure else 80)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    try:
        sock: socket.socket = socket.create_connection((host, port), timeout=timeout)
        if secure:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        sock.sendall(
            (
                f"GET {target} HTTP/1.1\r\nHost: {parts.netloc}\r\nUpgrade: websocket\r\n"
                f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        head = b""
        hs_deadline = time.perf_counter() + timeout
        while b"\r\n\r\n" not in head:
            if time.perf_counter() > hs_deadline:
                raise TimeoutError
            chunk = sock.recv(1)  # byte-wise so no frame bytes after the headers are swallowed
            if not chunk:
                raise WsClosed("closed during handshake")
            head += chunk
            if len(head) > 65536:
                raise CanaryError("WS handshake response too large")
    except (OSError, ssl.SSLError, TimeoutError, WsClosed) as exc:
        raise CanaryError(f"WS connect failed: {type(exc).__name__}") from None
    lines = head.decode("latin-1").split("\r\n")
    if " 101 " not in lines[0] + " ":
        raise CanaryError(f"WS handshake rejected: {redact(lines[0])[:80]}")
    expected = base64.b64encode(sha1((key + WS_GUID).encode()).digest()).decode()  # noqa: S324
    accept = next(
        (ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("sec-websocket-accept")),
        "",
    )
    if accept != expected:
        raise CanaryError("WS handshake: bad Sec-WebSocket-Accept")
    return sock


def ws_send_text(sock: socket.socket, text: str) -> None:
    """Send one masked text frame (client frames must be masked)."""
    payload = text.encode()
    header = bytearray([0x81])
    n = len(payload)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", n)
    mask = secrets.token_bytes(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(bytes(header) + mask + masked)


def ws_recv_text(sock: socket.socket, deadline: float | None = None) -> str:
    """Return the next complete text message; answers pings; raises WsClosed on close.

    ``deadline`` is an absolute perf_counter time bounding the whole message, so a server that
    trickles one byte at a time cannot stretch the wait past it (raises TimeoutError).
    """
    message = b""
    while True:
        b0, b1 = _recv_exact(sock, 2, deadline)
        fin, opcode = b0 & 0x80, b0 & 0x0F
        length = b1 & 0x7F
        if length == 126:
            (length,) = struct.unpack(">H", _recv_exact(sock, 2, deadline))
        elif length == 127:
            (length,) = struct.unpack(">Q", _recv_exact(sock, 8, deadline))
        if length > MAX_FRAME_BYTES or len(message) + length > MAX_FRAME_BYTES:
            raise CanaryError(f"WS frame length {length} exceeds {MAX_FRAME_BYTES} byte cap")
        masked = bool(b1 & 0x80)
        mask = _recv_exact(sock, 4, deadline) if masked else b""
        payload = _recv_exact(sock, length, deadline)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:
            code = struct.unpack(">H", payload[:2])[0] if len(payload) >= 2 else 1005
            raise WsClosed(f"close frame code {code}")
        if opcode == 0x9:  # ping -> pong (masked, same payload)
            m = secrets.token_bytes(4)
            sock.sendall(
                bytes([0x8A, 0x80 | len(payload)]) + m
                + bytes(b ^ m[i % 4] for i, b in enumerate(payload))
            )
            continue
        if opcode == 0xA:
            continue
        message += payload
        if fin:
            return message.decode("utf-8", "replace")


# --------------------------------------------------------------------------- assertions


@dataclasses.dataclass
class QueryResult:
    """Outcome of one canary query. Contains no URLs, tokens or headers."""

    query: str
    latency_ms: float | None = None
    answer: str = ""
    cards: int = 0
    crosschecked: int = 0
    failures: list[str] = dataclasses.field(default_factory=list)
    # Server said demo/rate limit hit (CAP_CODES / HTTP 429): not an outage, not a pass.
    capped: bool = False

    @property
    def ok(self) -> bool:
        return not self.failures and not self.capped


def _is_price(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def collect_cards(items_frame: list[dict[str, Any]] | None, final_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Cards the user sees: the ``items`` frame, else the first outfit variant's items."""
    if items_frame:
        return [c for c in items_frame if isinstance(c, dict)]
    variants = _as_list(final_state.get("outfit_variants"))
    if variants and isinstance(variants[0], dict):
        return [c for c in _as_list(variants[0].get("items")) if isinstance(c, dict)]
    return []


def _as_list(value: Any) -> list[Any]:
    """Tolerate hostile shapes: anything that is not a list is treated as empty."""
    return value if isinstance(value, list) else []


def reference_prices(final_state: dict[str, Any]) -> list[tuple[str, Any]]:
    """(article_id, price_inr) pairs from the other item lists in the same response."""
    pairs: list[tuple[str, Any]] = []
    lists: list[Any] = [v.get("items") for v in _as_list(final_state.get("outfit_variants"))
                        if isinstance(v, dict)]
    partner = final_state.get("partner_look")
    if isinstance(partner, dict):
        lists.append(partner.get("items"))
    for lst in lists:
        for it in _as_list(lst):
            if isinstance(it, dict) and it.get("article_id"):
                pairs.append((str(it["article_id"]), it.get("price_inr")))
    return pairs


def check_prices(cards: list[dict[str, Any]], refs: list[tuple[str, Any]]) -> tuple[list[str], int]:
    """Apply the PRICE INVARIANT. Returns (failure messages, number of cards cross-checked)."""
    failures: list[str] = []
    seen: dict[str, Any] = {}
    others: dict[str, list[Any]] = {}
    for aid, price in refs:
        others.setdefault(aid, []).append(price)
    crosschecked = 0
    for idx, card in enumerate(cards):
        if card.get("is_owned"):
            continue
        aid = card.get("article_id")
        price = card.get("price_inr")
        if not isinstance(aid, str) or not aid:
            failures.append(f"card[{idx}] has no article_id")
            continue
        if not _is_price(price):
            failures.append(f"card {aid} price_inr invalid: {price!r}")
            continue
        compare = others.get(aid, []) + ([seen[aid]] if aid in seen else [])
        if compare:
            crosschecked += 1
        for other in compare:
            if not _is_price(other) or abs(float(other) - float(price)) >= 0.005:
                failures.append(f"card {aid} price {price!r} != item data price {other!r}")
                break
        seen.setdefault(aid, price)
    return failures, crosschecked


def evaluate(
    query: str,
    *,
    answer: str,
    items_frame: list[dict[str, Any]] | None,
    final_state: dict[str, Any] | None,
    latency_ms: float,
    budget_ms: float,
    protocol_error: str | None = None,
) -> QueryResult:
    """Pure assertion logic for one query's collected response."""
    res = QueryResult(query=query, latency_ms=round(latency_ms, 1), answer=answer)
    if protocol_error:
        res.failures.append(protocol_error)
    if final_state is None:
        return res  # nothing further is meaningful without a done frame
    if not answer.strip():
        res.failures.append("answer is empty")
    cards = collect_cards(items_frame, final_state)
    res.cards = len(cards)
    if not cards:
        res.failures.append("no product cards returned")
    price_failures, res.crosschecked = check_prices(cards, reference_prices(final_state))
    res.failures += price_failures
    if latency_ms >= budget_ms:  # spec: latency must be < 15 s
        res.failures.append(f"latency {latency_ms:.0f} ms exceeds budget {budget_ms:.0f} ms")
    return res


# --------------------------------------------------------------------------- one query


def run_query(
    base_url: str,
    session_token: str,
    query: str,
    budget_ms: float,
    ws_timeout_s: float = WS_HARD_TIMEOUT_S,
) -> QueryResult:
    """Run ws-ticket -> WS chat for ``query`` and evaluate it. Never raises CanaryError."""
    base = base_url.rstrip("/")
    try:
        ticket = http_post_json(
            f"{base}/auth/ws-ticket",
            {"Content-Type": "application/json", "Authorization": f"Bearer {session_token}"},
        ).get("ticket")
        if not isinstance(ticket, str) or not ticket:
            raise CanaryError("POST /auth/ws-ticket response has no ticket")
        ws_url = base.replace("http", "ws", 1) + "/chat/stream?" + urllib.parse.urlencode(
            {"ticket": ticket}
        )
    except CanaryError as exc:
        return evaluate(query, answer="", items_frame=None, final_state=None, latency_ms=0,
                        budget_ms=budget_ms, protocol_error=str(exc))

    tokens: list[str] = []
    items_frame: list[dict[str, Any]] | None = None
    final_state: dict[str, Any] | None = None
    error: str | None = None
    capped = False
    started = time.perf_counter()
    sock: socket.socket | None = None
    try:
        sock = ws_connect(ws_url, ws_timeout_s)
        started = time.perf_counter()  # latency = user_message sent -> done received
        ws_send_text(sock, json.dumps({"type": "user_message", "conversation_id": None,
                                       "message": query}))
        deadline = started + ws_timeout_s
        while True:
            frame = json.loads(ws_recv_text(sock, deadline))
            if not isinstance(frame, dict):
                error = "WS frame is not a JSON object"
                break
            ftype = frame.get("type")
            if ftype == "items":
                raw_items = frame.get("items")
                if not isinstance(raw_items, list):
                    error = "items frame has non-list items"
                    break
                items_frame = raw_items
            elif ftype == "token":
                tokens.append(str(frame.get("text", "")))
            elif ftype == "done":
                final_state = frame.get("final_state")
                if not isinstance(final_state, dict):
                    final_state = None
                    error = "done frame final_state is not an object"
                break
            elif ftype in ("error", "cancelled"):
                if ftype == "error" and frame.get("code") in CAP_CODES:
                    capped = True  # demo/rate limit: not an outage (see QueryResult.capped)
                else:
                    error = (f"server sent {ftype} frame: "
                             f"{snippet(str(frame.get('message') or frame.get('code') or ''), 200)}")
                break
    except WsClosed as exc:
        error = f"WS closed before done frame ({exc})"
    except (TimeoutError, socket.timeout):
        error = f"no done frame within {ws_timeout_s:.0f} s"
    except ValueError:
        error = "WS frame was not valid JSON"
    except CanaryError as exc:
        error = str(exc)
    except (OSError, ssl.SSLError) as exc:
        error = f"WS I/O error: {type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - incl. MemoryError: a crash must still alert
        error = f"unexpected: {type(exc).__name__}"
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
    try:
        res = evaluate(query, answer="".join(tokens), items_frame=items_frame,
                       final_state=final_state, latency_ms=latency_ms, budget_ms=budget_ms,
                       protocol_error=error)
    except Exception as exc:  # noqa: BLE001 - hostile response shape must not skip the alert
        res = QueryResult(query=query, latency_ms=round(latency_ms, 1),
                          failures=[f"unexpected: {type(exc).__name__}"])
    res.capped = capped
    return res


# --------------------------------------------------------------------------- budget cap


def count_queries_today(run_rows: list[tuple[str, str, str]], current_run_id: str) -> int:
    """Queries already spent today: 3 per workflow run other than this one, excluding skipped.

    ``run_rows`` are (id, status, conclusion). Over-counts runs that died before querying; that
    only makes the cap stricter.
    """
    prior = [r for r in run_rows if r[0] != current_run_id and r[2] != "skipped"]
    return len(prior) * len(QUERIES)


def would_exceed_cap(spent: int, cap: int = DAILY_QUERY_CAP, forced: bool = False) -> bool:
    """True if running another len(QUERIES) queries would pass ``cap``.

    A forced-failure run (the alert-path proof) may exceed the cap by one run: it is allowed
    when spent + 3 <= cap + 3. That costs at most 3 extra queries.
    """
    allowed = cap + (len(QUERIES) if forced else 0)
    return spent + len(QUERIES) > allowed


GhRunner = Callable[[list[str], str | None], str]


def gh(args: list[str], stdin: str | None = None) -> str:
    """Run the gh CLI (auth via GH_TOKEN); raises CanaryError with a redacted message."""
    try:
        out = subprocess.run(  # noqa: S603
            ["gh", *args], input=stdin, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CanaryError(f"gh failed: {type(exc).__name__}") from None
    if out.returncode != 0:
        raise CanaryError(f"gh {args[0]} failed: {snippet(out.stderr, 200)}")
    return out.stdout


def fetch_run_rows(
    repo: str, workflow: str, today: str, runner: GhRunner = gh
) -> list[tuple[str, str, str]]:
    """List today's (UTC) runs of ``workflow`` via the workflow-runs API."""
    out = runner(
        ["api", "--method", "GET", f"repos/{repo}/actions/workflows/{workflow}/runs",
         "-f", f"created=>={today}", "-F", "per_page=100", "--paginate",
         "--jq", r'.workflow_runs[] | [(.id|tostring), .status, (.conclusion // "")] | @tsv'],
        None,
    )
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def lookup_spent(
    repo: str, workflow: str, today: str, run_id: str,
    runner: GhRunner = gh, attempts: int = 3, pause_s: float = 5.0,
) -> int:
    """Queries spent today, or CanaryError.

    The listing must contain the CURRENT run (it is in_progress and created today); empty or
    partial output without it is treated as a lookup failure, never as spent=0. Retried a few
    times because a just-created run can lag in the listing.
    """
    last = "unknown"
    for attempt in range(attempts):
        if attempt:
            time.sleep(pause_s)
        try:
            rows = fetch_run_rows(repo, workflow, today, runner)
        except CanaryError as exc:
            last = str(exc)
            continue
        if run_id and any(r[0] == run_id for r in rows):
            return count_queries_today(rows, run_id)
        last = f"current run {run_id!r} not in workflow-runs listing ({len(rows)} rows)"
    raise CanaryError(last)


def read_fail_count(path: str) -> int:
    """Consecutive cap-lookup failures recorded by earlier runs (0 if absent/garbled)."""
    try:
        return max(0, int(Path(path).read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return 0


def decide_lookup_failure(prev_count: int, forced: bool) -> tuple[str, int]:
    """Outcome of a failed cap lookup -> (status, new consecutive-failure count).

    skip  : do not query (protect budget), run stays green, count+1
    fail  : the Nth consecutive failure - the canary is effectively dead, so alert
    error : a forced run cannot prove anything while skipped; exit non-zero (status stays skip)

    The count is persisted across runs by actions/cache (no secrets): see canary.yml.
    """
    if forced:
        return "error", prev_count
    count = prev_count + 1
    return ("fail" if count >= LOOKUP_FAIL_LIMIT else "skip"), count


# --------------------------------------------------------------------------- issue lifecycle


@dataclasses.dataclass(frozen=True)
class IssueAction:
    """What to do with prod-alert issues: create | comment | close | none."""

    kind: str
    numbers: tuple[int, ...] = ()


def decide_issue_action(status: str, open_issues: list[int]) -> IssueAction:
    """Pure issue-lifecycle decision. ``status`` is pass | fail | skip | capped.

    Only ``fail`` opens/comments and only ``pass`` closes; skip and capped never touch issues.
    """
    if status == "fail":
        if open_issues:
            return IssueAction("comment", (min(open_issues),))
        return IssueAction("create")
    if status == "pass" and open_issues:
        return IssueAction("close", tuple(sorted(open_issues)))
    return IssueAction("none")


def build_issue_body(result: dict[str, Any], run_url: str, heading: str) -> str:
    """Markdown body: query, failed assertion, latency, run URL, <=500-char snippet."""
    lines = [f"{heading} at {result.get('timestamp', '')}", ""]
    if result.get("note"):
        lines += [f"**Failure:** {redact(str(result['note']))[:300]}", ""]
    for q in result.get("queries", []):
        if q.get("failures"):
            lines += [
                f"**Query:** `{q['query']}`",
                f"**Failed assertion(s):** {'; '.join(q['failures'])}",
                f"**Latency:** {q.get('latency_ms')} ms (budget {result.get('budget_ms')} ms)",
                f"**Response snippet (max {SNIPPET_CHARS} chars):**",
                "```",
                snippet(q.get("answer", ""), SNIPPET_CHARS).replace("```", "'''"),
                "```",
                "",
            ]
    if result.get("forced"):
        lines += ["_This was a forced-failure run (`force_fail=true`); it is a test of the alert path._", ""]
    lines.append(f"[Workflow run]({run_url})")
    return "\n".join(lines)


def apply_issue_action(repo: str, result: dict[str, Any], run_url: str, runner: GhRunner = gh) -> IssueAction:
    """Execute the lifecycle against GitHub via gh (needs only ``issues: write``).

    Only open issues with the canary's OWN fixed title are considered, so other `prod-alert`
    issues (e.g. a future part-B log-count alert) are never commented on or closed here.
    """
    status = result.get("status", "skip")
    listed = runner(["issue", "list", "--repo", repo, "--label", ALERT_LABEL, "--state", "open",
                     "--limit", "100", "--json", "number,title", "--jq",
                     r'.[] | [(.number|tostring), .title] | @tsv'], None)
    open_issues = []
    for line in listed.splitlines():
        num, _, title = line.partition("\t")
        if title.strip() == ALERT_TITLE and num.isdigit():
            open_issues.append(int(num))
    action = decide_issue_action(status, open_issues)
    if action.kind == "create":
        labels = runner(["label", "list", "--repo", repo, "--limit", "200", "--json", "name",
                         "--jq", ".[].name"], None).split("\n")
        if ALERT_LABEL not in labels:
            runner(["label", "create", ALERT_LABEL, "--repo", repo, "--color", "B60205",
                    "--description", "Production alert opened by the canary"], None)
        runner(["issue", "create", "--repo", repo, "--title", ALERT_TITLE, "--label", ALERT_LABEL,
                "--body-file", "-"], build_issue_body(result, run_url, "## Canary failed\n\nDetected"))
    elif action.kind == "comment":
        runner(["issue", "comment", str(action.numbers[0]), "--repo", repo, "--body-file", "-"],
               build_issue_body(result, run_url, "Still failing"))
    elif action.kind == "close":
        for n in action.numbers:
            runner(["issue", "comment", str(n), "--repo", repo, "--body-file", "-"],
                   f"**Recovered** at {result.get('timestamp', '')}: all canary queries passed. "
                   f"[Workflow run]({run_url})")
            runner(["issue", "close", str(n), "--repo", repo, "--reason", "completed"], None)
    return action


# --------------------------------------------------------------------------- CLI


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def summarize(results: list[QueryResult]) -> str:
    """One-line per-run summary, including how many cards were actually cross-checked."""
    return (f"SUMMARY: queries={len(results)} passed={sum(r.ok for r in results)} "
            f"capped={sum(r.capped for r in results)} "
            f"cards={sum(r.cards for r in results)} "
            f"crosschecked_total={sum(r.crosschecked for r in results)}")


def classify(results: list[QueryResult]) -> str:
    """fail if any real failure; else capped if any query hit a demo limit; else pass."""
    if any(r.failures for r in results):
        return "fail"
    if any(r.capped for r in results):
        return "capped"
    return "pass"


def _write(path: str, result: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def _run_inner(args: argparse.Namespace, result: dict[str, Any]) -> int:
    budget = result["budget_ms"]
    forced = bool(args.force_fail)
    state_file = getattr(args, "state_file", "")
    if args.repo:
        prev = read_fail_count(state_file) if state_file else 0
        try:
            spent = lookup_spent(args.repo, args.workflow, now_utc().strftime("%Y-%m-%d"),
                                 args.run_id)
        except CanaryError as exc:
            verdict, count = decide_lookup_failure(prev, forced)
            result["note"] = f"cap lookup failed ({count} consecutive): {exc}"
            if state_file:
                Path(state_file).write_text(str(count), encoding="utf-8")
            if verdict == "skip":
                result["status"] = "skip"
                print(f"::warning::Canary skipped, cap lookup failed ({count}/{LOOKUP_FAIL_LIMIT}"
                      f" before it alerts): {exc}")
                return 0
            if verdict == "error":
                result["status"] = "skip"
                print(f"::error::force_fail run could NOT run (cap lookup failed): {exc}")
                return 1
            result["status"] = "fail"
            print(f"::error::Canary dead: cap lookup failed {count} runs in a row: {exc}")
            return 1
        if state_file:
            Path(state_file).write_text("0", encoding="utf-8")
        if would_exceed_cap(spent, forced=forced):
            result["status"] = "skip"
            result["note"] = f"daily cap: {spent} queries already spent today (cap {DAILY_QUERY_CAP})"
            if forced:
                print(f"::error::force_fail run skipped by the daily cap - {result['note']}")
                return 1
            print(f"::notice::Canary skipped - {result['note']}")
            return 0
    base = args.base_url.rstrip("/")
    results: list[QueryResult] = []
    try:
        token = http_post_json(f"{base}/demo/session").get("session_token")
        if not isinstance(token, str) or not token:
            raise CanaryError("POST /demo/session response has no session_token")
    except CanaryError as exc:
        first = evaluate(QUERIES[0], answer="", items_frame=None, final_state=None,
                         latency_ms=0, budget_ms=budget, protocol_error=str(exc))
        if exc.http_status == 429:  # frontend treats 429 as "daily limit", not an outage
            first.failures.clear()
            first.capped = True
        results = [first]
    else:
        results = [run_query(base, token, q, budget) for q in QUERIES]
    result["queries"] = [dataclasses.asdict(r) | {"ok": r.ok, "answer": snippet(r.answer)}
                         for r in results]
    result["status"] = classify(results)
    result["crosschecked_total"] = sum(r.crosschecked for r in results)
    for r in results:
        print(f"{'PASS' if r.ok else 'CAPPED' if r.capped else 'FAIL'} {r.query!a}: "
              f"latency={r.latency_ms} ms cards={r.cards} crosschecked={r.crosschecked} "
              f"answer_chars={len(r.answer)}"
              + ("" if not r.failures else f" failures={ascii(r.failures)}"))
    print(summarize(results))
    if result["status"] == "capped":
        print("::warning::Demo daily/rate cap hit - canary inconclusive (not a pass, no alert change)")
    return 1 if result["status"] == "fail" else 0


def run_canary(args: argparse.Namespace) -> int:
    """`run` subcommand: cap check, 3 queries, ALWAYS write a result JSON. Exit 1 on failure."""
    budget = -1.0 if args.force_fail else float(LATENCY_BUDGET_MS)
    # Default to "fail": if anything below crashes, the issue step must still see a failure.
    result: dict[str, Any] = {"status": "fail", "timestamp": now_utc().isoformat(timespec="seconds"),
                              "budget_ms": budget, "forced": bool(args.force_fail), "queries": []}
    rc = 1
    try:
        rc = _run_inner(args, result)
    except BaseException as exc:  # noqa: BLE001 - incl. MemoryError/KeyboardInterrupt
        result["status"] = "fail"
        result["note"] = f"unexpected: {type(exc).__name__}"
        print(f"::error::canary crashed: {type(exc).__name__}")
        rc = 1
    finally:
        _write(args.out, result)
    return rc


def issue_cmd(args: argparse.Namespace) -> int:
    """`issue` subcommand: apply the prod-alert lifecycle from a result file."""
    path = Path(args.result)
    if not path.exists():
        print("::notice::no canary result file; nothing to do")
        return 0
    result = json.loads(path.read_text(encoding="utf-8"))
    action = apply_issue_action(args.repo, result, args.run_url)
    print(f"issue action: {action.kind} {list(action.numbers)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", default=DEFAULT_BASE_URL)
    r.add_argument("--out", default="canary-result.json")
    r.add_argument("--force-fail", action="store_true",
                   help="assert a -1 ms latency budget (impossible: even 0.0 ms exceeds it) without extra LLM calls")
    r.add_argument("--repo", default="", help="owner/name; enables the daily-cap check via gh")
    r.add_argument("--workflow", default="canary.yml")
    r.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    r.add_argument("--state-file", default="",
                   help="file holding the consecutive cap-lookup-failure count (actions/cache)")
    r.set_defaults(func=run_canary)
    i = sub.add_parser("issue")
    i.add_argument("--result", default="canary-result.json")
    i.add_argument("--repo", required=True)
    i.add_argument("--run-url", required=True)
    i.set_defaults(func=issue_cmd)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
