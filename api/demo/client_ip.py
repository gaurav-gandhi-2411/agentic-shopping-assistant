"""Client-IP extraction for the demo rate limiters.

uvicorn runs with `--proxy-headers --forwarded-allow-ips '*'` (Dockerfile).  With a wildcard
trust list uvicorn sets `request.client.host` to the LEFTMOST X-Forwarded-For entry, which is
whatever the client typed -- trivially spoofable.  This module therefore never uses
`request.client.host` while an X-Forwarded-For entry is available.

Measured Cloud Run header shapes (real captures; <real> is the true peer):
    normal request     X-Forwarded-For: <real>
    single spoof       X-Forwarded-For: <spoof>, <real>
    multi-value spoof  X-Forwarded-For: <s1>, <s2>, <real>
    IPv6 spoof         X-Forwarded-For: <v6 spoof>, <real v4>
Cloud Run APPENDS the real client as the LAST entry, so the trustworthy hop is counted from
the RIGHT: with N trusted proxies in front of the app (Cloud Run alone = 1, the default; behind
an external HTTPS LB = 2) it is entry [-N] (env TRUSTED_PROXY_HOPS).  Every entry to the LEFT
of that one is client-controlled and is never read.  The socket peer on Cloud Run is a private
shared address (all clients would collapse into one key), so it is used only when NO usable
entry exists (direct connection / local dev).

If the selected entry is present but unparseable we do NOT slide to another position (that
would hand the key to the client); we fall through to the socket peer exactly as for a missing
header.

There is no proxy in front of run.app today: the browser calls Cloud Run directly (REST + WS).
If one is ever added (e.g. a Vercel rewrite), the trusted-hop count must be raised for the routes
behind it, or a separate shared-secret-authenticated header introduced per route -- the single
global TRUSTED_PROXY_HOPS would otherwise key every user on the proxy's address.

Accepted entry forms: `1.2.3.4`, `1.2.3.4:5678`, `2001:db8::1`, `[2001:db8::1]`,
`[2001:db8::1]:443`.  IPv6 clients are collapsed to their /64 so rotating the host bits does
not mint new keys.
"""
from __future__ import annotations

import ipaddress
import os

from starlette.requests import HTTPConnection


def _trusted_hops() -> int:
    try:
        return max(1, int(os.environ.get("TRUSTED_PROXY_HOPS", "1")))
    except ValueError:
        return 1


def _strip_port(value: str) -> str:
    """Drop an optional port and IPv6 brackets: '1.2.3.4:80' -> '1.2.3.4', '[::1]:80' -> '::1'."""
    value = value.strip()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value
    if value.count(":") == 1:  # exactly one colon => IPv4:port (bare IPv6 has >= 2)
        return value.split(":", 1)[0]
    return value


def _normalise(value: str) -> str | None:
    try:
        addr = ipaddress.ip_address(_strip_port(value))
    except ValueError:
        return None
    if isinstance(addr, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{addr}/64", strict=False).network_address)
    return str(addr)


def get_client_ip(conn: HTTPConnection) -> str:
    """Return the rate-limit key for a request/websocket (never raises)."""
    header = conn.headers.get("x-forwarded-for", "")
    parts = [p for p in (s.strip() for s in header.split(",")) if p]
    hops = _trusted_hops()
    if len(parts) >= hops:
        # Only the trusted position is ever consulted; garbage there is NOT retried elsewhere.
        ip = _normalise(parts[-hops])
        if ip is not None:
            return ip
    # No usable trusted entry (direct connection / local dev): socket peer as seen by the app.
    host = conn.client.host if conn.client else ""
    return _normalise(host) or host or "0.0.0.0"  # noqa: S104 - sentinel key, not a bind
