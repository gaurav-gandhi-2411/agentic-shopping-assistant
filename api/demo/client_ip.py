"""Client-IP extraction for the demo rate limiters.

uvicorn runs with `--proxy-headers --forwarded-allow-ips '*'` (Dockerfile).  With a wildcard
trust list uvicorn sets `request.client.host` to the LEFTMOST X-Forwarded-For entry, which is
whatever the client typed -- trivially spoofable, so every request could claim a fresh IP and
dodge the per-IP limit.  Cloud Run's front end APPENDS the real peer address to the header,
so the trustworthy hop is counted from the RIGHT: with N trusted proxies in front of the app
(Cloud Run alone = 1, the default; behind an external HTTPS LB = 2) it is entry [-N].

IPv6 clients are collapsed to their /64 so rotating the host bits does not mint new keys.
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


def _normalise(value: str) -> str | None:
    try:
        addr = ipaddress.ip_address(value.strip())
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
        ip = _normalise(parts[-hops])
        if ip is not None:
            return ip
    # No usable header (direct connection / local dev): socket peer as seen by the app.
    host = conn.client.host if conn.client else ""
    return _normalise(host) or host or "0.0.0.0"  # noqa: S104 - sentinel key, not a bind
