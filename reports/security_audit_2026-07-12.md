# Report 7 — Security Audit

Compiled 2026-07-12. Every finding sourced to a file/line or a live command run this
session. Findings ordered most → least important.

## HIGH — Demo rate limits are still raised for testing (revert before public launch)

Live-verified via `gcloud run services describe`:
```
DEMO_PER_IP_HOUR_LIMIT=1000    (code default: 10   — api/demo/guards.py:22)
DEMO_DAILY_REQUEST_CAP=5000    (code default: 200  — api/demo/guards.py:29)
```
The code's own comment is explicit: *"set it on the Cloud Run service to raise the
limit for testing, then remove/reset it before opening to the public."* This is a
100x/25x relaxation from the coded defaults, still active. **Action: unset both env
vars (or set to the intended public values) before any public launch/traffic push.**
This is the same item flagged in the pre-public hardening checklist earlier this
session and has not yet been reverted.

There is also a separate `RATE_LIMIT_PER_MINUTE` (default 10/min, `api/rate_limit.py`)
covering a different code path — not observed overridden on the live service, but
worth double-checking it isn't set alongside the demo limits.

## MEDIUM — `ws` npm package has a HIGH-severity advisory (memory disclosure + DoS)

`npm audit --audit-level=moderate` (frontend/, this session): 10 vulnerabilities (1
high, 8 moderate, 1 low). The high one:
```
ws  8.0.0 - 8.20.1  (transitive dependency)
  - Uninitialized memory disclosure (GHSA-58qx-3vcg-4xpx)
  - Memory exhaustion DoS from tiny fragments (GHSA-96hv-2xvq-fx4p)
  fix available via `npm audit fix` (non-breaking)
```
Also moderate: `@babel/core` (arbitrary file read via sourcemap comment, non-breaking
fix available), `@opentelemetry/core` (unbounded memory allocation, pulled in via
`@sentry/nextjs`, non-breaking fix available), `postcss` (XSS via unescaped
`</style>`, **only fixable via `npm audit fix --force`, which downgrades `next` to a
pre-9.4 canary — almost certainly a false/over-broad advisory range, not a real
downgrade path; do not force this one blindly**).

**Note on exposure**: `ws` is very likely a server-side/build-time transitive
dependency here (this is a browser-rendered Next.js app; client WebSocket usage goes
through the browser's native `WebSocket` API, not the npm `ws` package) — worth
confirming its actual runtime reachability before treating this as directly
exploitable, but it should still be patched via `npm audit fix` (non-breaking) as
routine hygiene. **This audit did not modify any dependencies — reporting only, per
this item's scope.**

`pip-audit -r requirements.txt`: **zero known vulnerabilities.**

## LOW/INFORMATIONAL — Single-instance-only in-memory state (documented, not hidden)

Two components explicitly self-document as "single Cloud Run instance only":
- `api/rate_limit.py`: sliding-window counter is in-process memory; the module
  docstring says "replace with a Redis-backed limiter for multi-instance."
- `api/auth.py`'s WS ticket store: in-process dict, same caveat.

Not a live vulnerability today (min-instances config and current traffic don't
require multi-instance), but flagged here because it becomes a real gap the moment
this service scales beyond one instance — rate limits and ticket validation would
silently become per-instance rather than global, which could let a single client
exceed the intended global rate by hitting different instances. Worth a ticket, not
an emergency.

## GOOD — Secrets management

No `.env` files committed (only `.env.example`/`.env.local.example` templates;
`.gitignore` correctly excludes real `.env*`). Live-verified via `gcloud run services
describe`: all five sensitive values (`OPENROUTER_API_KEY`, `GROQ_API_KEY`,
`DEMO_JWT_SECRET`, `DATABASE_URL`, `SUPABASE_URL`) are sourced from GCP Secret Manager
(`secretKeyRef`), not plain env vars. `api/demo/session.py` has a hardcoded fallback
JWT secret (`"dev-demo-secret-CHANGE-ME"`) for local dev when `DEMO_JWT_SECRET` is
unset — confirmed NOT reachable in production (the real secret is properly wired).

## GOOD — WS auth design

`api/auth.py`: RS256 JWT verification against Supabase's JWKS endpoint (no shared
signing secret held by the backend). WebSocket auth uses a short-lived (60s), one-use,
32-byte-entropy nonce ticket (`POST /auth/ws-ticket` → `?ticket=` on the WS URL)
specifically so the full JWT never appears in server access logs — a deliberate,
well-reasoned design, not an accident.

## GOOD — CORS

`api/main.py`: `CORS_ORIGINS` env var restricts allowed origins (production value:
exactly `https://stylemaitri.vercel.app`, live-verified — not a wildcard). Combined
correctly with `allow_credentials=True` (browsers reject wildcard-origin +
credentials combinations anyway, so this couldn't have been silently misconfigured
without breaking functionally).

## GOOD — Debug endpoint gating

`/sentry-debug` (deliberately raises an exception to verify Sentry ingestion) is
gated behind `SENTRY_DEBUG_ENDPOINT` env var, 404s otherwise. **Live-verified**:
`curl https://asa-stylist-api-.../sentry-debug` → `404` in production.

## GOOD — Cost guard

`api/demo/guards.py`: a $0.50/brand/UTC-day cost cap, enforced via a Postgres
`demo_daily_stats` row as the source of truth (survives process restarts), with an
in-memory fast-path check to avoid a DB round-trip on every message. A reasonable,
cost-aware design matching this project's stated cost-conscious defaults.

## NOT DEEPLY AUDITED THIS PASS (flagged, not claimed clean)

- **Input validation**: Pydantic models are used at the API boundary (confirmed by
  spot-check of route signatures) but a full parameter-by-parameter injection/
  boundary-value audit across every endpoint was not performed this session — that
  would be its own dedicated pass, not squeezed into this multi-item turn.
- **SQL injection**: a targeted grep for f-string-interpolated SQL (`text(f"..."`,
  `.execute(f"..."` containing SELECT/INSERT/UPDATE/DELETE) across `api/` and
  `src/storage/` found **zero matches** — a good sign, but this is a pattern-match
  spot-check, not a full parameter-by-parameter audit of every query construction
  path; treat as "no obvious red flag found," not "certified clean."
