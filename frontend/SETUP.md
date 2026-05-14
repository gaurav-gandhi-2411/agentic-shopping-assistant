# Frontend Setup

Next.js 15 App Router frontend for the Shopping Assistant. Supabase for auth (magic link + Google OAuth). TypeScript, shadcn/ui, Tailwind CSS.

---

## Prerequisites

- Node.js 20 or later
- npm (or pnpm/yarn — adjust commands accordingly)

---

## Install

```bash
cd frontend
npm install
```

---

## Environment variables

Copy the example file and fill in the three required values:

```bash
cp .env.local.example .env.local
```

| Variable | Description | Local dev value |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL | `https://<project-ref>.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon/public key | from Supabase dashboard |
| `NEXT_PUBLIC_BACKEND_URL` | FastAPI backend base URL | `http://localhost:8000` |

All three are required. The `NEXT_PUBLIC_` prefix is intentional — these values are safe to expose in the browser.

Find `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` in the Supabase dashboard:
**Settings → API → Project URL** and **Project API keys → anon/public**

---

## Supabase dashboard configuration

Three steps required before auth will work.

### Step 1 — Register your app's callback URL with Supabase

> **Why:** Supabase validates that the redirect URL after login is on an allowlist. Without this step, both magic link and Google OAuth will fail with a "redirect_uri_mismatch" or similar error.

This is the URL that **Supabase redirects your browser to** after authentication completes. Your Next.js `/auth/callback` route handler sits here and exchanges the code for a session.

In the Supabase dashboard:
1. Go to **Authentication → URL Configuration**
2. Under **Redirect URLs**, click **Add URL**
3. Add: `http://localhost:3000/auth/callback`

For production, also add (before deploying):
```
https://<your-domain>/auth/callback
```

This URL goes in **Supabase only**. Do not put it in Google Cloud Console.

---

### Step 2 — Configure Google OAuth

Skip this step if you only want magic link auth.

There are two paths. **Pick one before testing the OAuth flow.**

#### Path 1 — Supabase-managed credentials (fastest, no Google Cloud setup)

Supabase provides a built-in Google OAuth app you can enable instantly. The consent screen will say "Supabase" rather than your project name.

1. In the Supabase dashboard: **Authentication → Providers → Google**
2. Toggle **Enable** (no Client ID or Secret required for the managed app)
3. Done — no Google Cloud Console steps needed.

Use this for getting things running quickly in development.

#### Path 2 — Your own Google Cloud credentials (cleaner branding)

Use this for production or if you want the consent screen to show your project name.

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth 2.0 Client ID**
3. Application type: **Web application**
4. Under **Authorised redirect URIs**, add:
   ```
   https://<project-ref>.supabase.co/auth/v1/callback
   ```
   > **Important:** this is the URL that **Google redirects to** — it points at Supabase, not at your app. Supabase then redirects to your `/auth/callback`. Do not put `localhost:3000` here.
5. Copy the **Client ID** and **Client Secret**
6. In the Supabase dashboard: **Authentication → Providers → Google → Enable**
7. Paste the Client ID and Client Secret, then save

---

**Summary of which URL goes where:**

| URL | Where it goes | Why |
|---|---|---|
| `http://localhost:3000/auth/callback` | Supabase → Authentication → URL Configuration → Redirect URLs | Supabase redirects your browser here after auth |
| `https://<project-ref>.supabase.co/auth/v1/callback` | Google Cloud Console → OAuth 2.0 → Authorised redirect URIs | Google redirects to Supabase; Supabase handles the Google callback (Path 2 only) |

---

### Step 3 — (Optional) Restrict sign-ups to the allow list

If the Supabase Auth Hook from `alembic/versions/0003_allowed_emails.py` is deployed, new sign-ups are automatically blocked unless the email is in `public.allowed_emails`. No frontend changes are needed — the hook fires server-side before a user account is created.

To add yourself during local dev:

```sql
INSERT INTO public.allowed_emails (email) VALUES ('you@example.com');
```

---

## Run locally

```bash
# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Next.js frontend (from repo root)
cd frontend
npm run dev
```

Open http://localhost:3000. You will be redirected to `/login`.

---

## How the auth flow works

```
Browser                     Next.js                    Supabase
  |                            |                           |
  |-- POST /auth/v1/otp ------>|-------------------------->|
  |   (magic link request)     |                           |-- email -->  User
  |                                                               |
  |<-- click link in email ------------------------------------------
  |                            |
  |-- GET /auth/callback?code= |
  |                            |-- exchangeCodeForSession() -->|
  |                            |<-- session (JWT) -------------|
  |                            |-- set-cookie (session) ------>|
  |<-- redirect /chat ---------|
  |                            |
  |-- GET /chat                |
  |                            |-- middleware: getUser() -->|
  |                            |<-- user -------------------|
  |<-- 200 /chat --------------|
```

Key points:

- Both magic link and Google OAuth land at `/auth/callback?code=<code>`. One handler, one flow.
- `middleware.ts` calls `supabase.auth.getUser()` on every request to silently refresh the session token before it expires. It also guards `/chat` — unauthenticated requests redirect to `/login`.
- Three separate Supabase client instances are required because cookie access works differently in each context:
  - `lib/supabase/client.ts` — browser (Client Components)
  - `lib/supabase/server.ts` — server (Server Components, Route Handlers) — uses async `cookies()` from `next/headers`
  - `middleware.ts` — edge (reads `request.cookies`, writes to both request and response cookies to propagate the refresh)
- The FastAPI backend receives the Supabase JWT as `Authorization: Bearer <token>` and verifies it independently via the JWKS endpoint. The frontend never passes the raw key to the backend.
