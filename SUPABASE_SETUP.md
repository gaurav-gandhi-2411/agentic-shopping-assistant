# Supabase Setup

This document covers deploying the Postgres schema to a Supabase project and verifying it.

## Prerequisites

- A Supabase project — [create one at supabase.com](https://supabase.com) (free tier is fine)
- Python environment with project dependencies installed: `pip install -r requirements.txt`

---

## 1. Get your DATABASE_URL

In the Supabase dashboard:

1. Open **Settings → Database**
2. Scroll to **Connection string** and select the **URI** tab
3. Copy the string — it looks like:
   ```
   postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
   ```
4. Add it to your `.env` file (create from template if needed):
   ```bash
   cp .env.example .env
   # open .env and set DATABASE_URL
   ```

> **Use the Transaction pooler (port 5432), not the Session pooler (port 6543).**
> psycopg3 uses extended query protocol, which conflicts with PgBouncer in session mode.

---

## 2. Run migrations

**Always dry-run against a local Docker Postgres before applying to Supabase.**

### 2a. Local dry-run (recommended before every Supabase apply)

Spin up a throwaway Postgres container and apply the migrations to it first.
This validates the SQL end-to-end without risking the Supabase database.

```bash
# Start a fresh local Postgres
docker run --rm -d --name pg-dryrun \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=shopping \
  -p 5434:5432 postgres:15

export DATABASE_URL=postgresql://postgres:test@localhost:5434/shopping

# The --create-auth-shim flag (or APPLY_AUTH_SHIM=true) creates the auth
# schema and a minimal auth.users stub before running migrations.
# This is required on plain Postgres because migration 0001 adds a FK that
# REFERENCES auth.users(id).  On Supabase, auth.users already exists.
python scripts/run_migrations.py --create-auth-shim
```

Expected output:
```
Connecting to: postgresql://postgres:***@localhost:5434/shopping
Creating auth schema shim (local Postgres mode)...
Auth shim ready.
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, initial_schema
INFO  [alembic.runtime.migration] Running upgrade 0001 -> 0002, add_conversation_state
Migrations applied successfully.
```

Verify the schema via psql:
```bash
psql postgresql://postgres:test@localhost:5434/shopping
```
```sql
\dt
-- Expected: alembic_version, conversations, feedback, messages, users

\d conversations
-- Expected: id, user_id, title, is_public, created_at, updated_at,
--           excluded_colours, summary, summary_message_count
```

Then stop the container:
```bash
docker stop pg-dryrun
```

You can also generate the raw SQL without connecting to any database at all:
```bash
python scripts/run_migrations.py --dry-run
```
This uses Alembic's offline mode (`sql=True`) — no connection is made, and the
auth shim is not needed.

### 2b. Apply to Supabase

Once you are satisfied with the local dry-run:

```bash
# Load environment variables from .env (must contain your Supabase DATABASE_URL)
export $(grep -v '^#' .env | xargs)

# Apply — no shim flag: Supabase already has auth.users
python scripts/run_migrations.py
```

Expected output:
```
Connecting to: postgresql://postgres.***:***@aws-0-<region>.pooler.supabase.com:5432/postgres
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, initial_schema
INFO  [alembic.runtime.migration] Running upgrade 0001 -> 0002, add_conversation_state
Migrations applied successfully.
```

---

## 3. Verify the schema

Run these queries in the Supabase **SQL Editor** or via `psql`:

```sql
-- All four application tables should exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('users', 'conversations', 'messages', 'feedback')
ORDER BY table_name;
-- Expected: 4 rows

-- Migration watermark
SELECT version_num FROM alembic_version;
-- Expected: 0002

-- RLS is enabled on every table
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('users', 'conversations', 'messages', 'feedback')
ORDER BY tablename;
-- Expected: rowsecurity = true on all four rows

-- RLS policies installed (requires Supabase — auth.uid() exists)
SELECT tablename, policyname
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
-- Expected: 12 policies:
--   users:         users_select, users_update
--   conversations: conversations_delete, conversations_insert,
--                  conversations_select, conversations_update
--   messages:      messages_insert, messages_select
--   feedback:      feedback_delete, feedback_insert,
--                  feedback_select, feedback_update
```

---

## 4. Auth trigger

The migration installs `handle_new_user()`, a `SECURITY DEFINER` trigger that syncs every new Supabase Auth sign-up to `public.users`. Verify:

```sql
SELECT trigger_name, event_object_schema, event_object_table
FROM information_schema.triggers
WHERE trigger_name = 'on_auth_user_created';
-- Expected: one row (event_object_schema = auth, event_object_table = users)
```

Supabase Auth will call this automatically on every new user registration — no application code needed.

---

## 5. Use the database in the API

Set `DATABASE_URL` in your environment (or `.env`) before starting the API:

```bash
export DATABASE_URL=postgresql://postgres.<ref>:<pass>@<host>:5432/postgres
uvicorn api.main:app --reload --port 8000
```

The API will log:
```
Startup complete (session_store=PostgresSessionStore)
```

Without `DATABASE_URL`, the API falls back to an in-memory session store (sessions lost on restart).

### Fly.io

```bash
fly secrets set DATABASE_URL="postgresql://postgres.<ref>:<pass>@<host>:5432/postgres"
```

---

## 6. Local development without Supabase

To test Postgres persistence locally without a Supabase account:

```bash
docker run --rm -d --name pg-dev \
  -e POSTGRES_PASSWORD=dev \
  -e POSTGRES_DB=shopping \
  -p 5432:5432 postgres:15

export DATABASE_URL=postgresql://postgres:dev@localhost:5432/shopping
python scripts/run_migrations.py
uvicorn api.main:app --reload --port 8000
```

On plain Postgres, RLS is enabled but policies are not installed (they require `auth.uid()` from Supabase). A superuser connection sees all rows regardless of RLS. Full RLS enforcement requires a real Supabase project and is tested in Phase 2 prompt 2.

---

## Notes

| Topic | Detail |
|---|---|
| RLS enforcement | Policies reference `auth.uid()` (Supabase only). On plain Postgres, RLS is enabled but policies are absent — superuser connections bypass it. Supabase automatically enforces policies for all JWT-authenticated client connections. |
| Superuser access | Use the service-role key (server-side only, never expose to clients) to bypass RLS for admin operations. |
| Session isolation | `PostgresSessionStore.get/set/delete` always filter by `user_id`, so even without RLS the application layer enforces ownership. |
| JWT auth | `get_current_user_id()` returns a hardcoded dev UUID until Phase 2 prompt 2 wires in JWT extraction. |
