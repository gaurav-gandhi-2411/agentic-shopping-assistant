# Testing Guide

## Test layers

| Layer | Command | What it covers |
|---|---|---|
| Unit | `pytest tests/unit/` | Pure logic, no DB |
| Integration (Docker) | `pytest tests/integration/` | Real Postgres via Docker; CRUD, constraints |
| RLS integration | — deferred; see below | Supabase Auth policies |

## Running integration tests locally

Requires Docker.

```bash
docker run --rm -d --name pg-test \
  -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=shopping \
  -p 5433:5432 postgres:15

DATABASE_URL=postgresql://postgres:test@localhost:5433/shopping \
  alembic upgrade head

DATABASE_URL=postgresql://postgres:test@localhost:5433/shopping \
  pytest tests/integration/

docker stop pg-test
```

Integration tests seed an `auth.users` row and a matching `public.users` row before each test, then pass the user id explicitly to every operation.  See `tests/integration/conftest.py`.

## RLS policy gap (deferred to Phase 2 prompt 2)

Row-level security is **enabled** on all four tables but the policies in `0001_initial_schema.py` reference `auth.uid()`, which only exists on Supabase.  On plain Postgres the policy block is skipped at migration time, so superuser connections see all rows regardless of RLS.

This means:

- Local Docker integration tests do **not** verify that user A cannot read user B's conversations.
- The trigger `on_auth_user_created` fires on the local auth stub (verified), but JWT-scoped policy enforcement requires a real Supabase project.

**What is deferred**: Integration tests against a Supabase staging project that:
1. Create two users via Supabase Auth.
2. Insert conversations under each user using JWT-authenticated connections.
3. Assert that a query scoped to user A returns zero rows from user B's data.

These tests will be added in Phase 2 prompt 2 alongside the JWT auth middleware.
