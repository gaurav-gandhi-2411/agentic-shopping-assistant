#!/usr/bin/env python3
"""Migration-INTEGRITY gate: run the repo's Alembic migrations against an EPHEMERAL Postgres.

This never talks to production and holds no production credentials. It answers "do the
migrations in this repo form one sane, applicable, reversible-where-promised chain?", not "is
production behind master" (that is a prod question; see DEPLOY.md "Database migration").

Usage (CI service container, or a throwaway local container):

    MIGRATION_CHECK_DB_URL=postgresql://postgres:pw@localhost:5432/postgres \\
        python scripts/ops/check_migrations.py

Safety: the target database is wiped (DROP SCHEMA public CASCADE) and must be on
localhost / 127.0.0.1 / ::1 with no URL query parameters beyond sslmode, connect_timeout and
application_name (libpq query params like ?host= would override the authority); anything else, or
an unparsable URL, is refused before any connection is made.

Known by-design gap: sanity checks cover only the hardcoded table lists below; a table created by
a future migration is not checked until it is added to them.

Checks, in order (any failure -> exit 1, all failures listed):
  a. exactly one Alembic head in alembic/versions/ (offline, no DB);
  b. the target DB is empty, then `alembic upgrade head` from empty succeeds;
  c. alembic_version after upgrade == the single head;
  d. downgrade round trips (see IRREVERSIBLE for what cannot be reversed and why);
  e. post-upgrade sanity: expected tables exist, RLS is on where migrations 0001/0003/0008 turn
     it on, and anon/authenticated hold zero grants on the 11 tables migration 0007 revokes.
"""
from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine, make_url

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Query keys that cannot redirect the connection. Case-sensitive on purpose (HOST is refused).
ALLOWED_QUERY_KEYS = frozenset({"sslmode", "connect_timeout", "application_name"})

# Migrations whose downgrade() cannot run. Rev -> reason. The gate does NOT skip them silently:
# it proves each one still fails (and leaves the DB untouched), so the entry goes stale loudly
# the day someone fixes it. Everything between them is cycled down and back up.
IRREVERSIBLE: dict[str, str] = {
    "0003": (
        "KNOWN BUG: downgrade() drops function normalise_allowed_email() before dropping "
        "allowed_emails, whose trigger depends on it (DependentObjectsStillExist); swap the "
        "order to fix, then remove this entry"
    ),
    "0007": "intentional: raises NotImplementedError (restoring the default grant reopens the hole)",
}

# Tables created by migrations 0001-0006 (+ alembic's own), per 0007's _TABLES.
EXPECTED_TABLES: tuple[str, ...] = (
    "alembic_version",
    "allowed_emails",
    "conversations",
    "demo_daily_stats",
    "demo_rate_limits",
    "feedback",
    "messages",
    "pairing_stats",
    "saved_looks",
    "styling_events",
    "users",
)
# RLS enabled by 0008 (deny-by-default, zero policies) ...
RLS_0008_TABLES: tuple[str, ...] = (
    "demo_daily_stats",
    "demo_rate_limits",
    "pairing_stats",
    "saved_looks",
    "styling_events",
)
# ... and by 0001 / 0003.
RLS_EARLIER_TABLES: tuple[str, ...] = ("allowed_emails", "conversations", "feedback", "messages", "users")
# Tables 0007 revokes anon/authenticated from: every EXPECTED table.
REVOKED_TABLES = EXPECTED_TABLES

# Roles Supabase provides and 0007 REVOKEs from; absent on plain Postgres.
STUB_ROLES: tuple[str, ...] = ("anon", "authenticated")


# --------------------------------------------------------------------------- pure logic


def check_single_head(heads: Sequence[str]) -> list[str]:
    """Return error strings unless there is exactly one head."""
    if len(heads) == 0:
        return ["no Alembic head found: alembic/versions/ is empty or the chain is broken"]
    if len(heads) > 1:
        return [
            f"multiple Alembic heads ({', '.join(sorted(heads))}): branch in the migration "
            "history; add a merge revision (`alembic merge`) or fix down_revision"
        ]
    return []


def check_applied_equals_head(applied: Sequence[str], heads: Sequence[str]) -> list[str]:
    """alembic_version rows after `upgrade head` must be exactly the head set."""
    if sorted(applied) != sorted(heads):
        return [
            f"alembic_version is {sorted(applied)} after `upgrade head` but `alembic heads` is "
            f"{sorted(heads)}"
        ]
    return []


def plan_round_trip(
    chain: Sequence[str], irreversible: Sequence[str]
) -> list[tuple[str, str, str]]:
    """Plan downgrade coverage for a linear base->head ``chain`` of revision ids.

    Downgrading from X to Y runs the downgrade() of every revision above Y up to X, so revision
    k's downgrade is exercised by "downgrade to the revision below k". Steps, ascending:
      * ``("cycle", low, high)``: at ``high``, downgrade to ``low`` ("base" for the first
        revision), upgrade back. ``high`` is the top of a maximal run of reversible revisions.
      * ``("irreversible", rev, below)``: at ``rev``, a downgrade to ``below`` must FAIL.
    """
    steps: list[tuple[str, str, str]] = []
    i, n = 0, len(chain)
    while i < n:
        below = chain[i - 1] if i > 0 else "base"
        if chain[i] in irreversible:
            steps.append(("irreversible", chain[i], below))
            i += 1
            continue
        j = i
        while j + 1 < n and chain[j + 1] not in irreversible:
            j += 1
        steps.append(("cycle", below, chain[j]))
        i = j + 1
    return steps


def check_localhost(url: URL) -> list[str]:
    """Refuse any non-local target: this script DROPs schema public.

    libpq lets query parameters (host, hostaddr, port, service, ...) override the URL authority,
    so checking ``url.host`` alone is bypassable. Only a tiny allow-list of harmless query keys
    is accepted; anything else (any case, any repetition) is refused. Fail closed.
    """
    errors: list[str] = []
    host = url.host or ""
    if host not in LOCAL_HOSTS:
        errors.append(f"refusing to run against host {host!r}: only {sorted(LOCAL_HOSTS)} allowed")
    bad = sorted(k for k in url.query if k not in ALLOWED_QUERY_KEYS)
    if bad:
        errors.append(
            f"refusing URL query parameter(s) {bad}: libpq lets them override the host; "
            f"only {sorted(ALLOWED_QUERY_KEYS)} are allowed"
        )
    return errors


def parse_target_url(raw: str) -> tuple[URL | None, list[str]]:
    """Parse and validate the target URL; any parse failure is a clean error, never a traceback.

    The error text deliberately omits the raw URL (it may contain a password).
    """
    try:
        url = make_url(raw)
    except Exception as exc:  # noqa: BLE001 - fail closed on any parse problem
        return None, [f"MIGRATION_CHECK_DB_URL is not a parsable database URL ({type(exc).__name__})"]
    errors = check_localhost(url)
    return (None, errors) if errors else (url, [])


def check_tables(present: Sequence[str]) -> list[str]:
    missing = sorted(set(EXPECTED_TABLES) - set(present))
    return [f"expected tables missing after upgrade: {', '.join(missing)}"] if missing else []


def check_rls(rls_by_table: dict[str, bool], expect_on: Sequence[str], expect_off: Sequence[str] = ()) -> list[str]:
    errs = []
    for t in expect_on:
        if rls_by_table.get(t) is not True:
            errs.append(f"RLS not enabled on public.{t} (relrowsecurity={rls_by_table.get(t)})")
    for t in expect_off:
        if rls_by_table.get(t) is not False:
            errs.append(f"RLS unexpectedly enabled on public.{t} after downgrade")
    return errs


def check_no_grants(grants: Sequence[tuple[str, str, str]]) -> list[str]:
    """``grants`` = (grantee, table, privilege) rows that should not exist."""
    return [f"{g} still holds {p} on public.{t} (0007 should have revoked it)" for g, t, p in grants]


# --------------------------------------------------------------------------- DB side


def _fail(errors: list[str]) -> None:
    for e in errors:
        print(f"::error::{e}", flush=True)
    raise SystemExit(1)


def _alembic_cfg():  # noqa: ANN202 - alembic Config imported lazily
    from alembic.config import Config

    return Config(str(REPO_ROOT / "alembic.ini"))


def repo_heads_and_chain() -> tuple[list[str], list[str]]:
    """Offline: (heads, linear base->head chain of the first head)."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_cfg())
    heads = list(script.get_heads())
    chain = [r.revision for r in script.walk_revisions()][::-1] if len(heads) == 1 else []
    return heads, chain


def setup_ephemeral(engine: Engine) -> None:
    """Make plain Postgres look enough like Supabase for the migrations to apply. Wipes public."""
    from scripts.run_migrations import apply_auth_shim

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS auth CASCADE"))
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        for role in STUB_ROLES:  # 0007 does REVOKE ... FROM anon, authenticated
            conn.execute(text(
                f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') "
                f"THEN CREATE ROLE {role} NOLOGIN; END IF; END $$"
            ))
        # Supabase auto-grants every new public table to these roles; replicate it so 0007's
        # REVOKE is actually exercised (on plain Postgres there would be nothing to revoke).
        conn.execute(text(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, authenticated"
        ))
    apply_auth_shim(engine)  # 0001 has a FK to auth.users(id)
    with engine.begin() as conn:
        # Supabase's auth.uid(); makes 0001/0003's `IF auth.uid() exists` policy branches run.
        conn.execute(text(
            "CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE "
            "AS $$ SELECT NULL::uuid $$"
        ))


def _scalars(engine: Engine, sql: str) -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(sql))]


def public_tables(engine: Engine) -> list[str]:
    return _scalars(engine, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")


def applied_versions(engine: Engine) -> list[str]:
    return _scalars(engine, "SELECT version_num FROM alembic_version")


def rls_map(engine: Engine) -> dict[str, bool]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT c.relname, c.relrowsecurity FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'r'"
        ))
        return {r[0]: bool(r[1]) for r in rows}


def leftover_grants(engine: Engine) -> list[tuple[str, str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT grantee, table_name, privilege_type FROM information_schema.role_table_grants "
            "WHERE table_schema = 'public' AND grantee IN ('anon', 'authenticated') "
            "AND table_name = ANY(:t) ORDER BY 1, 2, 3"
        ), {"t": list(REVOKED_TABLES)})
        return [(r[0], r[1], r[2]) for r in rows]


def sanity(engine: Engine, heads: Sequence[str]) -> list[str]:
    errs = check_applied_equals_head(applied_versions(engine), heads)
    errs += check_tables(public_tables(engine))
    errs += check_rls(rls_map(engine), RLS_0008_TABLES + RLS_EARLIER_TABLES)
    errs += check_no_grants(leftover_grants(engine))
    return errs


def run_step(engine: Engine, cfg, kind: str, x: str, y: str) -> list[str]:  # noqa: ANN001
    """Execute one plan_round_trip step against the live DB; return error strings."""
    from alembic import command

    if kind == "irreversible":  # x = rev, y = revision below it
        command.upgrade(cfg, x)
        try:
            command.downgrade(cfg, y)
        except Exception as exc:  # noqa: BLE001 - failing is the documented expectation
            print(f"(d) {x} downgrade fails as documented ({type(exc).__name__}): {IRREVERSIBLE[x]}")
            # Alembic runs PG DDL transactionally, so the failed downgrade must leave no trace.
            if applied_versions(engine) != [x]:
                return [f"failed downgrade of {x} was not rolled back"]
            return []
        return [f"{x} is listed in IRREVERSIBLE but its downgrade succeeded; remove the entry"]
    # cycle: x = low, y = high
    command.upgrade(cfg, y)
    command.downgrade(cfg, x)
    if x == "base":
        left = set(public_tables(engine)) - {"alembic_version"}
        errors = [f"downgrade base left tables behind: {sorted(left)}"] if left else []
    else:
        errors = check_applied_equals_head(applied_versions(engine), [x])
    if x == "0007" and y == "0008":  # 0008's downgrade must really switch RLS off again
        errors += check_rls(rls_map(engine), (), RLS_0008_TABLES)
    if errors:
        return errors
    command.upgrade(cfg, y)
    print(f"(d) {y} -> downgrade {x} -> upgrade {y}: OK")
    return []


def run(url: URL) -> list[str]:
    from alembic import command

    errors: list[str] = check_localhost(url)
    if errors:
        return errors

    heads, chain = repo_heads_and_chain()
    errors = check_single_head(heads)
    if errors:
        return errors
    print(f"(a) single head: {heads[0]}; chain: {' -> '.join(chain)}")

    # env.py reads DATABASE_URL only (ignores sqlalchemy.url); set it for this process alone.
    os.environ["DATABASE_URL"] = url.set(drivername="postgresql").render_as_string(hide_password=False)
    cfg = _alembic_cfg()
    engine = create_engine(url.set(drivername="postgresql+psycopg"))
    try:
        setup_ephemeral(engine)
        leftovers = public_tables(engine)
        if leftovers:
            return [f"database not empty after reset: {leftovers}"]
        print("(b) database empty; ephemeral roles/shim ready")

        try:
            command.upgrade(cfg, "head")
        except Exception as exc:  # noqa: BLE001 - annotate, then show the full traceback
            print(f"::error::alembic upgrade head from an empty database failed: "
                  f"{type(exc).__name__}", flush=True)
            traceback.print_exc()
            return ["alembic upgrade head from an empty database failed (see traceback above)"]
        print("(b) alembic upgrade head from empty: OK")
        errors = sanity(engine, heads)
        if errors:
            return errors
        print(f"(c)(e) alembic_version == head ({heads[0]}); tables, RLS, revokes: OK")

        # Downgrades cannot cross an IRREVERSIBLE revision, so walk up from a fresh DB.
        setup_ephemeral(engine)
        for kind, x, y in plan_round_trip(chain, list(IRREVERSIBLE)):
            errors = run_step(engine, cfg, kind, x, y)
            if errors:
                return errors
        command.upgrade(cfg, "head")
        errors = sanity(engine, heads)
        if errors:
            return errors
        print("(d)(e) round trips done, final upgrade head + sanity: OK")
    finally:
        engine.dispose()
    return []


def main() -> None:
    raw = os.environ.get("MIGRATION_CHECK_DB_URL", "").strip()
    if not raw:
        _fail(["MIGRATION_CHECK_DB_URL is not set (ephemeral Postgres URL, localhost only)"])
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    url, errors = parse_target_url(raw)
    if url is not None:
        errors = run(url)
    if errors:
        _fail(errors)
    print("Migration integrity: all checks passed.")


if __name__ == "__main__":
    main()
