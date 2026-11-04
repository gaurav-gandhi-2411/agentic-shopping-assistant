"""Pure (no database) unit tests for scripts/ops/check_migrations.py.

The DB-backed run is exercised by .github/workflows/migration-integrity.yml against an ephemeral
Postgres service container; these tests cover the decision logic only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy.engine import make_url

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ops" / "check_migrations.py"
_spec = importlib.util.spec_from_file_location("check_migrations", _PATH)
assert _spec and _spec.loader
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def test_single_head_ok() -> None:
    assert cm.check_single_head(["0008"]) == []


def test_no_head_fails() -> None:
    assert "no Alembic head" in cm.check_single_head([])[0]


def test_multiple_heads_fail_and_name_both() -> None:
    (msg,) = cm.check_single_head(["0008", "0008b"])
    assert "multiple Alembic heads" in msg and "0008" in msg and "0008b" in msg


def test_applied_equals_head() -> None:
    assert cm.check_applied_equals_head(["0008"], ["0008"]) == []
    assert cm.check_applied_equals_head(["0007"], ["0008"])
    assert cm.check_applied_equals_head([], ["0008"])  # empty alembic_version is a failure


def test_localhost_guard() -> None:
    assert cm.check_localhost(make_url("postgresql://u:p@localhost:5432/db")) == []
    assert cm.check_localhost(make_url("postgresql://u:p@127.0.0.1/db")) == []
    assert cm.check_localhost(make_url("postgresql://u:p@db.abc.supabase.co:5432/postgres"))
    assert cm.check_localhost(make_url("postgresql:///db"))  # no host (unix socket): refuse


_BYPASS_QUERIES = [
    "?host=evil.example",
    "?hostaddr=8.8.8.8",
    "?port=5432&host=evil.example",
    "?service=prod",
    "?HOST=evil.example",
    "?host=127.0.0.1",  # even a loopback value: the key itself is refused
    "?host=127.0.0.1&host=evil.example",  # repeated params
    "?sslmode=require&host=evil.example",  # allowed key must not mask a bad one
    "?%68ost=evil.example",  # percent-encoded param name
    "?ho%73t=evil.example",
]


def test_query_param_overrides_are_refused() -> None:
    for q in _BYPASS_QUERIES:
        url, errors = cm.parse_target_url("postgresql://dummy@localhost/postgres" + q)
        assert url is None and errors, q


def test_allowed_query_params_pass() -> None:
    url, errors = cm.parse_target_url(
        "postgresql://dummy@localhost/postgres?sslmode=disable&connect_timeout=5"
        "&application_name=x"
    )
    assert url is not None and errors == []


def test_unparsable_url_fails_closed_without_echoing_it() -> None:
    url, errors = cm.parse_target_url("not a url with secret-token")
    assert url is None and len(errors) == 1 and "secret-token" not in errors[0]


def test_url_with_special_chars_roundtrips_without_hand_parsing() -> None:
    url = make_url("postgresql://postgres:p%40ss%3Aw%2Frd%231@127.0.0.1:55432/postgres")
    assert url.password == "p@ss:w/rd#1" and url.host == "127.0.0.1" and url.port == 55432


def test_plan_no_irreversible_is_one_full_cycle() -> None:
    assert cm.plan_round_trip(["a", "b", "c"], []) == [("cycle", "base", "c")]


def test_plan_splits_around_irreversible() -> None:
    chain = ["1", "2", "3", "4", "5"]
    assert cm.plan_round_trip(chain, ["3"]) == [
        ("cycle", "base", "2"),
        ("irreversible", "3", "2"),
        ("cycle", "3", "5"),
    ]


def test_plan_irreversible_first_and_last() -> None:
    assert cm.plan_round_trip(["1", "2"], ["1", "2"]) == [
        ("irreversible", "1", "base"),
        ("irreversible", "2", "1"),
    ]


def test_plan_repo_chain_covers_every_reversible_revision() -> None:
    chain = [f"000{i}" for i in range(1, 9)]
    steps = cm.plan_round_trip(chain, list(cm.IRREVERSIBLE))
    assert [s[0] for s in steps] == ["cycle", "irreversible", "cycle", "irreversible", "cycle"]


def test_rls_check() -> None:
    assert cm.check_rls({"a": True}, ["a"]) == []
    assert cm.check_rls({"a": False}, ["a"])
    assert cm.check_rls({}, ["a"])  # missing table is a failure, not a pass
    assert cm.check_rls({"a": False}, [], ["a"]) == []
    assert cm.check_rls({"a": True}, [], ["a"])


def test_tables_and_grants() -> None:
    assert cm.check_tables(cm.EXPECTED_TABLES) == []
    assert "users" in cm.check_tables([t for t in cm.EXPECTED_TABLES if t != "users"])[0]
    assert cm.check_no_grants([]) == []
    assert cm.check_no_grants([("anon", "users", "DELETE")])


def test_expected_tables_match_migration_0007_and_0008() -> None:
    """Guard the hardcoded lists against drifting from the migrations they were derived from."""
    versions = _PATH.parent.parent.parent / "alembic" / "versions"

    def tables(name: str) -> set[str]:
        src = next(versions.glob(f"{name}_*.py")).read_text(encoding="utf-8")
        body = src.split("_TABLES = (", 1)[1].split(")", 1)[0]
        return {t.strip().strip('"') for t in body.split(",") if t.strip()}

    assert tables("0007") == set(cm.REVOKED_TABLES)
    assert tables("0008") == set(cm.RLS_0008_TABLES)
