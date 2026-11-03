"""DATABASE_URL parsing: passwords with raw reserved characters (hermetic, no network)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from src.utils.db_url import create_db_engine, normalize_database_url, parse_database_url
from src.utils.redact import install_log_redaction

# Fake password with '@' ':' '/' '#' '%' (stray, so it is read as raw) and a space.
# Pieces are distinctive words so leaked fragments are detectable.
FAKE_PW = "Alpha@Bravo:Charlie/Delta#Echo%Fox Golf@Hotel"
PIECES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Fox", "Golf", "Hotel"]
HOST = "db.example.invalid"  # RFC 6761: guaranteed never to resolve


def _raw_url(pw: str = FAKE_PW) -> str:
    return f"postgresql://postgres.projref:{pw}@{HOST}:5432/postgres"


def _assert_no_leak(text: str) -> None:
    assert FAKE_PW not in text
    for piece in PIECES:
        assert piece not in text, f"password fragment {piece!r} leaked"


def test_raw_at_in_password_round_trips() -> None:
    url = parse_database_url(_raw_url())
    assert url.drivername == "postgresql+psycopg"
    assert url.host == HOST
    assert url.port == 5432
    assert url.database == "postgres"
    assert url.username == "postgres.projref"
    assert url.password == FAKE_PW


def test_already_encoded_password_not_double_encoded() -> None:
    url = parse_database_url(f"postgresql://u:p%40ss%3Aw%2Frd@{HOST}/db")
    assert url.password == "p@ss:w/rd"
    # Re-normalising the output is stable (no %2540).
    again = parse_database_url(normalize_database_url(f"postgresql://u:p%40ss@{HOST}/db"))
    assert again.password == "p@ss"
    assert "%2540" not in normalize_database_url(f"postgresql://u:p%40ss@{HOST}/db")


def test_encoded_and_raw_forms_are_equivalent() -> None:
    raw = parse_database_url(f"postgres://u:a@b@{HOST}/db")
    enc = parse_database_url(f"postgres://u:a%40b@{HOST}/db")
    assert raw == enc
    assert raw.password == "a@b"


def test_stray_percent_kept_literally() -> None:
    assert parse_database_url(f"postgresql://u:100%@{HOST}/db").password == "100%"
    assert parse_database_url(f"postgresql://u:a%zz@{HOST}/db").password == "a%zz"


def test_no_password_and_no_userinfo() -> None:
    no_pw = parse_database_url(f"postgresql://postgres@{HOST}/db")
    assert (no_pw.username, no_pw.password) == ("postgres", None)
    bare = parse_database_url(f"postgresql://{HOST}:6543/db?sslmode=require")
    assert (bare.username, bare.password, bare.port) == (None, None, 6543)
    assert bare.query == {"sslmode": "require"}


def test_str_masks_password() -> None:
    assert FAKE_PW not in str(parse_database_url(_raw_url()))
    assert "***" in str(parse_database_url(_raw_url()))


def test_invalid_input_error_does_not_echo_input() -> None:
    with pytest.raises(ValueError) as exc:
        parse_database_url("not a url but has SecretWord123")
    assert "SecretWord123" not in str(exc.value)
    with pytest.raises(ValueError) as exc2:
        parse_database_url("postgresql://u:SecretWord123@host:notaport/db")
    assert "SecretWord123" not in str(exc2.value)


def test_create_db_engine_uses_parsed_url() -> None:
    engine = create_db_engine(_raw_url())
    try:
        assert engine.url.host == HOST
        assert engine.url.password == FAKE_PW
    finally:
        engine.dispose()


def test_connect_failure_never_leaks_password(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Architect's required test: failed connect, ALL output scanned for the password.

    Goes through the real call sites (feedback._get_engine and
    api.main._build_session_store), so it fails on the pre-fix code, where a
    password fragment ended up in the host passed to the driver.
    """
    import api.routes.feedback as feedback
    from api.main import _build_session_store

    monkeypatch.setenv("DATABASE_URL", _raw_url())
    monkeypatch.setattr(feedback, "_engine", None)
    install_log_redaction()
    caplog.set_level(logging.DEBUG)
    stream = logging.StreamHandler()  # stderr, captured by capsys
    logging.getLogger().addHandler(stream)
    install_log_redaction()

    failures: list[str] = []
    _, session_engine = _build_session_store(None, {})
    for engine in (feedback._get_engine(), session_engine):
        try:
            with pytest.raises(Exception) as exc:  # noqa: PT011 - driver-specific type
                with engine.connect():
                    pass
            failures.append(str(exc.value))
            logging.getLogger("t").error("db connect failed: %s", exc.value, exc_info=exc.value)
        finally:
            engine.dispose()

    logging.getLogger().removeHandler(stream)
    out = capsys.readouterr()
    everything = "\n".join([*failures, caplog.text, out.out, out.err])
    assert HOST in everything  # the failure really happened and names the true host
    _assert_no_leak(everything)


def test_alembic_env_uses_helper() -> None:
    src = (Path(__file__).resolve().parent.parent / "alembic" / "env.py").read_text()
    assert "parse_database_url" in src
    assert "install_log_redaction" in src
    assert not re.search(r"""startswith\(\s*prefix""", src), "hand-rolled prefix rewrite is back"


def test_no_call_site_builds_engine_from_raw_env_url() -> None:
    root = Path(__file__).resolve().parent.parent
    for rel in ("api/main.py", "api/routes/feedback.py", "scripts/run_migrations.py"):
        text = (root / rel).read_text()
        assert "create_engine(db_url" not in text, rel
        assert 'startswith(prefix)' not in text, rel
