"""sentry_sdk.init must run only when SENTRY_DSN is set (it costs seconds of imports)."""

from __future__ import annotations

import asyncio

import pytest
import sentry_sdk
from fastapi import FastAPI

import api.main as main

# Invented, non-routable DSN (RFC 2606 .invalid) -- never contacted, init is spied.
FAKE_DSN = "https://public@o0.ingest.invalid/1"


class _Stop(Exception):
    """Raised right after the Sentry block so lifespan never loads the heavy stack."""


def _run_lifespan(monkeypatch: pytest.MonkeyPatch, dsn: str | None) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    if dsn is None:
        monkeypatch.delenv("SENTRY_DSN", raising=False)
    else:
        monkeypatch.setenv("SENTRY_DSN", dsn)

    def _stop(*_a: object, **_k: object) -> None:
        raise _Stop

    monkeypatch.setattr("src.catalogue.loader.load_config", _stop)

    async def _enter() -> None:
        async with main.lifespan(FastAPI()):
            pass

    with pytest.raises(_Stop):
        asyncio.run(_enter())
    return calls


@pytest.mark.parametrize("dsn", [None, "", "   "])
def test_lifespan_skips_init_without_dsn(monkeypatch: pytest.MonkeyPatch, dsn: str | None) -> None:
    assert _run_lifespan(monkeypatch, dsn) == []


def test_lifespan_inits_once_with_same_options_when_dsn_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _run_lifespan(monkeypatch, FAKE_DSN)
    assert len(calls) == 1
    kw = calls[0]
    assert kw["dsn"] == FAKE_DSN
    assert kw["traces_sample_rate"] == 0.1
    assert kw["send_default_pii"] is False
    assert kw["max_request_body_size"] == "never"
    assert kw["before_breadcrumb"] is main._sentry_before_breadcrumb
    assert kw["before_send"] is main._sentry_before_send


def test_init_sentry_return_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    assert main._init_sentry(dsn=None) is False
    assert main._init_sentry(dsn=FAKE_DSN, traces_sample_rate=0.1) is True
    assert calls == [{"dsn": FAKE_DSN, "traces_sample_rate": 0.1}]


def test_chat_path_sentry_calls_are_noops_when_uninitialised() -> None:
    # chat.py calls these on every request; without init() they must not raise.
    sentry_sdk.set_user({"id": "u1"})
    sentry_sdk.set_tag("conversation_id", "c1")
