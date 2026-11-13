"""Client-IP derivation for the rate limiters, using the real measured Cloud Run header shapes."""

from __future__ import annotations

import pytest
from starlette.requests import Request

from api.demo.client_ip import get_client_ip

_PEER = "10.8.0.1"  # Cloud Run's shared private socket peer: must never be the key
_REAL = "203.0.113.9"
_REAL6 = "2001:db8:aa:bb:1111:2222:3333:4444"


def _req(xff: str | None, peer: str | None = _PEER) -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234) if peer else None})


@pytest.fixture(autouse=True)
def _hops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRUSTED_PROXY_HOPS", raising=False)


def test_single_real_entry() -> None:
    assert get_client_ip(_req(_REAL)) == _REAL


def test_spoof_plus_real_uses_real() -> None:
    assert get_client_ip(_req(f"6.6.6.6, {_REAL}")) == _REAL


def test_multi_value_spoof_plus_real_uses_real() -> None:
    assert get_client_ip(_req(f"6.6.6.6, 7.7.7.7, {_REAL}")) == _REAL


def test_ipv6_spoof_plus_real_v4_uses_real_v4() -> None:
    assert get_client_ip(_req(f"2001:db8:dead:beef::1, {_REAL}")) == _REAL


def test_real_ipv6_client_collapses_to_slash64_and_rotating_host_bits_share_a_key() -> None:
    a = get_client_ip(_req(f"1.1.1.1, {_REAL6}"))
    b = get_client_ip(_req("1.1.1.1, 2001:db8:aa:bb:ffff:eeee:dddd:cccc"))
    assert a == b == "2001:db8:aa:bb::"


def test_missing_header_uses_socket_peer() -> None:
    assert get_client_ip(_req(None, peer="198.51.100.4")) == "198.51.100.4"


def test_empty_header_uses_socket_peer() -> None:
    assert get_client_ip(_req("", peer="198.51.100.4")) == "198.51.100.4"
    assert get_client_ip(_req(" , ,", peer="198.51.100.4")) == "198.51.100.4"


def test_garbage_rightmost_does_not_fall_back_to_client_controlled_leftmost() -> None:
    # Leftmost is a valid, attacker-chosen IP; it must NOT become the key.
    assert get_client_ip(_req("6.6.6.6, not-an-ip", peer="198.51.100.4")) == "198.51.100.4"


def test_no_peer_and_no_header_returns_sentinel() -> None:
    assert get_client_ip(_req(None, peer=None)) == "0.0.0.0"  # noqa: S104


@pytest.mark.parametrize(
    ("xff", "expected"),
    [
        (f"6.6.6.6, {_REAL}:5678", _REAL),
        ("6.6.6.6, [2001:db8:aa:bb::1]:443", "2001:db8:aa:bb::"),
        ("6.6.6.6, [2001:db8:aa:bb::1]", "2001:db8:aa:bb::"),
        (f"6.6.6.6,   {_REAL}  ", _REAL),
        (f"6.6.6.6,, {_REAL}", _REAL),
    ],
)
def test_port_bracket_and_whitespace_forms(xff: str, expected: str) -> None:
    assert get_client_ip(_req(xff)) == expected


def test_trusted_hops_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert get_client_ip(_req(f"6.6.6.6, {_REAL}, 35.1.1.1")) == _REAL


def test_trusted_hops_two_with_too_few_entries_uses_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "2")
    assert get_client_ip(_req(_REAL, peer="198.51.100.4")) == "198.51.100.4"


def test_rotating_leftmost_never_changes_key() -> None:
    keys = {get_client_ip(_req(f"9.9.9.{i}, 8.8.{i}.1, {_REAL}")) for i in range(50)}
    assert keys == {_REAL}
