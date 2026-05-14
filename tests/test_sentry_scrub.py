"""Unit tests for Sentry token-scrubbing hooks in api.main."""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from api.main import _scrub_token_from_url, _sentry_before_breadcrumb, _sentry_before_send


# ---------------------------------------------------------------------------
# _scrub_token_from_url
# ---------------------------------------------------------------------------

def test_scrub_replaces_token_value():
    url = "wss://example.com/chat/stream?token=eyJfake&other=keep"
    result = _scrub_token_from_url(url)
    params = parse_qs(urlparse(result).query)
    assert params["token"] == ["[Filtered]"]
    assert params["other"] == ["keep"]


def test_scrub_leaves_no_token_url_unchanged():
    url = "https://example.com/chat?foo=bar&baz=qux"
    assert _scrub_token_from_url(url) == url


def test_scrub_url_without_query_unchanged():
    url = "https://example.com/health"
    assert _scrub_token_from_url(url) == url


def test_scrub_preserves_other_params_order():
    url = "https://example.com/?a=1&token=secret&b=2"
    result = _scrub_token_from_url(url)
    params = parse_qs(urlparse(result).query)
    assert params["token"] == ["[Filtered]"]
    assert params["a"] == ["1"]
    assert params["b"] == ["2"]


# ---------------------------------------------------------------------------
# _sentry_before_breadcrumb
# ---------------------------------------------------------------------------

def test_breadcrumb_http_url_scrubbed():
    crumb = {
        "type": "http",
        "category": "http",
        "data": {"url": "wss://example.com/chat/stream?token=eyJfake&other=keep"},
    }
    result = _sentry_before_breadcrumb(crumb, {})
    params = parse_qs(urlparse(result["data"]["url"]).query)
    assert params["token"] == ["[Filtered]"]
    assert params["other"] == ["keep"]


def test_breadcrumb_navigation_from_to_scrubbed():
    crumb = {
        "type": "default",
        "category": "navigation",
        "data": {
            "from": "/old?token=abc",
            "to": "/new?token=xyz&keep=1",
        },
    }
    result = _sentry_before_breadcrumb(crumb, {})
    from_params = parse_qs(urlparse(result["data"]["from"]).query)
    assert from_params["token"] == ["[Filtered]"]
    assert "abc" not in result["data"]["from"]
    to_params = parse_qs(urlparse(result["data"]["to"]).query)
    assert to_params["token"] == ["[Filtered]"]
    assert to_params["keep"] == ["1"]


def test_breadcrumb_no_token_untouched():
    crumb = {
        "category": "http",
        "data": {"url": "https://example.com/api?foo=bar"},
    }
    result = _sentry_before_breadcrumb(crumb, {})
    assert result["data"]["url"] == "https://example.com/api?foo=bar"


def test_breadcrumb_no_data_returned():
    crumb = {"category": "console", "message": "hello"}
    result = _sentry_before_breadcrumb(crumb, {})
    assert result is crumb


# ---------------------------------------------------------------------------
# _sentry_before_send
# ---------------------------------------------------------------------------

def test_before_send_scrubs_request_url():
    event = {
        "request": {"url": "https://example.com/chat/stream?token=eyJ123&x=y"},
    }
    result = _sentry_before_send(event, {})
    params = parse_qs(urlparse(result["request"]["url"]).query)
    assert params["token"] == ["[Filtered]"]
    assert params["x"] == ["y"]


def test_before_send_scrubs_query_string_str():
    event = {"request": {"query_string": "token=eyJsecret&other=keep"}}
    result = _sentry_before_send(event, {})
    qs_params = parse_qs(result["request"]["query_string"])
    assert qs_params["token"] == ["[Filtered]"]
    assert qs_params["other"] == ["keep"]


def test_before_send_scrubs_query_string_list():
    event = {
        "request": {
            "query_string": [["token", "eyJsecret"], ["keep", "value"]],
        }
    }
    result = _sentry_before_send(event, {})
    qs = dict(result["request"]["query_string"])
    assert qs["token"] == "[Filtered]"
    assert qs["keep"] == "value"


def test_before_send_no_request_key():
    event = {"exception": {"values": []}}
    result = _sentry_before_send(event, {})
    assert result is event
