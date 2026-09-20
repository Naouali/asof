"""Rate limiting, retries, offline refusal, and loud failure."""

from __future__ import annotations

import time

import httpx
import pytest

from quantlab.config import Settings, get_settings
from quantlab.data.catalogue import get_source
from quantlab.data.http import (
    HttpClient,
    OfflineError,
    RateLimiter,
    SourceError,
    SourceUnavailableError,
    require_key,
)

SPEC = get_source("binance")


def client_for(handler, settings: Settings, **kwargs) -> HttpClient:
    return HttpClient(
        SPEC,
        settings=settings,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


# ------------------------------------------------------------------ rate limiter --
def test_token_bucket_allows_a_burst_then_throttles() -> None:
    limiter = RateLimiter(20.0)
    began = time.monotonic()
    for _ in range(40):
        limiter.acquire()
    elapsed = time.monotonic() - began
    # 20 burst tokens, then 20 more at 20/s == ~1s.
    assert 0.8 < elapsed < 1.6


def test_penalise_empties_the_bucket() -> None:
    limiter = RateLimiter(100.0)
    limiter.penalise(0.3)
    began = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - began >= 0.25


def test_rate_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(0)


# ------------------------------------------------------------------------ client --
def test_successful_request(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbol"] == "BTCUSDT"
        return httpx.Response(200, json={"ok": True})

    with client_for(handler, settings) as client:
        assert client.get_json("https://example.com/x", params={"symbol": "BTCUSDT"}) == {
            "ok": True
        }


def test_user_agent_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """SEC EDGAR rejects requests without one, so this must never be dropped."""
    monkeypatch.setenv("QUANTLAB_HTTP_USER_AGENT", "QuantLab/test (me@example.com)")
    get_settings.cache_clear()
    settings = get_settings()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, json={})

    client = HttpClient(
        SPEC,
        settings=settings,
        client=httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"User-Agent": settings.http_user_agent},
        ),
    )
    with client:
        client.get_json("https://example.com/x")
    assert seen == ["QuantLab/test (me@example.com)"]


def test_offline_mode_refuses_before_any_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTLAB_OFFLINE", "true")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("offline mode must not reach the transport")

    with (
        client_for(handler, get_settings()) as client,
        pytest.raises(OfflineError, match="offline mode"),
    ):
        client.get_json("https://example.com/x")


def test_non_https_is_refused(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not be reached")

    with client_for(handler, settings) as client, pytest.raises(SourceError, match="non-HTTPS"):
        client.get_json("http://example.com/x")


def test_client_error_fails_immediately_without_retrying(settings: Settings) -> None:
    """A 404 is a real answer: the symbol does not exist. Retrying it wastes the
    provider's goodwill and hides the cause."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, text="not found")

    with client_for(handler, settings) as client, pytest.raises(SourceError, match="HTTP 404"):
        client.get_json("https://example.com/x")
    assert len(calls) == 1


def test_server_error_is_retried_then_raises(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(HttpClient, "_delay", staticmethod(lambda _attempt: 0.0))
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with client_for(handler, settings) as client, pytest.raises(SourceError, match="gave up"):
        client.get_json("https://example.com/x")
    assert len(calls) == settings.http_max_retries


def test_retry_then_success(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HttpClient, "_delay", staticmethod(lambda _attempt: 0.0))
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(500)
        return httpx.Response(200, json={"recovered": True})

    with client_for(handler, settings) as client:
        assert client.get_json("https://example.com/x") == {"recovered": True}
    assert len(attempts) == 3


def test_429_honours_retry_after(settings: Settings) -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "0.1"})
        return httpx.Response(200, json={})

    began = time.monotonic()
    with client_for(handler, settings) as client:
        client.get_json("https://example.com/x")
    assert time.monotonic() - began >= 0.1


def test_binance_weight_header_triggers_preemptive_backoff(settings: Settings) -> None:
    """Crossing Binance's weight limit earns an IP ban, which corrupts an ingest
    run halfway through instead of failing it."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={}, headers={"x-mbx-used-weight-1m": "1100"})

    with client_for(handler, settings) as client:
        client.get_json("https://example.com/x")
        began = time.monotonic()
        client.limiter.acquire()
        assert time.monotonic() - began > 5.0


def test_transport_error_is_retried(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HttpClient, "_delay", staticmethod(lambda _attempt: 0.0))
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"ok": 1})

    with client_for(handler, settings) as client:
        assert client.get_json("https://example.com/x") == {"ok": 1}


def test_non_json_response_names_the_content_type(settings: Settings) -> None:
    """An HTML error page parsed as JSON is how a scraper failure turns into a
    confusing crash three modules downstream."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>nope</html>", headers={"content-type": "text/html"})

    with (
        client_for(handler, settings) as client,
        pytest.raises(SourceError, match="rather than JSON"),
    ):
        client.get_json("https://example.com/x")


# --------------------------------------------------------------------- key check --
def test_require_key_names_the_variable_and_the_free_signup(settings: Settings) -> None:
    with pytest.raises(SourceUnavailableError) as excinfo:
        require_key(get_source("fred"), settings)
    message = str(excinfo.value)
    assert "QUANTLAB_FRED_API_KEY" in message
    assert "https://" in message


def test_require_key_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTLAB_FRED_API_KEY", "abc")
    get_settings.cache_clear()
    assert require_key(get_source("fred"), get_settings()) == "abc"


def test_keyless_source_rejects_a_key_request(settings: Settings) -> None:
    with pytest.raises(SourceError, match="takes no API key"):
        require_key(get_source("binance"), settings)
