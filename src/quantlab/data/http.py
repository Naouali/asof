"""Shared HTTP client for data sources.

Free data providers are generous and fragile in equal measure. This module holds
the politeness and failure policy in one place so that no source module has to
reinvent it, and so that the rules are auditable:

* **Rate limits come from the catalogue**, not from each source's own guesswork.
  A token bucket per source enforces ``max_requests_per_second``, and Binance's
  weight headers additionally throttle us before the venue does.
* **Offline mode is a hard refusal.** Once ingest has run, everything else is
  supposed to read the lake. With ``QUANTLAB_OFFLINE=true``, a stray network call raises
  instead of quietly re-fetching and producing results that depend on the day.
* **Failure is loud.** There is no fallback source, no empty frame on error, no
  swallowed exception. Never quietly substitute a different data
  source when one fails.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Any

import httpx

from quantlab.config import Settings, get_settings
from quantlab.data.catalogue import SourceSpec
from quantlab.logging import get_logger

__all__ = [
    "HttpClient",
    "OfflineError",
    "RateLimiter",
    "SourceError",
    "SourceUnavailableError",
    "require_key",
]

log = get_logger("quantlab.data.http")


class SourceError(RuntimeError):
    """A data source failed. Never caught and turned into an empty result."""

    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"[{source}] {message}")
        self.source = source


class SourceUnavailableError(SourceError):
    """A source cannot be used at all: missing credentials, or blocked access."""


class OfflineError(SourceError):
    """A network call was attempted while offline mode is on."""


class RateLimiter:
    """A token bucket, shared across threads.

    Burst capacity is one second's worth of requests, so a short burst is allowed
    but the sustained rate is exactly the catalogued limit.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.rate = requests_per_second
        self.capacity = max(1.0, requests_per_second)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may be made. Returns how long it waited."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait = (1.0 - self._tokens) / self.rate
            self._tokens = 0.0
            self._updated = now + wait
        time.sleep(wait)
        return wait

    def penalise(self, seconds: float) -> None:
        """Empty the bucket and hold off for ``seconds`` -- used after a 429."""
        with self._lock:
            self._tokens = 0.0
            self._updated = time.monotonic() + seconds


class HttpClient:
    """An HTTP client bound to one catalogued source."""

    #: Status codes worth retrying. 429 is rate limiting; 5xx is the provider.
    RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        spec: SourceSpec,
        *,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.spec = spec
        self.settings = settings or get_settings()
        self.limiter = rate_limiter or RateLimiter(spec.max_requests_per_second)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(self.settings.http_timeout_seconds),
            follow_redirects=True,
            headers={"User-Agent": self.settings.http_user_agent},
        )

    # ------------------------------------------------------------- lifecycle --
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---------------------------------------------------------------- request --
    def request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> httpx.Response:
        """One request with rate limiting, retries and loud failure.

        A GET, unless a body is given. ``data`` is sent as a form POST, which is
        how a site with a search form wants to be asked; ``json`` as a JSON POST,
        which is how an API that takes a query document does. Cookies persist for
        the life of the client, so a session opened by one request serves the next.
        """
        if self.settings.offline:
            raise OfflineError(
                self.spec.key,
                f"offline mode is on, refusing to fetch {url}. Readers must use the "
                "lake; if you meant to ingest, unset QUANTLAB_OFFLINE.",
            )
        if not url.startswith("https://"):
            raise SourceError(self.spec.key, f"refusing a non-HTTPS request to {url}")

        attempts = max(1, self.settings.http_max_retries)
        last_error: str = "no attempt was made"

        for attempt in range(1, attempts + 1):
            waited = self.limiter.acquire()
            if waited:
                log.debug("http.throttled", source=self.spec.key, waited_seconds=round(waited, 3))
            try:
                if data is None and json is None:
                    response = self._client.get(url, params=params, headers=headers)
                else:
                    response = self._client.post(
                        url, params=params, headers=headers, data=data, json=json
                    )
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "http.transport_error",
                    source=self.spec.key,
                    url=url,
                    attempt=attempt,
                    error=last_error,
                )
                self._backoff(attempt)
                continue

            self._observe(response)

            if response.status_code in self.RETRY_STATUS:
                last_error = f"HTTP {response.status_code}"
                retry_after = self._retry_after(response)
                log.warning(
                    "http.retryable_status",
                    source=self.spec.key,
                    url=url,
                    status=response.status_code,
                    attempt=attempt,
                    retry_after=retry_after,
                )
                if response.status_code == 429:
                    self.limiter.penalise(retry_after or self._delay(attempt))
                self._backoff(attempt, retry_after)
                continue

            if response.status_code >= 400:
                # A 4xx that is not rate limiting is a real answer: the symbol does
                # not exist, the key is wrong, the jurisdiction is blocked. Retrying
                # it wastes the provider's goodwill and hides the cause.
                raise SourceError(
                    self.spec.key,
                    f"HTTP {response.status_code} for {url}: {response.text[:300]}",
                )

            return response

        raise SourceError(
            self.spec.key,
            f"gave up on {url} after {attempts} attempts; last failure: {last_error}",
        )

    def get_json(self, url: str, **kwargs: Any) -> Any:
        response = self.request(url, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise SourceError(
                self.spec.key,
                f"{url} returned {response.headers.get('content-type', 'unknown')} "
                f"rather than JSON: {response.text[:200]!r}",
            ) from exc

    def get_text(self, url: str, **kwargs: Any) -> str:
        return self.request(url, **kwargs).text

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        """Raw body, for payloads that are neither JSON nor text -- a zip, or a
        CSV whose encoding the source does not declare."""
        return self.request(url, **kwargs).content

    # ---------------------------------------------------------------- helpers --
    def _observe(self, response: httpx.Response) -> None:
        """Back off pre-emptively when a venue tells us how close we are to a ban.

        Binance publishes used weight per minute. Crossing its limit earns an IP
        ban, which corrupts an ingest run halfway through rather than failing it.
        """
        used = response.headers.get("x-mbx-used-weight-1m")
        if used is None:
            return
        try:
            weight = int(used)
        except ValueError:
            return
        if weight > 1000:
            log.warning("http.weight_high", source=self.spec.key, used_weight_1m=weight)
            self.limiter.penalise(10.0)
        elif weight > 800:
            self.limiter.penalise(2.0)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    @staticmethod
    def _delay(attempt: int) -> float:
        # Deterministic exponential backoff, capped. No jitter: a reproducible log
        # is worth more here than the marginal benefit of decorrelating retries
        # from a single-process ingester.
        return min(60.0, 2.0 ** (attempt - 1))

    def _backoff(self, attempt: int, retry_after: float | None = None) -> None:
        time.sleep(retry_after if retry_after is not None else self._delay(attempt))


def require_key(spec: SourceSpec, settings: Settings | None = None) -> str:
    """Return the configured API key for a source, or fail with a usable message."""
    settings = settings or get_settings()
    if spec.key_setting is None:
        raise SourceError(spec.key, "this source takes no API key")
    value = settings.secret_for(spec.key_setting)
    if not value:
        raise SourceUnavailableError(
            spec.key,
            f"no API key configured. Set QUANTLAB_{spec.key_setting.upper()} in .env. "
            f"The key is free: {spec.url}",
        )
    return value
