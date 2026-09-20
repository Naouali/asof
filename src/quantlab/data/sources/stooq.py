"""Stooq daily bars.

**Status as of 2026-09-20: the CSV endpoint is gated behind a JavaScript
proof-of-work anti-bot challenge and does not serve data to programmatic
clients.** A request to ``https://stooq.com/q/d/l/?s=aapl.us&i=d`` returns HTTP 200
carrying an HTML interstitial that computes a SHA-256 hash puzzle in the browser
and posts the solution to ``/__verify`` before any CSV is released.

QuantLab does not solve that challenge. It is an access control the operator
deliberately deployed, and defeating it would be both a terms-of-service problem
and the kind of thing that gets an IP banned mid-ingest. So this fetcher detects
the interstitial and fails loudly: a different data source is never quietly
substituted when one fails.

**Consequence:** Yahoo is the primary free equity price source, and the
cross-check Stooq was meant to provide -- an independent opinion on Yahoo's
undocumented adjustments -- is not available. That is a real reduction in data
quality assurance and is recorded in docs/LIMITATIONS.md.

If Stooq later drops the challenge, or if you obtain access through their bulk
data offering, the parser below is complete and correct; only the transport is
blocked.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.calendars import session_close
from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError, SourceUnavailableError
from quantlab.data.sources.base import Source, register
from quantlab.logging import degraded, get_logger

__all__ = ["StooqDailyBars", "parse_stooq_csv"]

log = get_logger("quantlab.data.sources.stooq")

CSV_URL = "https://stooq.com/q/d/l/"

#: Markers of the proof-of-work interstitial described in the module docstring.
CHALLENGE_MARKERS = ("__verify", "requires JavaScript", "crypto.subtle.digest")


def parse_stooq_csv(text: str, symbol: str, venue: str) -> list[dict[str, Any]]:
    """Parse a Stooq daily CSV into canonical rows.

    Kept separate from the transport so it is testable against a recorded fixture
    while the live endpoint is blocked.
    """
    frame = pl.read_csv(io.StringIO(text), try_parse_dates=True)
    expected = {"Date", "Open", "High", "Low", "Close"}
    missing = expected - set(frame.columns)
    if missing:
        raise SourceError("stooq", f"{symbol}: CSV is missing columns {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for record in frame.iter_rows(named=True):
        day = record["Date"]
        if isinstance(day, dt.datetime):
            day = day.date()
        try:
            as_of = session_close(venue, day)
        except ValueError:
            log.warning("stooq.bar_off_calendar", symbol=symbol, venue=venue, day=str(day))
            continue
        close = record["Close"]
        if close is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "as_of": as_of,
                "known_at": as_of,
                "open": record["Open"],
                "high": record["High"],
                "low": record["Low"],
                "close": close,
                "volume": record.get("Volume"),
                # Stooq publishes no separate adjusted series and documents no
                # adjustment methodology, so there is nothing honest to put here.
                "adj_close": None,
                "currency": "",
                "venue": venue,
            }
        )
    return rows


@register
class StooqDailyBars(Source):
    """Daily bars from Stooq. Currently blocked -- see the module docstring."""

    name: ClassVar[str] = "stooq.ohlcv_daily"
    spec: ClassVar = get_source("stooq")
    dataset: ClassVar[str] = "ohlcv_daily"
    asset_class: ClassVar = AssetClass.EQUITY
    default_symbols: ClassVar[tuple[str, ...]] = ()
    blocked_reason: ClassVar[str | None] = (
        "blocked by a JavaScript proof-of-work anti-bot challenge (2026-09-20); "
        "QuantLab does not solve anti-bot challenges"
    )

    #: Stooq suffixes to calendar codes. Only the ones we would actually use.
    VENUE_BY_SUFFIX: ClassVar[dict[str, str]] = {
        "us": "XNYS",
        "uk": "XLON",
        "de": "XETR",
        "jp": "XTKS",
    }

    def venue_for(self, symbol: str) -> str:
        suffix = symbol.rsplit(".", 1)[-1].lower() if "." in symbol else ""
        try:
            return self.VENUE_BY_SUFFIX[suffix]
        except KeyError:
            raise SourceError(
                self.spec.key,
                f"cannot infer a venue for {symbol!r}. Stooq encodes the market in the "
                f"ticker suffix; known suffixes: {sorted(self.VENUE_BY_SUFFIX)}",
            ) from None

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            venue = self.venue_for(symbol)
            text = self.client.get_text(
                CSV_URL,
                params={
                    "s": symbol,
                    "i": "d",
                    "d1": start.strftime("%Y%m%d"),
                    "d2": end.strftime("%Y%m%d"),
                },
            )
            if any(marker in text for marker in CHALLENGE_MARKERS):
                degraded(
                    self.spec.key,
                    "anti-bot proof-of-work challenge returned instead of CSV",
                    symbol=symbol,
                )
                raise SourceUnavailableError(
                    self.spec.key,
                    "Stooq served a JavaScript proof-of-work challenge instead of CSV. "
                    "QuantLab does not solve anti-bot challenges. Equity prices are "
                    "coming from Yahoo instead, and the independent cross-check Stooq "
                    "was meant to provide is unavailable -- see docs/LIMITATIONS.md. "
                    "Disable this source in your ingest config rather than retrying.",
                )
            rows.extend(parse_stooq_csv(text, symbol, venue))

        return self.finalise(rows)
