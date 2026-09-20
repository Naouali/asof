"""Yahoo Finance daily bars and corporate actions.

Implemented directly against the public ``v8/finance/chart`` endpoint rather than
through the ``yfinance`` package. Two reasons: the dependency is a scraper that
changes shape without notice, and going direct makes the exact fields we rely on
visible in this file instead of buried behind a convenience layer.

**Two restatement problems you must understand before using this data.**

1. ``adj_close`` is Yahoo's own dividend-and-split adjustment, computed with an
   undocumented methodology and *recomputed* whenever Yahoo reprocesses a
   corporate action. It is stored, because it is useful for a quick look, but its
   ``known_at`` is a lie: we record it as the session close like the rest of the
   row, while in truth today's value was not knowable then. Do not rely
   on it. Build total returns from ``close`` plus the ``corporate_actions``
   dataset, whose dividends and splits each carry their own ex-date.

2. ``open``/``high``/``low``/``close`` are **split-adjusted retroactively**, and
   not only by splits that had already happened. A May 2014 Apple close comes back
   as $21.12; the real figure was $591.48, divided by the 7:1 split that June *and*
   by the 4:1 split six years later. This does not affect *returns*, which is what
   almost every signal consumes, but it makes any price-*level* signal --
   penny-stock filters, round-number effects, nominal price momentum -- wrong in a
   way that will not raise.

Both facts are recorded in the catalogue.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.calendars import (
    UnknownVenueError,
    resolve_venue,
    session_close,
    sessions,
)
from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["YahooCorporateActions", "YahooDailyBars"]

log = get_logger("quantlab.data.sources.yahoo")

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

#: Yahoo sometimes returns materially less history than requested, with no error
#: and no indication in the payload. Observed 2026-09-20: the same URL for SPY
#: returned 5030 bars starting 2006-09-20 on one call and 5462 bars starting
#: 2005-01-03 minutes later. A silently shortened history changes every
#: conclusion drawn from it without anyone noticing, so coverage is verified and the
#: request retried before the data is accepted.
COVERAGE_ATTEMPTS = 3

#: How far after the requested start the first bar may legitimately fall --
#: enough to absorb a weekend plus a holiday, not enough to hide a lost year.
COVERAGE_TOLERANCE = dt.timedelta(days=10)

#: When Yahoo truncates, it also reports ``firstTradeDate`` as the start of the
#: truncated range, so the payload is internally consistent and a metadata check
#: cannot detect the loss. The defence is therefore to AVOID the cap rather than
#: detect it: history is requested in windows of this length, which are served in
#: full. The checks below remain as backstops.
CHUNK = dt.timedelta(days=365 * 8)

#: Fraction of the venue's trading sessions that must actually come back between
#: the first and last bar received. Computed from our own exchange calendar with
#: no Yahoo metadata, so it holds even when the payload lies about itself, and it
#: catches holes in the middle of a history, which no metadata check can see.
MIN_SESSION_COVERAGE = 0.95


def _windows(start: dt.datetime, end: dt.datetime) -> list[tuple[dt.datetime, dt.datetime]]:
    """Split a range into windows Yahoo serves in full."""
    out: list[tuple[dt.datetime, dt.datetime]] = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + CHUNK)
        out.append((cursor, stop))
        cursor = stop + dt.timedelta(days=1)
    return out


def _check_session_coverage(symbol: str, venue: str, rows: list[dict[str, Any]]) -> None:
    """Assert the returned bars actually cover the venue's sessions."""
    received = {row["as_of"].date() for row in rows}
    first, last = min(received), max(received)
    expected = sessions(venue, first, last)
    if not expected:
        return
    coverage = len(received & set(expected)) / len(expected)
    if coverage >= MIN_SESSION_COVERAGE:
        return
    raise SourceError(
        "yahoo",
        f"{symbol}: only {coverage:.1%} of {venue} sessions between {first} and "
        f"{last} came back ({len(received)} bars for {len(expected)} sessions). "
        "Yahoo dropped days in the middle of the history without reporting it; "
        "refusing to store a series with silent holes.",
    )


class _YahooChart(Source):
    """Shared access to the chart endpoint."""

    spec: ClassVar = get_source("yahoo")
    asset_class: ClassVar = AssetClass.EQUITY

    def _chart_verified(self, symbol: str, start: dt.datetime, end: dt.datetime) -> dict[str, Any]:
        """Fetch a chart and verify Yahoo actually served the window we asked for.

        Yahoo truncates history non-deterministically. Retrying usually gets the
        full range; when it does not, this raises rather than accepting a shorter
        sample, because 2006-2026 instead of 2005-2026 is a different history
        and nothing else would say so.
        """
        shortfall = ""
        for attempt in range(1, COVERAGE_ATTEMPTS + 1):
            result = self._chart(symbol, start, end)
            timestamps = result.get("timestamp") or []
            if not timestamps:
                return result  # genuinely empty windows are handled by the caller

            meta = result.get("meta") or {}
            listed = meta.get("firstTradeDate")
            earliest_possible = (
                max(start, dt.datetime.fromtimestamp(int(listed), tz=dt.UTC)) if listed else start
            )
            first_bar = dt.datetime.fromtimestamp(int(timestamps[0]), tz=dt.UTC)
            gap = first_bar - earliest_possible
            if gap <= COVERAGE_TOLERANCE:
                return result

            shortfall = (
                f"asked for history from {earliest_possible.date()} but the earliest "
                f"bar returned was {first_bar.date()} ({gap.days} days short)"
            )
            log.warning(
                "yahoo.truncated_history",
                symbol=symbol,
                attempt=attempt,
                requested_start=earliest_possible.date().isoformat(),
                received_start=first_bar.date().isoformat(),
                days_short=gap.days,
            )

        raise SourceError(
            self.spec.key,
            f"{symbol}: {shortfall}, after {COVERAGE_ATTEMPTS} attempts. Yahoo "
            "truncates history without reporting it. Refusing to store a silently "
            "shortened sample -- re-run, or narrow the requested window.",
        )

    def _chart(self, symbol: str, start: dt.datetime, end: dt.datetime) -> dict[str, Any]:
        payload = self.client.get_json(
            CHART_URL.format(symbol=symbol),
            params={
                "period1": int(start.timestamp()),
                # Yahoo's period2 is exclusive of the bar starting at that instant;
                # one extra day makes the requested end date inclusive.
                "period2": int((end + dt.timedelta(days=1)).timestamp()),
                "interval": "1d",
                "events": "div,split",
                "includeAdjustedClose": "true",
            },
        )
        chart = payload.get("chart") or {}
        if chart.get("error"):
            raise SourceError(self.spec.key, f"{symbol}: {chart['error']}")
        results = chart.get("result")
        if not results:
            raise SourceError(
                self.spec.key,
                f"{symbol}: chart response carried neither a result nor an error; "
                f"Yahoo's unofficial API has probably changed shape: {payload!r:.300}",
            )
        return dict(results[0])

    @staticmethod
    def _venue(meta: dict[str, Any], symbol: str) -> str:
        for field in ("fullExchangeName", "exchangeName"):
            name = meta.get(field)
            if not name:
                continue
            try:
                return resolve_venue(str(name))
            except UnknownVenueError:
                continue
        raise UnknownVenueError(
            f"{symbol}: Yahoo reports exchange "
            f"{meta.get('fullExchangeName') or meta.get('exchangeName')!r}, which has "
            "no calendar mapping. Add it to quantlab.data.calendars.VENUE_ALIASES; do "
            "not let it default to a US calendar."
        )


@register
class YahooDailyBars(_YahooChart):
    """Daily OHLCV bars.

    ``as_of`` is the session close in UTC, taken from the venue's calendar rather
    than from Yahoo's timestamp (which is the session *open*). ``known_at`` equals
    ``as_of``: a closing price is knowable when the session closes, and is not
    revised -- with the split-adjustment caveat in the module docstring.
    """

    name: ClassVar[str] = "yahoo.ohlcv_daily"
    dataset: ClassVar[str] = "ohlcv_daily"
    default_symbols: ClassVar[tuple[str, ...]] = (
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "AAPL",
        "MSFT",
        "JPM",
        "XOM",
        "JNJ",
    )

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            symbol_rows: list[dict[str, Any]] = []
            venue: str | None = None

            for window_start, window_end in _windows(start, end):
                result = self._chart_verified(symbol, window_start, window_end)
                meta = result.get("meta") or {}
                timestamps = result.get("timestamp") or []
                if not timestamps:
                    # A window before the symbol listed. Legitimately empty.
                    continue

                venue = self._venue(meta, symbol)
                currency = str(meta.get("currency") or "")
                exchange_tz = _zone(str(meta.get("exchangeTimezoneName") or "UTC"))

                quote = (result["indicators"]["quote"] or [{}])[0]
                adjusted = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")

                unmatched = 0
                unclosed = 0
                now = utcnow()
                for index, epoch in enumerate(timestamps):
                    # Yahoo's daily timestamp is the session OPEN in exchange-local
                    # time. The session it belongs to is that instant's local date.
                    session_date = dt.datetime.fromtimestamp(epoch, tz=exchange_tz).date()
                    try:
                        as_of = session_close(venue, session_date)
                    except ValueError:
                        # A bar on a day the calendar has no session. Counted and
                        # reported rather than snapped to a neighbouring session,
                        # which would corrupt every later alignment.
                        unmatched += 1
                        continue

                    if as_of > now:
                        # The current, still-open session. Yahoo reports a live price
                        # for it; storing that as a close would let a snapshot taken
                        # later today read a mid-session quote as a settled close.
                        unclosed += 1
                        continue

                    close = _at(quote.get("close"), index)
                    if close is None:
                        # Yahoo pads holidays and halts with nulls. A bar with no
                        # close is not an observation.
                        continue

                    symbol_rows.append(
                        {
                            "symbol": symbol,
                            "as_of": as_of,
                            "known_at": as_of,
                            "open": _at(quote.get("open"), index),
                            "high": _at(quote.get("high"), index),
                            "low": _at(quote.get("low"), index),
                            "close": close,
                            "volume": _at(quote.get("volume"), index),
                            "adj_close": _at(adjusted, index),
                            "currency": currency,
                            "venue": venue,
                        }
                    )

                if unclosed:
                    log.info("yahoo.dropped_unclosed_session", symbol=symbol, bars=unclosed)
                if unmatched:
                    log.warning(
                        "yahoo.bars_off_calendar",
                        symbol=symbol,
                        venue=venue,
                        dropped=unmatched,
                        reason="Yahoo returned bars on days the exchange calendar has no session",
                    )

            if symbol_rows and venue is not None:
                _check_session_coverage(symbol, venue, symbol_rows)
            rows.extend(symbol_rows)

        return self.finalise(rows)


@register
class YahooCorporateActions(_YahooChart):
    """Dividends and splits, each carrying its own ex-date as ``as_of``.

    This is the honest route to total returns: apply these to raw closes yourself
    rather than trusting ``adj_close``, whose value changes under you.

    Delistings are **not** here, because Yahoo does not serve them. Missing
    delisting returns inflate short-leg performance, which is exactly why the
    equity limitations are stated as loudly as they are.
    """

    name: ClassVar[str] = "yahoo.corporate_actions"
    dataset: ClassVar[str] = "corporate_actions"
    default_symbols: ClassVar[tuple[str, ...]] = YahooDailyBars.default_symbols

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            for window_start, window_end in _windows(start, end):
                rows.extend(self._events_in(symbol, window_start, window_end))
        return self.finalise(rows)

    def _events_in(self, symbol: str, start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]]:
        result = self._chart_verified(symbol, start, end)
        events = result.get("events") or {}
        if not events:
            return []

        meta = result.get("meta") or {}
        venue = self._venue(meta, symbol)
        exchange_tz = _zone(str(meta.get("exchangeTimezoneName") or "UTC"))
        rows: list[dict[str, Any]] = []

        for event in (events.get("dividends") or {}).values():
            as_of = _event_instant(event, venue, exchange_tz)
            if as_of is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "as_of": as_of,
                    "known_at": as_of,
                    "action": "dividend",
                    "amount": float(event["amount"]),
                }
            )

        for event in (events.get("splits") or {}).values():
            as_of = _event_instant(event, venue, exchange_tz)
            if as_of is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "as_of": as_of,
                    "known_at": as_of,
                    "action": "split",
                    "split_numerator": float(event["numerator"]),
                    "split_denominator": float(event["denominator"]),
                }
            )

        return rows


def _event_instant(event: dict[str, Any], venue: str, exchange_tz: dt.tzinfo) -> dt.datetime | None:
    """The session close on a corporate action's ex-date."""
    raw = event.get("date")
    if raw is None:
        return None
    day = dt.datetime.fromtimestamp(int(raw), tz=exchange_tz).date()
    try:
        return session_close(venue, day)
    except ValueError:
        log.warning("yahoo.event_off_calendar", venue=venue, day=str(day))
        return None


def _at(values: Any, index: int) -> float | None:
    if not values or index >= len(values):
        return None
    value = values[index]
    return None if value is None else float(value)


def _zone(name: str) -> dt.tzinfo:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise SourceError("yahoo", f"unknown exchange timezone {name!r}") from exc
