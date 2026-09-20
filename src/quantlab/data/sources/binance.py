"""Binance spot klines, perpetual funding history, and instrument reference data.

Crypto is the best free data of any asset class: complete funding history, open
interest, and order books, all without a key. It is the cleanest free market
data there is, and a fair yardstick for how much worse the rest is.

Two things this module exists to get right:

**Survivorship.** ``binance.instruments`` snapshots the traded universe every time
it runs. Delisted symbols stop being served, so a universe built from "symbols
trading today" is a universe of survivors -- the crypto equivalent of the equity
problem, and worse, because listings churn fast. Running that fetcher daily is
what makes a 2022 universe reconstructible in 2026. Every day it does not run is a
day of universe history that cannot be recovered.

**Funding regimes.** The funding formula, cap and interval have all changed over
time and differ per contract. ``interval_hours`` is derived from the observed
spacing of payments rather than assumed to be eight, so a series spanning
a regime change can at least detect that it is comparing different instruments.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.calendars import CRYPTO_VENUE
from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["BinanceFunding", "BinanceInstruments", "BinanceKlines"]

log = get_logger("quantlab.data.sources.binance")

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"

KLINES_URL = f"{SPOT_BASE}/api/v3/klines"
EXCHANGE_INFO_URL = f"{SPOT_BASE}/api/v3/exchangeInfo"
FUNDING_URL = f"{FUTURES_BASE}/fapi/v1/fundingRate"

#: Binance caps both endpoints at 1000 rows per request.
PAGE_LIMIT = 1000

#: Guard against an unbounded pagination loop if the venue stops advancing.
MAX_PAGES = 5000


def _ms(moment: dt.datetime) -> int:
    return int(moment.timestamp() * 1000)


def _from_ms(value: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value / 1000, tz=dt.UTC)


class _Binance(Source):
    spec: ClassVar = get_source("binance")
    asset_class: ClassVar = AssetClass.CRYPTO


@register
class BinanceKlines(_Binance):
    """Spot bars at an explicit interval.

    ``as_of`` is Binance's own ``closeTime``, the last instant of the bar, so no
    calendar is needed -- the venue trades continuously and tells us exactly when
    each bar ended.
    """

    name: ClassVar[str] = "binance.ohlcv_bars"
    dataset: ClassVar[str] = "ohlcv_bars"
    default_symbols: ClassVar[tuple[str, ...]] = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
    )

    def __init__(self, *, interval: str = "1d", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.interval = interval

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            cursor = _ms(start)
            stop = _ms(end)
            unclosed = 0
            for page in range(MAX_PAGES):
                payload = self.client.get_json(
                    KLINES_URL,
                    params={
                        "symbol": symbol,
                        "interval": self.interval,
                        "startTime": cursor,
                        "endTime": stop,
                        "limit": PAGE_LIMIT,
                    },
                )
                if not isinstance(payload, list):
                    raise SourceError(
                        self.spec.key, f"{symbol}: unexpected klines payload {payload!r:.200}"
                    )
                if not payload:
                    break

                now = utcnow()
                for kline in payload:
                    close_time = _from_ms(int(kline[6]))
                    if close_time > now:
                        # Binance serves the CURRENT, incomplete bar with its close
                        # time in the future. Storing it would stamp a mid-session
                        # price as a settled close and make it visible to any
                        # snapshot taken later that day -- look-ahead bias produced
                        # by the venue's own convention.
                        unclosed += 1
                        continue
                    rows.append(
                        {
                            "symbol": symbol,
                            "as_of": close_time,
                            "known_at": close_time,
                            "interval": self.interval,
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                            "quote_volume": float(kline[7]),
                            "trades": int(kline[8]),
                        }
                    )

                last_open = int(payload[-1][0])
                if len(payload) < PAGE_LIMIT or last_open >= stop:
                    break
                cursor = last_open + 1
                if page == MAX_PAGES - 1:
                    raise SourceError(
                        self.spec.key,
                        f"{symbol}: pagination did not terminate after {MAX_PAGES} pages",
                    )

            if unclosed:
                log.info("binance.dropped_unclosed_bars", symbol=symbol, bars=unclosed)

        return self.finalise(rows)


@register
class BinanceFunding(_Binance):
    """Perpetual futures funding payments.

    ``interval_hours`` is measured from consecutive payment timestamps per symbol,
    not assumed, because the interval has changed over time and differs across
    contracts.
    """

    name: ClassVar[str] = "binance.funding_rate"
    dataset: ClassVar[str] = "funding_rate"
    default_symbols: ClassVar[tuple[str, ...]] = (
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
    )

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            cursor = _ms(start)
            stop = _ms(end)
            symbol_rows: list[dict[str, Any]] = []

            for page in range(MAX_PAGES):
                payload = self.client.get_json(
                    FUNDING_URL,
                    params={
                        "symbol": symbol,
                        "startTime": cursor,
                        "endTime": stop,
                        "limit": PAGE_LIMIT,
                    },
                )
                if not isinstance(payload, list):
                    raise SourceError(
                        self.spec.key, f"{symbol}: unexpected funding payload {payload!r:.200}"
                    )
                if not payload:
                    break

                for record in payload:
                    funding_time = _from_ms(int(record["fundingTime"]))
                    mark = record.get("markPrice")
                    symbol_rows.append(
                        {
                            "symbol": symbol,
                            "as_of": funding_time,
                            "known_at": funding_time,
                            "funding_rate": float(record["fundingRate"]),
                            "mark_price": float(mark) if mark not in (None, "") else None,
                        }
                    )

                last_time = int(payload[-1]["fundingTime"])
                if len(payload) < PAGE_LIMIT or last_time >= stop:
                    break
                cursor = last_time + 1
                if page == MAX_PAGES - 1:
                    raise SourceError(
                        self.spec.key,
                        f"{symbol}: pagination did not terminate after {MAX_PAGES} pages",
                    )

            rows.extend(_with_interval_hours(symbol_rows))

        return self.finalise(rows)


def _with_interval_hours(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the observed funding interval to each payment.

    The interval of a payment is the gap since the previous one, which is what
    annualising a funding rate needs. The first payment of a series
    has no predecessor, so it inherits the next gap.
    """
    if not rows:
        return rows
    ordered = sorted(rows, key=lambda r: r["as_of"])
    gaps: list[float | None] = [None]
    for previous, current in itertools.pairwise(ordered):
        # Rounded to 4dp: venue funding timestamps drift by a few milliseconds,
        # which would otherwise report a clean 8-hour interval as 7.9999997 and
        # make a regime change indistinguishable from clock noise.
        gaps.append(round((current["as_of"] - previous["as_of"]).total_seconds() / 3600.0, 4))
    if len(gaps) > 1:
        gaps[0] = gaps[1]
    for row, gap in zip(ordered, gaps, strict=True):
        row["interval_hours"] = gap
    return ordered


@register
class BinanceInstruments(_Binance):
    """A dated snapshot of the traded universe.

    ``as_of`` is the observation instant, not a market event: this row records
    "on this date, Binance listed this symbol with this status". Accumulating
    these daily is the only defence against crypto survivorship bias, since
    delisted symbols simply stop being returned.
    """

    name: ClassVar[str] = "binance.instruments"
    dataset: ClassVar[str] = "instruments"
    asset_class: ClassVar = AssetClass.CRYPTO

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        del start, end  # a universe snapshot has no history; it is always "now"

        payload = self.client.get_json(EXCHANGE_INFO_URL)
        listed = payload.get("symbols")
        if not listed:
            raise SourceError(self.spec.key, "exchangeInfo returned no symbols")

        observed = utcnow()
        wanted = {s.upper() for s in symbols} if symbols else None
        rows = [
            {
                "symbol": entry["symbol"],
                "as_of": observed,
                "known_at": observed,
                "venue": CRYPTO_VENUE,
                "status": str(entry.get("status", "UNKNOWN")),
                "name": entry["symbol"],
                "currency": str(entry.get("quoteAsset", "")),
                "base_asset": str(entry.get("baseAsset", "")),
                "quote_asset": str(entry.get("quoteAsset", "")),
            }
            for entry in listed
            if wanted is None or str(entry["symbol"]).upper() in wanted
        ]
        log.info("binance.universe_snapshot", symbols=len(rows), observed_at=observed.isoformat())
        return self.finalise(rows)
