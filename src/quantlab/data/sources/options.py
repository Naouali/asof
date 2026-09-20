"""Option chain snapshots.

**This is the one dataset that cannot be backfilled.** Every other source here
can be re-fetched from the beginning; no free source sells historical option
chains at all. The history starts the day this collector first runs, and every
day it does not run is a day that can never be recovered. That makes it the same
class of problem as the Binance instrument snapshots: the cost of not running it
is paid later and cannot be refunded.

Quotes come from CBOE's delayed-quote files rather than from a broker or a
scraped portal. They are the exchange's own, they carry bid, ask, size, volume,
open interest, implied volatility and greeks, and they need no credentials. The
cost is that they are delayed and that a snapshot is a snapshot: there is no
intraday path, so anything requiring one -- realised gamma P&L, intraday hedging
error -- cannot be reconstructed from this no matter how long it is collected.

**On the greeks and the implied volatilities.** They are CBOE's, computed with
CBOE's own model, dividend assumption and rate curve, none of which is published
alongside the numbers. They are stored because throwing away a free field is
worse than storing a documented one, and they should be recomputed from the mid
price before anything is traded on them. A greek you did not compute is a greek
whose assumptions you do not know.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["OCC_PATTERN", "CboeOptionChain", "parse_occ_symbol"]

log = get_logger("quantlab.data.sources.options")

QUOTE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"

#: The OCC option symbol: root, then YYMMDD, then C or P, then the strike in
#: thousandths. ``SPX261016C00200000`` is the SPX 200 call expiring 2026-10-16.
OCC_PATTERN = re.compile(r"^(?P<root>[A-Z0-9]+?)(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")

#: Strikes are encoded as thousandths of a currency unit.
STRIKE_SCALE = 1000.0

#: Index options are requested with a leading underscore; single names are not.
INDEX_TICKERS = frozenset({"SPX", "VIX", "NDX", "RUT", "DJX", "XSP"})


def parse_occ_symbol(symbol: str) -> tuple[str, dt.date, str, float] | None:
    """``(root, expiry, right, strike)`` from an OCC symbol, or ``None``.

    Returns ``None`` rather than raising for an unrecognised symbol, because CBOE
    occasionally lists non-standard series and one of them must not abort a whole
    snapshot -- but the count of them is logged, so a format change shows up as a
    number rather than as silence.
    """
    match = OCC_PATTERN.match(symbol.strip().upper())
    if match is None:
        return None
    try:
        expiry = dt.datetime.strptime(match["expiry"], "%y%m%d").replace(tzinfo=dt.UTC).date()
    except ValueError:
        return None
    return match["root"], expiry, match["right"], int(match["strike"]) / STRIKE_SCALE


@register
class CboeOptionChain(Source):
    """A point-in-time snapshot of one underlying's option chain."""

    name: ClassVar[str] = "options_snapshot.chain_snapshot"
    dataset: ClassVar[str] = "chain_snapshot"
    asset_class: ClassVar[AssetClass] = AssetClass.OPTIONS
    spec: ClassVar = get_source("options_snapshot")

    #: Kept very small. A single SPX chain is 27,000 quotes and 12 MB of JSON, so
    #: a wide default would fill a disk within months for data nobody asked for.
    default_symbols: ClassVar[tuple[str, ...]] = ("SPX",)

    def __init__(self, *args: Any, require_open_interest: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: Quotes on strikes nobody holds are the bulk of a chain and the least
        #: trustworthy part of it: wide, stale, sometimes crossed. Dropping them
        #: is both a storage decision and a data-quality one.
        self.require_open_interest = require_open_interest

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        del start  # a snapshot has no window: it is whatever the chain is now
        tickers = list(dict.fromkeys(symbols)) or list(self.default_symbols)

        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            rows.extend(self._rows_for(ticker, end))

        if not rows:
            raise SourceError(
                self.name,
                f"no option quotes for {tickers}. A snapshot cannot be backfilled, "
                "so an empty result is a day of history lost rather than a query "
                "that can be retried later with a wider window.",
            )
        return self.finalise(rows)

    def _rows_for(self, ticker: str, end: dt.datetime) -> list[dict[str, Any]]:
        request_ticker = f"_{ticker}" if ticker.upper() in INDEX_TICKERS else ticker
        payload = self.client.get_json(QUOTE_URL.format(ticker=request_ticker))
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict) or "options" not in data:
            raise SourceError(
                self.name,
                f"{ticker}: CBOE returned no options block. Index chains are "
                "requested with a leading underscore (_SPX); a single name is not.",
            )

        as_of = _quote_instant(data) or utcnow()
        if as_of > end:
            as_of = end
        collected = utcnow()
        underlying = _number(data.get("current_price"))

        rows: list[dict[str, Any]] = []
        unparsed = 0
        for quote in data["options"]:
            parsed = parse_occ_symbol(str(quote.get("option", "")))
            if parsed is None:
                unparsed += 1
                continue
            _root, expiry, right, strike = parsed

            open_interest = _number(quote.get("open_interest")) or 0.0
            if self.require_open_interest and open_interest <= 0:
                continue

            rows.append(
                {
                    "symbol": ticker.upper(),
                    "as_of": as_of,
                    "known_at": collected,
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "bid": _number(quote.get("bid")),
                    "ask": _number(quote.get("ask")),
                    "last": _number(quote.get("last_trade_price")),
                    "volume": _number(quote.get("volume")),
                    "open_interest": open_interest,
                    "implied_vol": _number(quote.get("iv")),
                    "delta": _number(quote.get("delta")),
                    "gamma": _number(quote.get("gamma")),
                    "vega": _number(quote.get("vega")),
                    "theta": _number(quote.get("theta")),
                    "underlying_price": underlying,
                }
            )

        if unparsed:
            log.warning(
                "options.unparsed_symbols",
                ticker=ticker,
                unparsed=unparsed,
                reason="OCC symbol did not match the expected root/YYMMDD/C-P/strike form",
            )
        log.info(
            "options.snapshot",
            ticker=ticker,
            quotes=len(rows),
            as_of=as_of.isoformat(),
            expiries=len({r["expiry"] for r in rows}),
            note="this snapshot cannot be re-fetched later",
        )
        return rows


def _quote_instant(data: dict[str, Any]) -> dt.datetime | None:
    """When the quotes describe, from CBOE's own timestamp.

    Used rather than the collection time because the file is delayed: dating a
    16:15 close as though it were observed at 18:40 would misstate by hours what
    a snapshot is for.
    """
    raw = data.get("last_trade_time")
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(str(raw)).replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
