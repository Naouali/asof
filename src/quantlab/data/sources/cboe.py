"""CBOE volatility indices.

Free daily history for VIX and its relatives, straight from CBOE as CSV. VIX
reaches back to 1990, which is longer than any free equity option data, and it is
the standard input for a volatility-regime filter.

**These are index levels, not instruments.** They are written to
``series_observations`` rather than ``ohlcv_daily``, deliberately. VIX spot cannot
be bought: the tradeable expressions are futures, options and ETPs, each with its
own roll cost and basis, and every one of them has underperformed spot VIX by a
wide margin over any long horizon. Filing an index under a dataset the backtest
engine treats as tradeable would let a strategy "buy VIX" and collect a return
nobody could have earned. The open, high and low are discarded for the same
reason: with no execution possible, an intraday range has no execution meaning.

**The series are not continuous instruments.** The methodology changed in 2003,
when CBOE moved VIX from the old OEX-implied-volatility calculation to the
model-free variance-swap formula; the pre-2003 series under the old method is
VXO, a different index. A backtest spanning 2003 is trading two instruments and
calling them one. That boundary is recorded per series here and repeated in
docs/LIMITATIONS.md.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Sequence
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.logging import get_logger

__all__ = ["CBOE_SERIES", "CboeSeries", "CboeVolatilityIndices"]

log = get_logger("quantlab.data.sources.cboe")

BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"

EASTERN = ZoneInfo("America/New_York")
#: VIX and its relatives settle at 16:15 Eastern, fifteen minutes after the
#: equity close, because the constituent SPX options trade until then. The close
#: is knowable at that instant and not before.
CLOSE_HOUR, CLOSE_MINUTE = 16, 15


class CboeSeries:
    """One published index, and what is known to be wrong with it."""

    def __init__(
        self,
        symbol: str,
        file_stem: str,
        description: str,
        *,
        value_column: str = "CLOSE",
        reliable_from: dt.date | None = None,
        methodology_break: dt.date | None = None,
    ) -> None:
        self.symbol = symbol
        self.file_stem = file_stem
        self.description = description
        self.value_column = value_column
        #: Observations before this are served but flagged: CBOE publishes them
        #: and they are not trustworthy.
        self.reliable_from = reliable_from
        #: A date on which the calculation changed, so the series either side is
        #: not the same index.
        self.methodology_break = methodology_break

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.file_stem}.csv"


CBOE_SERIES: dict[str, CboeSeries] = {
    "VIX": CboeSeries(
        "VIX",
        "VIX_History",
        "30-day implied volatility of the S&P 500",
        # CBOE moved to the model-free variance-swap calculation on 2003-09-22.
        # What is served before that date is the old series back-cast; the index
        # actually published then was VXO, computed a different way.
        methodology_break=dt.date(2003, 9, 22),
    ),
    "VIX9D": CboeSeries("VIX9D", "VIX9D_History", "9-day implied volatility of the S&P 500"),
    "VIX3M": CboeSeries("VIX3M", "VIX3M_History", "3-month implied volatility of the S&P 500"),
    "VIX6M": CboeSeries("VIX6M", "VIX6M_History", "6-month implied volatility of the S&P 500"),
    "VVIX": CboeSeries(
        "VVIX",
        "VVIX_History",
        "volatility of VIX itself",
        value_column="VVIX",
        # The first fortnight of the published file is visibly unusable: 71.73 on
        # 6 March 2006, then a nine-day gap, then 15.71. The series settles down
        # from April 2006.
        reliable_from=dt.date(2006, 4, 1),
    ),
}


@register
class CboeVolatilityIndices(Source):
    """Daily closes for CBOE's published volatility indices."""

    name: ClassVar[str] = "cboe.series_observations"
    dataset: ClassVar[str] = "series_observations"
    asset_class: ClassVar[AssetClass] = AssetClass.OPTIONS
    spec: ClassVar = get_source("cboe")
    default_symbols: ClassVar[tuple[str, ...]] = ("VIX", "VIX9D", "VIX3M", "VVIX")

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        wanted = list(dict.fromkeys(symbols)) or list(self.default_symbols)

        unknown = sorted(set(wanted) - set(CBOE_SERIES))
        if unknown:
            raise SourceError(
                self.name,
                f"unknown CBOE series {unknown}; published: {sorted(CBOE_SERIES)}",
            )

        rows: list[dict[str, Any]] = []
        for name in wanted:
            rows.extend(self._rows_for(CBOE_SERIES[name], start, end))

        if not rows:
            raise SourceError(
                self.name,
                f"no CBOE observations for {wanted} between {start:%Y-%m-%d} and {end:%Y-%m-%d}",
            )
        return self.finalise(rows)

    def _rows_for(
        self, series: CboeSeries, start: dt.datetime, end: dt.datetime
    ) -> list[dict[str, Any]]:
        text = self.client.get_text(series.url)
        frame = _parse_csv(text, series, self.name)

        rows: list[dict[str, Any]] = []
        suspect = 0
        for day, value in zip(frame["day"], frame["value"], strict=True):
            known_at = dt.datetime(
                day.year, day.month, day.day, CLOSE_HOUR, CLOSE_MINUTE, tzinfo=EASTERN
            ).astimezone(dt.UTC)
            if not (start <= known_at <= end):
                continue
            if series.reliable_from is not None and day < series.reliable_from:
                suspect += 1
                continue
            rows.append(
                {
                    "symbol": series.symbol,
                    "as_of": dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC),
                    "known_at": known_at,
                    "value": float(value),
                    "units": "index level",
                    # These are restated only in the sense that CBOE may correct a
                    # print; they are not a vintage series.
                    "vintage": False,
                }
            )

        if suspect:
            log.warning(
                "cboe.unreliable_history_dropped",
                symbol=series.symbol,
                dropped=suspect,
                reliable_from=series.reliable_from.isoformat() if series.reliable_from else None,
                reason="CBOE publishes these observations and they are not trustworthy",
            )
        if series.methodology_break is not None and rows:
            spans = [r for r in rows if r["as_of"].date() < series.methodology_break]
            if spans and len(spans) < len(rows):
                log.warning(
                    "cboe.methodology_break",
                    symbol=series.symbol,
                    changed_on=series.methodology_break.isoformat(),
                    before=len(spans),
                    after=len(rows) - len(spans),
                    reason=(
                        "the calculation changed on this date, so a series spanning "
                        "it is two indices reported under one name"
                    ),
                )
        return rows


def _parse_csv(text: str, series: CboeSeries, source_name: str) -> pl.DataFrame:
    """CBOE's CSV: ``DATE`` in US M/D/Y, plus either OHLC or a single column."""
    try:
        raw = pl.read_csv(io.StringIO(text))
    except Exception as exc:
        raise SourceError(source_name, f"{series.symbol}: CSV did not parse ({exc})") from exc

    if "DATE" not in raw.columns or series.value_column not in raw.columns:
        raise SourceError(
            source_name,
            f"{series.symbol}: expected columns DATE and {series.value_column}, got "
            f"{raw.columns}. CBOE changed the file layout; the parser must be "
            "updated rather than guessing which column is the close.",
        )

    try:
        parsed = raw.select(
            pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y", strict=True).alias("day"),
            pl.col(series.value_column).cast(pl.Float64).alias("value"),
        )
    except pl.exceptions.PolarsError as exc:
        # A date CBOE serves in an unexpected format is a layout change, not a
        # bad row. Letting `strict=False` null it instead would drop observations
        # silently and leave a series with holes nobody notices.
        raise SourceError(
            source_name,
            f"{series.symbol}: DATE is not in CBOE's documented M/D/Y format "
            f"({exc}). The file layout changed; the parser must be updated.",
        ) from exc

    return parsed.drop_nulls().sort("day")
