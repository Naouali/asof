"""US Energy Information Administration — inventories and production.

The fundamentals behind energy futures: crude and product stocks, natural gas in
storage, field production, refinery runs. Free with a key, and the only public
source with this coverage.

**Two things make this data dangerous, and they compound.**

*It is revised, and there is no vintage archive.* EIA restates weekly inventories
and monthly production, and the API serves only the current value. There is no
ALFRED equivalent: the number as first published is simply gone. Every row here
therefore carries ``vintage=False``, and a backtest reading it is reading a
figure that was corrected after the fact. That is look-ahead, it cannot be
removed by anything this module does, and the only honest mitigations are to lag
the series well past the revision window or to treat results built on it as an
upper bound.

*The release time is the signal.* A weekly petroleum report lands Wednesday at
10:30 Eastern and natural gas storage Thursday at 10:30 Eastern, and the market
moves on the release. The data's own period ends days earlier, so dating an
observation by its period would make it knowable before the number existed --
the same error as reading a COT report on its Tuesday. ``known_at`` is the
release instant, derived from the published schedule and rolled forward over
federal holidays, because EIA is a federal agency and does not publish on them.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import polars as pl

from quantlab.data.calendars import next_federal_workday
from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.logging import get_logger

__all__ = ["EIA_SERIES", "EiaEnergyStocks", "EiaSeries", "eia_release"]

log = get_logger("quantlab.data.sources.eia")

BASE_URL = "https://api.eia.gov/v2"
EASTERN = ZoneInfo("America/New_York")

#: Both weekly reports are published at 10:30 Eastern.
RELEASE_HOUR, RELEASE_MINUTE = 10, 30

#: Weekday each report is released, as `datetime.weekday()`: Wednesday is 2.
PETROLEUM_WEEKDAY = 2
NATURAL_GAS_WEEKDAY = 3


class EiaSeries:
    """One EIA series, its route, and when it is published."""

    def __init__(
        self,
        symbol: str,
        route: str,
        facets: dict[str, str],
        description: str,
        *,
        frequency: str = "weekly",
        release_weekday: int = PETROLEUM_WEEKDAY,
        units: str = "thousand barrels",
    ) -> None:
        self.symbol = symbol
        self.route = route
        self.facets = facets
        self.description = description
        self.frequency = frequency
        self.release_weekday = release_weekday
        self.units = units

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.route}/data/"


EIA_SERIES: dict[str, EiaSeries] = {
    "CRUDE_STOCKS": EiaSeries(
        "CRUDE_STOCKS",
        "petroleum/stoc/wstk",
        {"series": "WCESTUS1"},
        "US commercial crude oil inventories excluding the SPR",
    ),
    "GASOLINE_STOCKS": EiaSeries(
        "GASOLINE_STOCKS",
        "petroleum/stoc/wstk",
        {"series": "WGTSTUS1"},
        "US total motor gasoline inventories",
    ),
    "DISTILLATE_STOCKS": EiaSeries(
        "DISTILLATE_STOCKS",
        "petroleum/stoc/wstk",
        {"series": "WDISTUS1"},
        "US distillate fuel oil inventories",
    ),
    "CUSHING_STOCKS": EiaSeries(
        "CUSHING_STOCKS",
        "petroleum/stoc/wstk",
        {"series": "W_EPC0_SAX_YCUOK_MBBL"},
        "Crude oil stocks at Cushing, Oklahoma -- the WTI delivery point",
    ),
    "NATGAS_STORAGE": EiaSeries(
        "NATGAS_STORAGE",
        "natural-gas/stor/wkly",
        {"series": "NW2_EPG0_SWO_R48_BCF"},
        "Working natural gas in underground storage, lower 48",
        release_weekday=NATURAL_GAS_WEEKDAY,
        units="billion cubic feet",
    ),
}


def eia_release(period_end: dt.date, release_weekday: int) -> dt.datetime:
    """When the report covering a period became public, as a UTC instant.

    The weekly petroleum report covers the week ending Friday and is published
    the following Wednesday; natural gas storage covers the same week and is
    published the following Thursday. Both at 10:30 Eastern, both rolled forward
    when that day is a federal holiday -- EIA is a federal agency and does not
    publish on one.
    """
    ahead = (release_weekday - period_end.weekday()) % 7
    scheduled = period_end + dt.timedelta(days=ahead or 7)
    released = next_federal_workday(scheduled)
    return dt.datetime(
        released.year,
        released.month,
        released.day,
        RELEASE_HOUR,
        RELEASE_MINUTE,
        tzinfo=EASTERN,
    ).astimezone(dt.UTC)


@register
class EiaEnergyStocks(Source):
    """Weekly US energy inventories, dated by their release."""

    name: ClassVar[str] = "eia.series_observations"
    dataset: ClassVar[str] = "series_observations"
    asset_class: ClassVar[AssetClass] = AssetClass.COMMODITY
    spec: ClassVar = get_source("eia")
    default_symbols: ClassVar[tuple[str, ...]] = tuple(EIA_SERIES)

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        wanted = list(dict.fromkeys(symbols)) or list(self.default_symbols)

        unknown = sorted(set(wanted) - set(EIA_SERIES))
        if unknown:
            raise SourceError(
                self.name, f"unknown EIA series {unknown}; known: {sorted(EIA_SERIES)}"
            )

        key = self.settings.secret_for("eia_api_key")
        if not key:
            raise SourceError(
                self.name,
                "EIA needs a free API key. Register at "
                "https://www.eia.gov/opendata/register.php and set "
                "QUANTLAB_EIA_API_KEY in .env.",
            )

        rows: list[dict[str, Any]] = []
        for name in wanted:
            rows.extend(self._rows_for(EIA_SERIES[name], key, start, end))

        if not rows:
            raise SourceError(
                self.name,
                f"no EIA observations for {wanted} released between "
                f"{start:%Y-%m-%d} and {end:%Y-%m-%d}. The window filters on the "
                "RELEASE date, not the period end.",
            )
        log.warning(
            "eia.restated_without_vintages",
            rows=len(rows),
            reason=(
                "EIA revises these series and serves only the current value; the "
                "figure as first published is not archived anywhere, so a backtest "
                "reading them has look-ahead that cannot be removed here"
            ),
        )
        return self.finalise(rows)

    def _rows_for(
        self, series: EiaSeries, key: str, start: dt.datetime, end: dt.datetime
    ) -> list[dict[str, Any]]:
        params = {
            "api_key": key,
            "frequency": series.frequency,
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": "5000",
            **{f"facets[{k}][]": v for k, v in series.facets.items()},
        }
        payload = self.client.get_json(series.url, params=params)
        records = (payload or {}).get("response", {}).get("data")
        if records is None:
            raise SourceError(
                self.name,
                f"{series.symbol}: no response.data block. EIA changed the v2 "
                f"envelope, or the route {series.route} was retired.",
            )

        rows: list[dict[str, Any]] = []
        for record in records:
            period, value = record.get("period"), record.get("value")
            if not period or value is None:
                continue
            try:
                period_end = dt.date.fromisoformat(str(period)[:10])
            except ValueError:
                continue

            known_at = eia_release(period_end, series.release_weekday)
            if not (start <= known_at <= end):
                continue
            rows.append(
                {
                    "symbol": series.symbol,
                    "as_of": dt.datetime(
                        period_end.year, period_end.month, period_end.day, tzinfo=dt.UTC
                    ),
                    "known_at": known_at,
                    "value": float(value),
                    "units": series.units,
                    # Not a vintage series: EIA keeps no archive of first prints.
                    "vintage": False,
                }
            )
        return rows
