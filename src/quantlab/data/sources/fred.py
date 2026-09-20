"""FRED and ALFRED.

These are two fetchers against one API, and the difference between them is the
difference between an honest macro history and a fictional one.

``fred.series_observations``
    The current vintage of a series. For a series that is **never revised** --
    market-observed rates, VIX, credit spreads -- the current vintage *is* the
    historical one, and the only correction needed is the publication lag: a
    Treasury yield dated Monday is published Tuesday afternoon, so it was not
    knowable on Monday morning.

``alfred.series_observations``
    Every vintage of a series, with the real-time window in which each value was
    the published one. ``known_at`` is set to that vintage's start, so a snapshot
    taken in 2009 sees the GDP figure that was on the record in 2009, not the one
    revised three times since.

To make the distinction impossible to fudge, **this module refuses to ingest a
series it has not been told how to treat**. :data:`SERIES_POLICY` classifies each
series as never-revised (with its publication lag) or revised (ALFRED only).
Adding a series means making that call explicitly, which is the point: the failure
mode being prevented here is someone pulling `GDPC1` from FRED, reading 2008
through 2026's revisions, and never learning that they did.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError, require_key
from quantlab.data.sources.base import Source, register
from quantlab.logging import get_logger

__all__ = ["SERIES_POLICY", "AlfredVintageSeries", "FredSeries", "SeriesPolicy"]

log = get_logger("quantlab.data.sources.fred")

OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

#: ALFRED returns every vintage when asked for the widest possible real-time window.
ALFRED_REALTIME_START = "1776-07-04"
ALFRED_REALTIME_END = "9999-12-31"


@dataclass(frozen=True, slots=True)
class SeriesPolicy:
    """How a FRED series must be treated to avoid look-ahead bias."""

    series_id: str
    description: str
    units: str
    #: True when the series is never revised after publication, so the current
    #: vintage equals the historical one.
    never_revised: bool
    #: Hours between the observation date and its publication. Applied to
    #: ``known_at`` for never-revised series.
    publication_lag_hours: float = 24.0

    @property
    def requires_vintages(self) -> bool:
        return not self.never_revised


def _rate(series_id: str, description: str, lag_hours: float = 24.0) -> SeriesPolicy:
    return SeriesPolicy(
        series_id=series_id,
        description=description,
        units="percent",
        never_revised=True,
        publication_lag_hours=lag_hours,
    )


def _revised(series_id: str, description: str, units: str) -> SeriesPolicy:
    return SeriesPolicy(
        series_id=series_id, description=description, units=units, never_revised=False
    )


#: Series QuantLab knows how to treat. Extend deliberately, never by default.
SERIES_POLICY: dict[str, SeriesPolicy] = {
    policy.series_id: policy
    for policy in (
        # -- Treasury constant-maturity yields. Market observations, never revised.
        #    Published the next business day around 16:15 ET, hence a ~24h lag.
        _rate("DGS1MO", "1-month Treasury constant maturity"),
        _rate("DGS3MO", "3-month Treasury constant maturity"),
        _rate("DGS6MO", "6-month Treasury constant maturity"),
        _rate("DGS1", "1-year Treasury constant maturity"),
        _rate("DGS2", "2-year Treasury constant maturity"),
        _rate("DGS5", "5-year Treasury constant maturity"),
        _rate("DGS10", "10-year Treasury constant maturity"),
        _rate("DGS30", "30-year Treasury constant maturity"),
        _rate("T10Y2Y", "10-year minus 2-year Treasury spread"),
        _rate("T10Y3M", "10-year minus 3-month Treasury spread"),
        # -- Policy and money-market rates.
        _rate("DFF", "Effective federal funds rate"),
        _rate("SOFR", "Secured overnight financing rate"),
        _rate("IORB", "Interest on reserve balances"),
        # -- Credit spreads (ICE BofA OAS). Index values, not revised.
        _rate("BAMLH0A0HYM2", "ICE BofA US high yield option-adjusted spread"),
        _rate("BAMLC0A0CM", "ICE BofA US corporate option-adjusted spread"),
        _rate("BAMLH0A3HYC", "ICE BofA US high yield CCC and lower OAS"),
        # -- Volatility.
        _rate("VIXCLS", "CBOE volatility index, close"),
        # -- Revised macro. ALFRED only; using FRED for these is look-ahead bias.
        _revised("GDPC1", "Real gross domestic product", "billions of chained 2017 dollars"),
        _revised("CPIAUCSL", "CPI for all urban consumers, all items", "index 1982-1984=100"),
        _revised("PAYEMS", "All employees, total nonfarm", "thousands of persons"),
        _revised("UNRATE", "Unemployment rate", "percent"),
        _revised("INDPRO", "Industrial production index", "index 2017=100"),
        _revised("PCEPILFE", "Core PCE price index", "index 2017=100"),
        _revised("HOUST", "Housing starts", "thousands of units"),
        _revised("UMCSENT", "University of Michigan consumer sentiment", "index 1966:Q1=100"),
    )
}


def _policy(series_id: str) -> SeriesPolicy:
    try:
        return SERIES_POLICY[series_id]
    except KeyError:
        raise SourceError(
            "fred",
            f"series {series_id!r} has no revision policy. Add it to "
            "quantlab.data.sources.fred.SERIES_POLICY, declaring explicitly whether "
            "it is ever revised. Ingesting an unclassified series is how a macro "
            "history quietly starts holding values that did not exist at the time.",
        ) from None


def _parse_value(raw: Any) -> float | None:
    # FRED encodes a missing observation as ".".
    if raw in (None, ".", ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _as_utc_date(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.UTC)


class _FredBase(Source):
    dataset: ClassVar[str] = "series_observations"
    asset_class: ClassVar = AssetClass.MACRO

    def _observations(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        payload = self.client.get_json(
            OBSERVATIONS_URL,
            params={
                **params,
                "api_key": require_key(self.spec, self.settings),
                "file_type": "json",
            },
        )
        if "error_message" in payload:
            raise SourceError(self.spec.key, str(payload["error_message"]))
        observations = payload.get("observations")
        if observations is None:
            raise SourceError(self.spec.key, f"response carried no observations: {payload!r:.300}")
        return list(observations)


@register
class FredSeries(_FredBase):
    """Current-vintage observations, permitted only for never-revised series.

    ``known_at`` is the observation date plus the series' publication lag, because
    a rate dated Monday is not on the wire until Tuesday afternoon. Requesting a
    revisable series raises and points at :class:`AlfredVintageSeries`.
    """

    name: ClassVar[str] = "fred.series_observations"
    spec: ClassVar = get_source("fred")
    default_symbols: ClassVar[tuple[str, ...]] = (
        "DGS3MO",
        "DGS2",
        "DGS10",
        "DGS30",
        "T10Y2Y",
        "DFF",
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "VIXCLS",
    )

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for series_id in symbols:
            policy = _policy(series_id)
            if policy.requires_vintages:
                raise SourceError(
                    self.spec.key,
                    f"{series_id} ({policy.description}) is revised, and FRED serves "
                    "only the latest vintage. Using it here would mean reading 2008's "
                    "GDP as restated in 2026. Ingest it through "
                    "`alfred.series_observations` instead.",
                )

            lag = dt.timedelta(hours=policy.publication_lag_hours)
            for observation in self._observations(
                {
                    "series_id": series_id,
                    "observation_start": start.date().isoformat(),
                    "observation_end": end.date().isoformat(),
                }
            ):
                value = _parse_value(observation.get("value"))
                if value is None:
                    continue
                as_of = _as_utc_date(observation["date"])
                rows.append(
                    {
                        "symbol": series_id,
                        "as_of": as_of,
                        "known_at": as_of + lag,
                        "value": value,
                        "units": policy.units,
                        "vintage": False,
                    }
                )

        return self.finalise(rows)


@register
class AlfredVintageSeries(_FredBase):
    """Every vintage of a series, with ``known_at`` set to the vintage start.

    This is the only free way to read a macro series without knowing the
    future. It is also slow -- one row per (observation, vintage) pair, and a
    long revised series has many -- so ingest it incrementally.
    """

    name: ClassVar[str] = "alfred.series_observations"
    spec: ClassVar = get_source("alfred")
    default_symbols: ClassVar[tuple[str, ...]] = (
        "GDPC1",
        "CPIAUCSL",
        "PAYEMS",
        "UNRATE",
        "INDPRO",
    )

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for series_id in symbols:
            policy = _policy(series_id)
            observations = self._observations(
                {
                    "series_id": series_id,
                    "observation_start": start.date().isoformat(),
                    "observation_end": end.date().isoformat(),
                    "realtime_start": ALFRED_REALTIME_START,
                    "realtime_end": ALFRED_REALTIME_END,
                }
            )
            if not observations:
                continue

            for observation in observations:
                value = _parse_value(observation.get("value"))
                if value is None:
                    continue
                as_of = _as_utc_date(observation["date"])
                # `realtime_start` is the date this value became the published one.
                # That is exactly `known_at`; the store's latest-known-at-wins rule
                # then reconstructs any vintage for free.
                known_at = _as_utc_date(observation["realtime_start"])
                rows.append(
                    {
                        "symbol": series_id,
                        "as_of": as_of,
                        # A vintage cannot precede the observation it revises; ALFRED
                        # reports the first vintage of very old observations as the
                        # start of its own archive, which can predate the period end.
                        "known_at": max(known_at, as_of),
                        "value": value,
                        "units": policy.units,
                        "vintage": True,
                    }
                )

        return self.finalise(rows)
