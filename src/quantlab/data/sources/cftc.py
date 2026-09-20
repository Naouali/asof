"""CFTC Commitments of Traders.

Weekly aggregate positioning in US futures markets, broken out by trader
category. Free, public domain, and the only view of who is on which side of a
futures market that costs nothing.

**The publication lag is the whole problem.** A COT report counts positions as of
the close on Tuesday and is released the following Friday at 3:30 p.m. Eastern.
The CFTC states it plainly: *"The COT Report is generally published each Friday
at 3:30 pm Eastern Time (US), using the data from the immediately preceding
Tuesday of that week... It takes three days to process the data."* Nothing in any
CFTC payload carries that release date -- the Socrata API returns
``report_date_as_yyyy_mm_dd`` and no publication field at all -- so ``known_at``
has to be derived, and deriving it wrongly is the single most common look-ahead
bias in published COT research. A signal that reads Tuesday's positioning on
Tuesday is reading a number that did not exist for another three days.

**Three reporting regimes, not one.** Checking 1,933 reports for the CME
Eurodollar contract back to 1986:

* Before 1992-09-30 the report was **semi-monthly**, mid-month and month-end, on
  whatever weekday those fell. The CFTC says of this era that *"the mid-month data
  was not published before that time"* and that *"a significant period elapsed
  between the report date for that data and its eventual compilation"*. That data
  was never knowable contemporaneously at all, so there is no honest ``known_at``
  to give it, and this module refuses to ingest it by default.
* From 1992-09-30 the report is weekly. 1,772 of the reports fall on a Tuesday;
  161 do not, because a holiday moved the snapshot.
* Holidays also move the release. The CFTC warns only that *"holidays can change
  the COT release schedule"* and publishes no machine-readable calendar, so the
  release is derived from the federal holiday rules in
  :mod:`quantlab.data.calendars` and rolled forward, never backward.

**Grain units changed in 1998**, from bushels to contracts, and the CFTC notes
that changes from the last 1997 reports were not recalculated. A grain series
spanning that boundary is two different series.
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

__all__ = [
    "COT_REPORTS",
    "WEEKLY_REPORTING_BEGAN",
    "CftcPositioning",
    "PreWeeklyEraError",
    "cot_release",
]

log = get_logger("quantlab.data.sources.cftc")

BASE_URL = "https://publicreporting.cftc.gov/resource"

#: Releases are at 3:30 p.m. in Washington, which is a wall-clock time: the UTC
#: offset moves with US daylight saving, so it is stored as a zoned instant.
EASTERN = ZoneInfo("America/New_York")
RELEASE_HOUR, RELEASE_MINUTE = 15, 30

#: Weekly reporting began with the 1992-09-30 report. Before it, only mid-month
#: and month-end data exists, and the CFTC says the mid-month data was compiled
#: retrospectively rather than published at the time.
WEEKLY_REPORTING_BEGAN = dt.date(1992, 9, 30)

#: Processing time between the snapshot and the release, per the CFTC's own
#: description: Tuesday's data, published Friday.
PROCESSING_DAYS = 3


class PreWeeklyEraError(SourceError):
    """Raised for reports the CFTC never published contemporaneously."""


def cot_release(report_date: dt.date) -> dt.datetime:
    """The instant a report became public, as a UTC datetime.

    Tuesday's snapshot plus three days lands on Friday. If that Friday is a
    federal holiday the release rolls **forward** to the next working day, never
    backward: assuming an earlier release is the error that manufactures
    look-ahead, so the uncertainty is spent in the direction that costs a signal
    performance rather than inventing it.

    Raises :class:`PreWeeklyEraError` for reports before weekly publication began,
    because that data has no honest release date to give.
    """
    if report_date < WEEKLY_REPORTING_BEGAN:
        raise PreWeeklyEraError(
            "cftc_cot",
            f"the COT report for {report_date} predates weekly publication "
            f"({WEEKLY_REPORTING_BEGAN}). The CFTC states that mid-month data from "
            "this era was not published at the time and was compiled later, so no "
            "known_at can be assigned to it honestly. Pass "
            "allow_pre_weekly_era=True to ingest it anyway; it will be dated with "
            "a deliberately punitive lag and must not be used in a signal.",
        )

    scheduled = report_date + dt.timedelta(days=PROCESSING_DAYS)
    released = next_federal_workday(scheduled)
    return dt.datetime(
        released.year,
        released.month,
        released.day,
        RELEASE_HOUR,
        RELEASE_MINUTE,
        tzinfo=EASTERN,
    ).astimezone(dt.UTC)


#: A conservative stand-in for the pre-1992 era, used only when the caller opts
#: in. Ninety days is not a measurement -- the CFTC published no schedule for
#: retrospectively compiled data -- it is a lag long enough that any signal built
#: on this era is penalised rather than flattered.
PRE_WEEKLY_ERA_LAG = dt.timedelta(days=90)


# ======================================================================================
# Reports
# ======================================================================================
class CotReport:
    """One Socrata dataset, and how to read its positioning columns."""

    def __init__(
        self,
        dataset_id: str,
        label: str,
        categories: dict[str, dict[str, str]],
    ) -> None:
        self.dataset_id = dataset_id
        self.label = label
        #: ``{category: {measure: source column}}``
        self.categories = categories

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.dataset_id}.json"


#: The legacy report, published since 1986 and the only one with deep history.
#: "Non-commercial" is the speculative bucket every published COT signal uses.
LEGACY = CotReport(
    "6dca-aqww",
    "legacy futures-only",
    {
        "noncommercial": {
            "long": "noncomm_positions_long_all",
            "short": "noncomm_positions_short_all",
            "spreading": "noncomm_positions_spread",
        },
        "commercial": {
            "long": "comm_positions_long_all",
            "short": "comm_positions_short_all",
        },
        "nonreportable": {
            "long": "nonrept_positions_long_all",
            "short": "nonrept_positions_short_all",
        },
        "total": {"open_interest": "open_interest_all"},
    },
)

#: The disaggregated report, from 2006. Splits the legacy "commercial" bucket into
#: producers and swap dealers, which is the split that made the legacy commercial
#: series hard to interpret: a swap dealer hedging an index position is not a
#: producer hedging a crop.
DISAGGREGATED = CotReport(
    "72hh-3qpy",
    "disaggregated futures-only",
    {
        "producer_merchant": {
            "long": "prod_merc_positions_long",
            "short": "prod_merc_positions_short",
        },
        "swap_dealer": {
            "long": "swap_positions_long_all",
            "short": "swap__positions_short_all",
            "spreading": "swap__positions_spread_all",
        },
        "managed_money": {
            "long": "m_money_positions_long_all",
            "short": "m_money_positions_short_all",
            "spreading": "m_money_positions_spread",
        },
        "other_reportable": {
            "long": "other_rept_positions_long",
            "short": "other_rept_positions_short",
            "spreading": "other_rept_positions_spread",
        },
        "total": {"open_interest": "open_interest_all"},
    },
)

#: Traders in Financial Futures, from 2006. The financial-market analogue of the
#: disaggregated report: dealers, asset managers, leveraged funds.
TFF = CotReport(
    "gpe5-46if",
    "traders in financial futures",
    {
        "dealer": {
            "long": "dealer_positions_long_all",
            "short": "dealer_positions_short_all",
            "spreading": "dealer_positions_spread_all",
        },
        "asset_manager": {
            "long": "asset_mgr_positions_long",
            "short": "asset_mgr_positions_short",
            "spreading": "asset_mgr_positions_spread",
        },
        "leveraged_funds": {
            "long": "lev_money_positions_long",
            "short": "lev_money_positions_short",
            "spreading": "lev_money_positions_spread",
        },
        "other_reportable": {
            "long": "other_rept_positions_long",
            "short": "other_rept_positions_short",
            "spreading": "other_rept_positions_spread",
        },
        "total": {"open_interest": "open_interest_all"},
    },
)

COT_REPORTS: dict[str, CotReport] = {
    "legacy": LEGACY,
    "disaggregated": DISAGGREGATED,
    "tff": TFF,
}

#: A small default universe of liquid contracts, by CFTC contract market code.
#: Deliberately short: the full COT covers hundreds of contracts, and an ingest
#: that pulls all of them on a first run is an ingest nobody waits for.
#:
#: These codes are **verified against the live API**, not guessed. The naming is
#: not intuitive -- corn is 002602 and not 020601, SRW wheat is 001602 -- and a
#: wrong code returns an empty result rather than an error, so three of the
#: original guesses here ingested silently as nothing.
DEFAULT_CONTRACTS: tuple[str, ...] = (
    "13874A",  # E-mini S&P 500
    "209742",  # Nasdaq mini
    "002602",  # Corn (CBOT)
    "001602",  # Wheat, SRW (CBOT)
    "088691",  # Gold (COMEX)
    "084691",  # Silver (COMEX)
    "043602",  # UST 10-year note (CBOT)
    "042601",  # UST 2-year note (CBOT)
    "099741",  # Euro FX
    "097741",  # Japanese yen
    "098662",  # US Dollar Index
)


@register
class CftcPositioning(Source):
    """Weekly futures positioning, from whichever COT report is selected.

    One fetcher rather than three, because all three reports write the canonical
    ``positioning`` dataset and the registry is keyed ``<source>.<dataset>``.
    The report is an ingest option, as the bar interval is for Binance:

    .. code-block:: yaml

        - fetcher: cftc_cot.positioning
          options: {report: disaggregated}
    """

    name: ClassVar[str] = "cftc_cot.positioning"
    dataset: ClassVar[str] = "positioning"
    asset_class: ClassVar[AssetClass] = AssetClass.FUTURES
    spec: ClassVar = get_source("cftc_cot")
    default_symbols: ClassVar[tuple[str, ...]] = tuple(dict.fromkeys(DEFAULT_CONTRACTS))

    #: Socrata's default page size is small; this is its documented maximum.
    page_size: ClassVar[int] = 50_000

    def __init__(
        self,
        *args: Any,
        report: str = "legacy",
        allow_pre_weekly_era: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if report not in COT_REPORTS:
            raise ValueError(f"unknown COT report {report!r}; known: {sorted(COT_REPORTS)}")
        self.report_name = report
        self.report = COT_REPORTS[report]
        self.allow_pre_weekly_era = allow_pre_weekly_era

    def _fetch_symbols(self, symbols: Sequence[object]) -> list[str]:
        """Validated, de-duplicated contract market codes."""
        return [_validate_contract_code(code) for code in dict.fromkeys(symbols)]

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        contracts = [_validate_contract_code(c) for c in dict.fromkeys(symbols)] or list(
            self.default_symbols
        )

        rows: list[dict[str, Any]] = []
        skipped_pre_weekly = 0
        empty_contracts: list[str] = []
        for contract in contracts:
            payload = self._fetch_contract(contract, start, end)
            if not payload:
                empty_contracts.append(contract)
            for record in payload:
                report_date = _parse_date(record.get("report_date_as_yyyy_mm_dd"))
                if report_date is None:
                    continue
                try:
                    known_at = cot_release(report_date)
                except PreWeeklyEraError:
                    if not self.allow_pre_weekly_era:
                        skipped_pre_weekly += 1
                        continue
                    known_at = (
                        dt.datetime(
                            report_date.year, report_date.month, report_date.day, tzinfo=dt.UTC
                        )
                        + PRE_WEEKLY_ERA_LAG
                    )
                rows.extend(self._rows_for(record, report_date, known_at))

        if empty_contracts:
            # Socrata answers an unknown contract code with an empty array, not an
            # error, so a typo ingests silently as nothing. Spec section 13: fail
            # loudly rather than quietly return less than was asked for.
            log.warning(
                "cftc.contracts_returned_nothing",
                report=self.report_name,
                contracts=empty_contracts,
                requested=len(contracts),
                reason=(
                    "an unknown or retired contract market code returns an empty "
                    "array rather than an error -- check the code against "
                    "market_and_exchange_names"
                ),
            )
        if skipped_pre_weekly:
            log.warning(
                "cftc.pre_weekly_era_skipped",
                rows=skipped_pre_weekly,
                cutoff=WEEKLY_REPORTING_BEGAN.isoformat(),
                reason=(
                    "the CFTC states this data was compiled retrospectively rather "
                    "than published at the time, so it has no honest known_at"
                ),
            )
        if not rows:
            raise SourceError(
                self.name,
                f"no COT observations for {contracts} between "
                f"{start:%Y-%m-%d} and {end:%Y-%m-%d}. Contract market codes are "
                "six characters, e.g. '088691' for gold -- a ticker like 'GC' will "
                "match nothing.",
            )
        return self.finalise(rows)

    def _fetch_contract(
        self, contract: str, start: dt.datetime, end: dt.datetime
    ) -> list[dict[str, Any]]:
        params = {
            "$where": (
                f"cftc_contract_market_code='{contract}' "
                f"AND report_date_as_yyyy_mm_dd >= '{start:%Y-%m-%d}' "
                f"AND report_date_as_yyyy_mm_dd <= '{end:%Y-%m-%d}'"
            ),
            "$order": "report_date_as_yyyy_mm_dd",
            "$limit": str(self.page_size),
        }
        payload = self.client.get_json(self.report.url, params=params)
        if not isinstance(payload, list):
            raise SourceError(
                self.name,
                f"expected a JSON array from Socrata, got {type(payload).__name__}",
            )
        return payload

    def _rows_for(
        self, record: dict[str, Any], report_date: dt.date, known_at: dt.datetime
    ) -> list[dict[str, Any]]:
        as_of = dt.datetime(report_date.year, report_date.month, report_date.day, tzinfo=dt.UTC)
        symbol = str(record.get("cftc_contract_market_code", "")).strip()
        shared = {
            "symbol": symbol,
            "as_of": as_of,
            "known_at": known_at,
            "report": self.report_name,
            "name": _clean(record.get("market_and_exchange_names")),
            "contract_units": _clean(record.get("contract_units")),
            "exchange": _clean(record.get("cftc_market_code")),
        }
        rows: list[dict[str, Any]] = []
        for category, measures in self.report.categories.items():
            for measure, column in measures.items():
                value = _parse_float(record.get(column))
                if value is None:
                    continue
                rows.append({**shared, "category": category, "measure": measure, "value": value})
        return rows


#: A contract market code is six characters: digits, sometimes with a trailing
#: letter (13874A) or a '+' (20974+).
_CODE_LENGTH = 6


def _validate_contract_code(symbol: object) -> str:
    """Reject anything that is not a CFTC contract market code, loudly.

    YAML's legacy octal rule is the reason this exists. An unquoted ``002602``
    in a config is parsed as **octal 2602, i.e. the integer 1410**, and the code
    for corn silently becomes a number that matches nothing. Only codes whose
    digits are all 0-7 are affected, so ``088691`` (gold) survives and
    ``002602`` (corn) does not -- which makes the failure look like a
    CFTC-side gap rather than a parsing bug.

    The original value cannot be recovered from 1410, so this raises instead of
    guessing. Quote contract codes in YAML.
    """
    if isinstance(symbol, (bool, int)):
        raise SourceError(
            "cftc_cot",
            f"contract market code {symbol!r} arrived as an integer. YAML parses an "
            f"unquoted leading-zero code as octal -- 002602 becomes {0o2602}, which "
            "matches no contract -- and the original digits cannot be recovered. "
            'Quote contract codes in the ingest plan: symbols: ["002602"].',
        )
    text = str(symbol).strip()
    if len(text) != _CODE_LENGTH:
        raise SourceError(
            "cftc_cot",
            f"contract market code {text!r} is {len(text)} characters; CFTC codes "
            f"are {_CODE_LENGTH}, e.g. '088691' for gold. A ticker such as 'GC' "
            "matches nothing and returns an empty result rather than an error.",
        )
    return text


# ----------------------------------------------------------------------- parsing --
def _parse_date(value: Any) -> dt.date | None:
    if not value:
        return None
    return dt.date.fromisoformat(str(value)[:10])


def _parse_float(value: Any) -> float | None:
    """Socrata returns every number as a string, and blanks as empty or absent."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _clean(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
