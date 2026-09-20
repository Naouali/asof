"""SEC EDGAR XBRL company facts.

The only free source of **as-filed** US company fundamentals, and the reason the
`fundamentals` dataset exists. Every other free fundamentals source serves
restated history: today's view of what a company earned in 2015, silently revised
by every subsequent correction. EDGAR serves what was filed, with the filing date
attached, so a backtest can see 2015 through 2015's eyes.

No API key. The SEC's fair-access policy asks only for a User-Agent carrying real
contact details and caps callers at 10 requests a second; both are enforced by
:class:`~quantlab.data.http.HttpClient` from the catalogue entry.

**Three things here are easy to get silently wrong.**

*Duration.* A revenue figure is a flow measured over a window, and EDGAR reports
several windows for the same period end in the same filing. Apple's revenue facts
carry 3-, 6-, 9- and 12-month spans -- the quarterly figure, two year-to-date
cumulatives, and the annual total -- and a 10-K contains both the 12-month year
and the 3-month fourth quarter. Loading them into one series keyed by period end
mixes a quarter's revenue with a year's, and a profitability ratio built on the
result divides whichever happened to be last into total assets. The span is
therefore classified and written into ``fiscal_period``, which is part of the
dataset's key, so the two can never collide.

*Concept precedence.* Several XBRL tags mean the same thing, because tagging
practice changed and because filers differ. A company that reports both
``Revenues`` and ``RevenueFromContractWithCustomerExcludingAssessedTax`` would
otherwise produce two rows with one key and different values, and deduplication
would pick between them by filing date -- that is, arbitrarily. One tag is chosen
per filer per metric, by documented precedence, and the choice is recorded.

*Coverage is not random.* Measured across five large filers: revenue, total
assets, book equity and interest expense are present for all five; cost of goods
sold for three. JPMorgan has no COGS because banks do not report one. Dropping
filers with missing line items therefore drops banks, not a random sample, and
that is a sector tilt rather than noise.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.logging import get_logger

__all__ = ["METRIC_TAGS", "EdgarFundamentals", "classify_period", "filing_known_at"]

log = get_logger("quantlab.data.sources.edgar")

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

EASTERN = ZoneInfo("America/New_York")

#: EDGAR accepts filings until 17:30 Eastern; anything later carries the next
#: business day's date. A filing dated D was therefore public by 17:30 on D, and
#: that instant is used rather than midnight. Midnight UTC on the filed date is
#: 8 p.m. the previous evening in New York, which would make every filing
#: knowable before it was accepted.
ACCEPTANCE_HOUR, ACCEPTANCE_MINUTE = 17, 30

#: XBRL concepts per metric, in precedence order. The first concept a filer
#: actually reports is the one used, so a company reporting two of them does not
#: produce two rows under one key.
METRIC_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "cost_of_goods_sold": (
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
    ),
    "total_assets": ("Assets",),
    "sg_and_a": (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ),
    "interest_expense": ("InterestExpense", "InterestIncomeExpenseNet"),
    "book_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "accruals": ("IncreaseDecreaseInAccountsReceivable",),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
}

#: Forms worth keeping. Everything else -- 8-K, S-1, proxy statements -- either
#: repeats these numbers or is not periodic reporting at all.
PERIODIC_FORMS = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"})

#: Day spans, and the label each becomes. A span matching none of these is
#: dropped rather than guessed: a 7-month figure is a transition period from a
#: fiscal-year change, and calling it a quarter or a year is wrong both ways.
_SPANS: tuple[tuple[int, int, str], ...] = (
    (80, 100, "Q"),  # a quarter
    (170, 190, "H"),  # six months, year to date
    (260, 285, "9M"),  # nine months, year to date
    (350, 380, "FY"),  # a full year
)


def filing_known_at(filed: dt.date) -> dt.datetime:
    """When a filing dated ``filed`` became knowable, as a UTC instant."""
    return dt.datetime(
        filed.year,
        filed.month,
        filed.day,
        ACCEPTANCE_HOUR,
        ACCEPTANCE_MINUTE,
        tzinfo=EASTERN,
    ).astimezone(dt.UTC)


def classify_period(fact: dict[str, Any]) -> str | None:
    """How long a period this fact covers, or ``None`` if it cannot be told.

    The label carries the **span only** -- ``instant``, ``Q``, ``H``, ``9M``,
    ``FY`` -- and never a year. Two reasons. The period end is already in
    ``as_of``, so a year here is redundant; and the ``fy`` field on a fact is the
    *filing's* fiscal year rather than the fact's own. Apple's FY2023 revenue,
    restated as a comparative in the FY2025 10-K, carries ``fy: 2025``, so
    labelling it "2025FY" would file three different years under one name.

    Instantaneous facts -- balance sheet items, which have no start date -- are
    fully described by the date they are measured on, which ``as_of`` already
    holds.
    """
    if fact.get("start") is None:
        return "instant"

    try:
        days = (dt.date.fromisoformat(fact["end"]) - dt.date.fromisoformat(fact["start"])).days
    except (KeyError, TypeError, ValueError):
        return None

    for low, high, label in _SPANS:
        if low <= days <= high:
            return label
    return None


@register
class EdgarFundamentals(Source):
    """As-filed company fundamentals from EDGAR's XBRL company-facts API."""

    name: ClassVar[str] = "sec_edgar.fundamentals"
    dataset: ClassVar[str] = "fundamentals"
    asset_class: ClassVar[AssetClass] = AssetClass.EQUITY
    spec: ClassVar = get_source("sec_edgar")

    #: Kept short on purpose. Each company's facts payload is several megabytes --
    #: Apple's is 3.8 MB -- so a default universe of thousands would be gigabytes
    #: and an hour of polite request pacing on a first run.
    default_symbols: ClassVar[tuple[str, ...]] = (
        "AAPL",
        "MSFT",
        "JNJ",
        "KO",
        "PG",
        "XOM",
        "JPM",
        "PFE",
        "WMT",
        "HD",
    )

    def __init__(self, *args: Any, metrics: Sequence[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        unknown = sorted(set(metrics or ()) - set(METRIC_TAGS))
        if unknown:
            raise ValueError(
                f"unknown metrics {unknown}; known: {sorted(METRIC_TAGS)}. Add the "
                "XBRL concept to METRIC_TAGS rather than requesting it here."
            )
        self.metrics = tuple(metrics) if metrics else tuple(METRIC_TAGS)
        self._ticker_map: dict[str, int] | None = None

    # ------------------------------------------------------------- ticker map --
    def cik_for(self, ticker: str) -> int:
        """CIK for a ticker, from the SEC's own mapping."""
        if self._ticker_map is None:
            payload = self.client.get_json(TICKER_URL)
            if not isinstance(payload, dict):
                raise SourceError(self.name, "company_tickers.json was not an object")
            self._ticker_map = {
                str(row["ticker"]).upper(): int(row["cik_str"])
                for row in payload.values()
                if isinstance(row, dict) and row.get("ticker")
            }
            log.info("edgar.ticker_map", entries=len(self._ticker_map))

        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise SourceError(
                self.name,
                f"{ticker!r} is not in the SEC's ticker-to-CIK mapping. EDGAR covers "
                "US registrants only: a foreign listing, an ETF or a delisted ticker "
                "will not be there. It is not substituted with a guess.",
            )
        return cik

    # ------------------------------------------------------------------ fetch --
    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        tickers = list(dict.fromkeys(symbols)) or list(self.default_symbols)

        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            cik = self.cik_for(ticker)
            payload = self.client.get_json(FACTS_URL.format(cik=cik))
            if not isinstance(payload, dict):
                raise SourceError(self.name, f"{ticker}: company facts was not an object")
            rows.extend(self._rows_for(ticker, payload, start, end))

        if not rows:
            raise SourceError(
                self.name,
                f"no fundamentals for {tickers} filed between {start:%Y-%m-%d} and "
                f"{end:%Y-%m-%d}. The window filters on the FILING date, not the "
                "period end, so a narrow recent window legitimately returns nothing "
                "for a company that has not filed in it.",
            )
        return self.finalise(rows)

    def _rows_for(
        self, ticker: str, payload: dict[str, Any], start: dt.datetime, end: dt.datetime
    ) -> list[dict[str, Any]]:
        concepts = payload.get("facts", {}).get("us-gaap", {})
        rows: list[dict[str, Any]] = []
        chosen: dict[str, str] = {}
        unclassified = 0

        for metric in self.metrics:
            concept = _first_reported(METRIC_TAGS[metric], concepts)
            if concept is None:
                continue
            chosen[metric] = concept

            for unit, facts in concepts[concept].get("units", {}).items():
                for fact in facts:
                    row, skipped = self._row_for(ticker, metric, unit, fact, start, end)
                    unclassified += skipped
                    if row is not None:
                        rows.append(row)

        log.debug(
            "edgar.company",
            ticker=ticker,
            metrics=len(chosen),
            rows=len(rows),
            unclassified_spans=unclassified,
            concepts=chosen,
        )
        missing = sorted(set(self.metrics) - set(chosen))
        if missing:
            log.info(
                "edgar.metrics_not_reported",
                ticker=ticker,
                missing=missing,
                note=(
                    "absence is systematic, not random -- a bank reports no cost of "
                    "goods sold -- so dropping these filers is a sector tilt"
                ),
            )
        return rows

    def _row_for(
        self,
        ticker: str,
        metric: str,
        unit: str,
        fact: dict[str, Any],
        start: dt.datetime,
        end: dt.datetime,
    ) -> tuple[dict[str, Any] | None, int]:
        form = str(fact.get("form") or "")
        if form not in PERIODIC_FORMS:
            return None, 0
        filed_raw, end_raw, value = fact.get("filed"), fact.get("end"), fact.get("val")
        if not filed_raw or not end_raw or value is None:
            return None, 0

        known_at = filing_known_at(dt.date.fromisoformat(str(filed_raw)))
        if not (start <= known_at <= end):
            return None, 0

        period = classify_period(fact)
        if period is None:
            return None, 1

        period_end = dt.date.fromisoformat(str(end_raw))
        return (
            {
                "symbol": ticker,
                "as_of": dt.datetime(
                    period_end.year, period_end.month, period_end.day, tzinfo=dt.UTC
                ),
                "known_at": known_at,
                "metric": metric,
                "value": float(value),
                "unit": unit,
                "fiscal_period": period,
                "form": form,
            },
            0,
        )


def _first_reported(candidates: Iterable[str], concepts: dict[str, Any]) -> str | None:
    """The first candidate concept this filer actually reports.

    Choosing one rather than taking every match is what stops two XBRL tags for
    the same idea landing on one key with different values, where deduplication
    would then pick between them by filing date -- that is, arbitrarily.
    """
    for concept in candidates:
        if concept in concepts:
            return concept
    return None
