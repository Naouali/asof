"""SEC EDGAR XBRL company facts.

EDGAR matters because it is as-filed: `known_at` is the filing date, so a
backtest sees a year through that year's eyes rather than through every
subsequent restatement. These tests are mostly about the three ways that value
is silently destroyed -- mixing reporting spans, picking between duplicate XBRL
tags arbitrarily, and treating a filing as knowable before it was accepted.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from tests.unit.test_sources import fixture_source, load_json_fixture

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.edgar import (
    METRIC_TAGS,
    PERIODIC_FORMS,
    EdgarFundamentals,
    classify_period,
    filing_known_at,
)

EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- known_at --
def test_a_filing_is_knowable_at_the_acceptance_cutoff_not_at_midnight() -> None:
    """EDGAR accepts filings until 17:30 Eastern; anything later carries the next
    business day's date. Midnight UTC on the filed date is 8 p.m. the previous
    evening in New York, which would make every filing knowable before the SEC
    had it."""
    known = filing_known_at(dt.date(2024, 11, 1))

    assert known.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M") == "2024-11-01 17:30"
    assert known > dt.datetime(2024, 11, 1, tzinfo=dt.UTC)


def test_the_cutoff_is_a_wall_clock_time() -> None:
    """17:30 in Washington is 21:30 UTC in summer and 22:30 in winter."""
    assert filing_known_at(dt.date(2024, 7, 1)).hour == 21
    assert filing_known_at(dt.date(2024, 12, 1)).hour == 22


# ------------------------------------------------------------------- spans --
def test_an_instantaneous_fact_is_labelled_instant() -> None:
    """A balance sheet item has no duration; `as_of` fully describes it."""
    assert classify_period({"end": "2024-09-28", "fy": 2024, "fp": "FY"}) == "instant"


@pytest.mark.parametrize(
    ("start", "end", "label"),
    [
        ("2024-06-30", "2024-09-28", "Q"),
        ("2024-03-31", "2024-09-28", "H"),
        ("2023-12-31", "2024-09-28", "9M"),
        ("2023-09-30", "2024-09-28", "FY"),
    ],
)
def test_duration_facts_are_classified_by_span(start: str, end: str, label: str) -> None:
    assert classify_period({"start": start, "end": end, "fy": 2024, "fp": "FY"}) == label


def test_an_unrecognisable_span_is_dropped_rather_than_guessed() -> None:
    """A seven-month figure is a transition period from a fiscal-year change.
    Calling it a quarter or a year is wrong in both directions."""
    assert classify_period({"start": "2024-01-01", "end": "2024-08-01", "fy": 2024}) is None
    assert classify_period({"start": "bad", "end": "2024-08-01", "fy": 2024}) is None


def test_the_label_never_carries_a_year() -> None:
    """`fy` on a fact is the FILING's fiscal year, not the fact's. Apple's FY2023
    revenue, restated as a comparative in the FY2025 10-K, carries fy=2025, so a
    label built from it files three different years under one name. The period
    end is already in `as_of`."""
    restated = {"start": "2022-09-25", "end": "2023-09-30", "fy": 2025, "fp": "FY"}
    assert classify_period(restated) == "FY"
    assert "2025" not in classify_period(restated)


# ------------------------------------------------------------------- fetch --
def fetch_aapl(settings: Settings, **kwargs: object) -> pl.DataFrame:
    payload = load_json_fixture("edgar_companyfacts_aapl.json")
    with fixture_source(EdgarFundamentals, payload, settings, **kwargs) as source:
        source._ticker_map = {"AAPL": 320193}
        return source.fetch(
            ["AAPL"],
            dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2026, 9, 20, tzinfo=dt.UTC),
        )


def test_a_recorded_payload_parses(settings: Settings) -> None:
    frame = fetch_aapl(settings)

    assert frame["symbol"].unique().to_list() == ["AAPL"]
    assert set(frame["fiscal_period"].unique()) <= {"instant", "Q", "H", "9M", "FY"}
    assert (frame["known_at"] > frame["as_of"]).all(), "a filing follows its period"


def test_annual_and_quarterly_figures_stay_separable(settings: Settings) -> None:
    """The failure this guards: EDGAR reports revenue over four different windows
    in the same filing, and a ratio built from whichever was filed last divides a
    quarter's revenue into a year's assets."""
    frame = fetch_aapl(settings)
    revenue = frame.filter(pl.col("metric") == "revenue")

    annual = revenue.filter(pl.col("fiscal_period") == "FY")["value"]
    quarterly = revenue.filter(pl.col("fiscal_period") == "Q")["value"]

    assert annual.len() > 0
    assert quarterly.len() > 0
    assert annual.min() > quarterly.max(), "a year must exceed any quarter in it"


def test_only_one_xbrl_tag_is_used_per_metric(settings: Settings) -> None:
    """Apple reports both `RevenueFromContractWithCustomer...` and `Revenues`.
    Emitting both puts two values under one key, and deduplication then picks
    between them by filing date -- that is, arbitrarily."""
    payload = load_json_fixture("edgar_companyfacts_aapl.json")
    concepts = payload["facts"]["us-gaap"]
    assert {"RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"} <= set(concepts)

    frame = fetch_aapl(settings)
    revenue = frame.filter((pl.col("metric") == "revenue") & (pl.col("fiscal_period") == "FY"))
    duplicated = revenue.group_by("as_of", "known_at").len().filter(pl.col("len") > 1)
    assert duplicated.height == 0


def test_the_same_period_is_kept_at_every_filing(settings: Settings) -> None:
    """The point of as-filed data: a later filing restating an earlier year is a
    new row with a later known_at, and the superseded value stays visible to
    earlier snapshots."""
    frame = fetch_aapl(settings)
    annual = frame.filter((pl.col("metric") == "revenue") & (pl.col("fiscal_period") == "FY"))
    refiled = annual.group_by("as_of").len().filter(pl.col("len") > 1)

    assert refiled.height > 0, "a 10-K restates the prior year as a comparative"


def test_non_periodic_forms_are_ignored(settings: Settings) -> None:
    frame = fetch_aapl(settings)
    assert set(frame["form"].unique()) <= PERIODIC_FORMS


def test_the_window_filters_on_the_filing_date(settings: Settings) -> None:
    """Not on the period end: a 2015 period restated in a 2025 filing became
    knowable in 2025, and a window ending in 2020 must not contain it."""
    payload = load_json_fixture("edgar_companyfacts_aapl.json")
    with fixture_source(EdgarFundamentals, payload, settings) as source:
        source._ticker_map = {"AAPL": 320193}
        with pytest.raises(SourceError, match="FILING date"):
            source.fetch(
                ["AAPL"],
                dt.datetime(1995, 1, 1, tzinfo=dt.UTC),
                dt.datetime(1996, 1, 1, tzinfo=dt.UTC),
            )


def test_an_unknown_ticker_is_refused_not_guessed(settings: Settings) -> None:
    payload = load_json_fixture("edgar_companyfacts_aapl.json")
    with fixture_source(EdgarFundamentals, payload, settings) as source:
        source._ticker_map = {"AAPL": 320193}
        with pytest.raises(SourceError, match="US registrants only"):
            source.fetch(
                ["VOD.L"],
                dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
                dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            )


def test_an_unknown_metric_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown metrics"):
        EdgarFundamentals(metrics=["ebitda"])


def test_metric_tags_are_ordered_by_precedence() -> None:
    """The first tag a filer reports wins, so order is meaning, not decoration."""
    assert METRIC_TAGS["revenue"][0] == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert all(tags for tags in METRIC_TAGS.values())
    assert EdgarFundamentals.dataset == "fundamentals"
    assert EdgarFundamentals.name == "sec_edgar.fundamentals"
