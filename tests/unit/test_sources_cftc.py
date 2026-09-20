"""CFTC Commitments of Traders.

Most of these tests are about one number: `known_at`. A COT report counts
positions on Tuesday and is published the following Friday at 15:30 Eastern, and
no CFTC payload carries that release date, so it has to be derived. Deriving it
wrongly is the most common look-ahead bias in published COT research -- a signal
that reads Tuesday's positioning on Tuesday is reading a number that did not
exist for another three days.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from tests.unit.test_sources import fixture_source, load_json_fixture

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.schemas import get_schema
from quantlab.data.sources.cftc import (
    COT_REPORTS,
    WEEKLY_REPORTING_BEGAN,
    CftcPositioning,
    PreWeeklyEraError,
    cot_release,
)

EASTERN = ZoneInfo("America/New_York")


def eastern(release: dt.datetime) -> dt.datetime:
    return release.astimezone(EASTERN)


# ------------------------------------------------------------------ the release --
def test_tuesdays_report_is_published_on_friday_at_half_past_three() -> None:
    """The CFTC's own description: 'published each Friday at 3:30 pm Eastern
    Time, using the data from the immediately preceding Tuesday of that week'."""
    released = eastern(cot_release(dt.date(2026, 9, 15)))

    assert released.date() == dt.date(2026, 9, 18)
    assert released.strftime("%a %H:%M") == "Fri 15:30"


def test_the_gap_is_three_days_wide() -> None:
    """The whole point of the dataset's known_at. A signal reading Tuesday's
    number on Tuesday is reading something that does not exist yet."""
    report = dt.date(2026, 9, 15)
    assert (cot_release(report).date() - report).days == 3


def test_the_release_time_is_a_wall_clock_time_not_a_utc_offset() -> None:
    """15:30 in Washington is 19:30 UTC in summer and 20:30 in winter. Storing a
    fixed offset would put every winter release an hour early, and an hour early
    is enough to see a number before it exists."""
    summer = cot_release(dt.date(2026, 6, 16))
    winter = cot_release(dt.date(2026, 12, 15))

    assert summer.hour == 19
    assert winter.hour == 20
    assert eastern(summer).hour == eastern(winter).hour == 15


def test_a_holiday_rolls_the_release_forward_never_backward() -> None:
    """Uncertainty about the schedule is spent in the direction that costs a
    signal performance, rather than the direction that manufactures it."""
    # 1 January 2022 fell on a Saturday, so Friday 31 December 2021 was the
    # observed federal holiday. The release moves to the following Monday.
    released = eastern(cot_release(dt.date(2021, 12, 28)))
    assert released.date() == dt.date(2022, 1, 3)
    assert released.strftime("%a") == "Mon"


def test_a_holiday_ordinary_friday_still_releases_on_friday() -> None:
    """The Friday after Thanksgiving is not a federal holiday, whatever the
    exchanges do with it."""
    assert eastern(cot_release(dt.date(2024, 11, 26))).date() == dt.date(2024, 11, 29)


def test_a_snapshot_taken_on_a_non_tuesday_still_gets_a_release() -> None:
    """161 of 1,933 historical reports fall on some other weekday, because a
    holiday moved the snapshot."""
    released = eastern(cot_release(dt.date(2026, 7, 1)))  # a Wednesday
    assert released.date() > dt.date(2026, 7, 4)
    assert released.strftime("%H:%M") == "15:30"


def test_the_release_is_always_after_the_report() -> None:
    day = WEEKLY_REPORTING_BEGAN
    while day < dt.date(2041, 1, 1):
        release = cot_release(day)
        assert release.date() > day, day
        assert 3 <= (release.date() - day).days <= 7, day
        day += dt.timedelta(days=7)


# ------------------------------------------------------------ the pre-1992 era --
def test_data_the_cftc_never_published_at_the_time_is_refused() -> None:
    """Before 1992-09-30 the report was semi-monthly, and the CFTC states the
    mid-month data 'was not published before that time' and was compiled later.
    There is no honest known_at to give it, so it is not given one."""
    with pytest.raises(PreWeeklyEraError) as excinfo:
        cot_release(dt.date(1990, 6, 15))

    message = str(excinfo.value)
    assert "not published at the time" in message
    assert "allow_pre_weekly_era" in message  # and says how to override


def test_the_cutoff_is_inclusive_of_the_first_weekly_report() -> None:
    cot_release(WEEKLY_REPORTING_BEGAN)
    with pytest.raises(PreWeeklyEraError):
        cot_release(WEEKLY_REPORTING_BEGAN - dt.timedelta(days=1))


# ------------------------------------------------------------------- the fetch --
def test_a_recorded_payload_parses(settings: Settings) -> None:
    payload = load_json_fixture("cftc_cot_legacy_gold.json")
    with fixture_source(CftcPositioning, payload, settings) as source:
        frame = source.fetch(
            ["088691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )

    assert frame["symbol"].unique().to_list() == ["088691"]
    assert set(frame["category"].unique()) == {
        "noncommercial",
        "commercial",
        "nonreportable",
        "total",
    }
    assert frame["exchange"].unique().to_list() == ["CMX"]
    assert "TROY OUNCES" in frame["contract_units"][0]


def test_every_row_carries_the_derived_release(settings: Settings) -> None:
    payload = load_json_fixture("cftc_cot_legacy_gold.json")
    with fixture_source(CftcPositioning, payload, settings) as source:
        frame = source.fetch(
            ["088691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )

    lags = (frame["known_at"] - frame["as_of"]).dt.total_hours().unique().to_list()
    # 92 hours is three days plus 20:30 UTC, which is 15:30 Eastern in winter.
    assert set(lags) == {92}
    assert (frame["known_at"] > frame["as_of"]).all()


def test_the_accounting_identity_holds(settings: Settings) -> None:
    """COT's own invariant: open interest equals the sum of long positions across
    every category plus non-commercial spreading, and likewise for shorts.
    It holds exactly, so it verifies the column mapping rather than assuming it --
    a transposed column would break it immediately."""
    payload = load_json_fixture("cftc_cot_legacy_gold.json")
    with fixture_source(CftcPositioning, payload, settings) as source:
        frame = source.fetch(
            ["088691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )

    wide = frame.pivot(index="as_of", on=["category", "measure"], values="value")

    def column(fragment: str) -> pl.Expr:
        return pl.col(next(c for c in wide.columns if fragment in c))

    spreading = column('{"noncommercial","spreading"}')
    open_interest = column('{"total","open_interest"}')
    for side in ("long", "short"):
        total = (
            column(f'{{"noncommercial","{side}"}}')
            + column(f'{{"commercial","{side}"}}')
            + column(f'{{"nonreportable","{side}"}}')
            + spreading
        )
        assert wide.select((total - open_interest).abs().max()).item() == 0.0, side


def test_an_empty_result_fails_loudly_with_a_usable_hint(settings: Settings) -> None:
    """Spec section 13: never an empty frame to mean something went wrong. The
    message has to name the likely mistake, which is passing a ticker where a
    contract market code belongs."""
    with (
        fixture_source(CftcPositioning, [], settings) as source,
        pytest.raises(SourceError) as excinfo,
    ):
        source.fetch(
            ["088691"],
            dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 2, 1, tzinfo=dt.UTC),
        )
    assert "no COT observations" in str(excinfo.value)
    assert "six characters" in str(excinfo.value)


def test_pre_weekly_rows_are_dropped_by_default(settings: Settings) -> None:
    payload = [
        {
            "report_date_as_yyyy_mm_dd": "1990-06-15T00:00:00.000",
            "cftc_contract_market_code": "088691",
            "open_interest_all": "1000",
        }
    ]
    with fixture_source(CftcPositioning, payload, settings) as source, pytest.raises(SourceError):
        source.fetch(
            ["088691"],
            dt.datetime(1990, 1, 1, tzinfo=dt.UTC),
            dt.datetime(1991, 1, 1, tzinfo=dt.UTC),
        )


def test_pre_weekly_rows_can_be_opted_into_with_a_punitive_lag(settings: Settings) -> None:
    """Kept available because refusing outright loses data someone may want for
    a non-signal purpose. The lag is deliberately long enough that any signal
    built on it is penalised rather than flattered."""
    payload = [
        {
            "report_date_as_yyyy_mm_dd": "1990-06-15T00:00:00.000",
            "cftc_contract_market_code": "088691",
            "open_interest_all": "1000",
        }
    ]
    with fixture_source(CftcPositioning, payload, settings, allow_pre_weekly_era=True) as source:
        frame = source.fetch(
            ["088691"],
            dt.datetime(1990, 1, 1, tzinfo=dt.UTC),
            dt.datetime(1991, 1, 1, tzinfo=dt.UTC),
        )

    assert frame.height == 1
    assert (frame["known_at"] - frame["as_of"]).dt.total_days().item() == 90


# ------------------------------------------------------------------- the reports --
def test_one_fetcher_serves_every_report() -> None:
    """The registry is keyed <source>.<dataset> and all three reports write the
    canonical `positioning` dataset, so the report is an option rather than a
    separate fetcher -- as the bar interval is for Binance."""
    assert CftcPositioning.name == "cftc_cot.positioning"
    assert CftcPositioning.dataset == "positioning"
    assert CftcPositioning.spec.key == "cftc_cot"


@pytest.mark.parametrize("report", ["legacy", "disaggregated", "tff"])
def test_every_report_can_be_selected(report: str) -> None:
    assert CftcPositioning(report=report).report_name == report


def test_an_unknown_report_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown COT report"):
        CftcPositioning(report="supplemental")


def test_the_report_is_recorded_on_every_row(settings: Settings) -> None:
    """The reports overlap -- legacy and disaggregated both publish a total open
    interest for the same contract and Tuesday -- so without the report on the
    row those observations share a key and one silently replaces the other."""
    payload = load_json_fixture("cftc_cot_legacy_gold.json")
    with fixture_source(CftcPositioning, payload, settings) as source:
        frame = source.fetch(
            ["088691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )
    assert frame["report"].unique().to_list() == ["legacy"]
    assert "report" in get_schema("positioning").key


def test_the_three_reports_read_different_datasets() -> None:
    ids = {report.dataset_id for report in COT_REPORTS.values()}
    assert len(ids) == len(COT_REPORTS)


def test_the_disaggregated_report_splits_what_legacy_conflates() -> None:
    """The reason the disaggregated report exists: a swap dealer hedging an index
    position is not a producer hedging a crop, and legacy calls both commercial."""
    legacy = set(COT_REPORTS["legacy"].categories)
    disaggregated = set(COT_REPORTS["disaggregated"].categories)

    assert "commercial" in legacy
    assert "commercial" not in disaggregated
    assert {"producer_merchant", "swap_dealer"} <= disaggregated


# ------------------------------------------------------- the lag, end to end --
def test_a_snapshot_cannot_see_tuesdays_positioning_until_friday(
    settings: Settings, store: object
) -> None:
    """The bias this whole module exists to prevent, proven through the lake.

    A report counting positions on Tuesday 2024-12-17 was published at 15:30
    Eastern on Friday the 20th. Snapshots before that instant must see nothing,
    and the boundary must be sharp to the minute -- an hour of slack is an hour
    of look-ahead.
    """
    payload = load_json_fixture("cftc_cot_legacy_gold.json")
    with fixture_source(CftcPositioning, payload, settings) as source:
        frame = source.fetch(
            ["088691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )
    store.write(frame, asset_class="futures")  # type: ignore[attr-defined]

    tuesday = dt.datetime(2024, 12, 17, tzinfo=dt.UTC)

    def visible(instant: dt.datetime) -> int:
        snapshot = store.as_of(instant)  # type: ignore[attr-defined]
        return snapshot.frame("positioning").filter(pl.col("as_of") == tuesday).height

    # The snapshot date itself, and the three days it takes to process.
    assert visible(dt.datetime(2024, 12, 17, 23, 59, tzinfo=dt.UTC)) == 0
    assert visible(dt.datetime(2024, 12, 19, 23, 59, tzinfo=dt.UTC)) == 0

    # Friday, to the minute. 15:30 Eastern is 20:30 UTC in December.
    assert visible(dt.datetime(2024, 12, 20, 15, 29, tzinfo=EASTERN)) == 0
    assert visible(dt.datetime(2024, 12, 20, 15, 30, tzinfo=EASTERN)) > 0
    assert visible(dt.datetime(2024, 12, 23, tzinfo=dt.UTC)) > 0


# ------------------------------------------------------- contract code hygiene --
def test_a_yaml_octal_contract_code_is_rejected_not_guessed() -> None:
    """The bug this exists for, and it cost 57% of the COT data before it was
    found.

    YAML's legacy octal rule parses an unquoted `002602` as **octal**, giving the
    integer 1410. Only codes whose digits are all 0-7 are affected, so `088691`
    (gold) survives and `002602` (corn) does not -- which makes the failure look
    like a gap at the CFTC rather than a parsing bug in the config. Socrata then
    answers the bogus code with an empty array rather than an error.

    1410 cannot be turned back into 002602, so this raises rather than guessing.
    """
    with pytest.raises(SourceError) as excinfo:
        CftcPositioning()._fetch_symbols([1410])

    message = str(excinfo.value)
    assert "octal" in message
    assert "1410" in message
    assert 'symbols: ["002602"]' in message


def test_a_ticker_is_rejected_with_the_right_advice() -> None:
    with pytest.raises(SourceError, match="CFTC codes"):
        CftcPositioning()._fetch_symbols(["GC"])


def test_valid_codes_pass_through_unchanged() -> None:
    codes = ["088691", "13874A", "20974+", "002602"]
    assert CftcPositioning()._fetch_symbols(codes) == codes


def test_every_default_contract_code_is_well_formed() -> None:
    """The defaults were guessed once and three of them were wrong. They are now
    verified against the live API, and this keeps them the right shape."""
    from quantlab.data.sources.cftc import DEFAULT_CONTRACTS

    assert CftcPositioning()._fetch_symbols(list(DEFAULT_CONTRACTS))
    assert len(set(DEFAULT_CONTRACTS)) == len(DEFAULT_CONTRACTS)


def test_contracts_that_return_nothing_are_reported(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Socrata answers an unknown code with an empty array, not an error, so a
    typo ingests silently as nothing. Spec section 13: fail loudly."""
    payload = load_json_fixture("cftc_cot_legacy_gold.json")
    with caplog.at_level("WARNING"), fixture_source(CftcPositioning, payload, settings) as source:
        source.fetch(
            ["088691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )
    # The gold fixture is returned for every request, so nothing is empty here;
    # the warning path is exercised by an empty payload.
    with (
        caplog.at_level("WARNING"),
        fixture_source(CftcPositioning, [], settings) as source,
        pytest.raises(SourceError),
    ):
        source.fetch(
            ["088691", "084691"],
            dt.datetime(2024, 11, 1, tzinfo=dt.UTC),
            dt.datetime(2024, 12, 31, tzinfo=dt.UTC),
        )
    assert "contracts_returned_nothing" in caplog.text
