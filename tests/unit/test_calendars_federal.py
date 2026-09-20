"""US federal holidays.

Federal agencies do not publish on federal holidays, and the federal calendar is
not the exchange calendar: the NYSE trades on Columbus Day and Veterans Day and
closes on Good Friday, which is not a federal holiday at all. Using one to
predict the other is wrong in both directions.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantlab.data.calendars import (
    federal_holidays,
    is_federal_holiday,
    is_federal_workday,
    next_federal_workday,
)


@pytest.mark.parametrize(
    ("day", "name"),
    [
        (dt.date(2024, 1, 1), "New Year's Day"),
        (dt.date(2024, 1, 15), "Martin Luther King Jr. Day"),
        (dt.date(2024, 2, 19), "Washington's Birthday"),
        (dt.date(2024, 5, 27), "Memorial Day"),
        (dt.date(2024, 6, 19), "Juneteenth"),
        (dt.date(2024, 7, 4), "Independence Day"),
        (dt.date(2024, 9, 2), "Labor Day"),
        (dt.date(2024, 10, 14), "Columbus Day"),
        (dt.date(2024, 11, 11), "Veterans Day"),
        (dt.date(2024, 11, 28), "Thanksgiving"),
        (dt.date(2024, 12, 25), "Christmas"),
    ],
)
def test_the_eleven_federal_holidays(day: dt.date, name: str) -> None:
    assert is_federal_holiday(day), name


def test_there_are_eleven_of_them() -> None:
    assert len(federal_holidays(2024)) == 11


def test_juneteenth_did_not_exist_before_2021() -> None:
    """It was signed into law in June 2021. Backfilling it would close an office
    that was open, and delay a report that was not delayed."""
    assert not any(d.month == 6 for d in federal_holidays(2020))
    assert any(d.month == 6 for d in federal_holidays(2021))
    assert len(federal_holidays(2020)) == 10


@pytest.mark.parametrize(
    ("actual", "observed", "why"),
    [
        (dt.date(2021, 7, 4), dt.date(2021, 7, 5), "Sunday -> the Monday after"),
        (dt.date(2026, 7, 4), dt.date(2026, 7, 3), "Saturday -> the Friday before"),
        (dt.date(2021, 6, 19), dt.date(2021, 6, 18), "Saturday -> the Friday before"),
        (dt.date(2022, 12, 25), dt.date(2022, 12, 26), "Sunday -> the Monday after"),
    ],
)
def test_a_weekend_holiday_is_observed_on_the_nearest_weekday(
    actual: dt.date, observed: dt.date, why: str
) -> None:
    assert is_federal_holiday(observed), why
    assert not is_federal_holiday(actual) or actual.weekday() < 5


def test_new_years_day_can_be_observed_in_the_previous_year() -> None:
    """1 January 2022 fell on a Saturday, so it was observed on Friday 31
    December 2021. Looking a date up only in its own calendar year misses that,
    and the miss lands on a Friday -- exactly when a government report would
    otherwise be released."""
    assert is_federal_holiday(dt.date(2021, 12, 31))
    assert dt.date(2021, 12, 31) in federal_holidays(2021)
    assert is_federal_holiday(dt.date(2027, 12, 31))
    # And it does not fire when New Year's Day falls midweek.
    assert not is_federal_holiday(dt.date(2024, 12, 31))  # 1 Jan 2025 was a Wednesday


def test_the_federal_calendar_is_not_the_exchange_calendar() -> None:
    """Good Friday closes the NYSE and is not a federal holiday; Columbus Day and
    Veterans Day are federal holidays and the NYSE trades through both."""
    assert not is_federal_holiday(dt.date(2024, 3, 29))  # Good Friday
    assert is_federal_holiday(dt.date(2024, 10, 14))  # Columbus Day
    assert is_federal_holiday(dt.date(2024, 11, 11))  # Veterans Day


def test_workdays_exclude_weekends_and_holidays() -> None:
    assert is_federal_workday(dt.date(2024, 7, 3))  # Wednesday
    assert not is_federal_workday(dt.date(2024, 7, 4))  # holiday
    assert not is_federal_workday(dt.date(2024, 7, 6))  # Saturday


def test_the_next_workday_rolls_forward_over_a_long_weekend() -> None:
    # Thursday 4 July 2024; Friday the 5th was a normal working day.
    assert next_federal_workday(dt.date(2024, 7, 4)) == dt.date(2024, 7, 5)
    # Christmas 2021 fell on a Saturday, observed Friday the 24th.
    assert next_federal_workday(dt.date(2021, 12, 24)) == dt.date(2021, 12, 27)


def test_an_already_valid_workday_is_returned_unchanged() -> None:
    day = dt.date(2024, 3, 6)
    assert next_federal_workday(day) == day
    assert next_federal_workday(day, inclusive=False) == dt.date(2024, 3, 7)


def test_every_year_from_1986_has_a_sane_calendar() -> None:
    """A year that silently produced the wrong number of holidays would shift
    every derived release date in it.

    The count is 10 before Juneteenth and 11 after, give or take one: a Saturday
    New Year's Day is observed on 31 December of the *previous* year, so 1994
    genuinely has nine observances and 1993 has eleven. What must never happen is
    a date filed under the wrong year, because a lookup only ever searches its
    own year's set and the holiday would vanish.
    """
    for year in range(1986, 2041):
        holidays = federal_holidays(year)
        expected = 10 if year < 2021 else 11
        assert expected - 1 <= len(holidays) <= expected + 1, year
        assert all(d.year == year for d in holidays), year


def test_no_new_years_observance_is_lost_or_duplicated() -> None:
    """Every New Year's Day from 1986 to 2040 is observed exactly once somewhere
    in the calendars, whichever year it lands in. The 1994 bug was a date filed
    under 1994 while sitting in 1993, so it was in a set but reachable from
    neither."""
    observed = set()
    for year in range(1985, 2042):
        observed |= federal_holidays(year)

    for year in range(1986, 2041):
        new_year = dt.date(year, 1, 1)
        shift = {5: -1, 6: 1}.get(new_year.weekday(), 0)
        expected = new_year + dt.timedelta(days=shift)
        assert expected in observed, f"New Year {year} observed {expected} is missing"
        assert is_federal_holiday(expected), f"{expected} is filed under the wrong year"
