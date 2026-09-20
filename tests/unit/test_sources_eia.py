"""EIA energy inventories.

Untested against the live API: no free key was configured when this was written.
The parser runs against a fixture hand-built from EIA's documented v2 envelope,
so these tests assert that it handles the shape the documentation describes, not
the shape the API actually returns. Re-record once a key exists -- the same debt
FRED carried until Milestone 9 and the same way it was paid off.

What *is* fully tested is the release-date arithmetic, which needs no key and is
where the look-ahead risk lives.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest
from tests.unit.test_sources import fixture_source

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.eia import (
    EIA_SERIES,
    NATURAL_GAS_WEEKDAY,
    PETROLEUM_WEEKDAY,
    EiaEnergyStocks,
    eia_release,
)

EASTERN = ZoneInfo("America/New_York")
WINDOW = (dt.datetime(2024, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 9, 20, tzinfo=dt.UTC))


def eastern(moment: dt.datetime) -> dt.datetime:
    return moment.astimezone(EASTERN)


# ------------------------------------------------------------------ releases --
def test_the_weekly_petroleum_report_lands_on_wednesday_at_half_past_ten() -> None:
    """The week ends Friday and the report is published the following Wednesday.
    Dating the observation by its period would make it knowable five days before
    the number existed."""
    released = eastern(eia_release(dt.date(2026, 9, 11), PETROLEUM_WEEKDAY))

    assert released.strftime("%a %Y-%m-%d %H:%M") == "Wed 2026-09-16 10:30"


def test_natural_gas_storage_lands_a_day_later() -> None:
    released = eastern(eia_release(dt.date(2026, 9, 11), NATURAL_GAS_WEEKDAY))
    assert released.strftime("%a %Y-%m-%d") == "Thu 2026-09-17"


def test_a_federal_holiday_pushes_the_release_forward() -> None:
    """EIA is a federal agency and does not publish on a federal holiday. Rolling
    forward spends the uncertainty in the direction that costs a signal rather
    than the one that manufactures it."""
    # Christmas Day 2024 fell on a Wednesday.
    released = eastern(eia_release(dt.date(2024, 12, 20), PETROLEUM_WEEKDAY))

    assert released.date() > dt.date(2024, 12, 25)
    assert released.strftime("%H:%M") == "10:30"


def test_the_release_time_is_a_wall_clock() -> None:
    """10:30 in Washington is 14:30 UTC in summer and 15:30 in winter."""
    assert eia_release(dt.date(2026, 6, 12), PETROLEUM_WEEKDAY).hour == 14
    assert eia_release(dt.date(2026, 12, 11), PETROLEUM_WEEKDAY).hour == 15


def test_the_release_always_follows_the_period() -> None:
    day = dt.date(2020, 1, 3)
    while day < dt.date(2030, 1, 1):
        for weekday in (PETROLEUM_WEEKDAY, NATURAL_GAS_WEEKDAY):
            release = eia_release(day, weekday)
            assert release.date() > day, (day, weekday)
            assert (release.date() - day).days <= 12, (day, weekday)
        day += dt.timedelta(days=7)


def test_a_period_ending_on_the_release_weekday_waits_a_full_week() -> None:
    """Otherwise a Wednesday period would be published at 10:30 the same morning,
    before the week it covers has finished."""
    released = eia_release(dt.date(2026, 9, 16), PETROLEUM_WEEKDAY)  # a Wednesday
    assert released.date() == dt.date(2026, 9, 23)


# -------------------------------------------------------------------- fetch --
def envelope(rows: list[dict[str, object]]) -> dict[str, object]:
    """EIA's documented v2 response envelope."""
    return {"response": {"total": len(rows), "data": rows}}


def test_the_key_requirement_is_stated_plainly() -> None:
    """The platform runs with zero keys; a source that needs one says how to get
    it rather than failing obscurely."""
    settings = Settings(eia_api_key=None)
    with (
        fixture_source(EiaEnergyStocks, envelope([]), settings) as source,
        pytest.raises(SourceError, match=r"eia\.gov/opendata"),
    ):
        source.fetch(["CRUDE_STOCKS"], *WINDOW)


def test_observations_are_dated_by_release(settings: Settings) -> None:
    settings = Settings(eia_api_key="x" * 32)
    rows = [
        {"period": "2026-09-11", "value": 420000},
        {"period": "2026-09-04", "value": 418000},
    ]
    with fixture_source(EiaEnergyStocks, envelope(rows), settings) as source:
        frame = source.fetch(["CRUDE_STOCKS"], *WINDOW)

    assert frame.height == 2
    assert (frame["known_at"] > frame["as_of"]).all()
    assert not frame["vintage"].any(), "EIA keeps no archive of first prints"
    assert frame["units"].unique().to_list() == ["thousand barrels"]


def test_the_restatement_problem_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """It cannot be fixed here -- the first print is not archived anywhere -- so
    the only honest thing is to say so on every fetch."""
    settings = Settings(eia_api_key="x" * 32)
    rows = [{"period": "2026-09-11", "value": 420000}]
    with (
        caplog.at_level("WARNING"),
        fixture_source(EiaEnergyStocks, envelope(rows), settings) as source,
    ):
        source.fetch(["CRUDE_STOCKS"], *WINDOW)

    assert "restated_without_vintages" in caplog.text
    assert "look-ahead" in caplog.text


def test_an_unknown_series_is_refused() -> None:
    settings = Settings(eia_api_key="x" * 32)
    with (
        fixture_source(EiaEnergyStocks, envelope([]), settings) as source,
        pytest.raises(SourceError, match="unknown EIA series"),
    ):
        source.fetch(["URANIUM"], *WINDOW)


def test_a_changed_envelope_fails_rather_than_returning_nothing() -> None:
    settings = Settings(eia_api_key="x" * 32)
    with (
        fixture_source(EiaEnergyStocks, {"unexpected": {}}, settings) as source,
        pytest.raises(SourceError, match="v2 envelope"),
    ):
        source.fetch(["CRUDE_STOCKS"], *WINDOW)


def test_every_series_declares_a_release_day_and_units() -> None:
    for name, series in EIA_SERIES.items():
        assert series.symbol == name
        assert series.release_weekday in (PETROLEUM_WEEKDAY, NATURAL_GAS_WEEKDAY)
        assert series.units
        assert series.description
