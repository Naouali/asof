"""CBOE volatility indices.

The tests that matter here are about what VIX *is*: an index level that cannot be
bought, published under a name whose calculation changed in 2003, with an early
history CBOE serves and does not stand behind.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from tests.unit.test_sources import fixture_source, load_fixture

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.cboe import CBOE_SERIES, CboeVolatilityIndices

EASTERN = ZoneInfo("America/New_York")
WINDOW = (dt.datetime(1990, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 9, 20, tzinfo=dt.UTC))


def fetch(settings: Settings, fixture: str, symbol: str) -> pl.DataFrame:
    with fixture_source(CboeVolatilityIndices, load_fixture(fixture), settings) as source:
        return source.fetch([symbol], *WINDOW)


def test_an_index_level_is_not_a_tradeable_bar() -> None:
    """VIX spot cannot be bought. Filing it under ohlcv_daily would let a
    strategy buy it and collect a return nobody could have earned: the tradeable
    expressions are futures, options and ETPs, every one of which has trailed
    spot badly over any long horizon."""
    assert CboeVolatilityIndices.dataset == "series_observations"


def test_the_close_is_knowable_at_the_settlement_time(settings: Settings) -> None:
    """VIX settles at 16:15 Eastern, fifteen minutes after the equity close,
    because the constituent SPX options trade until then."""
    frame = fetch(settings, "cboe_vix_history.csv", "VIX")
    known = frame["known_at"][0].astimezone(EASTERN)

    assert known.strftime("%H:%M") == "16:15"
    assert (frame["known_at"] > frame["as_of"]).all()


def test_the_daily_close_is_what_is_kept(settings: Settings) -> None:
    frame = fetch(settings, "cboe_vix_history.csv", "VIX")

    assert frame["units"].unique().to_list() == ["index level"]
    assert not frame["vintage"].any()
    assert frame["value"].min() > 0
    assert set(frame.columns) >= {"symbol", "as_of", "known_at", "value"}


def test_the_2003_methodology_change_is_reported(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """CBOE moved VIX to the model-free variance-swap calculation on 2003-09-22.
    What is served before that is a back-cast; the index actually published then
    was VXO, computed a different way. A backtest spanning the date is trading
    two instruments under one name."""
    with caplog.at_level("WARNING"):
        fetch(settings, "cboe_vix_history.csv", "VIX")

    assert "methodology_break" in caplog.text
    assert CBOE_SERIES["VIX"].methodology_break == dt.date(2003, 9, 22)


def test_unreliable_early_history_is_dropped_and_said_so(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """CBOE publishes VVIX from March 2006, and the first fortnight is visibly
    unusable -- 71.73, then a nine-day gap, then 15.71. Serving it because the
    provider does would put a 70-point print into a volatility-of-volatility
    series that averages 93."""
    with caplog.at_level("WARNING"):
        frame = fetch(settings, "cboe_vvix_history.csv", "VVIX")

    assert "unreliable_history_dropped" in caplog.text
    assert frame["as_of"].min() >= dt.datetime(2006, 4, 1, tzinfo=dt.UTC)


def test_vvix_has_a_different_column_layout(settings: Settings) -> None:
    """VVIX publishes a close only, where VIX publishes OHLC. A parser assuming
    one layout silently produces nothing for the other."""
    assert CBOE_SERIES["VVIX"].value_column == "VVIX"
    assert CBOE_SERIES["VIX"].value_column == "CLOSE"
    assert fetch(settings, "cboe_vvix_history.csv", "VVIX").height > 0


def test_an_unknown_series_is_refused(settings: Settings) -> None:
    with (
        fixture_source(CboeVolatilityIndices, "DATE,CLOSE\n", settings) as source,
        pytest.raises(SourceError, match="unknown CBOE series"),
    ):
        source.fetch(["VIX99"], *WINDOW)


def test_a_changed_file_layout_fails_rather_than_guessing(settings: Settings) -> None:
    """If CBOE renames the close column, picking whichever column looks numeric
    would silently load the open as the close."""
    with (
        fixture_source(CboeVolatilityIndices, "DATE,SETTLE\n01/02/1990,17.24\n", settings) as src,
        pytest.raises(SourceError, match="changed the file layout"),
    ):
        src.fetch(["VIX"], *WINDOW)


def test_an_unparseable_date_is_not_silently_skipped(settings: Settings) -> None:
    with (
        fixture_source(
            CboeVolatilityIndices, "DATE,OPEN,HIGH,LOW,CLOSE\n1990-01-02,1,1,1,17.2\n", settings
        ) as source,
        pytest.raises(SourceError),
    ):
        source.fetch(["VIX"], *WINDOW)


def test_every_published_series_is_declared_with_its_file() -> None:
    for name, series in CBOE_SERIES.items():
        assert series.symbol == name
        assert series.url.endswith(".csv")
        assert series.description
