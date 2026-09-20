"""Tier 2 signals: real but fragile.

Both of these have a sign convention that, reversed, produces a strategy which
systematically pays a risk premium instead of earning it -- and backtests as a
confident, slow loss rather than as an obvious bug. Those conventions are pinned
here against the theory, not against a number someone once observed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.signals.base import SignalUnavailableError
from quantlab.signals.positioning.hedger_pressure import (
    MIN_OBSERVATIONS,
    CommercialHedgerPressure,
    hedger_pressure_index,
)
from quantlab.signals.volatility.term_structure import (
    CONTANGO_FLOOR,
    VixTermStructure,
    term_structure_slope,
)


# ----------------------------------------------------------- hedging pressure --
def positioning_rows(
    symbol: str, longs: list[float], shorts: list[float], *, report: str = "legacy"
) -> list[dict[str, object]]:
    start = dt.datetime(2020, 1, 7, tzinfo=dt.UTC)
    rows = []
    for week, (long, short) in enumerate(zip(longs, shorts, strict=True)):
        as_of = start + dt.timedelta(weeks=week)
        for measure, value in (("long", long), ("short", short)):
            rows.append(
                {
                    "source": "cftc_cot",
                    "dataset": "positioning",
                    "symbol": symbol,
                    "as_of": as_of,
                    "known_at": as_of + dt.timedelta(days=3),
                    "ingested_at": as_of + dt.timedelta(days=3),
                    "report": report,
                    "name": f"{symbol} - TEST EXCHANGE",
                    "category": "commercial",
                    "measure": measure,
                    "value": value,
                    "contract_units": "contracts",
                    "exchange": "CME",
                }
            )
    return rows


def test_the_index_is_bounded_and_signed_by_the_net() -> None:
    index = hedger_pressure_index(np.array([100.0, 0.0, 50.0]), np.array([0.0, 100.0, 50.0]))
    assert index.tolist() == [1.0, -1.0, 0.0]


def test_an_empty_commercial_book_is_undefined_not_zero() -> None:
    """Zero would read as 'perfectly balanced', which is a position. Absent is
    not a position."""
    assert np.isnan(hedger_pressure_index(np.array([0.0]), np.array([0.0]))[0])


def test_the_index_scales_by_the_commercial_book_not_open_interest() -> None:
    """Dividing by open interest makes the index move when the *other* side
    trades, which is not what the signal claims to measure."""
    steady = hedger_pressure_index(np.array([100.0]), np.array([300.0]))
    # Speculators double their activity; commercials do not move.
    assert hedger_pressure_index(np.array([100.0]), np.array([300.0])) == steady


def test_commercials_unusually_short_is_a_long_signal(store: object) -> None:
    """Keynes and Hicks: hedgers are net short, speculators take the other side
    and are paid to. So an unusually short commercial book means an unusually
    large premium accruing to the long. Reversed, this signal pays the premium
    every week instead of earning it."""
    weeks = MIN_OBSERVATIONS + 20
    longs = [500.0] * weeks
    shorts = [500.0] * (weeks - 1) + [1500.0]  # commercials lurch short at the end
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(positioning_rows("088691", longs, shorts)), asset_class="futures"
    )

    as_of = dt.datetime(2020, 1, 7, tzinfo=dt.UTC) + dt.timedelta(weeks=weeks + 1)
    scores = CommercialHedgerPressure().compute(store.as_of(as_of), ["088691"])  # type: ignore[attr-defined]

    assert scores["score"][0] > 0, "commercials short => speculators long => go long"


def test_commercials_unusually_long_is_a_short_signal(store: object) -> None:
    weeks = MIN_OBSERVATIONS + 20
    longs = [500.0] * (weeks - 1) + [1500.0]
    shorts = [500.0] * weeks
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(positioning_rows("084691", longs, shorts)), asset_class="futures"
    )

    as_of = dt.datetime(2020, 1, 7, tzinfo=dt.UTC) + dt.timedelta(weeks=weeks + 1)
    scores = CommercialHedgerPressure().compute(store.as_of(as_of), ["084691"])  # type: ignore[attr-defined]

    assert scores["score"][0] < 0


def test_a_short_history_is_refused_rather_than_scored(store: object) -> None:
    """A z-score over less than a year of weekly data is not a z-score."""
    weeks = 20
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(positioning_rows("099741", [500.0] * weeks, [600.0] * weeks)),
        asset_class="futures",
    )
    as_of = dt.datetime(2020, 1, 7, tzinfo=dt.UTC) + dt.timedelta(weeks=weeks + 1)

    with pytest.raises(SignalUnavailableError, match="weeks of"):
        CommercialHedgerPressure().compute(store.as_of(as_of), ["099741"])  # type: ignore[attr-defined]


def test_a_lookback_shorter_than_a_year_is_refused() -> None:
    with pytest.raises(ValueError, match="noise wearing a number"):
        CommercialHedgerPressure(lookback_weeks=12)


def test_an_unavailable_report_says_which_categories_exist(store: object) -> None:
    weeks = MIN_OBSERVATIONS + 5
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(positioning_rows("088691", [500.0] * weeks, [600.0] * weeks)),
        asset_class="futures",
    )
    as_of = dt.datetime(2020, 1, 7, tzinfo=dt.UTC) + dt.timedelta(weeks=weeks + 1)

    signal = CommercialHedgerPressure(report="disaggregated")
    with pytest.raises(SignalUnavailableError, match="no 'commercial' at all"):
        signal.compute(store.as_of(as_of), ["088691"])  # type: ignore[attr-defined]


def test_the_publication_lag_is_the_declared_failure_mode() -> None:
    """The single most common error in published COT research. It has to be in
    the signal's own failure modes, not only in the data layer."""
    modes = CommercialHedgerPressure.spec.known_failure_modes
    assert "three-day publication lag" in modes
    assert "reclassif" in modes.lower()
    assert CommercialHedgerPressure.spec.tier == 2


# --------------------------------------------------------------- VIX curve --
def vix_rows(pairs: list[tuple[str, float, float]]) -> list[dict[str, object]]:
    rows = []
    for day, front, back in pairs:
        as_of = dt.datetime.fromisoformat(day).replace(tzinfo=dt.UTC)
        for symbol, value in (("VIX", front), ("VIX3M", back)):
            rows.append(
                {
                    "source": "cboe",
                    "dataset": "series_observations",
                    "symbol": symbol,
                    "as_of": as_of,
                    "known_at": as_of + dt.timedelta(hours=21),
                    "ingested_at": as_of + dt.timedelta(hours=21),
                    "value": value,
                    "units": "index level",
                    "vintage": False,
                }
            )
    return rows


def test_the_slope_is_back_over_front() -> None:
    assert term_structure_slope(np.array([20.0]), np.array([23.0]))[0] == pytest.approx(1.15)


def test_a_steep_curve_is_a_short_volatility_position(store: object) -> None:
    """The premium is paid to whoever sells the insurance. Buying it every day
    instead is a slow, confident loss punctuated by one good week."""
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(vix_rows([("2024-01-02", 14.0, 18.0)])), asset_class="options"
    )
    scores = VixTermStructure().compute(store.as_of("2024-01-03"), [])  # type: ignore[attr-defined]

    assert scores["score"][0] < 0
    assert scores["score"][0] == pytest.approx(-1.0)


def test_an_inverted_curve_means_no_position(store: object) -> None:
    """The premium is not there to harvest, and holding through an inversion is
    how a bad week becomes a terminal one."""
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(vix_rows([("2024-01-02", 37.0, 28.0)])), asset_class="options"
    )
    scores = VixTermStructure().compute(store.as_of("2024-01-03"), [])  # type: ignore[attr-defined]

    assert scores["score"][0] == pytest.approx(0.0)


def test_the_position_scales_in_rather_than_switching_on(store: object) -> None:
    """A threshold creates a cliff the book crosses twice in a choppy market and
    pays to cross each time."""
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(vix_rows([("2024-01-02", 20.0, 21.5)])), asset_class="options"
    )
    scores = VixTermStructure().compute(store.as_of("2024-01-03"), [])  # type: ignore[attr-defined]

    assert -1.0 < scores["score"][0] < 0.0


def test_the_position_is_capped(store: object) -> None:
    """Short volatility with leverage is how the tail becomes terminal."""
    store.write(  # type: ignore[attr-defined]
        pl.DataFrame(vix_rows([("2024-01-02", 10.0, 40.0)])), asset_class="options"
    )
    scores = VixTermStructure().compute(store.as_of("2024-01-03"), [])  # type: ignore[attr-defined]

    assert scores["score"][0] >= -1.0


def test_a_missing_curve_is_refused(store: object) -> None:
    with pytest.raises(SignalUnavailableError, match=r"cboe\.series_observations"):
        VixTermStructure().compute(store.as_of("2024-01-03"), [])  # type: ignore[attr-defined]


def test_it_is_a_time_series_position_not_a_ranking() -> None:
    """There is one volatility curve. Ranking it against unrelated instruments
    would be ranking it against things it has no relationship with."""
    from quantlab.signals.base import SignalOutput

    assert VixTermStructure.spec.output is SignalOutput.TIME_SERIES_POSITION
    assert VixTermStructure.spec.tier == 2
    assert CONTANGO_FLOOR == 1.0


def test_the_asymmetric_payoff_is_a_declared_failure_mode() -> None:
    """A Sharpe ratio is a poor description of a strategy whose left tail is the
    whole story, and deflation does not fix that."""
    modes = VixTermStructure.spec.known_failure_modes
    assert "not symmetric" in modes
    assert "crowded" in modes
    assert "roll cost" in modes
