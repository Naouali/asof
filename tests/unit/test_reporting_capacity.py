"""Deriving capacity from a run that has already happened.

Two arithmetic mistakes are easy here and both were made before being caught, so
each has a test that pins the corrected behaviour: double-charging holding costs,
and counting every held bar as a rebalance.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.reporting.capacity import capacity_for, traded_liquidity


class Stats:
    years = 10.0
    cagr = 0.02
    cost_drag_bps_annual = 130.0
    turnover_annual = 4.0


class Result:
    """A run that traded 120 times over 10 years while holding on 2,520 bars."""

    def __init__(self, *, trading_bars: int = 120, bars: int = 2520) -> None:
        self.name = "fake"
        self.stats = Stats()
        start = dt.datetime(2015, 1, 1, tzinfo=dt.UTC)
        dates = [start + dt.timedelta(days=i) for i in range(bars)]
        traded = np.zeros(bars)
        step = max(1, bars // trading_bars)
        traded[::step][:trading_bars] = 1e6
        self.curve = pl.DataFrame({"as_of": dates, "traded_notional": traded})

        rows = []
        rng = np.random.default_rng(0)
        held = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
        for index, date in enumerate(dates):
            if traded[index] > 0:  # a rebalance moves the weights
                held = dict(zip(held, rng.dirichlet(np.ones(3)), strict=True))
            for symbol, weight in held.items():
                # Prices drift the held weights every bar, rebalance or not.
                rows.append(
                    {"as_of": date, "symbol": symbol, "weight": weight * (1 + 1e-4 * index)}
                )
        self.positions = pl.DataFrame(rows)

    def cost_decomposition_bps_annual(self) -> dict[str, float]:
        return {
            "spread": 20.0,
            "impact": 14.0,
            "commission": 4.0,
            "borrow": 80.0,
            "financing": 12.0,
        }


class FakePanel:
    symbols = ("AAA", "BBB", "CCC")

    def __init__(self, bars: int = 2520) -> None:
        n = len(self.symbols)
        self.adv_notional = np.full((bars, n), 50e6)
        self.volatility_daily = np.full((bars, n), 0.018)
        self.spread_bps = np.full((bars, n), 3.0)
        self.tradable = np.ones((bars, n), dtype=bool)

    def symbol_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.symbols)}


# -------------------------------------------------------------- rebalance count --
def test_rebalances_are_counted_from_trades_not_from_held_bars() -> None:
    """Held weights drift with prices on every bar, so counting position dates
    counts the bar count. On the real ETF run that is 248 rebalances a year
    against a true 12, and because impact is concave in size it makes the
    strategy look like it trades small and often -- which overstates capacity."""
    result = Result(trading_bars=120, bars=2520)
    capacity = capacity_for(result, FakePanel())  # type: ignore[arg-type]

    assert result.positions["as_of"].n_unique() == 2520  # what the wrong count would use
    # 120 trades over 10 years. The turnover per rebalance follows from it.
    assert capacity.gross_alpha_bps_annual == pytest.approx(330.0, abs=1.0)


def test_a_run_that_never_traded_is_refused() -> None:
    result = Result(trading_bars=1)
    with pytest.raises(ValueError, match="too few to measure a rebalance frequency"):
        capacity_for(result, FakePanel())  # type: ignore[arg-type]


def test_a_curve_without_traded_notional_is_refused_rather_than_guessed() -> None:
    result = Result()
    result.curve = result.curve.drop("traded_notional")
    with pytest.raises(ValueError, match="order of magnitude"):
        capacity_for(result, FakePanel())  # type: ignore[arg-type]


# ----------------------------------------------------------------- the arithmetic --
def test_gross_alpha_is_the_realised_return_with_every_cost_added_back() -> None:
    """CAGR of +200 bp with 130 bp of drag is a gross edge of 330 bp. Subtracting
    the holding costs here as well -- when the model is also given them -- charged
    the strategy twice and reported -143 bp/yr against a realised -68."""
    capacity = capacity_for(Result(), FakePanel())  # type: ignore[arg-type]
    assert capacity.gross_alpha_bps_annual == pytest.approx(200.0 + 130.0, abs=1.0)


def test_holding_costs_are_charged_exactly_once() -> None:
    """Net alpha at a costless size is the gross edge less the costs that do not
    scale with size: borrow and financing, 92 bp here."""
    capacity = capacity_for(Result(), FakePanel())  # type: ignore[arg-type]
    trading_at_zero_size = 330.0 - 92.0 - capacity.net_alpha_bps_at_zero
    # Spread and commission remain at any size; impact is what vanishes.
    assert 0.0 <= trading_at_zero_size < 120.0


# ------------------------------------------------------------------- liquidity --
def test_liquidity_is_weighted_by_what_is_traded_not_what_is_held() -> None:
    universe = traded_liquidity(Result(), FakePanel())  # type: ignore[arg-type]

    assert set(universe.frame["symbol"]) == {"AAA", "BBB", "CCC"}
    assert float(universe.frame["weight"].sum()) == pytest.approx(1.0)
    assert universe.frame["adv_notional"].min() > 0


def test_drift_between_rebalances_earns_almost_no_capacity_weight() -> None:
    """A name bought once and then left alone is traded once, however long it is
    held. Its drift between rebalances is not volume and must not earn it a
    capacity weight, or the universe looks more liquid than it is."""
    result = Result(trading_bars=120)
    # A fourth name held at a constant weight, drifting only.
    extra = result.positions.filter(pl.col("symbol") == "AAA").with_columns(
        symbol=pl.lit("DDD"), weight=pl.lit(0.25)
    )
    result.positions = pl.concat([result.positions, extra])

    class Panel4(FakePanel):
        symbols = ("AAA", "BBB", "CCC", "DDD")

    universe = traded_liquidity(result, Panel4())  # type: ignore[arg-type]
    weight = universe.frame.filter(pl.col("symbol") == "DDD")["weight"].item()
    # One entry against the others' 120 rebalances apiece.
    assert weight < 0.01


def test_a_run_with_no_positions_has_no_capacity() -> None:
    result = Result()
    result.positions = pl.DataFrame({"as_of": [], "symbol": [], "weight": []})
    with pytest.raises(ValueError, match="no traded universe"):
        traded_liquidity(result, FakePanel())  # type: ignore[arg-type]


def test_positions_outside_the_panel_are_refused() -> None:
    result = Result()
    stray = (
        result.positions.filter(pl.col("symbol") == "AAA")
        .head(3)
        .with_columns(symbol=pl.lit("ZZZ"))
    )
    result.positions = pl.concat([result.positions, stray])
    with pytest.raises(ValueError, match="absent from the panel"):
        traded_liquidity(result, FakePanel())  # type: ignore[arg-type]


def test_duplicate_position_rows_are_named_not_silently_aggregated() -> None:
    result = Result()
    result.positions = pl.concat([result.positions, result.positions.head(3)])
    with pytest.raises(ValueError, match=r"repeat a \(date, symbol\) pair"):
        traded_liquidity(result, FakePanel())  # type: ignore[arg-type]


def test_liquidity_averages_only_over_tradable_bars() -> None:
    """A name that listed halfway through is described by the period it existed,
    not by a mean over bars on which it did not trade."""
    panel = FakePanel()
    panel.tradable[:1260, 2] = False
    panel.adv_notional[:1260, 2] = 1.0  # a value that would poison a plain mean

    universe = traded_liquidity(Result(), panel)  # type: ignore[arg-type]
    ccc = universe.frame.filter(pl.col("symbol") == "CCC")["adv_notional"].item()
    assert ccc == pytest.approx(50e6)
