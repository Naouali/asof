"""The engine: conventions honoured, costs charged, and cheating made visible."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.backtest import (
    BacktestConfig,
    ExecutionTiming,
    Panel,
    StalenessPolicy,
    VectorisedBacktest,
)
from quantlab.costs.impact import ImpactParams, SquareRootImpact
from quantlab.costs.model import TransactionCostModel

START = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
# Panels in this module default to NEXT_CLOSE, so the configs must agree -- the
# engine refuses a mismatch rather than reporting a convention it did not use.
FREE = BacktestConfig(
    execution=ExecutionTiming.NEXT_CLOSE, borrow_bps_annual=0.0, financing_bps_annual=0.0
)


def panel_from(
    closes: dict[str, list[float]],
    *,
    opens: dict[str, list[float]] | None = None,
    spread_bps: float = 0.0,
    adv: float = 1e12,
    timing: ExecutionTiming = ExecutionTiming.NEXT_CLOSE,
    staleness: StalenessPolicy | None = None,
) -> Panel:
    out = []
    for symbol, series in closes.items():
        for bar, price in enumerate(series):
            if price != price:  # NaN means "no observation"
                continue
            out.append(
                {
                    "symbol": symbol,
                    "as_of": START + dt.timedelta(days=bar),
                    "close": price,
                    "open": (opens or {}).get(symbol, series)[bar],
                    "adv_notional": adv,
                    "volatility_daily": 0.02,
                    "spread_bps": spread_bps,
                }
            )
    return Panel.from_frame(pl.DataFrame(out), timing=timing, staleness=staleness)


def weights_at(bars: list[int], allocation: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"symbol": s, "as_of": START + dt.timedelta(days=b), "weight": w}
            for b in bars
            for s, w in allocation.items()
        ]
    )


def costless() -> TransactionCostModel:
    return TransactionCostModel(
        impact=SquareRootImpact(ImpactParams(y=1e-12)), use_asset_class_defaults=False
    )


# ----------------------------------------------------------------------------------
# Does it compute the right return?
# ----------------------------------------------------------------------------------
def test_fully_invested_position_earns_the_instrument_return() -> None:
    """A 10% price move on a 100% allocation is a 10% portfolio return. If this
    fails, nothing else in the engine matters."""
    panel = panel_from({"A": [100.0, 100.0, 110.0]})
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 1.0}))
    assert result.stats.total_return == pytest.approx(0.10, abs=1e-9)


def test_half_weight_earns_half_the_move() -> None:
    panel = panel_from({"A": [100.0, 100.0, 120.0]})
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 0.5}))
    assert result.stats.total_return == pytest.approx(0.10, abs=1e-9)


def test_a_short_earns_the_negative_of_the_move() -> None:
    panel = panel_from({"A": [100.0, 100.0, 90.0]})
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": -1.0}))
    assert result.stats.total_return == pytest.approx(0.10, abs=1e-9)


def test_uninvested_capital_earns_nothing() -> None:
    """Cash pays no interest here. Assuming it did would hand every low-exposure
    strategy a free return stream it never earned."""
    panel = panel_from({"A": [100.0, 150.0, 200.0]})
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 0.0}))
    assert result.stats.total_return == pytest.approx(0.0, abs=1e-12)


def test_weights_drift_between_rebalances() -> None:
    """Shares are held, not weights. Holding weights constant would imply trading
    every day for free -- and the difference is exactly the turnover this platform
    exists to charge for."""
    panel = panel_from({"A": [100.0, 100.0, 200.0, 200.0], "B": [100.0, 100.0, 100.0, 100.0]})
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 0.5, "B": 0.5}))
    final = result.positions.filter(pl.col("as_of") == START + dt.timedelta(days=3))
    weights = dict(zip(final["symbol"], final["weight"], strict=True))
    assert weights["A"] == pytest.approx(2 / 3, abs=1e-6), "the winner grew to two thirds"
    assert result.curve["traded_notional"].sum() > 0  # the initial buy only


# ----------------------------------------------------------------------------------
# Timing conventions
# ----------------------------------------------------------------------------------
def test_signal_is_traded_one_bar_after_it_is_computed() -> None:
    """A signal from the close of T cannot be traded at that same close. The bar
    the weight is dated on is the bar it is *computed* on."""
    panel = panel_from({"A": [100.0, 100.0, 100.0]})
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 1.0}))
    traded = result.curve["traded_notional"].to_list()
    assert traded[0] == 0.0, "nothing trades on bar 0"
    assert traded[1] > 0.0, "the bar-0 signal is executed on bar 1"


def test_open_and_close_execution_differ_by_exactly_the_intraday_move() -> None:
    """Entering at the open of T+1 captures that session's move; entering at its
    close captures none of it. Which is better depends on the path -- the point is
    that the choice is economically real and must be stated, not defaulted into.
    """
    closes = {"A": [100.0, 120.0, 120.0]}
    opens = {"A": [100.0, 115.0, 120.0]}
    at_open = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.NEXT_OPEN, borrow_bps_annual=0.0, financing_bps_annual=0.0
        ),
        costless(),
    ).run(
        panel_from(closes, opens=opens, timing=ExecutionTiming.NEXT_OPEN),
        weights_at([0], {"A": 1.0}),
    )
    at_close = VectorisedBacktest(FREE, costless()).run(
        panel_from(closes, opens=opens, timing=ExecutionTiming.NEXT_CLOSE),
        weights_at([0], {"A": 1.0}),
    )
    # Buying at 115 and marking at 120 earns the intraday move; buying at the
    # close of the same bar earns nothing over the remaining flat bar.
    assert at_open.stats.total_return == pytest.approx(120 / 115 - 1, rel=1e-6)
    assert at_close.stats.total_return == pytest.approx(0.0, abs=1e-9)


def test_look_ahead_execution_is_refused_without_an_acknowledgement() -> None:
    """Trading at the close that produced the signal is not a convention, it is a
    bug. It exists so the red-team suite can test a strategy that cheats."""
    with pytest.raises(ValueError, match="look-ahead bias, not a convention"):
        BacktestConfig(execution=ExecutionTiming.SAME_CLOSE)


def test_acknowledged_look_ahead_stamps_the_result_as_contaminated() -> None:
    config = BacktestConfig(
        execution=ExecutionTiming.SAME_CLOSE,
        acknowledge_look_ahead=True,
        borrow_bps_annual=0.0,
        financing_bps_annual=0.0,
    )
    panel = panel_from({"A": [100.0, 100.0, 130.0]}, timing=ExecutionTiming.SAME_CLOSE)
    result = VectorisedBacktest(config, costless()).run(panel, weights_at([1], {"A": 1.0}))
    assert result.contaminated
    assert "CONTAMINATED" in result.summary()
    assert config.signal_lag_bars == 0


def test_look_ahead_beats_the_honest_convention() -> None:
    """The whole point of keeping the cheating mode: a strategy that can see the
    bar it trades on must outperform, and by a visible margin."""
    closes = {"A": [100.0, 100.0, 140.0, 140.0]}
    weights = weights_at([1], {"A": 1.0})
    honest = VectorisedBacktest(FREE, costless()).run(panel_from(closes), weights)
    cheating = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.SAME_CLOSE,
            acknowledge_look_ahead=True,
            borrow_bps_annual=0.0,
            financing_bps_annual=0.0,
        ),
        costless(),
    ).run(panel_from(closes, timing=ExecutionTiming.SAME_CLOSE), weights)
    assert cheating.stats.total_return > honest.stats.total_return


def test_a_positive_delisting_return_is_rejected() -> None:
    with pytest.raises(ValueError, match="should be negative"):
        BacktestConfig(delisting_return=0.10)


# ----------------------------------------------------------------------------------
# Costs actually bite
# ----------------------------------------------------------------------------------
def test_spread_is_charged_on_every_trade() -> None:
    flat = panel_from({"A": [100.0] * 6}, spread_bps=20.0)
    result = VectorisedBacktest(FREE).run(flat, weights_at([0, 2, 4], {"A": 1.0}))
    assert result.stats.total_return < 0, "trading a flat market must lose money"
    assert result.curve["spread_cost"].sum() > 0


def test_impact_grows_with_size_relative_to_volume() -> None:
    weights = weights_at([0, 2], {"A": 1.0})
    deep = VectorisedBacktest(FREE).run(panel_from({"A": [100.0] * 5}, adv=1e12), weights)
    thin = VectorisedBacktest(FREE).run(panel_from({"A": [100.0] * 5}, adv=1e7), weights)
    assert thin.curve["impact_cost"].sum() > 10 * deep.curve["impact_cost"].sum()


def test_borrow_is_charged_on_shorts_only() -> None:
    panel = panel_from({"A": [100.0] * 30})
    long_only = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE, financing_bps_annual=0.0), costless()
    ).run(panel, weights_at([0], {"A": 1.0}))
    short_only = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE, financing_bps_annual=0.0), costless()
    ).run(panel, weights_at([0], {"A": -1.0}))
    assert long_only.curve["borrow_cost"].sum() == 0.0
    assert short_only.curve["borrow_cost"].sum() > 0.0


def test_financing_is_charged_on_leverage_only() -> None:
    panel = panel_from({"A": [100.0] * 30})
    unlevered = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE, borrow_bps_annual=0.0), costless()
    ).run(panel, weights_at([0], {"A": 1.0}))
    levered = VectorisedBacktest(
        BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE, borrow_bps_annual=0.0), costless()
    ).run(panel, weights_at([0], {"A": 3.0}))
    assert unlevered.curve["financing_cost"].sum() == pytest.approx(0.0, abs=1e-6)
    assert levered.curve["financing_cost"].sum() > 0.0


def test_holding_costs_accrue_over_calendar_days_not_trading_days() -> None:
    """Borrow accrues over weekends. Charging 252 days a year instead of 365
    understates financing by nearly a third."""
    dates = [START, START + dt.timedelta(days=1), START + dt.timedelta(days=31)]
    frame = pl.DataFrame(
        [
            {
                "symbol": "A",
                "as_of": d,
                "close": 100.0,
                "open": 100.0,
                "adv_notional": 1e12,
                "volatility_daily": 0.02,
                "spread_bps": 0.0,
            }
            for d in dates
        ]
    )
    panel = Panel.from_frame(frame, timing=ExecutionTiming.NEXT_CLOSE)
    weights = pl.DataFrame([{"symbol": "A", "as_of": dates[0], "weight": -1.0}])
    result = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.NEXT_CLOSE, borrow_bps_annual=100.0, financing_bps_annual=0.0
        ),
        costless(),
    ).run(panel, weights)
    # 30 calendar days of a 100 bp/year borrow on the whole book. A trading-day
    # convention would charge 21 days here and understate it by nearly a third.
    assert result.curve["borrow_cost"].sum() == pytest.approx(
        10_000_000 * 0.01 * 30 / 365, rel=0.02
    )


def test_gross_and_net_sharpe_differ_by_exactly_the_costs() -> None:
    panel = panel_from({"A": [100.0 * (1.01**i) for i in range(60)]}, spread_bps=10.0)
    result = VectorisedBacktest(FREE).run(panel, weights_at(list(range(0, 60, 5)), {"A": 1.0}))
    assert result.stats.sharpe_gross_undeflated > result.stats.sharpe_undeflated
    assert result.stats.cost_drag_bps_annual > 0


# ----------------------------------------------------------------------------------
# Delisting and staleness
# ----------------------------------------------------------------------------------
def test_a_delisting_applies_its_return_and_closes_the_position() -> None:
    """Treating the last observed price as a final settlement assumes the holder
    got out whole, which for a performance-related delisting is badly wrong."""
    closes = {"A": [100.0, 100.0, 100.0, float("nan"), float("nan")], "B": [100.0] * 5}
    panel = panel_from(closes, staleness=StalenessPolicy.strict())
    result = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.NEXT_CLOSE,
            delisting_return=-0.50,
            borrow_bps_annual=0.0,
            financing_bps_annual=0.0,
        ),
        costless(),
    ).run(panel, weights_at([0], {"A": 1.0, "B": 0.0}))

    assert result.data_quality["delisted_symbols"] == 1
    assert result.stats.total_return == pytest.approx(-0.50, abs=1e-6)
    final = result.positions.filter(pl.col("as_of") == START + dt.timedelta(days=4))
    assert final.height == 0, "nothing is held after the delisting"


def test_delisting_size_changes_the_answer() -> None:
    closes = {"A": [100.0, 100.0, 100.0, float("nan")], "B": [100.0] * 4}
    weights = weights_at([0], {"A": 1.0, "B": 0.0})
    mild = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.NEXT_CLOSE,
            delisting_return=-0.10,
            borrow_bps_annual=0.0,
            financing_bps_annual=0.0,
        ),
        costless(),
    ).run(panel_from(closes, staleness=StalenessPolicy.strict()), weights)
    severe = VectorisedBacktest(
        BacktestConfig(
            execution=ExecutionTiming.NEXT_CLOSE,
            delisting_return=-0.80,
            borrow_bps_annual=0.0,
            financing_bps_annual=0.0,
        ),
        costless(),
    ).run(panel_from(closes, staleness=StalenessPolicy.strict()), weights)
    assert severe.stats.total_return < mild.stats.total_return


def test_a_stale_instrument_is_liquidated_not_held_at_a_frozen_price() -> None:
    closes = {"A": [100.0, 100.0] + [float("nan")] * 6 + [100.0], "B": [100.0] * 9}
    panel = panel_from(closes, staleness=StalenessPolicy(max_bars=2))
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 1.0, "B": 0.0}))
    held = result.positions.filter(
        (pl.col("symbol") == "A") & (pl.col("as_of") == START + dt.timedelta(days=6))
    )
    assert held.height == 0


# ----------------------------------------------------------------------------------
# Result shape
# ----------------------------------------------------------------------------------
def test_books_balance_across_a_realistic_run() -> None:
    rng = np.random.default_rng(7)
    closes = {f"S{i}": list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))) for i in range(6)}
    panel = panel_from(closes, spread_bps=5.0, adv=5e7)
    weights = weights_at(list(range(0, 120, 10)), {f"S{i}": 1 / 6 for i in range(6)})
    result = VectorisedBacktest(BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE)).run(
        panel, weights
    )
    assert result.reconciliation.bars_checked == 119
    assert abs(result.reconciliation.worst_absolute_gap) < 0.01


def test_summary_never_reports_a_bare_sharpe() -> None:
    """Spec section 13: no Sharpe without its deflated counterpart and trial count.

    Enforced two ways. The raw statistic is named ``sharpe_undeflated`` so it
    cannot be misread, and the summary carries the deflated value and the trial
    count that produced it.
    """
    panel = panel_from({"A": [100.0 * (1.001**i) for i in range(50)]})
    result = VectorisedBacktest(FREE, costless()).run(
        panel, weights_at([0], {"A": 1.0}), family="summary-check"
    )
    summary = result.summary()
    assert "deflated" in summary
    assert "trial(s)" in summary
    assert not hasattr(result.stats, "sharpe")
    assert hasattr(result.stats, "sharpe_undeflated")
    assert result.evidence is not None


def test_cost_decomposition_is_reported_component_by_component() -> None:
    panel = panel_from({"A": [100.0] * 40}, spread_bps=8.0, adv=1e8)
    result = VectorisedBacktest(BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE)).run(
        panel, weights_at([0, 10, 20], {"A": -1.0})
    )
    decomposition = result.cost_decomposition_bps_annual()
    assert set(decomposition) == {"spread", "impact", "commission", "borrow", "financing"}
    assert decomposition["spread"] > 0
    assert decomposition["borrow"] > 0


def test_data_quality_facts_travel_with_the_result() -> None:
    # Two symbols, so the bar containing A's gap still exists in the panel's
    # calendar. With one symbol a missing bar is simply not a bar at all.
    panel = panel_from(
        {"A": [100.0, 100.0, float("nan"), 100.0], "B": [100.0] * 4},
        staleness=StalenessPolicy(max_bars=3),
    )
    result = VectorisedBacktest(FREE, costless()).run(panel, weights_at([0], {"A": 1.0}))
    quality = result.data_quality
    assert quality["forward_filled_cells"] == 1
    assert quality["execution"] == "next_close"
    assert quality["signal_lag_bars"] == 1
    assert "forward-filled" in quality["staleness_policy"]
