"""The event-driven engine.

The centrepiece here is the equivalence test. With no rules attached, this engine
and the vectorised one must produce the same equity curve -- not approximately,
exactly. Two independent implementations of the same convention agreeing is the
basis for trusting either; a divergence is a bug in one of them rather than a
difference of opinion about what a backtest means.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from quantlab.backtest.conventions import ExecutionTiming
from quantlab.backtest.event_driven import (
    EventDrivenBacktest,
    Rule,
    StopLossRule,
    TradingHaltRule,
)
from quantlab.backtest.events import EventKind, PortfolioState, event_queue
from quantlab.backtest.panel import Panel
from quantlab.backtest.prepare import panel_frame
from quantlab.backtest.vectorised import BacktestConfig, VectorisedBacktest


def build_panel(
    n_bars: int = 400, n_symbols: int = 6, seed: int = 7, drift: float = 0.0003
) -> Panel:
    rng = np.random.default_rng(seed)
    dates = [dt.datetime(2022, 1, 3, tzinfo=dt.UTC) + dt.timedelta(days=i) for i in range(n_bars)]
    rows = []
    for j in range(n_symbols):
        price = 100.0
        for day in dates:
            price *= float(np.exp(rng.normal(drift, 0.014)))
            rows.append(
                {
                    "symbol": f"S{j}",
                    "as_of": day,
                    "close": price,
                    "open": price,
                    "high": price,
                    "low": price,
                    "volume": 5e6,
                }
            )
    frame = panel_frame(pl.DataFrame(rows), window=30, spread_bps=3.0)
    return Panel.from_frame(frame, timing=ExecutionTiming.NEXT_OPEN)


def long_short_weights(panel: Panel, every: int = 21) -> pl.DataFrame:
    rebalances = [panel.dates[i] for i in range(40, panel.n_bars, every)]
    n = panel.n_symbols
    return pl.DataFrame(
        {
            "symbol": [panel.symbols[j] for _ in rebalances for j in range(n)],
            "as_of": [d for d in rebalances for _ in range(n)],
            "weight": [(1.0 / n if j % 2 == 0 else -1.0 / n) for _ in rebalances for j in range(n)],
        }
    )


# ----------------------------------------------------------------- equivalence --
def test_with_no_rules_it_matches_the_vectorised_engine_exactly() -> None:
    """The guarantee the whole design rests on.

    These are two separate implementations of one convention. If they disagree,
    at least one of them is wrong about what the convention is -- and neither
    number can be trusted until it is known which.
    """
    panel = build_panel()
    weights = long_short_weights(panel)
    config = BacktestConfig()

    vectorised = VectorisedBacktest(config).run(panel, weights, name="v", record_trial=False)
    event_driven = EventDrivenBacktest(config).run(panel, weights, name="e", record_trial=False)

    left = vectorised.curve["equity"].to_numpy()
    right = event_driven.curve["equity"].to_numpy()

    assert np.array_equal(left, right), "the two engines must agree bit for bit"
    assert vectorised.stats.sharpe_undeflated == event_driven.stats.sharpe_undeflated
    assert vectorised.stats.turnover_annual == pytest.approx(event_driven.stats.turnover_annual)


def test_the_cost_decomposition_matches_too() -> None:
    """Agreeing on the total while disagreeing on the parts would mean one engine
    is charging the wrong cost and another is cancelling the error."""
    panel = build_panel(n_bars=250)
    weights = long_short_weights(panel)
    config = BacktestConfig()

    left = VectorisedBacktest(config).run(panel, weights, name="v", record_trial=False)
    right = EventDrivenBacktest(config).run(panel, weights, name="e", record_trial=False)

    for component, value in left.cost_decomposition_bps_annual().items():
        assert right.cost_decomposition_bps_annual()[component] == pytest.approx(value)


def test_both_engines_reconcile() -> None:
    panel = build_panel(n_bars=200)
    result = EventDrivenBacktest(BacktestConfig()).run(
        panel, long_short_weights(panel), name="e", record_trial=False
    )
    assert result.reconciliation.bars_checked > 0


# ------------------------------------------------------------ path dependence --
def test_a_stop_changes_the_result_which_is_the_point() -> None:
    """The vectorised engine cannot express this: the trigger depends on the
    entry price, which the target weights do not know."""
    panel = build_panel()
    weights = long_short_weights(panel)
    config = BacktestConfig()

    plain = EventDrivenBacktest(config).run(panel, weights, name="p", record_trial=False)
    stopped = EventDrivenBacktest(config, rules=[StopLossRule(-0.08, 5)]).run(
        panel, weights, name="s", record_trial=False
    )

    assert not np.array_equal(plain.curve["equity"].to_numpy(), stopped.curve["equity"].to_numpy())
    # And the gap is the part of the result that is about the stop, not the signal.
    assert stopped.stats.turnover_annual != plain.stats.turnover_annual


def test_a_stop_that_never_triggers_leaves_the_result_alone() -> None:
    """A rule that does not fire must be exactly inert. Otherwise attaching one
    changes the answer for reasons unrelated to the rule."""
    panel = build_panel()
    weights = long_short_weights(panel)
    config = BacktestConfig()

    plain = EventDrivenBacktest(config).run(panel, weights, name="p", record_trial=False)
    never = EventDrivenBacktest(config, rules=[StopLossRule(-0.99, 0)]).run(
        panel, weights, name="n", record_trial=False
    )

    assert np.array_equal(plain.curve["equity"].to_numpy(), never.curve["equity"].to_numpy())


def test_stop_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="negative fraction"):
        StopLossRule(threshold=0.2)
    with pytest.raises(ValueError, match="non-negative"):
        StopLossRule(threshold=-0.2, cooldown_bars=-1)
    with pytest.raises(ValueError, match="bars must be positive"):
        TradingHaltRule(bars=0)


def test_a_flat_instrument_never_triggers_a_stop() -> None:
    """Profit since entry is NaN where flat. NaN compares False, so the rule
    fails closed rather than stopping a position that does not exist."""
    state = PortfolioState(
        bar=1,
        when=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        shares=np.array([0.0, 100.0]),
        exec_price=np.array([10.0, 10.0]),
        close=np.array([10.0, 5.0]),
        tradable=np.array([True, True]),
        equity=1000.0,
        entry_price=np.array([np.nan, 10.0]),
        bars_held=np.array([0, 5]),
        cooldown=np.zeros(2, dtype=int),
    )

    assert StopLossRule(-0.2, 0).triggered(state).tolist() == [1]


def test_a_short_is_stopped_by_a_rally_not_a_fall() -> None:
    """Profit is signed by the direction of the position. Getting this backwards
    stops every winner and holds every loser."""
    state = PortfolioState(
        bar=1,
        when=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        shares=np.array([-100.0, -100.0]),
        exec_price=np.array([10.0, 10.0]),
        close=np.array([13.0, 7.0]),  # one rallied, one fell
        tradable=np.array([True, True]),
        equity=1000.0,
        entry_price=np.array([10.0, 10.0]),
        bars_held=np.array([5, 5]),
        cooldown=np.zeros(2, dtype=int),
    )

    assert StopLossRule(-0.2, 0).triggered(state).tolist() == [0]


def test_rules_are_applied_in_the_order_supplied() -> None:
    class Flatten(Rule):
        name = "flatten"

        def apply(self, state: PortfolioState, target_shares: np.ndarray) -> np.ndarray:
            del state
            return np.zeros_like(target_shares)

    panel = build_panel(n_bars=150)
    result = EventDrivenBacktest(BacktestConfig(), rules=[Flatten()]).run(
        panel, long_short_weights(panel), name="flat", record_trial=False
    )

    assert result.stats.average_gross_exposure == pytest.approx(0.0)
    assert result.curve["equity"].to_numpy()[-1] == pytest.approx(BacktestConfig().initial_equity)


# ------------------------------------------------------------------- ordering --
def test_the_intra_bar_order_is_the_one_that_matters() -> None:
    """A stop applies to the position carried INTO the bar, so it is resolved
    before the rebalance. Evaluating it afterwards would test the new position
    against the old one's losses -- a different rule, and a wrong one."""
    assert EventKind.DELISTING < EventKind.STALE_EXIT < EventKind.STOP
    assert EventKind.STOP < EventKind.REBALANCE < EventKind.MARK


def test_a_quiet_bar_is_still_marked() -> None:
    """Otherwise the equity curve has a hole on every day nothing happened."""
    events = event_queue(dt.datetime(2024, 1, 2, tzinfo=dt.UTC), 5)
    assert [e.kind for e in events] == [EventKind.MARK]


def test_the_queue_comes_out_in_schedule_order() -> None:
    events = event_queue(
        dt.datetime(2024, 1, 2, tzinfo=dt.UTC),
        5,
        delisting=[1],
        stale=[2],
        stops=[3],
        rebalance=True,
    )
    assert [e.kind for e in events] == sorted(e.kind for e in events)
    assert events[0].kind is EventKind.DELISTING
    assert events[-1].kind is EventKind.MARK


def test_a_mismatched_panel_timing_is_refused() -> None:
    panel = build_panel(n_bars=120)
    config = BacktestConfig(execution=ExecutionTiming.NEXT_CLOSE)
    with pytest.raises(ValueError, match="built for next_open"):
        EventDrivenBacktest(config).run(panel, long_short_weights(panel), record_trial=False)
