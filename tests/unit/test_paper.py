"""Paper trading: the broker seam, the loop, and the decay monitor.

The spec puts a hard boundary here -- simulate execution, leave a clean
interface where a live adapter would attach, route nothing. These tests hold
that line and the three places where a paper book quietly stops being evidence
about anything: filling for free, running a cycle twice, and annualising a
fortnightly Sharpe as though it were daily.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest

from quantlab.paper.broker import Broker, PaperBroker, TargetOrder
from quantlab.paper.decay import (
    DEFAULT_HAIRCUT,
    MIN_OBSERVATIONS,
    assess_decay,
    sharpe_standard_error,
)
from quantlab.paper.loop import scores_to_weights
from quantlab.paper.state import CycleRecord, PaperState, PositionRecord

WHEN = dt.datetime(2026, 9, 18, tzinfo=dt.UTC)


def order(symbol: str, shares: float, price: float = 100.0, **kwargs: float) -> TargetOrder:
    return TargetOrder(
        symbol=symbol,
        target_shares=shares,
        reference_price=price,
        adv_notional=kwargs.get("adv", 50e6),
        volatility_daily=kwargs.get("vol", 0.02),
        spread_bps=kwargs.get("spread", 3.0),
    )


# ------------------------------------------------------------------ the seam --
def test_the_platform_ships_exactly_one_broker() -> None:
    """Spec section 1: research and paper trading only, with a clean interface
    where a live adapter would attach. A second implementation appearing here is
    a scope change, not a refactor."""
    assert Broker.__abstractmethods__ == frozenset({"positions", "equity", "submit"})
    assert issubclass(PaperBroker, Broker)
    assert [c.__name__ for c in Broker.__subclasses__()] == ["PaperBroker"]


def test_a_fill_costs_what_the_backtest_would_charge() -> None:
    """Not approximately -- the same cost model object. A paper book that filled
    at mid would beat its own backtest for no reason, and the difference would
    look like the strategy working."""
    broker = PaperBroker(cash=1_000_000.0)
    fills = broker.submit([order("AAA", 1000.0)], WHEN)

    assert len(fills) == 1
    assert fills[0].cost > 0
    assert fills[0].fill_price > fills[0].reference_price, "a buy fills above the mid"
    assert broker.cash < 1_000_000.0 - 1000.0 * 100.0


def test_a_sell_fills_below_the_mid() -> None:
    broker = PaperBroker(cash=1_000_000.0, shares={"AAA": 1000.0})
    fills = broker.submit([order("AAA", 0.0)], WHEN)

    assert fills[0].shares == -1000.0
    assert fills[0].fill_price < fills[0].reference_price


def test_an_order_with_no_liquidity_statistics_is_refused() -> None:
    """Trading it at zero cost is how a paper loop stops being evidence."""
    broker = PaperBroker(cash=1_000_000.0)
    bare = TargetOrder(symbol="AAA", target_shares=100.0, reference_price=100.0)

    with pytest.raises(ValueError, match="cannot be costed"):
        broker.submit([bare], WHEN)


def test_an_instrument_with_no_price_cannot_be_filled() -> None:
    """Filling at zero would create a free position out of missing data."""
    broker = PaperBroker(cash=1_000_000.0)
    with pytest.raises(ValueError, match="free position"):
        broker.submit([order("AAA", 100.0, price=0.0)], WHEN)


def test_reaching_a_target_already_held_trades_nothing() -> None:
    broker = PaperBroker(cash=1_000_000.0, shares={"AAA": 500.0})
    assert broker.submit([order("AAA", 500.0)], WHEN) == []
    assert broker.cash == 1_000_000.0


def test_slippage_is_recorded_per_fill_not_only_in_aggregate() -> None:
    """A mean that looks fine can hide a handful of orders far too large for the
    instrument, and the distribution is the diagnostic."""
    broker = PaperBroker(cash=100_000_000.0)
    small = broker.submit([order("AAA", 100.0)], WHEN)[0]
    broker.shares.clear()
    huge = broker.submit([order("BBB", 400_000.0)], WHEN)[0]

    assert huge.slippage_bps > small.slippage_bps, "impact is concave, not linear"


def test_equity_marks_the_book_at_supplied_prices() -> None:
    broker = PaperBroker(cash=50_000.0, shares={"AAA": 100.0, "BBB": -50.0})
    assert broker.equity({"AAA": 10.0, "BBB": 20.0}) == pytest.approx(50_000.0 + 1000.0 - 1000.0)


# ------------------------------------------------------------------ weights --
def test_scores_become_a_dollar_neutral_book() -> None:
    import polars as pl

    scores = pl.DataFrame({"symbol": ["A", "B", "C", "D"], "score": [2.0, 1.0, -1.0, -2.0]})
    weights = scores_to_weights(scores, gross=1.0)

    assert sum(weights.values()) == pytest.approx(0.0, abs=1e-12)
    assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0)
    assert weights["A"] > 0 > weights["D"]


def test_a_signal_with_no_dispersion_produces_no_position() -> None:
    """If every instrument scores the same the signal has no view, and
    manufacturing one out of rounding error is not a strategy."""
    import polars as pl

    flat = pl.DataFrame({"symbol": ["A", "B", "C"], "score": [1.0, 1.0, 1.0]})
    assert scores_to_weights(flat) == {}
    assert scores_to_weights(pl.DataFrame({"symbol": [], "score": []})) == {}


def test_non_finite_scores_do_not_poison_the_book() -> None:
    import polars as pl

    scores = pl.DataFrame({"symbol": ["A", "B", "C"], "score": [1.0, float("nan"), -1.0]})
    weights = scores_to_weights(scores)

    assert all(math.isfinite(w) for w in weights.values())
    assert "B" not in weights


# -------------------------------------------------------------------- state --
def record(as_of: str, equity: float) -> CycleRecord:
    return CycleRecord(
        strategy="demo",
        as_of=as_of,
        ran_at=as_of,
        equity=equity,
        cash=equity,
        gross_exposure=1.0,
        net_exposure=0.0,
        positions=[PositionRecord("AAA", 1.0, 100.0, 100.0, 0.1)],
    )


def test_state_is_append_only(tmp_path: Path) -> None:
    """The history of what the book thought at each point is the only way to
    tell later whether a decision was bad or merely unlucky."""
    state = PaperState(tmp_path / "demo.jsonl", "demo")
    state.append(record("2026-01-01T00:00:00+00:00", 100.0))
    state.append(record("2026-01-02T00:00:00+00:00", 101.0))

    assert len(state.history()) == 2
    assert state.latest()["equity"] == 101.0


def test_a_repeated_cycle_is_detectable(tmp_path: Path) -> None:
    """Running twice for one date doubles the trades and puts a step in the
    equity curve that nothing explains."""
    state = PaperState(tmp_path / "demo.jsonl", "demo")
    moment = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    assert not state.has_run_for(moment)

    state.append(record(moment.isoformat(), 100.0))
    assert state.has_run_for(moment)


def test_the_observation_frequency_is_inferred_not_assumed(tmp_path: Path) -> None:
    """A fortnightly book annualised by the square root of 252 reports a Sharpe
    three times too large, which turns 'too early to tell' into a number someone
    acts on."""
    state = PaperState(tmp_path / "demo.jsonl", "demo")
    for week in range(10):
        day = dt.datetime(2026, 1, 5, tzinfo=dt.UTC) + dt.timedelta(weeks=2 * week)
        state.append(record(day.isoformat(), 100.0 + week))

    assert 20 <= state.periods_per_year() <= 30


def test_a_daily_book_infers_a_daily_frequency(tmp_path: Path) -> None:
    state = PaperState(tmp_path / "demo.jsonl", "demo")
    for day in range(10):
        moment = dt.datetime(2026, 1, 5, tzinfo=dt.UTC) + dt.timedelta(days=day)
        state.append(record(moment.isoformat(), 100.0 + day))

    assert state.periods_per_year() > 250


def test_a_book_with_no_history_is_empty_not_an_error(tmp_path: Path) -> None:
    state = PaperState(tmp_path / "missing.jsonl", "demo")
    assert state.history() == []
    assert state.latest() is None


def test_the_filename_is_sanitised(tmp_path: Path) -> None:
    state = PaperState.for_strategy(tmp_path, "trend/../../etc/passwd")
    assert state.path.parent == tmp_path / "paper"
    assert "/" not in state.path.name


# -------------------------------------------------------------------- decay --
def test_too_few_observations_concludes_nothing() -> None:
    """The question people skip, and the one that usually has an answer."""
    rng = np.random.default_rng(0)
    report = assess_decay(rng.normal(0.0004, 0.01, 30), strategy="demo", backtest_sharpe=1.2)

    assert not report.is_conclusive
    assert not report.has_decayed
    assert "too few to conclude" in report.verdict()


def test_the_comparison_is_against_the_haircut_not_the_backtest() -> None:
    """A strategy delivering half its backtest Sharpe is doing exactly what the
    platform's prior predicts. Comparing against the printed number would
    declare decay on every strategy that behaved as expected."""
    rng = np.random.default_rng(1)
    # True Sharpe of about 0.6 -- half a backtest of 1.2.
    daily = 0.6 / math.sqrt(252)
    returns = rng.normal(daily * 0.01, 0.01, 2000)

    report = assess_decay(returns, strategy="demo", backtest_sharpe=1.2)

    assert report.expected_sharpe == pytest.approx(1.2 * DEFAULT_HAIRCUT)
    assert not report.has_decayed, "performing as predicted is not decay"


def test_a_genuine_collapse_is_detected() -> None:
    rng = np.random.default_rng(2)
    returns = rng.normal(-0.0008, 0.01, 1500)  # comfortably negative

    report = assess_decay(returns, strategy="demo", backtest_sharpe=1.2)

    assert report.has_decayed
    assert "has decayed" in report.verdict()
    assert report.t_statistic < -2.0


def test_outperformance_is_reported_with_suspicion() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0025, 0.01, 1500)

    report = assess_decay(returns, strategy="demo", backtest_sharpe=1.2)

    assert report.is_conclusive
    assert not report.has_decayed
    assert "same suspicion as a good backtest" in report.verdict()


def test_the_sample_needed_is_reported_when_nothing_is_conclusive() -> None:
    """Saying 'no conclusion' without saying what would settle it leaves the
    reader with the number they already had."""
    rng = np.random.default_rng(4)
    report = assess_decay(rng.normal(0.0002, 0.01, 200), strategy="demo", backtest_sharpe=1.2)

    if not report.is_conclusive:
        assert report.observations_needed > report.observations
        assert "would be" in report.verdict() or "too few" in report.verdict()


def test_the_standard_error_widens_with_the_sharpe() -> None:
    """A higher Sharpe is harder to pin down, not easier."""
    assert sharpe_standard_error(2.0, 252) > sharpe_standard_error(0.0, 252)
    assert sharpe_standard_error(1.0, 252) < sharpe_standard_error(1.0, 60)


def test_a_standard_error_needs_a_sample() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sharpe_standard_error(1.0, 1)
    with pytest.raises(ValueError, match="at least two live"):
        assess_decay(np.array([0.01]), strategy="demo", backtest_sharpe=1.0)


def test_the_minimum_sample_is_a_quarter() -> None:
    """Below it the standard error is so wide any verdict would be noise."""
    assert MIN_OBSERVATIONS == 60
