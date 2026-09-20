"""The accounting identity, including under arbitrary inputs.

Spec section 11 asks for property-based tests where the accounting identities hold
for arbitrary input sequences. That is the right shape of test for this module: the
identity is not a behaviour to be sampled at a few points, it is an invariant, and
a hand-picked example can pass while the general case fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quantlab.backtest.accounting import CENT, AccountingError, Ledger

# Bounded so the arithmetic stays in a range where float64 can carry a cent.
# Unbounded floats would test numpy's precision limits, not QuantLab's accounting.
prices = st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
shares = st.floats(min_value=-10_000.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
costs = st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)


def price_vector(n: int, values: list[float]) -> np.ndarray:
    return np.array(values[:n], dtype=np.float64)


# ----------------------------------------------------------------------------------
# The invariant
# ----------------------------------------------------------------------------------
@given(
    n=st.integers(min_value=1, max_value=6),
    data=st.data(),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_books_balance_for_arbitrary_sequences(n: int, data: st.DataObject) -> None:
    """Balance sheet and income statement agree, whatever is thrown at them.

    ``Ledger.settle`` reconciles internally and raises on a breach, so a run that
    completes is itself the assertion. The explicit check below repeats it from the
    outside, so a future change that softened the internal one would still fail.
    """
    ledger = Ledger(n_symbols=n, initial_equity=1_000_000.0)
    previous_close = price_vector(n, data.draw(st.lists(prices, min_size=n, max_size=n)))

    for bar in range(1, data.draw(st.integers(min_value=1, max_value=8)) + 1):
        exec_price = price_vector(n, data.draw(st.lists(prices, min_size=n, max_size=n)))
        close = price_vector(n, data.draw(st.lists(prices, min_size=n, max_size=n)))
        target = np.array(data.draw(st.lists(shares, min_size=n, max_size=n)), dtype=np.float64)
        dividends = np.zeros(n)

        equity_before = ledger.equity
        shares_before = ledger.shares.copy()

        flows = ledger.settle(
            bar=bar,
            shares_after=target,
            exec_price=exec_price,
            previous_close=previous_close,
            close=close,
            dividend_per_share=dividends,
            trade_costs=data.draw(costs),
            holding_costs=data.draw(costs),
        )

        # Balance sheet, computed here rather than taken from the ledger.
        independent = ledger.cash + float(np.dot(ledger.shares, close))
        assert independent == pytest.approx(ledger.equity, abs=CENT, rel=1e-9)

        # Income statement, likewise.
        expected_pnl = float(
            np.dot(shares_before, exec_price - previous_close) + np.dot(target, close - exec_price)
        )
        assert flows["mark_pnl"] == pytest.approx(expected_pnl, abs=CENT, rel=1e-9)
        assert ledger.equity - equity_before == pytest.approx(flows["net_pnl"], abs=CENT, rel=1e-9)
        previous_close = close


@given(
    n=st.integers(min_value=1, max_value=5),
    dividend=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
@settings(max_examples=50, deadline=None)
def test_dividends_accrue_to_the_holder_at_the_previous_close(n: int, dividend: float) -> None:
    """Paid on the position held *into* the bar, not the one traded during it."""
    ledger = Ledger(n_symbols=n, initial_equity=1_000_000.0)
    held = np.full(n, 100.0)
    price = np.full(n, 50.0)

    ledger.settle(
        bar=1,
        shares_after=held,
        exec_price=price,
        previous_close=price,
        close=price,
        dividend_per_share=np.zeros(n),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    equity_before = ledger.equity

    flows = ledger.settle(
        bar=2,
        shares_after=np.zeros(n),  # sold during the bar, but the dividend is earned
        exec_price=price,
        previous_close=price,
        close=price,
        dividend_per_share=np.full(n, dividend),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    assert flows["dividends"] == pytest.approx(100.0 * dividend * n)
    assert ledger.equity == pytest.approx(equity_before + 100.0 * dividend * n, abs=CENT)


# ----------------------------------------------------------------------------------
# Worked examples
# ----------------------------------------------------------------------------------
def test_buy_and_hold_earns_exactly_the_price_move() -> None:
    ledger = Ledger(n_symbols=1, initial_equity=100_000.0)
    price = np.array([100.0])
    ledger.settle(
        bar=1,
        shares_after=np.array([500.0]),
        exec_price=price,
        previous_close=price,
        close=price,
        dividend_per_share=np.zeros(1),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    assert ledger.cash == pytest.approx(50_000.0)
    assert ledger.equity == pytest.approx(100_000.0), "buying at the mark changes nothing"

    ledger.settle(
        bar=2,
        shares_after=np.array([500.0]),
        exec_price=np.array([110.0]),
        previous_close=price,
        close=np.array([110.0]),
        dividend_per_share=np.zeros(1),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    assert ledger.equity == pytest.approx(105_000.0), "500 shares x $10"


def test_intra_bar_execution_splits_the_move_at_the_trade() -> None:
    """The reason open-execution and close-execution backtests differ at all.

    Hold 100 shares into a bar that opens at 105 and closes at 120, and double the
    position at the open. The first 5 points are earned on 100 shares and the next
    15 on 200, not 20 points on either.
    """
    ledger = Ledger(n_symbols=1, initial_equity=100_000.0)
    ledger.settle(
        bar=1,
        shares_after=np.array([100.0]),
        exec_price=np.array([100.0]),
        previous_close=np.array([100.0]),
        close=np.array([100.0]),
        dividend_per_share=np.zeros(1),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    flows = ledger.settle(
        bar=2,
        shares_after=np.array([200.0]),
        exec_price=np.array([105.0]),
        previous_close=np.array([100.0]),
        close=np.array([120.0]),
        dividend_per_share=np.zeros(1),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    assert flows["mark_pnl"] == pytest.approx(100 * 5 + 200 * 15)


def test_costs_come_straight_out_of_equity() -> None:
    ledger = Ledger(n_symbols=1, initial_equity=100_000.0)
    price = np.array([100.0])
    ledger.settle(
        bar=1,
        shares_after=np.array([100.0]),
        exec_price=price,
        previous_close=price,
        close=price,
        dividend_per_share=np.zeros(1),
        trade_costs=250.0,
        holding_costs=125.0,
    )
    assert ledger.equity == pytest.approx(100_000.0 - 375.0)


def test_shorting_produces_cash_and_negative_exposure() -> None:
    ledger = Ledger(n_symbols=1, initial_equity=100_000.0)
    price = np.array([50.0])
    ledger.settle(
        bar=1,
        shares_after=np.array([-1_000.0]),
        exec_price=price,
        previous_close=price,
        close=price,
        dividend_per_share=np.zeros(1),
        trade_costs=0.0,
        holding_costs=0.0,
    )
    assert ledger.cash == pytest.approx(150_000.0)
    assert ledger.net_exposure(price) == pytest.approx(-50_000.0)
    assert ledger.gross_exposure(price) == pytest.approx(50_000.0)
    assert ledger.leverage(price) == pytest.approx(0.5)


# ----------------------------------------------------------------------------------
# Failure is loud
# ----------------------------------------------------------------------------------
def test_a_broken_ledger_raises_rather_than_reporting_a_curve() -> None:
    """Simulated by corrupting cash behind the ledger's back. Without the check,
    the run would continue and produce a plausible, wrong equity curve."""
    ledger = Ledger(n_symbols=1, initial_equity=100_000.0)
    price = np.array([100.0])
    ledger.cash += 5_000.0  # a bug, anywhere upstream

    with pytest.raises(AccountingError, match="do not balance"):
        ledger.settle(
            bar=1,
            shares_after=np.array([10.0]),
            exec_price=price,
            previous_close=price,
            close=price,
            dividend_per_share=np.zeros(1),
            trade_costs=0.0,
            holding_costs=0.0,
        )


def test_a_nan_price_reaching_a_live_position_is_caught() -> None:
    ledger = Ledger(n_symbols=1, initial_equity=100_000.0)
    ledger.equity = float("nan")
    with pytest.raises(AccountingError, match="not finite"):
        ledger.settle(
            bar=1,
            shares_after=np.array([1.0]),
            exec_price=np.array([100.0]),
            previous_close=np.array([100.0]),
            close=np.array([100.0]),
            dividend_per_share=np.zeros(1),
            trade_costs=0.0,
            holding_costs=0.0,
        )


def test_reconciliation_is_reported_as_evidence() -> None:
    """The claim that the books balanced should be auditable, not asserted."""
    ledger = Ledger(n_symbols=2, initial_equity=1_000_000.0)
    price = np.array([10.0, 20.0])
    for bar in range(1, 6):
        ledger.settle(
            bar=bar,
            shares_after=np.array([100.0 * bar, -50.0 * bar]),
            exec_price=price,
            previous_close=price,
            close=price,
            dividend_per_share=np.zeros(2),
            trade_costs=1.0,
            holding_costs=0.5,
        )
    assert ledger.report.bars_checked == 5
    assert abs(ledger.report.worst_absolute_gap) < CENT
    assert "reconciled" in ledger.report.describe()


def test_zero_initial_equity_is_rejected() -> None:
    with pytest.raises(ValueError, match="initial_equity must be positive"):
        Ledger(n_symbols=1, initial_equity=0.0)
