"""Drawdown controls and stop-losses.

Both are risk management, not alpha, and the tests are written to hold that line:
alongside the mechanical checks there are tests that measure what each control
*costs*, because a control that appears to improve returns in a backtest is
almost always fitted to the particular drawdowns in the sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.risk.controls import DrawdownControl, StopLossPolicy, drawdown_series


# ------------------------------------------------------------- drawdown series --
def test_drawdown_is_measured_from_the_running_peak() -> None:
    equity = np.array([100.0, 110.0, 88.0, 99.0, 121.0, 108.9])
    result = drawdown_series(equity)

    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)  # a new peak is not a drawdown
    assert result[2] == pytest.approx(-0.2)  # 88 from a peak of 110
    assert result[3] == pytest.approx(-0.1)
    assert result[4] == pytest.approx(0.0)  # new peak resets it
    assert result[5] == pytest.approx(-0.1)


def test_a_monotone_path_never_draws_down() -> None:
    assert np.allclose(drawdown_series(np.linspace(100.0, 200.0, 50)), 0.0)


def test_drawdown_rejects_an_empty_or_multidimensional_series() -> None:
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        drawdown_series(np.array([]))
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        drawdown_series(np.zeros((5, 2)))


# ----------------------------------------------------------- drawdown control --
def test_scaling_is_linear_between_the_thresholds() -> None:
    control = DrawdownControl(soft=-0.10, hard=-0.30, floor=0.0)

    assert control.multiplier(0.0) == pytest.approx(1.0)
    assert control.multiplier(-0.05) == pytest.approx(1.0)
    assert control.multiplier(-0.10) == pytest.approx(1.0)
    assert control.multiplier(-0.20) == pytest.approx(0.5)  # halfway
    assert control.multiplier(-0.30) == pytest.approx(0.0)
    assert control.multiplier(-0.50) == pytest.approx(0.0)


def test_the_floor_is_respected_below_the_hard_threshold() -> None:
    control = DrawdownControl(soft=-0.10, hard=-0.25, floor=0.25)
    assert control.multiplier(-0.25) == pytest.approx(0.25)
    assert control.multiplier(-0.90) == pytest.approx(0.25)
    assert control.multiplier(-0.175) == pytest.approx(0.625)  # halfway to the floor


def test_thresholds_must_be_negative_and_ordered() -> None:
    with pytest.raises(ValueError, match="negative fractions"):
        DrawdownControl(soft=0.10, hard=-0.25)
    with pytest.raises(ValueError, match="deeper than soft"):
        DrawdownControl(soft=-0.25, hard=-0.10)
    with pytest.raises(ValueError, match=r"floor must be in \[0, 1\]"):
        DrawdownControl(soft=-0.10, hard=-0.25, floor=1.5)


def test_the_multiplier_uses_only_the_drawdown_so_far() -> None:
    """A control that saw the whole path would look prescient, which is the
    single easiest way to make risk management appear to add return."""
    control = DrawdownControl(soft=-0.05, hard=-0.20)
    equity = np.array([100.0, 90.0, 80.0, 200.0])

    applied = control.apply(equity)
    # The huge final bar must not reach backward and relieve the earlier scaling.
    prefix = control.apply(equity[:3])
    assert np.allclose(applied[:3], prefix)


def test_a_control_caps_the_left_tail_and_pays_for_it_in_the_recovery() -> None:
    """The honest claim for a drawdown control, measured on a V-shaped path.

    The control delevers into the decline -- which is the point -- and is still
    delevered through the bounce, because the drawdown it reacts to is only
    repaired on the way back up. Capping the loss and giving up the recovery are
    the same mechanism, not a trade that can be tuned away.
    """
    down = np.full(40, -0.01)
    up = np.full(60, 0.01)
    returns = np.concatenate([down, up])

    control = DrawdownControl(soft=-0.05, hard=-0.30)
    equity, scaled_equity, multiplier = 1.0, 1.0, 1.0
    path, scaled_path = [1.0], [1.0]
    for step in returns:
        equity *= 1.0 + step
        scaled_equity *= 1.0 + multiplier * step
        path.append(equity)
        scaled_path.append(scaled_equity)
        multiplier = control.multiplier(float(drawdown_series(np.array(scaled_path))[-1]))

    worst_unscaled = drawdown_series(np.array(path)).min()
    worst_scaled = drawdown_series(np.array(scaled_path)).min()

    assert worst_scaled > worst_unscaled  # the left tail is genuinely capped
    assert scaled_path[-1] < path[-1]  # and the recovery is genuinely given up


def test_describe_states_the_shape() -> None:
    text = DrawdownControl(soft=-0.10, hard=-0.25, floor=0.0).describe()
    assert "-10%" in text
    assert "-25%" in text


# --------------------------------------------------------------- stop-losses --
def test_a_losing_position_is_closed_after_the_threshold_is_breached() -> None:
    policy = StopLossPolicy(threshold=-0.10, cooldown_bars=2)
    positions = np.ones((8, 1))
    returns = np.zeros((8, 1))
    returns[1:4, 0] = -0.04  # cumulative -11.5% by bar 3

    result = policy.apply(positions, returns)

    assert result[0, 0] == 1.0
    # Closed on the bar after the breach is observed, and that bar is the first
    # of the two cooldown bars -- so the book is flat for exactly two bars.
    assert list(result[:, 0]) == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_a_stop_acts_only_on_information_already_observed() -> None:
    """The position is closed the bar *after* the loss is visible, never on the
    bar that produced it."""
    policy = StopLossPolicy(threshold=-0.05, cooldown_bars=0)
    positions = np.ones((5, 1))
    returns = np.zeros((5, 1))
    returns[3, 0] = -0.50  # a crash arriving in bar 3

    result = policy.apply(positions, returns)

    # Bar 2 held into bar 3 and takes the loss; the stop closes bar 3 onward.
    assert result[2, 0] == 1.0
    assert result[3, 0] == 0.0


def test_a_profitable_position_is_never_stopped() -> None:
    policy = StopLossPolicy(threshold=-0.10, cooldown_bars=3)
    positions = np.ones((20, 1))
    returns = np.full((20, 1), 0.005)

    assert np.array_equal(policy.apply(positions, returns), positions)


def test_a_short_position_stops_on_a_rally_not_a_decline() -> None:
    """The loss is measured in the direction of the position, so the sign matters.
    Getting this wrong stops every winner and holds every loser."""
    policy = StopLossPolicy(threshold=-0.10, cooldown_bars=0)
    returns = np.zeros((6, 1))
    returns[1:4, 0] = 0.05  # the market rallies

    shorts = policy.apply(-np.ones((6, 1)), returns)
    longs = policy.apply(np.ones((6, 1)), returns)

    assert 0.0 in shorts[:, 0]  # the short is stopped out
    assert np.all(longs[:, 0] == 1.0)  # the long, holding the same path, is not


def test_the_loss_is_counted_from_entry_not_from_the_start_of_the_sample() -> None:
    """A position re-entered after a cooldown starts its count again. Carrying the
    old loss forward would stop the new position on the previous one's evidence."""
    policy = StopLossPolicy(threshold=-0.10, cooldown_bars=1)
    positions = np.ones((10, 1))
    returns = np.zeros((10, 1))
    returns[1, 0] = -0.12  # stops out
    returns[6, 0] = -0.03  # would breach only if the old loss were still counted

    result = policy.apply(positions, returns)

    assert result[1, 0] == 0.0  # stopped
    assert result[3, 0] == 1.0  # re-entered after the cooldown
    assert result[7, 0] == 1.0  # and not stopped again by a 3% move


def test_a_flat_position_accrues_no_loss() -> None:
    policy = StopLossPolicy(threshold=-0.05, cooldown_bars=0)
    positions = np.zeros((6, 1))
    positions[4:, 0] = 1.0
    returns = np.full((6, 1), -0.10)

    result = policy.apply(positions, returns)
    assert result[4, 0] == 1.0  # entering into a falling market is allowed


def test_the_cooldown_is_what_stops_a_stop_becoming_a_cost_generator() -> None:
    """Without a cooldown a stop re-enters the next bar, so a choppy market pays
    the spread over and over for no change in exposure."""
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.03, (1000, 1))
    positions = np.ones((1000, 1))

    def turnover(book: np.ndarray) -> float:
        return float(np.abs(np.diff(book[:, 0])).sum())

    immediate = turnover(StopLossPolicy(threshold=-0.04, cooldown_bars=0).apply(positions, returns))
    patient = turnover(StopLossPolicy(threshold=-0.04, cooldown_bars=25).apply(positions, returns))

    assert immediate > 3 * patient, f"{immediate:.0f} vs {patient:.0f} units of turnover"


def test_a_stopped_position_does_not_participate_in_the_bounce() -> None:
    """The cost no backtest of a stop shows on its own.

    The position is closed at the bottom by construction -- that is what a stop
    does -- and the bounce is where mean reversion lives.
    """
    returns = np.zeros((30, 1))
    returns[1:5, 0] = -0.08  # sharp decline
    returns[5:12, 0] = 0.07  # and a full recovery

    positions = np.ones((30, 1))
    stopped = StopLossPolicy(threshold=-0.15, cooldown_bars=10).apply(positions, returns)

    held_pnl = float(np.sum(positions[:-1, 0] * returns[1:, 0]))
    stopped_pnl = float(np.sum(stopped[:-1, 0] * returns[1:, 0]))

    assert stopped_pnl < held_pnl


def test_an_open_profit_is_protected_rather_than_given_back() -> None:
    """The difference between the two bases, on a position that ran up first.

    A trailing stop measures the loss from the high-water mark since entry, so a
    position up 50% that gives back 20% of its peak is closed. Measuring from the
    entry price instead, the same position is still up 20% and the stop does not
    fire -- the open profit has become a buffer.
    """
    returns = np.zeros((20, 1))
    returns[1:6, 0] = 0.085  # runs up roughly 50%
    returns[6:9, 0] = -0.08  # then gives back about a fifth of the peak
    positions = np.ones((20, 1))

    trailing = StopLossPolicy(threshold=-0.15, cooldown_bars=2, basis="peak")
    from_entry = StopLossPolicy(threshold=-0.15, cooldown_bars=2, basis="entry")

    assert 0.0 in trailing.apply(positions, returns)[:, 0]
    assert np.all(from_entry.apply(positions, returns)[:, 0] == 1.0)


@pytest.mark.slow
def test_a_from_entry_stop_switches_itself_off_as_a_position_works() -> None:
    """Why ``basis="entry"`` is kept only for comparison.

    On a driftless random walk the two bases face identical paths, and the
    from-entry stop barely fires at all: once wealth has compounded away from
    the entry price, the threshold sits far below anything the position will
    revisit. A risk control that weakens the longer a position is held is not a
    risk control.
    """
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.03, (1000, 1))
    positions = np.ones((1000, 1))

    def stops(policy: StopLossPolicy) -> int:
        book = policy.apply(positions, returns)[:, 0]
        return int(((book == 0.0) & (np.roll(book, 1) != 0.0))[1:].sum())

    trailing = stops(StopLossPolicy(threshold=-0.04, cooldown_bars=0, basis="peak"))
    from_entry = stops(StopLossPolicy(threshold=-0.04, cooldown_bars=0, basis="entry"))

    assert trailing > 10 * from_entry, f"{trailing} trailing stops vs {from_entry} from entry"


def test_the_basis_must_be_one_of_the_two_supported() -> None:
    with pytest.raises(ValueError, match="basis must be"):
        StopLossPolicy(basis="trailing")  # type: ignore[arg-type]


def test_describe_names_the_basis() -> None:
    assert "from the peak since entry" in StopLossPolicy(basis="peak").describe()
    assert "from entry" in StopLossPolicy(basis="entry").describe()


def test_stop_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="negative fraction"):
        StopLossPolicy(threshold=0.20)
    with pytest.raises(ValueError, match="non-negative"):
        StopLossPolicy(threshold=-0.20, cooldown_bars=-1)


def test_positions_and_returns_must_be_the_same_shape() -> None:
    with pytest.raises(ValueError, match="same shape"):
        StopLossPolicy().apply(np.ones((10, 2)), np.ones((10, 3)))


def test_instruments_are_stopped_independently() -> None:
    """A stop acts on one instrument's own evidence -- the distinction from a
    portfolio volatility target that waits for the move to show up diluted."""
    policy = StopLossPolicy(threshold=-0.10, cooldown_bars=2)
    positions = np.ones((8, 3))
    returns = np.zeros((8, 3))
    returns[2, 0] = -0.20  # only the first instrument collapses

    result = policy.apply(positions, returns)

    assert result[2, 0] == 0.0
    assert np.all(result[:, 1] == 1.0)
    assert np.all(result[:, 2] == 1.0)


def test_describe_states_the_rule() -> None:
    text = StopLossPolicy(threshold=-0.20, cooldown_bars=5).describe()
    assert "-20%" in text
    assert "5 bars" in text
