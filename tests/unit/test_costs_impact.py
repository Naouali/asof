"""Market impact: closed-form checks, scaling laws, and cross-model consistency.

Every assertion here is either an algebraic identity or an empirical regularity
from the literature. Nothing is pinned to a number this implementation happens to
produce -- a test that only says "it still does what it did" cannot catch a model
that was wrong from the start.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quantlab.costs.impact import (
    AlmgrenChriss,
    ImpactParams,
    PropagatorImpact,
    SquareRootImpact,
)

ADV = 1e9
VOL = 0.02


# ----------------------------------------------------------------------------------
# The law itself
# ----------------------------------------------------------------------------------
def test_square_root_law_matches_its_closed_form() -> None:
    """``Y · σ · (Q/ADV)^δ``, computed by hand.

    With Y=0.5, σ=2%/day and 1% of ADV: 0.5 × 0.02 × 0.1 = 0.001 = 10 bp. That
    also matches the practitioner rule of thumb that 1% of ADV costs about 10 bp,
    which is the sanity check that the calibration is in the right universe.
    """
    model = SquareRootImpact(ImpactParams(y=0.5, delta=0.5))
    assert model.peak_impact_bps(0.01 * ADV, ADV, VOL) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("participation", "expected_bps"),
    [
        (0.0001, 1.0),
        (0.0025, 5.0),
        (0.01, 10.0),
        (0.04, 20.0),
        (0.09, 30.0),
    ],
)
def test_known_points_on_the_curve(participation: float, expected_bps: float) -> None:
    model = SquareRootImpact(ImpactParams(y=0.5, delta=0.5))
    assert model.peak_impact_bps(participation * ADV, ADV, VOL) == pytest.approx(expected_bps)


def test_doubling_impact_requires_quadrupling_size() -> None:
    """The headline consequence of the square root, and the reason capacity is
    finite. Spec section 5 states it explicitly."""
    model = SquareRootImpact()
    small = model.peak_impact_bps(1e6, ADV, VOL)
    large = model.peak_impact_bps(4e6, ADV, VOL)
    assert large / small == pytest.approx(2.0)


def test_currency_cost_grows_like_q_to_the_three_halves() -> None:
    """Cost per share grows like sqrt(Q) and you pay it on every share, so dollars
    of cost grow like Q^1.5 while dollars of alpha grow at best linearly. That gap
    is the whole of capacity."""
    model = SquareRootImpact()
    base = model.cost_bps(1e6, ADV, VOL) * 1e6
    quadrupled = model.cost_bps(4e6, ADV, VOL) * 4e6
    assert quadrupled / base == pytest.approx(4**1.5)


def test_impact_is_zero_for_a_zero_order() -> None:
    assert SquareRootImpact().peak_impact_bps(0.0, ADV, VOL) == 0.0


def test_exponent_is_configurable_across_the_published_range() -> None:
    """Empirical estimates of δ span roughly 0.4-0.7; the platform must not hard-code
    0.5, because the choice materially changes capacity for large orders."""
    size = 0.05 * ADV
    shallow = SquareRootImpact(ImpactParams(delta=0.4)).peak_impact_bps(size, ADV, VOL)
    canonical = SquareRootImpact(ImpactParams(delta=0.5)).peak_impact_bps(size, ADV, VOL)
    steep = SquareRootImpact(ImpactParams(delta=0.7)).peak_impact_bps(size, ADV, VOL)
    # A larger exponent penalises large orders more.
    assert steep < canonical < shallow


@pytest.mark.parametrize(
    "bad",
    [
        {"y": 0.0},
        {"y": -1.0},
        {"delta": 0.0},
        {"delta": 1.5},
        {"permanent_fraction": 1.2},
    ],
)
def test_invalid_parameters_are_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ImpactParams(**bad)


# ----------------------------------------------------------------------------------
# Permanent / temporary split
# ----------------------------------------------------------------------------------
def test_split_sums_to_the_peak_at_a_full_day_horizon() -> None:
    model = SquareRootImpact()
    permanent, temporary = model.split(0.01 * ADV, ADV, VOL, horizon_days=1.0)
    assert permanent + temporary == pytest.approx(model.peak_impact_bps(0.01 * ADV, ADV, VOL))


def test_only_half_the_permanent_impact_is_paid() -> None:
    """Permanent impact accrues while the order is worked, so the early fills
    execute before most of it has happened."""
    model = SquareRootImpact(ImpactParams(permanent_fraction=1.0))
    peak = model.peak_impact_bps(0.01 * ADV, ADV, VOL)
    assert model.cost_bps(0.01 * ADV, ADV, VOL) == pytest.approx(peak / 2.0)


def test_purely_temporary_impact_is_paid_in_full() -> None:
    model = SquareRootImpact(ImpactParams(permanent_fraction=0.0))
    peak = model.peak_impact_bps(0.01 * ADV, ADV, VOL)
    assert model.cost_bps(0.01 * ADV, ADV, VOL) == pytest.approx(peak)


def test_permanent_impact_does_not_depend_on_urgency() -> None:
    """Permanent impact reflects the information the trade reveals, not the rate.
    Only the temporary component responds to trading more slowly -- which is the
    entire lever execution has."""
    model = SquareRootImpact()
    fast_perm, fast_temp = model.split(0.01 * ADV, ADV, VOL, horizon_days=0.25)
    slow_perm, slow_temp = model.split(0.01 * ADV, ADV, VOL, horizon_days=4.0)
    assert fast_perm == pytest.approx(slow_perm)
    assert fast_temp > slow_temp


def test_trading_faster_costs_more() -> None:
    model = SquareRootImpact()
    costs = [model.cost_bps(0.01 * ADV, ADV, VOL, horizon_days=h) for h in (0.1, 0.5, 1.0, 5.0)]
    assert costs == sorted(costs, reverse=True)


def test_extrapolation_beyond_the_calibrated_range_is_flagged() -> None:
    """Published fits are dominated by orders below ~10% of ADV. Beyond that the
    law understates, so a capacity number resting on it is an upper bound."""
    model = SquareRootImpact()
    assert not model.is_extrapolating(0.05 * ADV, ADV)
    assert model.is_extrapolating(0.25 * ADV, ADV)


def test_zero_adv_is_rejected() -> None:
    with pytest.raises(ValueError, match="adv_notional must be positive"):
        SquareRootImpact().participation(1e6, 0.0)


# ----------------------------------------------------------------------------------
# Scalar and vectorised must agree
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize("participation", [1e-5, 1e-3, 0.01, 0.05, 0.2])
@pytest.mark.parametrize("horizon", [0.25, 1.0, 3.0])
def test_vectorised_matches_scalar(participation: float, horizon: float) -> None:
    """The fast engine and the readable implementation must not drift apart; a
    backtest that costs trades differently from the tests is untested."""
    model = SquareRootImpact()
    scalar = model.cost_bps(participation * ADV, ADV, VOL, horizon_days=horizon)
    frame = pl.DataFrame(
        {"notional": [participation * ADV], "adv": [ADV], "vol": [VOL]}
    ).with_columns(
        cost=model.cost_bps_expr(
            pl.col("notional"), pl.col("adv"), pl.col("vol"), horizon_days=horizon
        )
    )
    assert frame["cost"].item() == pytest.approx(scalar)


# ----------------------------------------------------------------------------------
# Almgren-Chriss
# ----------------------------------------------------------------------------------
def _linearised() -> tuple[AlmgrenChriss, SquareRootImpact, float]:
    impact = SquareRootImpact()
    notional = 0.02 * ADV
    model = AlmgrenChriss.from_square_root(
        impact, notional=notional, adv_notional=ADV, volatility_daily=VOL
    )
    return model, impact, notional / ADV


def test_risk_neutral_schedule_is_twap() -> None:
    model, _, participation = _linearised()
    plan = model.schedule(participation, horizon_days=1.0, periods=10)
    assert plan.kappa == 0.0
    assert plan.is_twap
    assert plan.half_life_days() == pytest.approx(0.5, abs=0.11)


def test_schedule_liquidates_exactly() -> None:
    model, _, participation = _linearised()
    plan = model.schedule(participation, 1.0, 25)
    assert plan.trades.sum() == pytest.approx(participation)
    assert plan.holdings[0] == pytest.approx(participation)
    assert plan.holdings[-1] == 0.0
    assert (plan.trades > 0).all(), "a liquidation must never buy"


def test_risk_neutral_cost_converges_to_the_square_root_law() -> None:
    """The two models are calibrated from the same law, so at zero risk aversion
    they must agree in the continuous limit. The discrete gap is exactly
    ``γ·τ·X/2`` and vanishes as the period shrinks -- which is asserted rather
    than assumed.
    """
    model, impact, participation = _linearised()
    direct = impact.cost_bps(participation * ADV, ADV, VOL, horizon_days=1.0)
    errors = [
        abs(model.schedule(participation, 1.0, periods=n).expected_cost_bps - direct)
        for n in (10, 100, 1000)
    ]
    assert errors == sorted(errors, reverse=True), "error must shrink with more periods"
    assert errors[-1] / direct < 1e-3


def test_risk_aversion_front_loads_the_schedule() -> None:
    model, _, participation = _linearised()
    half_lives = [
        AlmgrenChriss(volatility_daily=VOL, gamma=model.gamma, eta=model.eta, risk_aversion=lam)
        .schedule(participation, 1.0, 40)
        .half_life_days()
        for lam in (0.0, 1e3, 1e4, 1e5)
    ]
    assert half_lives == sorted(half_lives, reverse=True)


def test_efficient_frontier_is_monotone() -> None:
    """Less risk must cost more. A frontier that is not monotone means the solver
    is wrong, not that a free lunch was found."""
    model, _, participation = _linearised()
    frontier = model.efficient_frontier(participation, 1.0, [0.0, 1e3, 1e4, 1e5], periods=40)
    costs = [cost for _, cost, _ in frontier]
    stdevs = [sd for _, _, sd in frontier]
    assert costs == sorted(costs)
    assert stdevs == sorted(stdevs, reverse=True)


def test_ill_posed_discretisation_is_rejected() -> None:
    """With too few periods the effective temporary impact goes negative and the
    discrete problem has no solution. Raising beats returning a negative cost."""
    model = AlmgrenChriss(volatility_daily=VOL, gamma=10.0, eta=0.1)
    with pytest.raises(ValueError, match="well posed"):
        model.schedule(0.01, horizon_days=10.0, periods=1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"periods": 0}, "periods"),
        ({"horizon_days": 0.0}, "horizon_days"),
        ({"participation": 0.0}, "participation"),
    ],
)
def test_schedule_rejects_bad_arguments(kwargs: dict[str, float], match: str) -> None:
    model, _, participation = _linearised()
    call = {"participation": participation, "horizon_days": 1.0, "periods": 20, **kwargs}
    with pytest.raises(ValueError, match=match):
        model.schedule(**call)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# Propagator
# ----------------------------------------------------------------------------------
def test_kernel_starts_at_one_and_decays() -> None:
    prop = PropagatorImpact()
    kernel = prop.kernel(np.array([0.0, 1.0, 5.0, 50.0]))
    assert kernel[0] == pytest.approx(1.0)
    assert list(kernel) == sorted(kernel, reverse=True)
    assert kernel[-1] < 0.5


def test_kernel_half_life_matches_its_closed_form() -> None:
    """``τ0·(2^(1/β) − 1)`` is where ``(1 + τ/τ0)^(-β) = 1/2``."""
    prop = PropagatorImpact(tau0=2.0, beta=0.5)
    expected = 2.0 * (2.0 ** (1.0 / 0.5) - 1.0)
    assert prop.half_life() == pytest.approx(expected)
    assert prop.kernel(np.array([prop.half_life()]))[0] == pytest.approx(0.5)


def test_impact_builds_while_trading_and_decays_after() -> None:
    """The decay is why splitting an order helps at all. Under purely permanent
    impact the schedule would not matter."""
    prop = PropagatorImpact()
    participations = np.array([0.01] * 5 + [0.0] * 10)
    path = prop.impact_path(participations, np.ones(15), VOL)
    assert path[4] == path.max(), "impact peaks at the last child order"
    assert path[-1] < path[4], "and decays once trading stops"
    assert path[-1] > 0, "but a power-law kernel never fully reverts"


def test_opposing_trades_cancel_impact() -> None:
    prop = PropagatorImpact()
    signs = np.array([1.0, -1.0])
    path = prop.impact_path(np.array([0.01, 0.01]), signs, VOL)
    assert abs(path[-1]) < path[0]


def test_propagator_cost_rises_with_size() -> None:
    prop = PropagatorImpact()
    costs = [
        prop.implied_cost_bps(np.full(10, size), np.ones(10), VOL) for size in (0.001, 0.005, 0.02)
    ]
    assert costs == sorted(costs)


def test_propagator_rejects_negative_participation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PropagatorImpact().impact_path(np.array([-0.01]), np.array([1.0]), VOL)


@pytest.mark.parametrize("beta", [0.0, 1.0, 1.5])
def test_propagator_rejects_non_decaying_kernels(beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        PropagatorImpact(beta=beta)
