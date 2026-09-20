"""Gârleanu-Pedersen dynamic trading.

The framework's claim is that with quadratic costs the optimal policy is to trade
a constant fraction of the way toward a slow-moving *aim* portfolio, rather than
rebalancing all the way to the Markowitz optimum each period. The tests check the
two limits where the answer is known analytically, the monotonicity the
parameters are supposed to produce, and that a simulated path actually converges
where the solved policy says it will.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.portfolio.dynamic import TradingPolicy, markowitz_portfolio


def covariance(n: int, *, rho: float = 0.3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    deviations = rng.uniform(0.01, 0.02, n)
    matrix = rho * np.outer(deviations, deviations)
    np.fill_diagonal(matrix, deviations**2)
    return matrix


# ------------------------------------------------------------------- Markowitz --
def test_the_costless_optimum_solves_the_first_order_condition() -> None:
    matrix = covariance(5)
    mu = np.linspace(0.0002, 0.001, 5)
    weights = markowitz_portfolio(mu, matrix, risk_aversion=4.0)
    assert np.allclose(4.0 * matrix @ weights, mu)


def test_a_singular_covariance_is_refused_with_advice() -> None:
    matrix = np.ones((3, 3))  # rank 1
    with pytest.raises(ValueError, match="Shrink the estimate"):
        markowitz_portfolio(np.ones(3), matrix, risk_aversion=2.0)


def test_risk_aversion_must_be_positive() -> None:
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        markowitz_portfolio(np.ones(3), covariance(3), risk_aversion=0.0)


def test_doubling_risk_aversion_halves_the_position() -> None:
    matrix, mu = covariance(4), np.full(4, 0.0005)
    assert np.allclose(
        markowitz_portfolio(mu, matrix, risk_aversion=8.0),
        0.5 * markowitz_portfolio(mu, matrix, risk_aversion=4.0),
    )


# ----------------------------------------------------------------- the solution --
def test_costless_trading_goes_straight_to_markowitz() -> None:
    """The limit where the framework has nothing to say: with no cost there is no
    reason to lag, so the aim *is* the optimum and the trade rate is one."""
    policy = TradingPolicy.solve(risk_aversion=4.0, trading_cost=0.0, signal_persistence=0.95)
    assert policy.trade_rate == 1.0
    assert policy.aim_scaling == 1.0
    assert policy.half_life_periods == 0.0


def test_costlier_trading_means_a_slower_trade_rate() -> None:
    rates = [
        TradingPolicy.solve(
            risk_aversion=4.0, trading_cost=cost, signal_persistence=0.95, discount=0.999
        ).trade_rate
        for cost in (0.1, 1.0, 10.0, 100.0)
    ]
    assert rates == sorted(rates, reverse=True)
    assert all(0.0 < r <= 1.0 for r in rates)


def test_a_faster_decaying_signal_is_chased_less_far() -> None:
    """The aim shrinks toward zero as the signal's half-life shortens: there is no
    point paying to reach a position the signal will have abandoned."""
    scalings = [
        TradingPolicy.solve(
            risk_aversion=4.0, trading_cost=5.0, signal_persistence=phi, discount=0.999
        ).aim_scaling
        for phi in (0.5, 0.9, 0.99)
    ]
    assert scalings == sorted(scalings)


def test_the_trade_rate_does_not_depend_on_the_signal() -> None:
    """The framework's practical claim: the rate is a property of costs and risk
    alone, so it is solved once rather than per period."""
    rates = {
        TradingPolicy.solve(
            risk_aversion=3.0, trading_cost=2.0, signal_persistence=phi, discount=0.99
        ).trade_rate
        for phi in (0.1, 0.5, 0.9, 0.99)
    }
    assert len(rates) == 1


def test_the_half_life_matches_the_trade_rate() -> None:
    policy = TradingPolicy.solve(
        risk_aversion=4.0, trading_cost=20.0, signal_persistence=0.95, discount=0.999
    )
    remaining = (1.0 - policy.trade_rate) ** policy.half_life_periods
    assert remaining == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"risk_aversion": 0.0}, "risk_aversion must be positive"),
        ({"trading_cost": -1.0}, "trading_cost must be non-negative"),
        ({"signal_persistence": 1.5}, r"signal_persistence must be in \[0, 1\]"),
        ({"discount": 0.0}, r"discount must be in \(0, 1\]"),
    ],
)
def test_parameters_are_validated(kwargs: dict[str, float], message: str) -> None:
    base = {"risk_aversion": 4.0, "trading_cost": 1.0, "signal_persistence": 0.9}
    with pytest.raises(ValueError, match=message):
        TradingPolicy.solve(**{**base, **kwargs})  # type: ignore[arg-type]


# ------------------------------------------------------------------- the policy --
def test_a_step_moves_the_stated_fraction_toward_the_aim() -> None:
    policy = TradingPolicy.solve(
        risk_aversion=4.0, trading_cost=10.0, signal_persistence=0.95, discount=0.999
    )
    matrix, mu = covariance(4), np.full(4, 0.0006)
    start = np.zeros(4)

    decision = policy.step(start, mu, matrix)

    assert np.allclose(decision.trade, policy.trade_rate * (decision.aim - start))
    assert np.allclose(decision.target, start + decision.trade)
    assert decision.turnover == pytest.approx(float(np.abs(decision.trade).sum()))


def test_a_position_already_at_the_aim_does_not_trade() -> None:
    policy = TradingPolicy.solve(
        risk_aversion=4.0, trading_cost=10.0, signal_persistence=0.95, discount=0.999
    )
    matrix, mu = covariance(3), np.full(3, 0.0005)
    aim = policy.aim_scaling * markowitz_portfolio(mu, matrix, 4.0)

    decision = policy.step(aim, mu, matrix)
    assert decision.turnover == pytest.approx(0.0, abs=1e-12)


def test_repeated_steps_converge_to_the_aim() -> None:
    """The solved rate is a claim about a path, so walk one."""
    policy = TradingPolicy.solve(
        risk_aversion=4.0, trading_cost=10.0, signal_persistence=0.95, discount=0.999
    )
    matrix, mu = covariance(5), np.full(5, 0.0006)
    position = np.zeros(5)
    turnovers = []
    for _ in range(300):
        decision = policy.step(position, mu, matrix)
        position = decision.target
        turnovers.append(decision.turnover)

    aim = policy.aim_scaling * markowitz_portfolio(mu, matrix, 4.0)
    assert np.allclose(position, aim, atol=1e-8)
    # And it approaches monotonically, trading less each period.
    assert turnovers == sorted(turnovers, reverse=True)


def test_the_aim_lags_the_markowitz_portfolio() -> None:
    """The whole point: with costs, the portfolio you aim at is a shrunk version
    of the one you would hold for free."""
    policy = TradingPolicy.solve(
        risk_aversion=4.0, trading_cost=50.0, signal_persistence=0.9, discount=0.999
    )
    decision = policy.step(np.zeros(4), np.full(4, 0.0006), covariance(4))

    assert 0.0 < decision.aim_shrinkage < 1.0
    assert np.abs(decision.aim).sum() < np.abs(decision.markowitz).sum()


def test_shrinkage_is_zero_when_trading_is_free() -> None:
    policy = TradingPolicy.solve(risk_aversion=4.0, trading_cost=0.0, signal_persistence=0.9)
    decision = policy.step(np.zeros(3), np.full(3, 0.0005), covariance(3))
    assert decision.aim_shrinkage == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(decision.target, decision.markowitz)


def test_shrinkage_of_a_zero_signal_is_defined() -> None:
    policy = TradingPolicy.solve(risk_aversion=4.0, trading_cost=5.0, signal_persistence=0.9)
    decision = policy.step(np.zeros(3), np.zeros(3), covariance(3))
    assert decision.aim_shrinkage == 0.0


def test_describe_states_the_policy() -> None:
    text = TradingPolicy.solve(
        risk_aversion=4.0, trading_cost=10.0, signal_persistence=0.95, discount=0.999
    ).describe()
    assert "of the way to the aim" in text
    assert "half-life" in text
