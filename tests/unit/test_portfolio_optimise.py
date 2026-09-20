"""Allocation schemes, constraints, and the dynamic trading policy."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.portfolio.constraints import Constraints, GroupLimit, build_groups
from quantlab.portfolio.covariance import ledoit_wolf_covariance
from quantlab.portfolio.dynamic import TradingPolicy, markowitz_portfolio
from quantlab.portfolio.optimise import (
    equal_weight,
    inverse_volatility,
    mean_variance,
    risk_contributions,
    risk_parity,
)


def covariance(n: int = 10, *, seed: int = 0, rho: float = 0.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.008, 0.030, n)
    common = rng.normal(0, 1, (600, 1))
    specific = rng.normal(0, 1, (600, n))
    returns = (np.sqrt(rho) * common + np.sqrt(1 - rho) * specific) * vols
    return ledoit_wolf_covariance(returns).matrix


# ==================================================================================
# Risk parity
# ==================================================================================
def test_risk_parity_equalises_risk_contributions() -> None:
    """The defining property, and the one the obvious-looking fixed point fails."""
    matrix = covariance(12)
    allocation = risk_parity(matrix)
    contributions = risk_contributions(allocation.weights, matrix)
    assert allocation.converged
    assert np.abs(contributions - 1 / 12).max() < 1e-8


def test_risk_parity_does_not_equalise_weights() -> None:
    """Equal *risk*, not equal money. The volatile assets get less."""
    matrix = covariance(12)
    weights = risk_parity(matrix).weights
    volatilities = np.sqrt(np.diag(matrix))
    assert np.ptp(weights) > 0.01
    # The most volatile asset gets the smallest weight.
    assert int(np.argmin(weights)) == int(np.argmax(volatilities))


def test_risk_parity_matches_inverse_volatility_when_correlations_are_equal() -> None:
    """A known closed form: with a single common factor and equal pairwise
    correlation, equal risk contribution *is* inverse volatility."""
    matrix = covariance(10, rho=0.4)
    parity = risk_parity(matrix).weights
    inverse = inverse_volatility(matrix).weights
    assert np.abs(parity - inverse).max() < 0.01


def test_risk_parity_differs_from_inverse_volatility_when_correlations_vary() -> None:
    rng = np.random.default_rng(3)
    vols = np.linspace(0.008, 0.03, 12)
    factors = rng.normal(0, 1, (600, 3))
    loadings = rng.uniform(0.2, 1.0, (3, 12))
    matrix = ledoit_wolf_covariance(
        (factors @ loadings + rng.normal(0, 1, (600, 12)) * 0.5) * vols
    ).matrix
    parity = risk_parity(matrix)
    inverse = inverse_volatility(matrix)
    assert np.ptp(risk_contributions(parity.weights, matrix)) < 1e-8
    assert np.ptp(risk_contributions(inverse.weights, matrix)) > 1e-4


def test_risk_parity_is_long_only_and_fully_invested() -> None:
    weights = risk_parity(covariance(8), gross=1.0).weights
    assert np.all(weights > 0), "risk parity has no way to express a short"
    assert weights.sum() == pytest.approx(1.0)


def test_risk_parity_on_a_single_asset() -> None:
    assert risk_parity(np.array([[0.04]])).weights == pytest.approx([1.0])


# ==================================================================================
# Diagnostics
# ==================================================================================
def test_effective_positions_exposes_hidden_concentration() -> None:
    """A book of fifty names where two carry most of the risk is a book of two."""
    spread = equal_weight(50)
    concentrated = equal_weight(50)
    weights = np.full(50, 0.001)
    weights[:2] = 0.499
    from quantlab.portfolio.optimise import Allocation

    concentrated = Allocation(weights=weights, method="t", volatility=0.0)
    assert spread.effective_positions == pytest.approx(50.0)
    assert concentrated.effective_positions < 3.0


def test_risk_contributions_sum_to_one() -> None:
    matrix = covariance(10)
    weights = risk_parity(matrix).weights
    assert risk_contributions(weights, matrix).sum() == pytest.approx(1.0)


def test_a_zero_variance_asset_is_rejected() -> None:
    """It has infinite risk-adjusted appeal and will consume the whole book."""
    matrix = covariance(4)
    matrix[2, 2] = 0.0
    with pytest.raises(ValueError, match="positive variance"):
        inverse_volatility(matrix)


# ==================================================================================
# Mean-variance and constraints
# ==================================================================================
def test_unconstrained_mean_variance_matches_its_closed_form() -> None:
    matrix = covariance(6)
    mu = np.linspace(0.0001, 0.0006, 6)
    allocation = mean_variance(
        mu, matrix, risk_aversion=20.0, constraints=Constraints.unconstrained()
    )
    closed_form = np.linalg.solve(20.0 * matrix, mu)
    assert allocation.weights == pytest.approx(closed_form, rel=1e-3, abs=1e-6)


def test_constraints_are_actually_binding() -> None:
    """Left alone on real estimates, the optimiser puts most of the book in a few
    names and levers the result. The constraints are what make it survivable."""
    matrix = covariance(10)
    mu = np.linspace(-0.0004, 0.0010, 10)
    free = mean_variance(mu, matrix, risk_aversion=1.0, constraints=Constraints.unconstrained())
    limited = mean_variance(
        mu,
        matrix,
        risk_aversion=1.0,
        constraints=Constraints(max_position=0.10, max_gross=1.0, net_range=(0.0, 0.0)),
    )
    assert free.gross > limited.gross
    assert np.abs(limited.weights).max() <= 0.10 + 1e-6
    assert limited.net == pytest.approx(0.0, abs=1e-6)


def test_dollar_neutral_constraints_hold() -> None:
    matrix = covariance(8)
    mu = np.linspace(-0.0003, 0.0007, 8)
    allocation = mean_variance(
        mu, matrix, risk_aversion=5.0, constraints=Constraints.dollar_neutral()
    )
    assert Constraints.dollar_neutral().violations(allocation.weights) == []


def test_long_only_constraints_hold() -> None:
    # Sixteen names, so a 10% position cap can still reach 100% net.
    matrix = covariance(16)
    mu = np.linspace(-0.0003, 0.0007, 16)
    limits = Constraints.long_only()
    allocation = mean_variance(mu, matrix, risk_aversion=5.0, constraints=limits)
    assert allocation.converged
    assert allocation.weights.min() >= -1e-6
    assert limits.violations(allocation.weights) == []


def test_an_infeasible_constraint_set_is_named_not_silently_answered() -> None:
    """The trap: long_only() caps positions at 10%, so eight names can reach only
    80% of capital while the net constraint demands 100%. Left to scipy this comes
    back as violating weights with a cryptic message."""
    with pytest.raises(ValueError, match="unreachable"):
        mean_variance(
            np.linspace(0.0, 0.001, 8),
            covariance(8),
            constraints=Constraints.long_only(),
        )


def test_net_exposure_beyond_the_gross_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot fit inside a gross limit"):
        Constraints(max_position=1.0, max_gross=0.5, net_range=(1.0, 1.0)).check_feasible(10)


def test_group_limits_cap_a_sector_bet() -> None:
    """A dollar-neutral book concentrated in one sector is a sector bet wearing a
    hedge. Constraining it beforehand is cheaper than detecting it afterwards."""
    matrix = covariance(9)
    mu = np.array([0.001] * 3 + [0.0] * 6)  # the whole edge sits in one group
    groups = (GroupLimit(name="tech", members=(0, 1, 2), max_gross=0.20),)
    allocation = mean_variance(
        mu,
        matrix,
        risk_aversion=1.0,
        constraints=Constraints(max_position=0.5, max_gross=2.0, groups=groups),
    )
    assert np.abs(allocation.weights[:3]).sum() <= 0.20 + 1e-6


def test_group_limits_can_be_built_from_labels() -> None:
    groups = build_groups(["tech", "tech", "energy", "energy", "tech"], max_gross=0.3)
    by_name = {g.name: g.members for g in groups}
    assert by_name["tech"] == (0, 1, 4)
    assert by_name["energy"] == (2, 3)


def test_violations_are_reported_in_words() -> None:
    limits = Constraints(max_position=0.05, max_gross=1.0, net_range=(0.0, 0.0))
    breaches = limits.violations(np.array([0.5, 0.5, 0.5]))
    assert any("largest position" in b for b in breaches)
    assert any("gross exposure" in b for b in breaches)
    assert any("net exposure" in b for b in breaches)


def test_contradictory_constraints_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        Constraints(min_position=0.2, max_position=0.1)
    with pytest.raises(ValueError, match="net_range must be"):
        Constraints(net_range=(1.0, 0.0))


def test_a_group_that_constrains_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="constrains nothing"):
        GroupLimit(name="empty", members=(0,))


def test_mismatched_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="disagree about the universe"):
        mean_variance(np.zeros(3), covariance(5))


# ==================================================================================
# Dynamic trading: Garleanu-Pedersen
# ==================================================================================
def test_no_trading_cost_means_trade_straight_to_markowitz() -> None:
    policy = TradingPolicy.solve(risk_aversion=1.0, trading_cost=0.0, signal_persistence=0.95)
    assert policy.trade_rate == pytest.approx(1.0)
    assert policy.aim_scaling == pytest.approx(1.0)


def test_an_enormous_cost_means_barely_trading() -> None:
    policy = TradingPolicy.solve(risk_aversion=1.0, trading_cost=1e6, signal_persistence=0.95)
    assert policy.trade_rate < 0.01
    assert policy.half_life_periods > 50


def test_a_permanent_signal_aims_at_markowitz() -> None:
    """With nothing decaying, there is no reason to aim short of the optimum."""
    policy = TradingPolicy.solve(
        risk_aversion=1.0, trading_cost=5.0, signal_persistence=1.0, discount=1.0
    )
    assert policy.aim_scaling == pytest.approx(1.0, abs=1e-9)


def test_a_fast_decaying_signal_aims_lower() -> None:
    """The result naive turnover filters miss: a decaying signal gets a smaller
    target as well as a slower approach to it."""
    slow = TradingPolicy.solve(
        risk_aversion=1.0, trading_cost=20.0, signal_persistence=0.99, discount=0.999
    )
    fast = TradingPolicy.solve(
        risk_aversion=1.0, trading_cost=20.0, signal_persistence=0.50, discount=0.999
    )
    assert fast.aim_scaling < slow.aim_scaling
    assert fast.trade_rate == pytest.approx(slow.trade_rate), (
        "the trade rate depends on cost, not on how fast the signal decays"
    )


def test_a_higher_cost_slows_the_approach() -> None:
    rates = [
        TradingPolicy.solve(
            risk_aversion=1.0, trading_cost=cost, signal_persistence=0.9, discount=0.999
        ).trade_rate
        for cost in (0.5, 5.0, 50.0, 500.0)
    ]
    assert rates == sorted(rates, reverse=True)


def test_the_policy_trades_partway_to_the_aim_not_to_markowitz() -> None:
    matrix = covariance(6)
    mu = np.linspace(0.0002, 0.0008, 6)
    policy = TradingPolicy.solve(
        risk_aversion=10.0, trading_cost=20.0, signal_persistence=0.9, discount=0.999
    )
    trade = policy.step(np.zeros(6), mu, matrix)

    assert np.abs(trade.aim).sum() < np.abs(trade.markowitz).sum(), "aim sits short of Markowitz"
    assert np.abs(trade.target).sum() < np.abs(trade.aim).sum(), "and we only go partway"
    assert trade.aim_shrinkage > 0


def test_repeated_steps_converge_toward_the_aim() -> None:
    matrix = covariance(5)
    mu = np.linspace(0.0002, 0.0006, 5)
    policy = TradingPolicy.solve(
        risk_aversion=10.0, trading_cost=10.0, signal_persistence=0.95, discount=0.999
    )
    position = np.zeros(5)
    distances = []
    for _ in range(30):
        trade = policy.step(position, mu, matrix)
        position = trade.target
        distances.append(float(np.abs(trade.aim - position).sum()))
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] < 0.05 * distances[0]


def test_turnover_falls_as_the_position_approaches_the_aim() -> None:
    matrix = covariance(5)
    mu = np.linspace(0.0002, 0.0006, 5)
    policy = TradingPolicy.solve(risk_aversion=10.0, trading_cost=10.0, signal_persistence=0.95)
    position = np.zeros(5)
    turnovers = []
    for _ in range(10):
        trade = policy.step(position, mu, matrix)
        turnovers.append(trade.turnover)
        position = trade.target
    assert turnovers == sorted(turnovers, reverse=True)


def test_a_singular_covariance_has_no_markowitz_portfolio() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="singular"):
        markowitz_portfolio(np.array([0.1, 0.1]), singular, 1.0)


def test_invalid_policy_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="risk_aversion"):
        TradingPolicy.solve(risk_aversion=0.0, trading_cost=1.0, signal_persistence=0.9)
    with pytest.raises(ValueError, match="signal_persistence"):
        TradingPolicy.solve(risk_aversion=1.0, trading_cost=1.0, signal_persistence=1.5)
    with pytest.raises(ValueError, match="discount"):
        TradingPolicy.solve(
            risk_aversion=1.0, trading_cost=1.0, signal_persistence=0.9, discount=0.0
        )


def test_the_policy_describes_itself() -> None:
    policy = TradingPolicy.solve(
        risk_aversion=1.0, trading_cost=10.0, signal_persistence=0.9, discount=0.999
    )
    text = policy.describe()
    assert "half-life" in text and "Markowitz" in text
