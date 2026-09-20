"""Capacity: the AUM at which a strategy's own trading eats its edge.

The headline assertions are the two the spec names: capacity scales with roughly
the **square** of alpha, and a strategy without a capacity number is not finished.
The closed form exists so the solver is checked against algebra rather than
against itself.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quantlab.costs.capacity import (
    CapacityModel,
    UniverseLiquidity,
    analytic_break_even_single_name,
)

ADV = 50e6
VOL = 0.02
SPREAD = 5.0


def universe(names: int = 100, **kwargs: float) -> UniverseLiquidity:
    return UniverseLiquidity.equal_weight(
        [f"S{i}" for i in range(names)],
        adv_notional=kwargs.get("adv_notional", ADV),
        volatility_daily=kwargs.get("volatility_daily", VOL),
        spread_bps=kwargs.get("spread_bps", SPREAD),
    )


def solve(alpha: float = 40.0, **kwargs: float):
    return CapacityModel().solve(
        kwargs.pop("universe", None) or universe(),  # type: ignore[arg-type]
        gross_alpha_bps_per_rebalance=alpha,
        turnover_per_rebalance=kwargs.pop("turnover", 0.30),  # type: ignore[arg-type]
        rebalances_per_year=kwargs.pop("rebalances", 12.0),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------------------
# Correctness against the closed form
# ----------------------------------------------------------------------------------
def test_solver_matches_the_closed_form_for_one_name() -> None:
    """Checks the numerics against algebra, not against a previous run."""
    analytic = analytic_break_even_single_name(
        gross_alpha_bps_per_rebalance=40.0,
        turnover_per_rebalance=0.3,
        adv_notional=ADV,
        volatility_daily=VOL,
        spread_bps=SPREAD,
    )
    numeric = solve(universe=universe(names=1)).break_even_aum
    assert analytic is not None and numeric is not None
    assert numeric == pytest.approx(analytic, rel=1e-6)


def test_net_alpha_is_zero_at_the_break_even() -> None:
    model = CapacityModel()
    result = solve()
    assert result.break_even_aum is not None
    net = model.net_alpha_bps_annual(
        result.break_even_aum,
        universe(),
        gross_alpha_bps_per_rebalance=40.0,
        turnover_per_rebalance=0.30,
        rebalances_per_year=12.0,
    )
    assert net == pytest.approx(0.0, abs=1e-6)


def test_net_alpha_declines_monotonically_with_size() -> None:
    model = CapacityModel()
    nets = [
        model.net_alpha_bps_annual(
            aum,
            universe(),
            gross_alpha_bps_per_rebalance=40.0,
            turnover_per_rebalance=0.30,
            rebalances_per_year=12.0,
        )
        for aum in (1e6, 1e7, 1e8, 1e9, 1e10)
    ]
    assert nets == sorted(nets, reverse=True)


# ----------------------------------------------------------------------------------
# The scaling laws the spec names
# ----------------------------------------------------------------------------------
def test_capacity_scales_with_roughly_the_square_of_alpha() -> None:
    """Spec section 5. Cost grows like Q^1.5 and gross alpha at best linearly, so
    with a square-root law the break-even moves with alpha squared. With a fixed
    spread to overcome first, the ratio is slightly more than four."""
    small = solve(alpha=20.0).break_even_aum
    large = solve(alpha=40.0).break_even_aum
    assert small is not None and large is not None
    assert 4.0 <= large / small <= 4.6


def test_zero_spread_gives_exactly_the_square_law() -> None:
    """With no fixed cost the relationship is clean, which isolates the impact
    exponent from the spread term."""
    frictionless = universe(spread_bps=0.0)
    small = solve(alpha=20.0, universe=frictionless).break_even_aum
    large = solve(alpha=40.0, universe=frictionless).break_even_aum
    assert small is not None and large is not None
    assert large / small == pytest.approx(4.0, rel=0.01)


def test_more_liquidity_means_more_capacity() -> None:
    thin = solve(universe=universe(adv_notional=10e6)).break_even_aum
    deep = solve(universe=universe(adv_notional=500e6)).break_even_aum
    assert thin is not None and deep is not None
    assert deep > thin


def test_higher_turnover_reduces_capacity() -> None:
    slow = solve(turnover=0.20).break_even_aum
    medium = solve(turnover=0.30).break_even_aum
    fast = solve(turnover=0.80).break_even_aum
    assert slow is not None and medium is not None and fast is not None
    assert slow > medium > fast


def test_capacity_beyond_the_search_bracket_is_reported_not_extrapolated() -> None:
    """A very low turnover strategy on deep names has a break-even beyond any
    plausible AUM. Returning a number there would be extrapolation; in practice it
    means the liquidity inputs are wrong, so it is reported as such."""
    result = solve(turnover=0.05)
    assert result.break_even_aum is None
    assert "beyond the search bracket" in result.note


def test_trading_more_slowly_raises_capacity() -> None:
    """The one lever execution has: temporary impact responds to urgency."""
    urgent = solve(horizon_days=0.5).break_even_aum
    patient = solve(horizon_days=5.0).break_even_aum
    assert urgent is not None and patient is not None
    assert patient > urgent


def test_holding_costs_reduce_capacity() -> None:
    free = solve().break_even_aum
    with_borrow = solve(holding_cost_bps_annual=100.0).break_even_aum
    assert free is not None and with_borrow is not None
    assert with_borrow < free


# ----------------------------------------------------------------------------------
# Honesty about the answer
# ----------------------------------------------------------------------------------
def test_a_strategy_that_never_pays_reports_no_capacity_not_a_small_one() -> None:
    """Reporting a tiny break-even would imply the strategy works if kept small.
    It does not: the spread alone exceeds the edge at every size."""
    result = solve(alpha=0.5)
    assert result.break_even_aum is None
    assert not result.is_viable
    assert result.net_alpha_bps_at_zero < 0
    assert "the signal does not pay" in result.note
    assert "no viable capacity" in result.summary()


def test_capacity_beyond_the_calibrated_range_is_labelled_an_upper_bound() -> None:
    """The square-root law is fitted on orders below ~10% of ADV. A capacity that
    requires 70% of a name's daily volume is extrapolation, and the law understates
    impact there -- so the number is a ceiling, not an estimate."""
    result = solve(alpha=40.0)
    assert result.extrapolated
    assert result.max_participation > 0.10
    assert "UPPER BOUND ONLY" in result.summary()


def test_small_capacity_within_the_calibrated_range_is_not_flagged() -> None:
    result = solve(alpha=40.0, universe=universe(adv_notional=50e9))
    assert not result.extrapolated
    assert "UPPER BOUND" not in result.summary()


def test_binding_constraint_is_identified() -> None:
    assert "spread" in solve(universe=universe(spread_bps=50.0)).binding_constraint
    assert "holding" in solve(holding_cost_bps_annual=500.0).binding_constraint


def test_curve_is_produced_for_the_tearsheet() -> None:
    """Spec section 9 requires the capacity estimate on every tearsheet; the curve
    is what makes it legible rather than a single unexplained number."""
    curve = solve().curve
    assert set(curve.columns) == {"aum", "gross_bps_annual", "cost_bps_annual", "net_bps_annual"}
    assert curve.height > 10
    assert curve["cost_bps_annual"].to_list() == sorted(curve["cost_bps_annual"].to_list())
    assert curve["net_bps_annual"].to_list() == sorted(
        curve["net_bps_annual"].to_list(), reverse=True
    )


def test_summary_is_readable() -> None:
    assert "break-even AUM $" in solve().summary()


# ----------------------------------------------------------------------------------
# Universe construction
# ----------------------------------------------------------------------------------
def test_splitting_across_identical_names_is_cost_neutral() -> None:
    """Impact depends on participation, not absolute size: a hundred names at 1%
    of ADV each costs exactly what one name at 1% costs. Worth pinning, because
    the intuition that diversification is itself cheap is wrong."""
    model = CapacityModel()
    split = model.cost_bps_of_aum(1e9, universe(names=100), turnover_per_rebalance=0.3)
    single = model.cost_bps_of_aum(
        1e9, universe(names=1, adv_notional=ADV * 100), turnover_per_rebalance=0.3
    )
    assert split == pytest.approx(single, rel=1e-9)


def test_a_heterogeneous_universe_costs_far_more_than_its_aggregate_adv_suggests() -> None:
    """This is why capacity is computed name by name. Because impact is concave in
    participation, a universe's thin names cost disproportionately more, and
    collapsing the universe into one deep instrument makes them vanish."""
    model = CapacityModel()
    mixed = UniverseLiquidity(
        pl.DataFrame(
            {
                "symbol": ["DEEP", "THIN"],
                "weight": [0.5, 0.5],
                "adv_notional": [4.9e9, 100e6],
                "volatility_daily": [VOL, VOL],
                "spread_bps": [SPREAD, SPREAD],
            }
        )
    )
    heterogeneous = model.cost_bps_of_aum(1e9, mixed, turnover_per_rebalance=0.3)
    homogeneous = model.cost_bps_of_aum(
        1e9, universe(names=2, adv_notional=2.5e9), turnover_per_rebalance=0.3
    )
    assert heterogeneous > 2 * homogeneous


def test_least_liquid_name_binds_first() -> None:
    """A universe's worst name hits the extrapolation range long before the
    aggregate does, which is why an equal-weight approximation is optimistic."""
    mixed = UniverseLiquidity(
        pl.DataFrame(
            {
                "symbol": ["DEEP", "THIN"],
                "weight": [0.5, 0.5],
                "adv_notional": [1e10, 1e6],
                "volatility_daily": [VOL, VOL],
                "spread_bps": [SPREAD, SPREAD],
            }
        )
    )
    model = CapacityModel()
    assert model.max_participation(1e8, mixed, turnover_per_rebalance=0.3) == pytest.approx(
        1e8 * 0.3 * 0.5 / 1e6
    )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="weights must sum to 1"):
        UniverseLiquidity(
            pl.DataFrame(
                {
                    "symbol": ["A", "B"],
                    "weight": [0.3, 0.3],
                    "adv_notional": [ADV, ADV],
                    "volatility_daily": [VOL, VOL],
                    "spread_bps": [SPREAD, SPREAD],
                }
            )
        )


def test_zero_volume_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="no measurable capacity"):
        UniverseLiquidity(
            pl.DataFrame(
                {
                    "symbol": ["A"],
                    "weight": [1.0],
                    "adv_notional": [0.0],
                    "volatility_daily": [VOL],
                    "spread_bps": [SPREAD],
                }
            )
        )


def test_missing_columns_are_named() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        UniverseLiquidity(pl.DataFrame({"symbol": ["A"], "weight": [1.0]}))


def test_empty_universe_has_no_capacity() -> None:
    schema = {
        "symbol": pl.Utf8(),
        "weight": pl.Float64(),
        "adv_notional": pl.Float64(),
        "volatility_daily": pl.Float64(),
        "spread_bps": pl.Float64(),
    }
    with pytest.raises(ValueError, match="empty"):
        UniverseLiquidity(pl.DataFrame(schema=schema))


def test_zero_turnover_costs_nothing() -> None:
    assert CapacityModel().cost_bps_of_aum(1e9, universe(), turnover_per_rebalance=0.0) == 0.0


def test_cost_of_aum_rejects_nonsense() -> None:
    model = CapacityModel()
    with pytest.raises(ValueError, match="aum must be positive"):
        model.cost_bps_of_aum(0.0, universe(), turnover_per_rebalance=0.3)
    with pytest.raises(ValueError, match="non-negative"):
        model.cost_bps_of_aum(1e9, universe(), turnover_per_rebalance=-0.1)


def test_analytic_returns_none_when_fixed_costs_already_exceed_alpha() -> None:
    assert (
        analytic_break_even_single_name(
            gross_alpha_bps_per_rebalance=1.0,
            turnover_per_rebalance=1.0,
            adv_notional=ADV,
            volatility_daily=VOL,
            spread_bps=50.0,
        )
        is None
    )


def test_capacity_is_deterministic() -> None:
    """Spec section 1: a given commit and snapshot must produce identical results."""
    first = solve().break_even_aum
    second = solve().break_even_aum
    assert first == second
    assert np.isfinite(first)  # type: ignore[arg-type]
