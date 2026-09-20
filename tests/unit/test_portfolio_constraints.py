"""Position limits, and what happens when they cannot all be met.

The class exists to stop an optimiser doing something stupid, so the tests are
mostly about the cases where it would: an infeasible problem, a group cap that
bites, a long-only book whose gross limit is redundant.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.portfolio.constraints import Constraints, GroupLimit, build_groups


# ------------------------------------------------------------------ validation --
def test_a_group_must_constrain_something() -> None:
    with pytest.raises(ValueError, match="constrains nothing"):
        GroupLimit(name="tech", members=(0, 1))
    with pytest.raises(ValueError, match="no members"):
        GroupLimit(name="tech", members=(), max_net=0.3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_position": -0.1}, "max_position must be positive"),
        ({"max_gross": 0.0}, "max_gross must be positive"),
        ({"min_position": 0.5, "max_position": 0.1}, "cannot exceed"),
        ({"net_range": (1.0, 0.0)}, r"net_range must be \(low, high\)"),
    ],
)
def test_incoherent_limits_are_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Constraints(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- bounds --
def test_a_symmetric_box_is_implied_by_max_position_alone() -> None:
    """Without a floor, a position cap bounds the short side too. Leaving the
    lower bound open would let the optimiser take an unlimited short."""
    assert Constraints(max_position=0.2).bounds(3) == [(-0.2, 0.2)] * 3


def test_a_floor_of_zero_makes_the_book_long_only() -> None:
    assert Constraints(max_position=0.2, min_position=0.0).bounds(2) == [(0.0, 0.2)] * 2


def test_clip_forces_a_vector_into_the_box() -> None:
    limits = Constraints(max_position=0.1, min_position=0.0)
    assert np.allclose(limits.clip(np.array([0.5, -0.3, 0.05])), [0.1, 0.0, 0.05])


# --------------------------------------------------------------- the presets --
def test_long_only_is_fully_invested_and_never_short() -> None:
    limits = Constraints.long_only()
    assert limits.min_position == 0.0
    assert limits.net_range == (1.0, 1.0)
    assert limits.violations(np.full(10, 0.1)) == []


def test_dollar_neutral_forces_the_legs_to_offset() -> None:
    limits = Constraints.dollar_neutral()
    weights = np.array([0.05, 0.05, -0.05, -0.05])
    assert limits.violations(weights) == []
    assert "dollar neutral" in limits.describe()
    assert limits.violations(np.array([0.05, 0.05, 0.05, -0.05])) != []


def test_unconstrained_permits_anything() -> None:
    limits = Constraints.unconstrained()
    assert limits.violations(np.array([50.0, -80.0])) == []
    assert limits.describe() == "unconstrained"
    assert limits.bounds(2) == [(None, None)] * 2


# ------------------------------------------------------------------ feasibility --
def test_a_fully_invested_book_needs_enough_names() -> None:
    """Ten names at a 10% cap can just reach 100%; nine cannot."""
    Constraints.long_only().check_feasible(10)
    with pytest.raises(ValueError, match="unreachable"):
        Constraints.long_only().check_feasible(9)


def test_a_net_target_below_what_the_floor_allows_is_rejected() -> None:
    """Three positions each at least 20% sum to at least 60%, so a net target of
    at most 50% cannot be met."""
    limits = Constraints(max_position=0.9, min_position=0.2, net_range=(0.0, 0.5))
    with pytest.raises(ValueError, match="unreachable"):
        limits.check_feasible(3)


def test_a_net_target_outside_the_gross_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot fit inside a gross limit"):
        Constraints(max_position=1.0, max_gross=0.5, net_range=(1.0, 1.0)).check_feasible(10)


def test_feasibility_needs_at_least_one_asset() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        Constraints().check_feasible(0)


def test_a_dollar_neutral_book_is_feasible_at_any_size() -> None:
    for n in (2, 10, 500):
        Constraints.dollar_neutral().check_feasible(n)


# ------------------------------------------------------------------ violations --
def test_every_broken_limit_is_named() -> None:
    limits = Constraints(max_position=0.1, min_position=0.0, max_gross=0.5, net_range=(0.0, 0.2))
    reported = limits.violations(np.array([0.4, -0.3, 0.2]))

    assert len(reported) == 4
    assert any("largest position" in r for r in reported)
    assert any("smallest position" in r for r in reported)
    assert any("gross exposure" in r for r in reported)
    assert any("net exposure" in r for r in reported)


def test_a_small_numerical_breach_is_tolerated() -> None:
    """SLSQP satisfies constraints to a tolerance, not exactly. Reporting every
    1e-9 overshoot would bury the breaches that matter."""
    limits = Constraints(max_position=0.1, net_range=(1.0, 1.0))
    assert limits.violations(np.full(10, 0.1 + 1e-9)) == []


def test_group_caps_bite_on_the_members_only() -> None:
    groups = (
        GroupLimit(name="tech", members=(0, 1), max_net=0.15),
        GroupLimit(name="energy", members=(2, 3), max_gross=0.10),
    )
    limits = Constraints(max_position=0.5, max_gross=None, groups=groups)

    assert limits.violations(np.array([0.05, 0.05, 0.03, -0.03])) == []

    reported = limits.violations(np.array([0.2, 0.2, 0.2, -0.2]))
    assert any("'tech' net" in r for r in reported)
    assert any("'energy' gross" in r for r in reported)


def test_a_group_net_cap_is_on_the_absolute_net() -> None:
    """A -40% net sector bet is as much a sector bet as a +40% one."""
    limits = Constraints(
        max_position=0.5, max_gross=None, groups=(GroupLimit("tech", (0, 1), max_net=0.1),)
    )
    assert limits.violations(np.array([-0.3, -0.3])) != []


# -------------------------------------------------------------------- plumbing --
def test_the_redundant_gross_constraint_is_skipped_for_a_long_only_book() -> None:
    """With every weight non-negative the sum of weights *is* the sum of absolute
    weights, so imposing both adds a non-smooth constraint for nothing."""
    long_only = Constraints.long_only()
    assert long_only._gross_is_implied()
    kinds = [c["type"] for c in long_only.scipy_constraints(10)]
    assert kinds.count("eq") == 1
    assert "ineq" not in kinds

    # A book that can go short needs the gross limit stated explicitly.
    assert not Constraints.dollar_neutral()._gross_is_implied()
    assert "ineq" in [c["type"] for c in Constraints.dollar_neutral().scipy_constraints(10)]


def test_scipy_constraints_evaluate_correctly() -> None:
    limits = Constraints(max_position=0.5, max_gross=1.5, net_range=(0.0, 0.0))
    weights = np.array([0.4, -0.4, 0.2, -0.2])
    for constraint in limits.scipy_constraints(4):
        value = np.atleast_1d(constraint["fun"](weights))
        if constraint["type"] == "eq":
            assert np.allclose(value, 0.0, atol=1e-9)
        else:
            assert np.all(value >= -1e-9)


def test_describe_lists_what_is_imposed() -> None:
    text = Constraints(
        max_position=0.1,
        min_position=0.0,
        max_gross=1.0,
        net_range=(0.5, 1.0),
        groups=(GroupLimit("tech", (0,), max_net=0.2),),
    ).describe()
    assert "position <= 10.0%" in text
    assert "gross <= 1.0x" in text
    assert "net in [0.5, 1.0]" in text
    assert "tech capped" in text


# ----------------------------------------------------------------- build_groups --
def test_groups_are_built_from_a_label_column() -> None:
    groups = build_groups(["tech", "energy", "tech", "banks"], max_net=0.25)

    assert [g.name for g in groups] == ["banks", "energy", "tech"]  # sorted, so deterministic
    assert {g.name: g.members for g in groups}["tech"] == (0, 2)
    assert all(g.max_net == 0.25 for g in groups)
    assert all(g.max_gross is None for g in groups)


def test_building_groups_with_no_cap_is_an_error() -> None:
    with pytest.raises(ValueError, match="constrains nothing"):
        build_groups(["tech", "energy"])
