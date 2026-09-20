"""Dynamic trading with transaction costs: the Gârleanu-Pedersen framework.

Spec section 8 is specific about this: turnover penalties belong **inside** the
optimiser, not as a post-hoc filter. The difference is not stylistic. A post-hoc
filter -- rebalance monthly instead of daily, or skip trades below a threshold --
throws away information without asking what it was worth. Putting the cost inside
the objective asks the right question: *given that this signal will decay anyway,
how much of the way toward it is worth paying to travel?*

The answer has a shape worth internalising. The optimal policy is **not** "trade
to the Markowitz portfolio slowly". It is:

1. Compute an **aim** portfolio, which is a weighted average of the Markowitz
   portfolios you expect in future. A fast-decaying signal pulls the aim back
   toward zero, because by the time you arrive the opportunity has gone.
2. Trade a **constant fraction** of the way from where you are to the aim. Not to
   the Markowitz portfolio -- to the aim.

So a costly-to-trade, fast-decaying signal gets both a smaller target *and* a
slower approach to it, which is why naive turnover filters underperform: they
only do the second.

The derivation, for the tractable case where the cost matrix is proportional to
covariance (``Λ = λΣ``). Working in risk units the problem decouples, and each
scalar problem is::

    max  Σ ρᵗ [ yₜ·aₜ − (γ/2)·yₜ² − (λ/2)·(yₜ − yₜ₋₁)² ]

with the signal decaying as ``aₜ₊₁ = φ·aₜ``. Guessing a linear policy
``yₜ = A·yₜ₋₁ + B·aₜ`` and matching coefficients in the first-order condition
gives::

    ρλA² − A(γ + λ(1+ρ)) + λ = 0        (take the root in (0, 1))
    B = A / (λ(1 − ρφA))

from which the trade rate is ``1 − A`` and the aim is ``B·aₜ`` divided by that
rate. The limits are the sanity check, and they are asserted by test: no cost
means trade straight to Markowitz; infinite cost means do not trade; a permanent
signal makes the aim the Markowitz portfolio; an instantly-decaying one collapses
it toward zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quantlab.logging import get_logger

__all__ = ["DynamicTrade", "TradingPolicy", "markowitz_portfolio"]

log = get_logger("quantlab.portfolio.dynamic")


def markowitz_portfolio(
    expected_returns: np.ndarray, covariance: np.ndarray, risk_aversion: float
) -> np.ndarray:
    """``(γΣ)⁻¹μ`` -- the costless optimum, and the thing you never actually hold."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    mu = np.asarray(expected_returns, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    try:
        return np.linalg.solve(risk_aversion * matrix, mu)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "covariance is singular, so the Markowitz portfolio is undefined. "
            "Shrink the estimate before optimising against it."
        ) from exc


@dataclass(frozen=True, slots=True)
class DynamicTrade:
    """One period's decision, with the reasoning kept."""

    #: Where to be at the end of this period.
    target: np.ndarray
    #: The change from the current position.
    trade: np.ndarray
    #: The slowly-moving portfolio being traded toward.
    aim: np.ndarray
    #: The costless optimum, for comparison.
    markowitz: np.ndarray
    #: Fraction of the distance to the aim covered this period.
    trade_rate: float

    @property
    def turnover(self) -> float:
        return float(np.abs(self.trade).sum())

    @property
    def aim_shrinkage(self) -> float:
        """How far the aim sits from Markowitz, as a fraction.

        Zero means the aim *is* the Markowitz portfolio -- a permanent signal with
        no cost. Approaching one means the signal decays so fast, or trading costs
        so much, that chasing it is not worth doing.
        """
        scale = float(np.abs(self.markowitz).sum())
        if scale <= 0:
            return 0.0
        return 1.0 - float(np.abs(self.aim).sum()) / scale


@dataclass(frozen=True, slots=True)
class TradingPolicy:
    """The solved policy: a trade rate and an aim scaling, both constant.

    Both depend only on the parameters, not on the current position or signal, so
    they are solved once and applied every period. That is the practical appeal of
    the framework -- the hard part is a scalar quadratic, not a per-period
    optimisation.
    """

    risk_aversion: float
    #: Cost parameter in ``Λ = λΣ``. Higher means trading is more expensive.
    trading_cost: float
    #: Per-period persistence of the signal. ``0.97`` daily is a ~23-day half-life.
    signal_persistence: float
    discount: float
    trade_rate: float
    aim_scaling: float

    @classmethod
    def solve(
        cls,
        *,
        risk_aversion: float,
        trading_cost: float,
        signal_persistence: float,
        discount: float = 1.0,
    ) -> TradingPolicy:
        """Solve the scalar quadratic for the trade rate and aim scaling."""
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if trading_cost < 0:
            raise ValueError("trading_cost must be non-negative")
        if not 0.0 <= signal_persistence <= 1.0:
            raise ValueError("signal_persistence must be in [0, 1]")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")

        gamma, lam, phi, rho = (
            risk_aversion,
            trading_cost,
            signal_persistence,
            discount,
        )

        if lam == 0.0:
            # Costless: trade straight to the Markowitz portfolio every period.
            return cls(gamma, lam, phi, rho, trade_rate=1.0, aim_scaling=1.0)

        # rho*lam*A^2 - A*(gamma + lam*(1+rho)) + lam = 0, stable root in (0, 1).
        b = gamma + lam * (1.0 + rho)
        if rho * lam == 0.0:
            persistence = lam / b
        else:
            discriminant = b * b - 4.0 * rho * lam * lam
            persistence = (b - math.sqrt(max(discriminant, 0.0))) / (2.0 * rho * lam)
        persistence = float(np.clip(persistence, 0.0, 1.0 - 1e-15))

        trade_rate = 1.0 - persistence
        denominator = lam * (1.0 - rho * phi * persistence)
        response = persistence / denominator if denominator > 0 else 0.0

        # y = A*y_prev + B*a  ==  (1 - rate)*y_prev + rate*aim, so aim = B*a/rate.
        # In Markowitz units (M = a/gamma) the aim is M scaled by gamma*B/rate.
        aim_scaling = gamma * response / trade_rate if trade_rate > 0 else 0.0

        return cls(
            risk_aversion=gamma,
            trading_cost=lam,
            signal_persistence=phi,
            discount=rho,
            trade_rate=float(trade_rate),
            aim_scaling=float(aim_scaling),
        )

    @property
    def half_life_periods(self) -> float:
        """Periods to close half the distance to the aim."""
        if self.trade_rate >= 1.0:
            return 0.0
        if self.trade_rate <= 0.0:
            return float("inf")
        return float(math.log(0.5) / math.log(1.0 - self.trade_rate))

    def step(
        self,
        current: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> DynamicTrade:
        """One period's trade, from where you are toward where you are going."""
        position = np.asarray(current, dtype=float)
        markowitz = markowitz_portfolio(expected_returns, covariance, self.risk_aversion)
        aim = self.aim_scaling * markowitz
        target = position + self.trade_rate * (aim - position)
        return DynamicTrade(
            target=target,
            trade=target - position,
            aim=aim,
            markowitz=markowitz,
            trade_rate=self.trade_rate,
        )

    def describe(self) -> str:
        return (
            f"trade {self.trade_rate:.1%} of the way to the aim each period "
            f"(half-life {self.half_life_periods:.1f} periods); the aim is "
            f"{self.aim_scaling:.1%} of the Markowitz portfolio"
        )
