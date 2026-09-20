"""Market impact.

Three models, in increasing order of detail:

:class:`SquareRootImpact`
    The headline law, ``impact = Y · σ_daily · (Q/ADV)^δ``. It is one of the most
    robust empirical regularities in market microstructure, holding across
    equities, futures, FX and crypto over several orders of magnitude of size. Its
    practical consequence is the reason capacity is finite: **doubling your impact
    requires roughly quadrupling your size**, so cost in currency grows like
    ``Q^1.5`` while gross alpha grows at best linearly in ``Q``.

:class:`AlmgrenChriss`
    Optimal execution scheduling. Answers "how fast should this be worked?" by
    trading off temporary impact (worse when fast) against price risk (worse when
    slow), and produces the efficient frontier of that trade-off.

:class:`PropagatorImpact`
    A transient-impact model in which each child order displaces the price and that
    displacement decays as a power law. Needed by the event-driven engine, where
    the path matters and not just the average. Its aggregate behaviour reproduces
    the square-root law, which is a useful consistency check between the two.

All three take impact as a fraction of price and convert to basis points at the
boundary, so the 1e4 factors live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quantlab.costs.base import BPS
from quantlab.logging import get_logger

__all__ = [
    "AlmgrenChriss",
    "ExecutionSchedule",
    "ImpactParams",
    "PropagatorImpact",
    "SquareRootImpact",
]

log = get_logger("quantlab.costs.impact")


@dataclass(frozen=True, slots=True)
class ImpactParams:
    """Calibration of the square-root law.

    Defaults are deliberately toward the punitive end of the published range. An
    impact model that is too gentle produces a strategy that looks deployable and
    is not; one that is too harsh produces a strategy that looks undeployable and
    might be. The first error costs money, the second costs an opportunity, and
    this platform is built to prefer the second.
    """

    #: ``Y`` in the law. Published estimates cluster in 0.5-1.0 depending on the
    #: market and the period. 0.5 with 2% daily vol puts a 1%-of-ADV order at
    #: about 10 bp, which matches practitioner rules of thumb.
    y: float = 0.5
    #: ``δ``. Empirical estimates range roughly 0.4-0.7; 0.5 is the canonical
    #: square root and the value most often replicated.
    delta: float = 0.5
    #: Fraction of the peak displacement that is permanent. Published
    #: decompositions put this between a third and two thirds; 0.5 is the midpoint
    #: and the value to revisit first when calibrating against real fills.
    permanent_fraction: float = 0.5
    #: How temporary impact scales with urgency. Working an order in half a day
    #: raises the trade rate, and temporary impact responds to rate. Defaults to
    #: ``delta`` so a full-day execution reproduces the headline law exactly.
    urgency_exponent: float | None = None
    #: Participation above which the law is extrapolation rather than calibration.
    #: Published fits are dominated by orders below ~10% of ADV; beyond that the
    #: law tends to *understate*, so crossing it is flagged rather than silently
    #: trusted.
    max_reliable_participation: float = 0.10

    def __post_init__(self) -> None:
        if self.y <= 0:
            raise ValueError("y must be positive")
        if not 0 < self.delta <= 1:
            raise ValueError("delta must be in (0, 1]")
        if not 0 <= self.permanent_fraction <= 1:
            raise ValueError("permanent_fraction must be in [0, 1]")

    @property
    def urgency(self) -> float:
        return self.delta if self.urgency_exponent is None else self.urgency_exponent


class SquareRootImpact:
    """``impact = Y · σ_daily · (Q/ADV)^δ``, split into permanent and temporary.

    The split matters for accounting, not just description. Permanent impact
    accumulates as the order is worked, so on average only **half** of it is paid
    by the order causing it -- the early fills execute before most of it has
    happened. Temporary impact is paid in full and then reverts, which is why it
    is the part that responds to trading more slowly.
    """

    def __init__(self, params: ImpactParams | None = None) -> None:
        self.params = params or ImpactParams()

    # ------------------------------------------------------------------ scalar --
    def participation(self, notional: float, adv_notional: float) -> float:
        if adv_notional <= 0:
            raise ValueError("adv_notional must be positive")
        return notional / adv_notional

    def peak_impact_bps(
        self, notional: float, adv_notional: float, volatility_daily: float
    ) -> float:
        """Total price displacement caused by the order, in basis points."""
        rate = self.participation(notional, adv_notional)
        if rate == 0:
            return 0.0
        return float(self.params.y * volatility_daily * rate**self.params.delta / BPS)

    def split(
        self,
        notional: float,
        adv_notional: float,
        volatility_daily: float,
        horizon_days: float = 1.0,
    ) -> tuple[float, float]:
        """``(permanent_bps, temporary_bps)`` for this order.

        Permanent impact is independent of how fast the order is worked: it
        reflects the information and inventory the trade reveals, not the rate.
        Temporary impact scales with urgency, which is the lever execution has.
        """
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        peak = self.peak_impact_bps(notional, adv_notional, volatility_daily)
        permanent = self.params.permanent_fraction * peak
        temporary = (1.0 - self.params.permanent_fraction) * peak
        temporary *= (1.0 / horizon_days) ** self.params.urgency
        return permanent, temporary

    def cost_bps(
        self,
        notional: float,
        adv_notional: float,
        volatility_daily: float,
        horizon_days: float = 1.0,
    ) -> float:
        """Impact actually paid: temporary in full, half the permanent."""
        permanent, temporary = self.split(notional, adv_notional, volatility_daily, horizon_days)
        return temporary + permanent / 2.0

    def is_extrapolating(self, notional: float, adv_notional: float) -> bool:
        """Whether this order is beyond the range the law was calibrated on."""
        return self.participation(notional, adv_notional) > self.params.max_reliable_participation

    # -------------------------------------------------------------- vectorised --
    def cost_bps_expr(
        self,
        notional: pl.Expr,
        adv_notional: pl.Expr,
        volatility_daily: pl.Expr,
        horizon_days: pl.Expr | float = 1.0,
    ) -> pl.Expr:
        """Polars expression matching :meth:`cost_bps`, for the vectorised engine."""
        params = self.params
        horizon = (
            pl.lit(float(horizon_days)) if isinstance(horizon_days, int | float) else horizon_days
        )
        rate = notional / adv_notional
        peak = pl.lit(params.y) * volatility_daily * rate.pow(params.delta) / pl.lit(BPS)
        permanent = pl.lit(params.permanent_fraction) * peak
        temporary = (
            (pl.lit(1.0) - pl.lit(params.permanent_fraction))
            * peak
            * (pl.lit(1.0) / horizon).pow(params.urgency)
        )
        return temporary + permanent / pl.lit(2.0)


# ======================================================================================
# Optimal execution
# ======================================================================================
@dataclass(frozen=True, slots=True)
class ExecutionSchedule:
    """The output of an Almgren-Chriss solve.

    Quantities are **participation units** -- fractions of average daily volume --
    not shares or currency. Keeping the whole problem dimensionless is what makes
    the impact coefficients comparable with the square-root law and the costs
    directly expressible in basis points.
    """

    #: Participation remaining at the end of each period, starting with the whole order.
    holdings: np.ndarray
    #: Participation traded in each period. Sums to the original order.
    trades: np.ndarray
    #: Length of one period, in days.
    period_days: float
    #: Expected implementation shortfall, in basis points of traded notional.
    expected_cost_bps: float
    #: Variance of that shortfall, in (basis points)^2.
    cost_variance_bps2: float
    #: Trajectory urgency. Zero is a linear (TWAP) schedule; larger front-loads.
    kappa: float

    @property
    def horizon_days(self) -> float:
        return self.period_days * len(self.trades)

    @property
    def cost_stdev_bps(self) -> float:
        return float(np.sqrt(max(self.cost_variance_bps2, 0.0)))

    def half_life_days(self) -> float:
        """Time by which half the order has been executed."""
        executed = np.cumsum(self.trades)
        target = executed[-1] / 2.0
        index = int(np.searchsorted(executed, target))
        return self.period_days * (index + 1)

    @property
    def is_twap(self) -> bool:
        return bool(np.allclose(self.trades, self.trades[0], rtol=1e-6))


class AlmgrenChriss:
    """Optimal execution under linear permanent and linear temporary impact.

    The trade-off: trading fast pays temporary impact; trading slowly leaves the
    unexecuted remainder exposed to price risk. ``risk_aversion`` prices that
    exposure. At zero the solution is a linear (TWAP) schedule; as it rises the
    schedule front-loads, paying more impact to carry less risk.

    **Units.** The whole problem is dimensionless: quantities are participation
    (fraction of ADV) and impact coefficients are price-fractions per unit
    participation rate. This is what lets :meth:`from_square_root` derive the
    coefficients from the calibrated empirical law instead of requiring a second,
    independent calibration -- and it makes the risk-neutral solution reproduce
    :meth:`SquareRootImpact.cost_bps` exactly, which is asserted by test.

    ``risk_aversion`` is therefore in those units too, where meaningful values are
    of order 1 to 10,000 rather than the tiny numbers seen in share-based
    formulations. The trajectory's curvature is ``κ² = λσ²/η̃``, so if a schedule
    comes back as TWAP the aversion is small relative to the impact scale.
    """

    def __init__(
        self,
        *,
        volatility_daily: float,
        gamma: float,
        eta: float,
        epsilon: float = 0.0,
        risk_aversion: float = 0.0,
    ) -> None:
        if eta <= 0:
            raise ValueError("eta (temporary impact coefficient) must be positive")
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative")
        self.sigma = volatility_daily
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.risk_aversion = risk_aversion

    @classmethod
    def from_square_root(
        cls,
        impact: SquareRootImpact,
        *,
        notional: float,
        adv_notional: float,
        volatility_daily: float,
        risk_aversion: float = 0.0,
        epsilon_bps: float = 0.0,
    ) -> AlmgrenChriss:
        """Linearise the square-root law around the size actually being traded.

        Almgren-Chriss needs linear coefficients; the empirical law is concave. The
        honest move is to linearise **at this order's own participation**, so the
        schedule is optimal for this trade rather than for a generic one. The
        approximation degrades for orders far from that size, which is fine,
        because a different order gets its own linearisation.
        """
        participation = impact.participation(notional, adv_notional)
        if participation <= 0:
            raise ValueError("notional must be positive to linearise impact")

        peak_frac = impact.peak_impact_bps(notional, adv_notional, volatility_daily) * BPS
        permanent_frac = impact.params.permanent_fraction * peak_frac
        temporary_frac = peak_frac - permanent_frac
        # Secant slopes through the origin, so gamma*X and eta*X reproduce the
        # square-root law's permanent and temporary displacements at this size.
        return cls(
            volatility_daily=volatility_daily,
            gamma=permanent_frac / participation,
            eta=temporary_frac / participation,
            epsilon=epsilon_bps * BPS,
            risk_aversion=risk_aversion,
        )

    # ------------------------------------------------------------------ solve ---
    def schedule(
        self, participation: float, horizon_days: float, periods: int = 20
    ) -> ExecutionSchedule:
        """Optimal trajectory for executing ``participation`` over ``horizon_days``."""
        if periods < 1:
            raise ValueError("periods must be at least 1")
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if participation <= 0:
            raise ValueError("participation must be positive")

        tau = horizon_days / periods
        # Effective temporary impact, net of the permanent impact already booked
        # inside the period. Almgren-Chriss eta-tilde.
        eta_tilde = self.eta - 0.5 * self.gamma * tau
        if eta_tilde <= 0:
            raise ValueError(
                "eta - gamma*tau/2 must be positive; the period is too long relative "
                "to the impact coefficients for the discrete model to be well posed. "
                "Use more periods, or a shorter horizon."
            )

        kappa_tilde_squared = self.risk_aversion * self.sigma**2 / eta_tilde
        # cosh(kappa*tau) = 1 + kappa_tilde^2 * tau^2 / 2
        argument = 1.0 + kappa_tilde_squared * tau**2 / 2.0
        kappa = float(np.arccosh(argument)) / tau if argument > 1.0 else 0.0

        times = np.arange(periods + 1) * tau
        if kappa * horizon_days < 1e-8:
            # Risk-neutral limit: the sinh ratio degenerates to a straight line.
            holdings = participation * (1.0 - times / horizon_days)
        else:
            holdings = (
                participation
                * np.sinh(kappa * (horizon_days - times))
                / np.sinh(kappa * horizon_days)
            )
        holdings[-1] = 0.0
        trades = -np.diff(holdings)

        # Almgren-Chriss (2000) implementation shortfall, summed directly so the
        # discrete problem is solved exactly rather than approximated by its
        # continuous-time closed form.
        expected = (
            0.5 * self.gamma * participation**2
            + self.epsilon * float(np.abs(trades).sum())
            + eta_tilde * float((trades**2).sum()) / tau
        )
        variance = self.sigma**2 * tau * float((holdings[1:] ** 2).sum())

        # Per unit traded, so the answer is a price fraction, then basis points.
        return ExecutionSchedule(
            holdings=holdings,
            trades=trades,
            period_days=tau,
            expected_cost_bps=float(expected / participation / BPS),
            cost_variance_bps2=float(variance / participation**2 / BPS**2),
            kappa=kappa,
        )

    def efficient_frontier(
        self,
        participation: float,
        horizon_days: float,
        risk_aversions: list[float],
        periods: int = 20,
    ) -> list[tuple[float, float, float]]:
        """``(risk_aversion, expected_cost_bps, cost_stdev_bps)`` for each level.

        The frontier is the honest way to present execution: there is no single
        "optimal" schedule, only a menu of cost-for-certainty trades, and which
        point to pick depends on how much the signal decays while you wait.
        """
        out: list[tuple[float, float, float]] = []
        for aversion in risk_aversions:
            model = AlmgrenChriss(
                volatility_daily=self.sigma,
                gamma=self.gamma,
                eta=self.eta,
                epsilon=self.epsilon,
                risk_aversion=aversion,
            )
            plan = model.schedule(participation, horizon_days, periods)
            out.append((aversion, plan.expected_cost_bps, plan.cost_stdev_bps))
        return out


# ======================================================================================
# Transient impact
# ======================================================================================
class PropagatorImpact:
    """Transient impact with a power-law decay kernel.

    Each child order displaces the price by ``g0 · f(v)`` and that displacement
    decays as ``G(τ) = (1 + τ/τ0)^(-β)``. Unlike the square-root law, which gives
    an average cost, this produces a *path*, which is what the event-driven engine
    needs to model partial fills and to let a later order trade against the residual
    impact of an earlier one.

    The decay is why splitting an order helps at all: with permanent impact only,
    execution schedule would not matter.
    """

    def __init__(
        self,
        *,
        g0: float = 0.5,
        tau0: float = 1.0,
        beta: float = 0.4,
        exponent: float = 0.5,
    ) -> None:
        if g0 <= 0:
            raise ValueError("g0 must be positive")
        if tau0 <= 0:
            raise ValueError("tau0 must be positive")
        if not 0 < beta < 1:
            # beta >= 1 makes the kernel non-integrable in the relevant regime and
            # the model loses the diffusive price behaviour it is meant to produce.
            raise ValueError("beta must be in (0, 1)")
        self.g0 = g0
        self.tau0 = tau0
        self.beta = beta
        self.exponent = exponent

    def kernel(self, lags: np.ndarray) -> np.ndarray:
        """Decay of a unit displacement after ``lags`` periods."""
        lags = np.asarray(lags, dtype=float)
        if np.any(lags < 0):
            raise ValueError("lags must be non-negative")
        decay: np.ndarray = (1.0 + lags / self.tau0) ** (-self.beta)
        return decay

    def half_life(self) -> float:
        """Periods until a displacement has decayed to half its initial size."""
        return float(self.tau0 * (2.0 ** (1.0 / self.beta) - 1.0))

    def impact_path(
        self,
        participations: np.ndarray,
        signs: np.ndarray,
        volatility_daily: float,
    ) -> np.ndarray:
        """Cumulative price displacement, as a fraction of price, at each step.

        ``participations[i]`` is the child order's size as a fraction of ADV and
        ``signs[i]`` its direction. The result is the displacement *after* each
        step, so element ``i`` already includes step ``i``'s own impact.
        """
        participations = np.asarray(participations, dtype=float)
        signs = np.asarray(signs, dtype=float)
        if participations.shape != signs.shape:
            raise ValueError("participations and signs must have the same shape")
        if np.any(participations < 0):
            raise ValueError("participations must be non-negative; use `signs` for direction")

        steps = len(participations)
        contributions = self.g0 * volatility_daily * participations**self.exponent * signs
        # Lower-triangular kernel matrix: row i weights every contribution at or
        # before i by its decay. Steps are small (an execution schedule, not a
        # backtest), so the O(n^2) build is not worth avoiding.
        lags = np.arange(steps)[:, None] - np.arange(steps)[None, :]
        weights = np.where(lags >= 0, self.kernel(np.maximum(lags, 0)), 0.0)
        path: np.ndarray = weights @ contributions
        return path

    def implied_cost_bps(
        self,
        participations: np.ndarray,
        signs: np.ndarray,
        volatility_daily: float,
    ) -> float:
        """Execution cost of a schedule, in basis points of the traded notional.

        Each child order executes against the displacement standing when it
        arrives, including the displacement it causes itself.
        """
        path = self.impact_path(participations, signs, volatility_daily)
        participations = np.asarray(participations, dtype=float)
        traded = float(participations.sum())
        if traded <= 0:
            return 0.0
        signed_cost = float((path * participations * np.asarray(signs, dtype=float)).sum())
        return signed_cost / traded / BPS
