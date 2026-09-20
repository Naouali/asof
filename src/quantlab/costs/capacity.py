"""Capacity: the AUM at which a strategy's own trading eats its edge.

The arithmetic that makes capacity finite:

* Gross alpha grows at best **linearly** in assets. Twice the money, twice the
  dollars of edge -- and that is the optimistic case, since crowding usually makes
  it sublinear.
* Cost grows like **Q^1.5**. Impact per share grows like the square root of size,
  and you pay it on every share, so dollars of cost grow faster than dollars of
  edge no matter how good the signal is.

The two curves cross. Where they cross is the capacity, and with a square-root law
it moves with roughly the **square** of alpha: a signal twice as strong is four
times as scalable. That is also why a small improvement in execution is worth more
than it looks.

Spec section 5: every tearsheet displays an estimated break-even AUM, and a
strategy without one is not a finished strategy. This module is what produces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import brentq

from quantlab.costs.base import BPS
from quantlab.costs.model import TransactionCostModel
from quantlab.logging import get_logger

__all__ = ["CapacityModel", "CapacityResult", "UniverseLiquidity"]

log = get_logger("quantlab.costs.capacity")

#: Columns a liquidity frame must carry.
REQUIRED_COLUMNS = ("symbol", "weight", "adv_notional", "volatility_daily", "spread_bps")

#: Search bracket for the solver, in currency units. One thousand to one trillion
#: spans every plausible answer; falling outside it is reported, not extrapolated.
_MIN_AUM = 1e3
_MAX_AUM = 1e12


@dataclass(frozen=True, slots=True)
class UniverseLiquidity:
    """The traded universe's liquidity, as one row per instrument.

    ``weight`` is each name's share of the **traded notional**, not of the
    portfolio. For most strategies these coincide; for one that rebalances a few
    names heavily and the rest rarely, they do not, and using portfolio weights
    would overstate capacity by spreading the trading across names that are not
    actually being traded.
    """

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.frame.columns]
        if missing:
            raise ValueError(f"liquidity frame is missing columns {missing}")
        if self.frame.height == 0:
            raise ValueError("liquidity frame is empty; capacity is undefined")
        if self.frame["adv_notional"].min() <= 0:  # type: ignore[operator]
            raise ValueError(
                "every instrument needs a positive adv_notional. A name with no "
                "measurable volume has no measurable capacity -- drop it from the "
                "universe rather than trading it at zero cost."
            )
        total = float(self.frame["weight"].sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1, got {total:.6f}")

    @classmethod
    def equal_weight(
        cls,
        symbols: list[str],
        *,
        adv_notional: float,
        volatility_daily: float,
        spread_bps: float,
    ) -> UniverseLiquidity:
        """A homogeneous universe. Useful for intuition, optimistic for planning.

        Real universes are dominated by their least liquid members once size grows,
        because the impact cost of the small names rises fastest. An equal-weight
        homogeneous approximation therefore **overstates** capacity.
        """
        count = len(symbols)
        return cls(
            pl.DataFrame(
                {
                    "symbol": symbols,
                    "weight": [1.0 / count] * count,
                    "adv_notional": [adv_notional] * count,
                    "volatility_daily": [volatility_daily] * count,
                    "spread_bps": [spread_bps] * count,
                }
            )
        )

    @property
    def total_adv(self) -> float:
        return float(self.frame["adv_notional"].sum())


@dataclass(frozen=True, slots=True)
class CapacityResult:
    """Break-even AUM and the curve that produced it."""

    break_even_aum: float | None
    gross_alpha_bps_annual: float
    #: Cost in annualised basis points of AUM, evaluated at the break-even point
    #: (or at the reference AUM when there is no break-even).
    cost_bps_annual: float
    #: Net alpha per annum at a small, effectively costless size. If this is
    #: already negative the strategy does not work at *any* size.
    net_alpha_bps_at_zero: float
    curve: pl.DataFrame
    binding_constraint: str
    #: Largest single-name participation implied at the break-even AUM, as a
    #: fraction of that name's ADV.
    max_participation: float = 0.0
    #: True when the break-even requires trading beyond the range the square-root
    #: law was calibrated on, where it is known to **understate** impact. The
    #: reported capacity is then an upper bound, not an estimate.
    extrapolated: bool = False
    note: str = ""

    @property
    def is_viable(self) -> bool:
        return self.break_even_aum is not None and self.break_even_aum > _MIN_AUM

    def summary(self) -> str:
        if not self.is_viable or self.break_even_aum is None:
            return f"no viable capacity: {self.note}"
        text = (
            f"break-even AUM ${self.break_even_aum / 1e6:,.1f}m "
            f"(gross {self.gross_alpha_bps_annual:.0f} bp/yr, "
            f"binding constraint: {self.binding_constraint})"
        )
        if self.extrapolated:
            text += (
                f" -- UPPER BOUND ONLY: requires {self.max_participation:.1%} of a "
                "name's daily volume, beyond the square-root law's calibrated range, "
                "where it understates impact. Slow the rebalance or shrink the number."
            )
        return text


class CapacityModel:
    """Solve for the AUM at which net alpha reaches zero."""

    def __init__(self, costs: TransactionCostModel | None = None) -> None:
        self.costs = costs or TransactionCostModel(use_asset_class_defaults=False)

    # ------------------------------------------------------------------ costs --
    def cost_bps_of_aum(
        self,
        aum: float,
        universe: UniverseLiquidity,
        *,
        turnover_per_rebalance: float,
        horizon_days: float = 1.0,
    ) -> float:
        """One rebalance's trading cost, in basis points of AUM.

        Costed **name by name**, which is not the same as costing the whole trade
        against the universe's total ADV.

        Splitting equally across *identical* names is exactly neutral -- impact
        depends on participation, not absolute size, so a hundred names at 1% each
        costs what one name at 1% costs. The reason aggregation is wrong is
        heterogeneity: because impact is concave in participation, the thin names
        cost disproportionately more, and collapsing the universe into one deep
        instrument makes them disappear. On a realistic mix the difference is
        large -- a 50/50 book of a $4.9bn and a $100m name costs roughly 2.6x what
        its combined ADV would suggest.
        """
        if aum <= 0:
            raise ValueError("aum must be positive")
        if turnover_per_rebalance < 0:
            raise ValueError("turnover_per_rebalance must be non-negative")
        if turnover_per_rebalance == 0:
            return 0.0

        frame = universe.frame
        traded = aum * turnover_per_rebalance * frame["weight"].to_numpy()
        costed = frame.with_columns(traded_notional=pl.Series(traded)).with_columns(
            cost_currency=self.costs.cost_currency_expr(
                notional=pl.col("traded_notional"),
                adv_notional=pl.col("adv_notional"),
                volatility_daily=pl.col("volatility_daily"),
                spread_bps=pl.col("spread_bps"),
                horizon_days=horizon_days,
                commission_bps=(
                    pl.col("commission_bps") if "commission_bps" in frame.columns else pl.lit(0.0)
                ),
            )
        )
        return float(costed["cost_currency"].sum()) / aum / BPS

    def net_alpha_bps_annual(
        self,
        aum: float,
        universe: UniverseLiquidity,
        *,
        gross_alpha_bps_per_rebalance: float,
        turnover_per_rebalance: float,
        rebalances_per_year: float,
        horizon_days: float = 1.0,
        holding_cost_bps_annual: float = 0.0,
    ) -> float:
        """Gross alpha less trading and holding costs, annualised."""
        per_rebalance = gross_alpha_bps_per_rebalance - self.cost_bps_of_aum(
            aum, universe, turnover_per_rebalance=turnover_per_rebalance, horizon_days=horizon_days
        )
        return per_rebalance * rebalances_per_year - holding_cost_bps_annual

    # ------------------------------------------------------------------ solve --
    def solve(
        self,
        universe: UniverseLiquidity,
        *,
        gross_alpha_bps_per_rebalance: float,
        turnover_per_rebalance: float,
        rebalances_per_year: float,
        horizon_days: float = 1.0,
        holding_cost_bps_annual: float = 0.0,
        curve_points: int = 40,
    ) -> CapacityResult:
        """Find the break-even AUM and build the curve for the tearsheet."""

        def net(aum: float) -> float:
            return self.net_alpha_bps_annual(
                aum,
                universe,
                gross_alpha_bps_per_rebalance=gross_alpha_bps_per_rebalance,
                turnover_per_rebalance=turnover_per_rebalance,
                rebalances_per_year=rebalances_per_year,
                horizon_days=horizon_days,
                holding_cost_bps_annual=holding_cost_bps_annual,
            )

        grid = np.geomspace(_MIN_AUM, _MAX_AUM, curve_points)
        nets = np.array([net(float(a)) for a in grid])
        costs = np.array(
            [
                self.cost_bps_of_aum(
                    float(a),
                    universe,
                    turnover_per_rebalance=turnover_per_rebalance,
                    horizon_days=horizon_days,
                )
                * rebalances_per_year
                for a in grid
            ]
        )
        curve = pl.DataFrame(
            {
                "aum": grid,
                "gross_bps_annual": np.full_like(
                    grid, gross_alpha_bps_per_rebalance * rebalances_per_year
                ),
                "cost_bps_annual": costs + holding_cost_bps_annual,
                "net_bps_annual": nets,
            }
        )

        net_at_floor = float(nets[0])
        binding = self._binding_constraint(
            universe,
            turnover_per_rebalance=turnover_per_rebalance,
            rebalances_per_year=rebalances_per_year,
            holding_cost_bps_annual=holding_cost_bps_annual,
        )

        if net_at_floor <= 0:
            # Costs beat the signal even at a size small enough to be costless.
            # There is no capacity to report, and reporting a small number would
            # imply the strategy works if you keep it tiny. It does not.
            note = (
                f"net alpha is already {net_at_floor:.1f} bp/yr at ${_MIN_AUM:,.0f}. "
                "Spread, commission and holding costs exceed the gross edge at any "
                "size; this is not a capacity problem, the signal does not pay."
            )
            log.warning("costs.capacity_zero", note=note, binding_constraint=binding)
            return CapacityResult(
                break_even_aum=None,
                gross_alpha_bps_annual=gross_alpha_bps_per_rebalance * rebalances_per_year,
                cost_bps_annual=float(costs[0]) + holding_cost_bps_annual,
                net_alpha_bps_at_zero=net_at_floor,
                curve=curve,
                binding_constraint=binding,
                note=note,
            )

        if nets[-1] > 0:
            note = (
                f"net alpha is still {nets[-1]:.1f} bp/yr at ${_MAX_AUM:,.0f}. The "
                "break-even lies beyond the search bracket, which in practice means "
                "the liquidity inputs are implausible rather than that the strategy "
                "is infinitely scalable."
            )
            log.warning("costs.capacity_unbounded", note=note)
            return CapacityResult(
                break_even_aum=None,
                gross_alpha_bps_annual=gross_alpha_bps_per_rebalance * rebalances_per_year,
                cost_bps_annual=float(costs[-1]) + holding_cost_bps_annual,
                net_alpha_bps_at_zero=net_at_floor,
                curve=curve,
                binding_constraint=binding,
                note=note,
            )

        break_even = float(brentq(net, _MIN_AUM, _MAX_AUM, xtol=1.0, rtol=1e-8))
        participation = self.max_participation(
            break_even, universe, turnover_per_rebalance=turnover_per_rebalance
        )
        limit = self.costs.impact.params.max_reliable_participation
        extrapolated = participation > limit
        if extrapolated:
            log.warning(
                "costs.capacity_extrapolated",
                break_even_aum=round(break_even, 0),
                max_participation=round(participation, 4),
                limit=limit,
                consequence="capacity is an upper bound; the law understates impact here",
            )
        return CapacityResult(
            break_even_aum=break_even,
            gross_alpha_bps_annual=gross_alpha_bps_per_rebalance * rebalances_per_year,
            cost_bps_annual=self.cost_bps_of_aum(
                break_even,
                universe,
                turnover_per_rebalance=turnover_per_rebalance,
                horizon_days=horizon_days,
            )
            * rebalances_per_year
            + holding_cost_bps_annual,
            net_alpha_bps_at_zero=net_at_floor,
            curve=curve,
            binding_constraint=binding,
            max_participation=participation,
            extrapolated=extrapolated,
        )

    def max_participation(
        self,
        aum: float,
        universe: UniverseLiquidity,
        *,
        turnover_per_rebalance: float,
    ) -> float:
        """Largest single-name participation implied at this AUM.

        The binding name is not the average one: a universe's least liquid member
        hits the extrapolation range long before the aggregate does, which is why
        capacity from an equal-weighted approximation is optimistic.
        """
        frame = universe.frame
        traded = aum * turnover_per_rebalance * frame["weight"].to_numpy()
        return float(np.max(traded / frame["adv_notional"].to_numpy()))

    def _binding_constraint(
        self,
        universe: UniverseLiquidity,
        *,
        turnover_per_rebalance: float,
        rebalances_per_year: float,
        holding_cost_bps_annual: float,
    ) -> str:
        """Which cost dominates at a size small enough that impact is negligible."""
        frame = universe.frame
        spread = float((frame["weight"] * frame["spread_bps"] / 2.0).sum())
        commission = (
            float((frame["weight"] * frame["commission_bps"]).sum())
            if "commission_bps" in frame.columns
            else 0.0
        )
        annual_spread = spread * turnover_per_rebalance * rebalances_per_year
        annual_commission = commission * turnover_per_rebalance * rebalances_per_year
        ranked = {
            "spread": annual_spread,
            "commission": annual_commission,
            "holding costs (financing and borrow)": holding_cost_bps_annual,
        }
        top = max(ranked, key=lambda key: ranked[key])
        if ranked[top] <= 0:
            return "impact (no fixed costs)"
        return f"{top} at small size; impact once scaled"


def analytic_break_even_single_name(
    *,
    gross_alpha_bps_per_rebalance: float,
    turnover_per_rebalance: float,
    adv_notional: float,
    volatility_daily: float,
    spread_bps: float,
    commission_bps: float = 0.0,
    y: float = 0.5,
    delta: float = 0.5,
    permanent_fraction: float = 0.5,
    horizon_days: float = 1.0,
    urgency: float | None = None,
) -> float | None:
    """Closed-form capacity for a single instrument.

    Exists to check the numerical solver against algebra rather than against
    itself, and to make the ``alpha²`` scaling visible:

        A* = (ADV / T) · [ (α − T·(s/2 + c)) · 1e-4 / (T · m · Y · σ) ]^(1/δ)

    with ``m`` the fraction of peak impact actually paid. At ``δ = 0.5`` the
    exponent is 2, so halving alpha quarters capacity.
    """
    fixed = turnover_per_rebalance * (spread_bps / 2.0 + commission_bps)
    residual = gross_alpha_bps_per_rebalance - fixed
    if residual <= 0:
        return None

    paid_fraction = (1.0 - permanent_fraction) * (1.0 / horizon_days) ** (
        delta if urgency is None else urgency
    ) + permanent_fraction / 2.0
    denominator = turnover_per_rebalance * paid_fraction * y * volatility_daily
    if denominator <= 0:
        return None

    participation = (residual * BPS / denominator) ** (1.0 / delta)
    return float(adv_notional * participation / turnover_per_rebalance)
