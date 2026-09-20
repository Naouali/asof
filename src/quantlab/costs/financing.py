"""Financing, borrow, funding and roll costs.

These are the costs a backtest forgets, and forgetting them is not a small error.
A punitive short borrow assumption alone is enough to erase most published equity
anomaly alphas, which is exactly why the default here is punitive rather than
absent: an unknown borrow fee is charged as if the name were hard to borrow, not
as if it were free.

Conventions:

* Rates are **annualised basis points** of the notional they apply to.
* Period charges use ACT/365, because these are financing accruals and a 360-day
  convention would understate every one of them by 1.4%.
* Costs are positive numbers. A *receipt* -- short rebate, negative funding -- is a
  negative cost, and is allowed to be.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.costs.base import BPS, Instrument
from quantlab.logging import get_logger

__all__ = ["FinancingModel", "FinancingParams", "HoldingCost"]

log = get_logger("quantlab.costs.financing")

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class HoldingCost:
    """Cost of holding a position for a period, in basis points of the notional."""

    borrow_bps: float = 0.0
    financing_bps: float = 0.0
    funding_bps: float = 0.0
    assumed_borrow: bool = False

    @property
    def total_bps(self) -> float:
        return self.borrow_bps + self.financing_bps + self.funding_bps


@dataclass(frozen=True, slots=True)
class FinancingParams:
    """Defaults, chosen to be punitive where data is missing."""

    #: Charged on any short whose actual borrow fee is unknown. Spec section 5
    #: names 10-20 bp per month as the range that erases most published anomaly
    #: alphas; 15 is the midpoint. This is the single most consequential default in
    #: the package: a long/short equity strategy's net alpha is roughly linear in
    #: it, so it should be replaced with real borrow data before anything is traded.
    default_borrow_bps_per_month: float = 15.0
    #: Spread over the base rate paid to finance long positions on margin.
    margin_spread_bps_annual: float = 150.0
    #: Fraction of the base rate earned on short sale proceeds. Retail and small
    #: institutional accounts typically earn none of it; assuming the full rate is
    #: how a backtest invents a risk-free income stream that does not exist.
    short_rebate_fraction: float = 0.0
    #: Hard-to-borrow threshold above which a short is flagged as fragile: a name
    #: this expensive to borrow can be recalled, and a forced buy-in is a cost no
    #: model here captures.
    hard_to_borrow_bps_annual: float = 500.0

    @property
    def default_borrow_bps_annual(self) -> float:
        return self.default_borrow_bps_per_month * 12.0


class FinancingModel:
    """Borrow, margin financing, perpetual funding and futures roll."""

    def __init__(self, params: FinancingParams | None = None) -> None:
        self.params = params or FinancingParams()

    # ----------------------------------------------------------------- borrow --
    def borrow_bps(
        self, instrument: Instrument, days: float, *, is_short: bool = True
    ) -> tuple[float, bool]:
        """Borrow cost for holding a short, and whether the rate was assumed.

        Returns ``(bps_of_notional, assumed)``. ``assumed`` is propagated all the
        way to the tearsheet: a strategy whose viability rests on an assumed borrow
        fee has not been tested, it has been guessed at.
        """
        if not is_short:
            return 0.0, False
        if days < 0:
            raise ValueError("days must be non-negative")

        annual = instrument.borrow_bps_annual
        assumed = annual is None
        if assumed:
            annual = self.params.default_borrow_bps_annual
        assert annual is not None  # noqa: S101 - narrowing for type checkers

        if annual >= self.params.hard_to_borrow_bps_annual:
            log.warning(
                "costs.hard_to_borrow",
                symbol=instrument.symbol,
                borrow_bps_annual=annual,
                risk="recall and forced buy-in are not modelled",
            )
        return annual * days / DAYS_PER_YEAR, assumed

    # ---------------------------------------------------------------- margin ---
    def financing_bps(
        self,
        *,
        gross_notional: float,
        equity: float,
        days: float,
        base_rate_annual_bps: float,
    ) -> float:
        """Cost of financing leverage, in basis points of gross notional.

        Only the notional beyond the account's own equity is financed. A strategy
        running gross 1.0 pays nothing here; one running gross 3.0 pays on two
        units of equity's worth, which is how leverage quietly consumes a carry
        trade's edge.
        """
        if equity <= 0:
            raise ValueError("equity must be positive")
        if gross_notional < 0 or days < 0:
            raise ValueError("gross_notional and days must be non-negative")

        borrowed = max(0.0, gross_notional - equity)
        if borrowed == 0 or gross_notional == 0:
            return 0.0
        rate = base_rate_annual_bps + self.params.margin_spread_bps_annual
        cost_currency = borrowed * rate * BPS * days / DAYS_PER_YEAR
        return cost_currency / gross_notional / BPS

    def short_rebate_bps(self, *, days: float, base_rate_annual_bps: float) -> float:
        """Interest earned on short sale proceeds. Negative, because it is income.

        Defaults to zero: most accounts earn none of it, and assuming otherwise
        manufactures a risk-free return that flatters every short book.
        """
        earned = base_rate_annual_bps * self.params.short_rebate_fraction
        return -earned * days / DAYS_PER_YEAR

    # --------------------------------------------------------------- funding ---
    @staticmethod
    def funding_bps(funding_rates: np.ndarray, side: int) -> float:
        """Perpetual funding paid over a sequence of funding events.

        Positive funding means longs pay shorts. ``side`` is +1 long, -1 short, so
        a long in a positive-funding regime pays and a short receives. This is a
        cost *and* a signal -- crypto carry is precisely the decision to be on the
        receiving side -- so the sign convention is worth being exact about.
        """
        if side not in (-1, 1):
            raise ValueError("side must be +1 or -1")
        rates = np.asarray(funding_rates, dtype=float)
        rates = rates[np.isfinite(rates)]
        if rates.size == 0:
            return 0.0
        return float(side * rates.sum() / BPS)

    # ------------------------------------------------------------------ roll ---
    @staticmethod
    def roll_bps(
        *,
        half_spread_bps: float,
        impact_bps: float,
        rolls_per_year: float,
        days: float,
    ) -> float:
        """Cost of rolling a futures position, in basis points of notional.

        Each roll is two trades -- out of the expiring contract and into the next --
        so a quarterly roll pays eight one-way crossings a year. For a trend
        strategy on liquid futures this is often larger than the signal's own
        turnover cost, and leaving it out is why paper trend-following looks better
        than the real thing.
        """
        if rolls_per_year < 0 or days < 0:
            raise ValueError("rolls_per_year and days must be non-negative")
        per_roll = 2.0 * (half_spread_bps + impact_bps)
        return per_roll * rolls_per_year * days / DAYS_PER_YEAR

    # ------------------------------------------------------------- aggregate ---
    def holding_cost(
        self,
        instrument: Instrument,
        *,
        days: float,
        is_short: bool,
        gross_notional: float = 1.0,
        equity: float = 1.0,
        base_rate_annual_bps: float = 400.0,
        funding_rates: np.ndarray | None = None,
    ) -> HoldingCost:
        """Everything charged for holding a position over ``days``."""
        borrow, assumed = self.borrow_bps(instrument, days, is_short=is_short)
        if is_short:
            borrow += self.short_rebate_bps(days=days, base_rate_annual_bps=base_rate_annual_bps)
        financing = self.financing_bps(
            gross_notional=gross_notional,
            equity=equity,
            days=days,
            base_rate_annual_bps=base_rate_annual_bps,
        )
        funding = (
            self.funding_bps(funding_rates, -1 if is_short else 1)
            if funding_rates is not None
            else 0.0
        )
        return HoldingCost(
            borrow_bps=borrow,
            financing_bps=financing,
            funding_bps=funding,
            assumed_borrow=assumed,
        )
