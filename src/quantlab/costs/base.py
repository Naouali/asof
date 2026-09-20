"""Core cost abstractions.

Units, fixed once so nothing downstream has to guess:

* **Trade costs** are quoted in **basis points of the traded notional**. A 10 bp
  cost on a $1m order is $1,000.
* **Holding costs** (financing, borrow) are quoted in **basis points of the held
  notional per year**, and converted to a period charge by the caller.
* **Volatility** is a daily decimal: 0.02 means 2% per day, not 2.
* **Participation** is traded notional divided by average daily notional volume.

The one prohibition that shapes this package: **there is no flat basis-point cost
model available as a default anywhere** (spec section 13). Flat costs are wrong in
the direction that matters -- they are roughly right for the small trades you test
on and wildly optimistic for the size you would actually deploy, so they make every
strategy look scalable. :class:`FlatBpsCostModel` exists only to demonstrate that
gap, and refuses to be constructed without an explicit acknowledgement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import polars as pl

from quantlab.data.catalogue import AssetClass
from quantlab.logging import get_logger

__all__ = [
    "BPS",
    "CostBreakdown",
    "CostModel",
    "FlatBpsCostModel",
    "Instrument",
    "Order",
    "SpreadSource",
]

log = get_logger("quantlab.costs")

#: One basis point as a fraction. Written once so the 1e-4 conversions in this
#: package are greppable rather than scattered magic numbers.
BPS = 1e-4

#: Where an instrument's spread came from. Carried so that a cost -- and any
#: capacity number built on it -- can say how much of itself is measurement and
#: how much is assumption.
SpreadSource = Literal["observed", "estimated", "assumed"]


@dataclass(frozen=True, slots=True)
class Instrument:
    """The liquidity characteristics a cost model needs about one instrument.

    Every field here is an *estimate* with an error bar, and the cost model is only
    as good as they are. `adv_notional` in particular is usually measured over a
    trailing window and is itself a forecast; a strategy whose capacity estimate is
    sensitive to it should say so.
    """

    symbol: str
    asset_class: AssetClass
    #: Average daily traded value in the quote currency. Not share volume.
    adv_notional: float
    #: Daily return volatility as a decimal (0.02 == 2% per day).
    volatility_daily: float
    #: Full quoted bid-ask spread in basis points. Crossing it once costs half.
    spread_bps: float
    #: Provenance of ``spread_bps``. Defaults to ``"assumed"``, because an
    #: unlabelled number is an assumption whatever its origin.
    spread_source: SpreadSource = "assumed"
    #: Minimum price increment as a fraction of price, where known. Used to floor
    #: spread estimates: no spread can be narrower than one tick.
    tick_bps: float | None = None
    #: Annualised short borrow fee in basis points. ``None`` means unknown, and
    #: unknown is charged punitively rather than assumed free.
    borrow_bps_annual: float | None = None
    #: Per-side commission in basis points, where a venue charges one.
    commission_bps: float = 0.0
    #: Free-form provenance, e.g. which dataset the ADV came from.
    source: str = ""

    def __post_init__(self) -> None:
        if self.adv_notional <= 0:
            raise ValueError(
                f"{self.symbol}: adv_notional must be positive, got {self.adv_notional}. "
                "An instrument with no measurable volume has no measurable capacity; "
                "exclude it from the universe rather than trading it at zero cost."
            )
        if self.volatility_daily < 0:
            raise ValueError(f"{self.symbol}: volatility_daily must be non-negative")
        if self.spread_bps < 0:
            raise ValueError(f"{self.symbol}: spread_bps must be non-negative")

    @property
    def half_spread_bps(self) -> float:
        """Cost of crossing the spread once, in basis points."""
        return self.spread_bps / 2.0


@dataclass(frozen=True, slots=True)
class Order:
    """A single trade to be costed.

    ``notional`` is the absolute value traded. ``side`` is kept separately because
    borrow applies only to shorts and funding is signed.
    """

    symbol: str
    notional: float
    #: +1 to buy, -1 to sell.
    side: int = 1
    #: Fraction of a trading day over which the order is worked. 1.0 is a full day;
    #: 0.1 is roughly 40 minutes of a US equity session. Shorter horizons pay more
    #: temporary impact for the same size.
    horizon_days: float = 1.0

    def __post_init__(self) -> None:
        if self.notional < 0:
            raise ValueError(
                f"{self.symbol}: notional must be the absolute traded value; use "
                "`side` for direction"
            )
        if self.side not in (-1, 1):
            raise ValueError(f"{self.symbol}: side must be +1 or -1, got {self.side}")
        if self.horizon_days <= 0:
            raise ValueError(f"{self.symbol}: horizon_days must be positive")

    @property
    def is_short(self) -> bool:
        return self.side < 0


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Where the money went, in basis points of traded notional.

    Kept decomposed rather than summed because the decomposition is the diagnostic:
    a strategy killed by spread needs a slower rebalance, one killed by impact needs
    less size, and one killed by borrow needs a different short book. A single
    number tells you none of that, which is why spec section 9 requires the
    breakdown on every tearsheet.
    """

    spread_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    permanent_impact_bps: float = 0.0
    commission_bps: float = 0.0
    financing_bps: float = 0.0
    borrow_bps: float = 0.0
    #: Diagnostics that are not themselves costs.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def impact_bps(self) -> float:
        """Impact actually paid on this order.

        Only half the permanent displacement is paid on average: it builds up as
        the order is worked, so the early fills execute before most of it has
        happened. The temporary component is paid in full. This is the standard
        Almgren-Chriss implementation-shortfall accounting for a uniform schedule;
        a front-loaded schedule pays more, a back-loaded one less.
        """
        return self.temporary_impact_bps + self.permanent_impact_bps / 2.0

    @property
    def total_bps(self) -> float:
        return (
            self.spread_bps
            + self.impact_bps
            + self.commission_bps
            + self.financing_bps
            + self.borrow_bps
        )

    def currency(self, notional: float) -> float:
        """Total cost in currency units for a given traded notional."""
        return self.total_bps * BPS * notional

    def as_dict(self) -> dict[str, float]:
        return {
            "spread_bps": self.spread_bps,
            "temporary_impact_bps": self.temporary_impact_bps,
            "permanent_impact_bps": self.permanent_impact_bps,
            "impact_bps": self.impact_bps,
            "commission_bps": self.commission_bps,
            "financing_bps": self.financing_bps,
            "borrow_bps": self.borrow_bps,
            "total_bps": self.total_bps,
        }

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        if not isinstance(other, CostBreakdown):  # pragma: no cover - type guard
            return NotImplemented
        return CostBreakdown(
            spread_bps=self.spread_bps + other.spread_bps,
            temporary_impact_bps=self.temporary_impact_bps + other.temporary_impact_bps,
            permanent_impact_bps=self.permanent_impact_bps + other.permanent_impact_bps,
            commission_bps=self.commission_bps + other.commission_bps,
            financing_bps=self.financing_bps + other.financing_bps,
            borrow_bps=self.borrow_bps + other.borrow_bps,
            detail={**self.detail, **other.detail},
        )

    def scaled(self, factor: float) -> CostBreakdown:
        return replace(
            self,
            spread_bps=self.spread_bps * factor,
            temporary_impact_bps=self.temporary_impact_bps * factor,
            permanent_impact_bps=self.permanent_impact_bps * factor,
            commission_bps=self.commission_bps * factor,
            financing_bps=self.financing_bps * factor,
            borrow_bps=self.borrow_bps * factor,
        )


class CostModel(ABC):
    """Interface every cost model implements.

    Two entry points on purpose. :meth:`estimate` is scalar and readable, and is
    what the tests and the event-driven engine use. :meth:`cost_bps_expr` returns a
    polars expression so the vectorised engine can cost a whole cross-section
    without a Python loop -- a 3,000-name daily backtest calls this ~7,500 times,
    and a per-order Python call there would dominate the runtime budget.

    Implementations must agree: a scalar estimate and its vectorised counterpart
    are required by test to produce the same number.
    """

    name: str = "cost-model"

    @abstractmethod
    def estimate(self, order: Order, instrument: Instrument) -> CostBreakdown:
        """Cost of one order against one instrument."""

    @abstractmethod
    def cost_bps_expr(
        self,
        notional: pl.Expr,
        adv_notional: pl.Expr,
        volatility_daily: pl.Expr,
        spread_bps: pl.Expr,
        horizon_days: pl.Expr | float = 1.0,
    ) -> pl.Expr:
        """Vectorised total trade cost in basis points of traded notional."""

    def estimate_many(
        self, orders: list[Order], instruments: dict[str, Instrument]
    ) -> CostBreakdown:
        """Notional-weighted aggregate cost across a basket.

        Weighting by notional is what makes the result comparable to a headline
        "the strategy paid N bps": an equal-weighted average of per-order bps would
        let a tiny cheap order offset a large expensive one.
        """
        total_notional = sum(order.notional for order in orders)
        if total_notional <= 0:
            return CostBreakdown()
        aggregate = CostBreakdown()
        for order in orders:
            breakdown = self.estimate(order, instruments[order.symbol])
            aggregate = aggregate + breakdown.scaled(order.notional / total_notional)
        return aggregate


class FlatBpsCostModel(CostModel):
    """A constant cost per trade. **Not usable as a default, by design.**

    This exists for one purpose: to show, on a tearsheet, how much a flat
    assumption understates the cost of real size. Flat costs are roughly right for
    the small trades a researcher tests on and wildly optimistic at deployment
    size, so they make every strategy look scalable and every capacity estimate
    infinite.

    Constructing it requires ``acknowledge_unrealistic=True``, and it logs a
    warning each time, so it can never end up as a default through a missing
    argument.
    """

    name = "flat-bps"

    def __init__(self, cost_bps: float, *, acknowledge_unrealistic: bool = False) -> None:
        if not acknowledge_unrealistic:
            raise ValueError(
                "Flat basis-point costs are prohibited as a default (spec section 13). "
                "They are roughly right for research-sized trades and wildly optimistic "
                "at deployment size, which makes every strategy look scalable. Use "
                "quantlab.costs.model.TransactionCostModel. If you genuinely want a flat "
                "model -- to show on a tearsheet how much it understates -- pass "
                "acknowledge_unrealistic=True."
            )
        if cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        self.cost_bps = cost_bps
        log.warning(
            "costs.flat_model_constructed",
            cost_bps=cost_bps,
            reason="flat costs ignore size entirely and imply unlimited capacity",
        )

    def estimate(self, order: Order, instrument: Instrument) -> CostBreakdown:
        del order, instrument
        return CostBreakdown(
            spread_bps=self.cost_bps,
            detail={"model": self.name, "warning": "size-independent; capacity is meaningless"},
        )

    def cost_bps_expr(
        self,
        notional: pl.Expr,
        adv_notional: pl.Expr,
        volatility_daily: pl.Expr,
        spread_bps: pl.Expr,
        horizon_days: pl.Expr | float = 1.0,
    ) -> pl.Expr:
        del notional, adv_notional, volatility_daily, spread_bps, horizon_days
        return pl.lit(self.cost_bps, dtype=pl.Float64())
