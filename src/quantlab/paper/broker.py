"""The broker interface, and the only implementation this platform ships.

Spec section 1 is explicit: *"No live broker execution in v1. The system is
research and paper-trading only. Do not write order-routing code to a live
brokerage. Build the execution simulator and the paper-trading loop; leave a
clean interface where a live adapter would attach later, and document it."*

This module is that interface. :class:`Broker` is the seam; :class:`PaperBroker`
is the simulator; there is no third implementation and adding one is out of
scope for v1 by instruction, not by oversight.

**Where a live adapter would attach.** Subclass :class:`Broker` and implement the
three methods. Everything above this line -- the loop, the decay monitor, the
reporting -- talks only to the abstract class and would not change. What *would*
change, and what makes this a deliberate boundary rather than a small task:

* Fills stop being deterministic. :class:`PaperBroker` fills the whole order at a
  price derived from the panel and the cost model; a real venue partially fills,
  rejects, and moves while you work the order.
* Position state stops being yours. The broker's record becomes authoritative and
  has to be reconciled against the local ledger every cycle, because they will
  disagree — on corporate actions, on fees, on borrow.
* Failure stops being local. A submission that times out may or may not have
  reached the exchange, and the only safe assumption is that it did.

None of those are hard to write and all of them are easy to write wrongly, which
is why the spec puts them outside v1.

**The simulator charges the same costs as the backtest.** Not approximately, the
same :class:`~quantlab.costs.model.TransactionCostModel` object. A paper loop
that filled at mid would beat its own backtest for no reason at all, and the
difference would look like the strategy working.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quantlab.costs.base import BPS, Instrument, Order
from quantlab.costs.model import TransactionCostModel
from quantlab.data.catalogue import AssetClass
from quantlab.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = ["Broker", "Fill", "LiveBrokerNotImplementedError", "PaperBroker", "TargetOrder"]

log = get_logger("quantlab.paper.broker")


class LiveBrokerNotImplementedError(NotImplementedError):
    """Raised if anything tries to route an order to a real venue."""


@dataclass(frozen=True, slots=True)
class TargetOrder:
    """An instruction to reach a target position, in shares."""

    symbol: str
    target_shares: float
    #: Reference price used for sizing, so a fill can be compared against it.
    reference_price: float
    #: Liquidity inputs the cost model needs. Absent for an instrument the panel
    #: has no statistics for, which is refused rather than costed at zero.
    adv_notional: float | None = None
    volatility_daily: float | None = None
    spread_bps: float | None = None
    asset_class: AssetClass = AssetClass.EQUITY


@dataclass(frozen=True, slots=True)
class Fill:
    """What actually happened to an order."""

    symbol: str
    shares: float
    #: Price before costs -- the reference the order was sized against.
    reference_price: float
    #: Price including half-spread, impact and commission, signed by direction.
    fill_price: float
    cost: float
    when: dt.datetime

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.reference_price

    @property
    def slippage_bps(self) -> float:
        """Cost as basis points of the traded notional.

        Recorded per fill rather than aggregated, because the distribution is the
        diagnostic: a mean slippage that looks fine can hide a handful of orders
        that were far too large for the instrument.
        """
        if self.notional <= 0:
            return 0.0
        return self.cost / self.notional / BPS


class Broker(ABC):
    """The seam between the strategy and wherever its orders go."""

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Current holdings in shares, by symbol."""

    @abstractmethod
    def equity(self, marks: dict[str, float]) -> float:
        """Account equity, marked at the supplied prices."""

    @abstractmethod
    def submit(self, orders: Sequence[TargetOrder], when: dt.datetime) -> list[Fill]:
        """Reach the target positions, returning what filled."""


@dataclass(slots=True)
class PaperBroker(Broker):
    """A simulator that charges the backtest's costs and nothing less."""

    cash: float
    shares: dict[str, float] = field(default_factory=dict)
    costs: TransactionCostModel = field(default_factory=lambda: TransactionCostModel())
    #: Days over which each order is assumed to be worked. The same knob the
    #: backtest has, and it matters: impact is concave in participation, so an
    #: order worked over a day costs far less than the same order in one print.
    horizon_days: float = 1.0
    fills: list[Fill] = field(default_factory=list)

    def positions(self) -> dict[str, float]:
        return {symbol: qty for symbol, qty in self.shares.items() if qty != 0.0}

    def equity(self, marks: dict[str, float]) -> float:
        held = sum(qty * marks.get(symbol, 0.0) for symbol, qty in self.shares.items())
        return self.cash + held

    def submit(self, orders: Sequence[TargetOrder], when: dt.datetime) -> list[Fill]:
        filled: list[Fill] = []
        for order in orders:
            held = self.shares.get(order.symbol, 0.0)
            delta = order.target_shares - held
            if delta == 0.0 or not np.isfinite(delta):
                continue
            if order.reference_price <= 0 or not np.isfinite(order.reference_price):
                raise ValueError(
                    f"{order.symbol}: cannot fill at a reference price of "
                    f"{order.reference_price}. An instrument with no price is one "
                    "nobody could have traded, and filling it at zero would create "
                    "a free position."
                )

            cost = self._cost_of(order, delta)
            # Costs are paid in cash, and the fill price is the reference moved
            # against the trader by exactly that amount. Reporting a mid fill and
            # a separate cost line would let a careless reader net the position
            # value without the costs.
            notional = abs(delta) * order.reference_price
            fill_price = order.reference_price + np.sign(delta) * (
                cost / abs(delta) if delta else 0.0
            )

            self.shares[order.symbol] = held + delta
            self.cash -= delta * order.reference_price + cost

            fill = Fill(
                symbol=order.symbol,
                shares=delta,
                reference_price=order.reference_price,
                fill_price=float(fill_price),
                cost=cost,
                when=when,
            )
            filled.append(fill)
            self.fills.append(fill)
            log.debug(
                "paper.fill",
                symbol=order.symbol,
                shares=round(delta, 4),
                notional=round(notional, 2),
                slippage_bps=round(fill.slippage_bps, 2),
            )
        return filled

    def _cost_of(self, order: TargetOrder, delta: float) -> float:
        missing = [
            name
            for name, value in (
                ("adv_notional", order.adv_notional),
                ("volatility_daily", order.volatility_daily),
                ("spread_bps", order.spread_bps),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{order.symbol}: no {', '.join(missing)}, so this order cannot be "
                "costed. Trading it at zero cost would make the paper book beat "
                "its own backtest for no reason -- which is exactly how a paper "
                "loop stops being evidence about anything."
            )

        instrument = Instrument(
            symbol=order.symbol,
            asset_class=order.asset_class,
            adv_notional=float(order.adv_notional or 0.0),
            volatility_daily=float(order.volatility_daily or 0.0),
            spread_bps=float(order.spread_bps or 0.0),
        )
        notional = abs(delta) * order.reference_price
        breakdown = self.costs.estimate(
            Order(order.symbol, notional, horizon_days=self.horizon_days), instrument
        )
        return breakdown.currency(notional)
