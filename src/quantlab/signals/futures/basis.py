"""Commodity basis, also called roll yield.

The slope of a futures curve is the carry of a futures position: a market in
backwardation pays you to hold it, one in contango charges you. It is one of the
strongest effects in commodities and the construction is trivial -- given two
points on the curve.

**Free data does not supply two points on the curve.** Spec section 3.4 is explicit
about this: continuous front-month series are all that is freely available, their
roll methodology is undocumented, and they carry no second contract. The only free
route to a real curve is exchange settlement bulletins, whose history starts when
you start collecting and whose terms of service may forbid automated access.

So this signal is implemented and refuses to run. That is the honest state of it,
and it is stated here rather than approximated with an ETF pair -- a USO/DBO spread
is a proxy for a proxy, and the error it introduces is of the same size as the
signal being measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from quantlab.conventions import RebalanceFrequency
from quantlab.data.catalogue import AssetClass
from quantlab.signals.base import (
    EvidenceGrade,
    Signal,
    SignalOutput,
    SignalSpec,
    SignalUnavailableError,
    register_signal,
)

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.data.pit import Snapshot

__all__ = ["CommodityBasis", "annualised_basis"]


def annualised_basis(near_price: float, far_price: float, months_between: float) -> float:
    """Annualised roll yield between two contracts on the same curve.

    Positive means backwardation: the near contract is dearer, so a long position
    rolls into a cheaper contract and earns the difference.
    """
    if near_price <= 0 or far_price <= 0:
        raise ValueError("prices must be positive")
    if months_between <= 0:
        raise ValueError("months_between must be positive")
    return (near_price / far_price - 1.0) * (12.0 / months_between)


@register_signal
class CommodityBasis(Signal):
    """Roll yield from the shape of the futures curve."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="carry.commodity_basis",
        asset_class=AssetClass.COMMODITY,
        tier=1,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("ohlcv_daily",),
        rebalance=RebalanceFrequency.MONTHLY,
        expected_turnover_annual=2.5,
        evidence=EvidenceGrade.STRONG,
        reference=(
            "Gorton & Rouwenhorst (2006), 'Facts and Fantasies about Commodity "
            "Futures'; Koijen, Moskowitz, Pedersen & Vrugt (2018), 'Carry'"
        ),
        known_failure_modes=(
            "Backwardation reflects scarcity, and scarcity ends -- often abruptly, "
            "when the supply shock that caused it resolves. The signal is therefore "
            "long exactly the markets most exposed to a supply normalisation. Curve "
            "shape is also heavily seasonal in agriculture and natural gas, so a "
            "naive basis rank is partly a calendar effect. Above all, the second "
            "contract point is not freely available: any implementation on free data "
            "is either proxied or reconstructed from exchange bulletins whose history "
            "begins when collection begins."
        ),
        warmup_days=0,
        notes=(
            "Cannot run on free data. Spec section 3.4: continuous front-month "
            "series carry no second contract, and an ETF-pair proxy introduces error "
            "of the same size as the signal."
        ),
    )

    def __init__(self, curve_symbols: dict[str, tuple[str, str, float]] | None = None) -> None:
        #: Commodity to ``(near symbol, far symbol, months between)``.
        self.curve_symbols = curve_symbols or {}

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        del symbols
        if not self.curve_symbols:
            raise SignalUnavailableError(
                f"{self.spec.name}: no futures curves configured, and free data "
                "supplies none. Continuous front-month series carry a single "
                "contract with an undocumented roll rule (spec section 3.4). "
                "Configure `curve_symbols` against a real curve source, or leave "
                "this signal out -- an ETF-pair proxy has error of the same size as "
                "the effect."
            )

        wanted = [s for pair in self.curve_symbols.values() for s in pair[:2]]
        bars = snapshot.ohlcv_daily(symbols=wanted)
        if bars.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name}: no prices for {wanted} at {snapshot.as_of:%Y-%m-%d}"
            )

        latest = bars.sort("as_of").group_by("symbol").agg(pl.col("close").last())
        prices = dict(zip(latest["symbol"], latest["close"], strict=True))

        scores: dict[str, float] = {}
        for commodity, (near, far, months) in self.curve_symbols.items():
            if near in prices and far in prices:
                scores[commodity] = annualised_basis(prices[near], prices[far], months)
        return self.finalise(snapshot, scores)
