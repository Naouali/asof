"""Transaction costs, market impact, financing and capacity.

This module decides whether the platform is useful or decorative.

The rule that shapes it (spec section 13): **flat basis-point costs are not
available as a default anywhere**. They are roughly right for the small trades a
researcher tests on and wildly optimistic at deployment size, which makes every
strategy look scalable and every capacity estimate infinite.

Typical use::

    from quantlab.costs import Instrument, Order, TransactionCostModel

    model = TransactionCostModel.for_asset_class(AssetClass.EQUITY)
    breakdown = model.estimate(
        Order("AAPL", notional=5_000_000),
        Instrument("AAPL", AssetClass.EQUITY, adv_notional=8e9,
                   volatility_daily=0.018, spread_bps=1.2),
    )
    breakdown.total_bps       # what it cost
    breakdown.as_dict()       # where it went

and for the number every finished strategy needs::

    from quantlab.costs import CapacityModel, UniverseLiquidity

    CapacityModel().solve(universe, gross_alpha_bps_per_rebalance=25,
                          turnover_per_rebalance=0.3, rebalances_per_year=12).summary()
"""

from __future__ import annotations

from quantlab.costs.base import (
    BPS,
    CostBreakdown,
    CostModel,
    FlatBpsCostModel,
    Instrument,
    Order,
)
from quantlab.costs.capacity import CapacityModel, CapacityResult, UniverseLiquidity
from quantlab.costs.financing import FinancingModel, FinancingParams, HoldingCost
from quantlab.costs.impact import (
    AlmgrenChriss,
    ExecutionSchedule,
    ImpactParams,
    PropagatorImpact,
    SquareRootImpact,
)
from quantlab.costs.model import (
    ASSET_CLASS_DEFAULTS,
    AssetClassDefaults,
    TransactionCostModel,
)
from quantlab.costs.spread import (
    AbdiRanaldoSpread,
    CorwinSchultzSpread,
    ObservedSpread,
    RollSpread,
    SpreadCorrection,
    SpreadEstimate,
    SpreadEstimator,
    calibrate_correction,
)

__all__ = [
    "ASSET_CLASS_DEFAULTS",
    "BPS",
    "AbdiRanaldoSpread",
    "AlmgrenChriss",
    "AssetClassDefaults",
    "CapacityModel",
    "CapacityResult",
    "CorwinSchultzSpread",
    "CostBreakdown",
    "CostModel",
    "ExecutionSchedule",
    "FinancingModel",
    "FinancingParams",
    "FlatBpsCostModel",
    "HoldingCost",
    "ImpactParams",
    "Instrument",
    "ObservedSpread",
    "Order",
    "PropagatorImpact",
    "RollSpread",
    "SpreadCorrection",
    "SpreadEstimate",
    "SpreadEstimator",
    "SquareRootImpact",
    "TransactionCostModel",
    "UniverseLiquidity",
    "calibrate_correction",
]
