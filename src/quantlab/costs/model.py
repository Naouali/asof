"""The composite transaction cost model.

This is what a backtest actually calls. It assembles spread, impact, commission
and -- for a holding period -- financing and borrow into one
:class:`~quantlab.costs.base.CostBreakdown`.

Defaults per asset class live in :data:`ASSET_CLASS_DEFAULTS`. They differ because
the markets differ, not to be conservative for its own sake: crypto perpetuals have
observable spreads and continuous trading, US equities have a borrow problem, and
futures pay a roll that equities do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from quantlab.costs.base import BPS, CostBreakdown, CostModel, Instrument, Order
from quantlab.costs.financing import FinancingModel, FinancingParams
from quantlab.costs.impact import ImpactParams, SquareRootImpact
from quantlab.data.catalogue import AssetClass
from quantlab.logging import get_logger

__all__ = ["ASSET_CLASS_DEFAULTS", "AssetClassDefaults", "TransactionCostModel"]

log = get_logger("quantlab.costs.model")


@dataclass(frozen=True, slots=True)
class AssetClassDefaults:
    """Per-asset-class cost assumptions, with the reasoning attached."""

    impact: ImpactParams
    #: Default per-side commission where the venue charges one.
    commission_bps: float
    #: Rolls per year for a futures position. Zero for everything else.
    rolls_per_year: float
    rationale: str


ASSET_CLASS_DEFAULTS: dict[AssetClass, AssetClassDefaults] = {
    AssetClass.EQUITY: AssetClassDefaults(
        impact=ImpactParams(y=0.5, delta=0.5, permanent_fraction=0.5),
        commission_bps=0.5,
        rolls_per_year=0.0,
        rationale=(
            "Canonical square-root calibration. The binding cost for equity "
            "long/short is usually borrow, not impact -- see FinancingParams."
        ),
    ),
    AssetClass.FUTURES: AssetClassDefaults(
        impact=ImpactParams(y=0.4, delta=0.5, permanent_fraction=0.4),
        commission_bps=0.2,
        rolls_per_year=4.0,
        rationale=(
            "Liquid futures absorb size better than the equity average, so Y is "
            "lower. The roll is the cost that gets forgotten: a quarterly roll is "
            "eight one-way crossings a year, often more than the signal's own "
            "turnover."
        ),
    ),
    AssetClass.FX: AssetClassDefaults(
        impact=ImpactParams(y=0.3, delta=0.5, permanent_fraction=0.3),
        commission_bps=0.0,
        rolls_per_year=0.0,
        rationale=(
            "Deepest market, lowest impact. But free FX data is a single daily "
            "ECB fix with no bid/ask at all, so the spread fed to this model is "
            "always an assumption -- see docs/LIMITATIONS.md."
        ),
    ),
    AssetClass.CRYPTO: AssetClassDefaults(
        impact=ImpactParams(y=0.8, delta=0.5, permanent_fraction=0.5),
        commission_bps=4.0,
        rolls_per_year=0.0,
        rationale=(
            "Thinner books and higher Y than equities. Taker fees around 4 bp "
            "dominate the cost of a small trade, which is why crypto strategies "
            "die of turnover rather than of impact. Spreads here are *observed*, "
            "so this is the one asset class whose spread input is not a guess."
        ),
    ),
    AssetClass.CREDIT: AssetClassDefaults(
        impact=ImpactParams(y=1.0, delta=0.5, permanent_fraction=0.6),
        commission_bps=0.0,
        rolls_per_year=0.0,
        rationale=(
            "Corporate bonds trade by appointment. Reported prices are often stale "
            "or matrix-derived, so realised costs dwarf anything estimated from "
            "them; Y is set at the top of the published range and should still be "
            "treated as optimistic."
        ),
    ),
}


def _defaults_for(asset_class: AssetClass) -> AssetClassDefaults:
    if asset_class in ASSET_CLASS_DEFAULTS:
        return ASSET_CLASS_DEFAULTS[asset_class]
    # Unknown asset class falls back to the most punitive calibration rather than
    # the gentlest: an unclassified instrument is not evidence of a liquid one.
    return ASSET_CLASS_DEFAULTS[AssetClass.CREDIT]


@dataclass
class TransactionCostModel(CostModel):
    """Spread plus square-root impact plus commission, with financing on request.

    The model is per-asset-class by default; pass ``impact`` to override the
    calibration for a specific study. There is deliberately no way to configure it
    into a flat per-trade cost.
    """

    impact: SquareRootImpact = field(default_factory=SquareRootImpact)
    financing: FinancingModel = field(default_factory=FinancingModel)
    #: When true, the per-asset-class impact calibration is used in preference to
    #: ``impact``, which then only applies to unrecognised asset classes.
    use_asset_class_defaults: bool = True
    name: str = "square-root"

    @classmethod
    def for_asset_class(
        cls, asset_class: AssetClass, *, financing: FinancingParams | None = None
    ) -> TransactionCostModel:
        return cls(
            impact=SquareRootImpact(_defaults_for(asset_class).impact),
            financing=FinancingModel(financing),
            use_asset_class_defaults=False,
        )

    def _impact_for(self, instrument: Instrument) -> SquareRootImpact:
        if not self.use_asset_class_defaults:
            return self.impact
        return SquareRootImpact(_defaults_for(instrument.asset_class).impact)

    @staticmethod
    def _check_spread_provenance(instrument: Instrument) -> None:
        """Warn when a spread is estimated for an instrument whose quotes are free.

        Measured on 993 days of real Binance daily bars: against a true quoted
        BTCUSDT spread of 0.001 bp, Roll returned 167 bp, Corwin-Schultz 106 bp and
        Abdi-Ranaldo 62 bp -- and Abdi-Ranaldo was undefined outright for seven of
        eight majors. On a liquid, volatile instrument these estimators do not
        measure the spread at all; they measure daily volatility.

        Crypto venues publish real books for free, so estimating there is a choice,
        and a costly one: a 60 bp error on a strategy turning over weekly is tens
        of percent a year of imaginary cost.
        """
        if instrument.asset_class is AssetClass.CRYPTO and instrument.spread_source != "observed":
            log.warning(
                "costs.estimated_crypto_spread",
                symbol=instrument.symbol,
                spread_bps=instrument.spread_bps,
                spread_source=instrument.spread_source,
                remedy="crypto order books are free -- observe the spread, do not estimate it",
            )

    def _commission_for(self, instrument: Instrument) -> float:
        if instrument.commission_bps:
            return instrument.commission_bps
        return _defaults_for(instrument.asset_class).commission_bps

    # ----------------------------------------------------------------- scalar --
    def estimate(self, order: Order, instrument: Instrument) -> CostBreakdown:
        self._check_spread_provenance(instrument)
        impact = self._impact_for(instrument)
        permanent, temporary = impact.split(
            order.notional,
            instrument.adv_notional,
            instrument.volatility_daily,
            order.horizon_days,
        )
        participation = impact.participation(order.notional, instrument.adv_notional)
        extrapolating = impact.is_extrapolating(order.notional, instrument.adv_notional)
        if extrapolating:
            log.warning(
                "costs.impact_extrapolated",
                symbol=instrument.symbol,
                participation=round(participation, 4),
                limit=impact.params.max_reliable_participation,
                direction="the square-root law understates impact beyond its calibrated range",
            )

        return CostBreakdown(
            spread_bps=instrument.half_spread_bps,
            temporary_impact_bps=temporary,
            permanent_impact_bps=permanent,
            commission_bps=self._commission_for(instrument),
            detail={
                "model": self.name,
                "participation": participation,
                "extrapolating": extrapolating,
                "spread_source": instrument.spread_source,
                "y": impact.params.y,
                "delta": impact.params.delta,
            },
        )

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
    ) -> CostBreakdown:
        """Financing, borrow and funding for holding a position."""
        holding = self.financing.holding_cost(
            instrument,
            days=days,
            is_short=is_short,
            gross_notional=gross_notional,
            equity=equity,
            base_rate_annual_bps=base_rate_annual_bps,
            funding_rates=funding_rates,
        )
        return CostBreakdown(
            financing_bps=holding.financing_bps + holding.funding_bps,
            borrow_bps=holding.borrow_bps,
            detail={"assumed_borrow": holding.assumed_borrow, "holding_days": days},
        )

    def round_trip_bps(self, order: Order, instrument: Instrument) -> float:
        """Cost of entering and exiting a position of this size.

        Turnover is usually quoted one-way while the cost of a position is two-way,
        and conflating them halves every cost estimate in a strategy summary.
        """
        return 2.0 * self.estimate(order, instrument).total_bps

    # -------------------------------------------------------------- vectorised --
    def cost_bps_expr(
        self,
        notional: pl.Expr,
        adv_notional: pl.Expr,
        volatility_daily: pl.Expr,
        spread_bps: pl.Expr,
        horizon_days: pl.Expr | float = 1.0,
        commission_bps: pl.Expr | float = 0.0,
    ) -> pl.Expr:
        """Total one-way trade cost, matching :meth:`estimate`, for the fast engine.

        Uses ``self.impact`` rather than the per-asset-class calibration: a
        vectorised cross-section is costed with one parameter set, so build the
        frame per asset class, or pass an already-specialised model.
        """
        commission = (
            pl.lit(float(commission_bps))
            if isinstance(commission_bps, int | float)
            else commission_bps
        )
        impact = self.impact.cost_bps_expr(notional, adv_notional, volatility_daily, horizon_days)
        return spread_bps / pl.lit(2.0) + impact + commission

    def cost_currency_expr(
        self,
        notional: pl.Expr,
        adv_notional: pl.Expr,
        volatility_daily: pl.Expr,
        spread_bps: pl.Expr,
        horizon_days: pl.Expr | float = 1.0,
        commission_bps: pl.Expr | float = 0.0,
    ) -> pl.Expr:
        """Trade cost in currency. Grows like ``Q^1.5``, which is why capacity is finite."""
        return (
            self.cost_bps_expr(
                notional, adv_notional, volatility_daily, spread_bps, horizon_days, commission_bps
            )
            * pl.lit(BPS)
            * notional
        )
