"""The vectorised research engine.

Daily, weekly or monthly rebalancing over a dense price panel, with full
square-root transaction costs, financing and borrow. Built for throughput: this is
the engine that screens signals and sweeps parameters, so it runs thousands of
times and its speed sets how much research is practical in a day.

**Shape of the computation.** The cross-section is vectorised; the time axis is a
loop. Equity at bar ``t`` depends on equity at ``t-1``, and costs depend on the
absolute size of each trade -- which depends on equity -- so the recursion is real
and cannot be collapsed into a cumulative product without approximating the cost
model. Approximating it is exactly the shortcut this platform exists to refuse, so
the loop stays, in numpy, with the whole cross-section vectorised inside it.

**What the engine assumes about its inputs**, none of which it can verify:

* Prices are on a consistent adjustment basis, and dividends are on the *same*
  basis. The data layer produces split-adjusted prices and split-adjusted dividend
  amounts, which are mutually consistent. Mixing bases misstates every total return
  by the split factor.
* Weights are targets known at the bar they are dated on. The engine applies the
  configured execution lag itself; pre-lagging them as well double-counts it and
  quietly makes the strategy worse.
* The panel's ADV, volatility and spread are point-in-time. The engine costs trades
  with them and cannot tell whether they were.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from quantlab.backtest.accounting import Ledger
from quantlab.backtest.conventions import (
    DEFAULT_DELISTING_RETURN,
    ExecutionTiming,
    StalenessPolicy,
)
from quantlab.backtest.panel import Panel
from quantlab.backtest.results import BacktestResult
from quantlab.costs.base import BPS
from quantlab.costs.model import TransactionCostModel
from quantlab.logging import get_logger

__all__ = ["BacktestConfig", "VectorisedBacktest"]

log = get_logger("quantlab.backtest.vectorised")

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Everything about a run that is not the data or the signal."""

    initial_equity: float = 10_000_000.0
    execution: ExecutionTiming = ExecutionTiming.NEXT_OPEN
    staleness: StalenessPolicy = field(default_factory=StalenessPolicy)
    #: Return applied on the final bar of an instrument that vanishes mid-sample.
    delisting_return: float = DEFAULT_DELISTING_RETURN
    #: Annualised borrow charged on short positions.
    borrow_bps_annual: float = 180.0
    #: Annualised all-in rate charged on notional beyond equity.
    financing_bps_annual: float = 550.0
    #: Fraction of a day over which each rebalance is worked. Shorter pays more
    #: temporary impact for the same size.
    execution_horizon_days: float = 1.0
    #: Per-side commission in basis points.
    commission_bps: float = 0.0
    #: Acknowledgement required to run with look-ahead execution.
    acknowledge_look_ahead: bool = False

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.execution is ExecutionTiming.SAME_CLOSE and not self.acknowledge_look_ahead:
            raise ValueError(
                "SAME_CLOSE execution trades at the very close that produced the "
                "signal, which is look-ahead bias, not a convention. It exists so "
                "the red-team suite can test a strategy that cheats in a known way. "
                "If that is what you want, pass acknowledge_look_ahead=True; the "
                "result will be stamped as contaminated."
            )
        if self.delisting_return > 0:
            raise ValueError(
                "delisting_return should be negative: instruments that vanish "
                "mid-sample overwhelmingly do so after doing badly"
            )

    @property
    def signal_lag_bars(self) -> int:
        """Bars between a signal being computed and being traded on."""
        return 0 if self.execution is ExecutionTiming.SAME_CLOSE else 1

    @property
    def is_contaminated(self) -> bool:
        return self.execution is ExecutionTiming.SAME_CLOSE


class VectorisedBacktest:
    """Run a weight schedule over a panel and report what it would have cost."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        costs: TransactionCostModel | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.costs = costs or TransactionCostModel(use_asset_class_defaults=False)

    # ------------------------------------------------------------------- run ---
    def run(
        self,
        panel: Panel,
        weights: pl.DataFrame,
        *,
        name: str = "strategy",
        family: str | None = None,
        record_trial: bool = True,
    ) -> BacktestResult:
        """Execute the weight schedule and return the full result.

        Every run records itself as a trial against ``family`` (defaulting to
        ``name``), and the resulting count feeds the deflated Sharpe on the result.
        Spec section 7 is explicit that this must be automatic: nobody remembers
        that they tried forty lookbacks last Tuesday, and the deflation is only as
        honest as the count behind it.

        Sweeping a parameter should keep one ``family`` and vary the config, so the
        forty variations count as forty trials rather than as forty families of one.
        ``record_trial=False`` exists for re-running a known configuration for
        reporting; it does not create a way to search without being counted,
        because an unrecorded run also gets no deflated Sharpe.
        """
        config = self.config
        if panel.timing is not config.execution:
            raise ValueError(
                f"the panel was built for {panel.timing.value} execution but the run "
                f"is configured for {config.execution.value}. The panel decides which "
                "price trades fill at and the config decides the signal lag; if they "
                "disagree the result reports a convention it did not use. Rebuild the "
                "panel with timing=ExecutionTiming."
                f"{config.execution.name}."
            )
        targets, is_rebalance = panel.align_weights(weights)
        n_bars, n_symbols = panel.shape

        lag = config.signal_lag_bars
        if lag:
            # A target dated on bar t is acted on at bar t+lag. Shifting here, once,
            # means no caller has to remember to do it -- and pre-lagged weights
            # would be double-lagged, which looks like the strategy simply being
            # worse rather than like a bug.
            targets = np.vstack([np.zeros((lag, n_symbols)), targets[:-lag]])
            is_rebalance = np.concatenate([np.zeros(lag, dtype=bool), is_rebalance[:-lag]])

        ledger = Ledger(n_symbols=n_symbols, initial_equity=config.initial_equity)
        day_fractions = _day_fractions(panel.dates)

        equity = np.empty(n_bars, dtype=np.float64)
        gross_pnl = np.zeros(n_bars, dtype=np.float64)
        trade_cost = np.zeros(n_bars, dtype=np.float64)
        holding_cost = np.zeros(n_bars, dtype=np.float64)
        spread_cost = np.zeros(n_bars, dtype=np.float64)
        impact_cost = np.zeros(n_bars, dtype=np.float64)
        commission_cost = np.zeros(n_bars, dtype=np.float64)
        borrow_cost = np.zeros(n_bars, dtype=np.float64)
        financing_cost = np.zeros(n_bars, dtype=np.float64)
        traded_notional = np.zeros(n_bars, dtype=np.float64)
        gross_exposure = np.zeros(n_bars, dtype=np.float64)
        net_exposure = np.zeros(n_bars, dtype=np.float64)
        n_positions = np.zeros(n_bars, dtype=np.int32)
        weights_held = np.zeros((n_bars, n_symbols), dtype=np.float64)

        previous_close = panel.close[0].copy()
        equity[0] = ledger.equity
        delisting_bar = np.where(panel.delisted, panel.last_observed, -1)

        for bar in range(1, n_bars):
            close = panel.close[bar]
            exec_price = panel.exec_price[bar].copy()
            tradable = panel.tradable[bar]

            # --- positions we can still hold ---------------------------------
            shares_before = ledger.shares
            forced_exit = (shares_before != 0) & ~tradable
            delisting_now = delisting_bar == bar

            # --- holding costs, charged on the position carried into the bar --
            days = day_fractions[bar]
            values_before = shares_before * np.nan_to_num(previous_close, nan=0.0)
            short_notional = float(np.abs(values_before[values_before < 0]).sum())
            gross_before = float(np.abs(values_before).sum())
            borrow = short_notional * config.borrow_bps_annual * BPS * days / DAYS_PER_YEAR
            borrowed = max(0.0, gross_before - ledger.equity)
            financing = borrowed * config.financing_bps_annual * BPS * days / DAYS_PER_YEAR

            # --- target position ---------------------------------------------
            if is_rebalance[bar]:
                # Size off pre-trade equity: the money you have when you decide,
                # not the money you will have once the costs are paid.
                pre_trade_equity = ledger.equity + float(
                    np.dot(shares_before, np.nan_to_num(exec_price - previous_close, nan=0.0))
                )
                target_notional = targets[bar] * pre_trade_equity
                with np.errstate(divide="ignore", invalid="ignore"):
                    shares_after = np.where(
                        tradable & (exec_price > 0), target_notional / exec_price, 0.0
                    )
                shares_after = np.nan_to_num(shares_after, nan=0.0)
            else:
                shares_after = shares_before.copy()

            # An untradable instrument cannot be held, whatever the target says.
            shares_after = np.where(tradable, shares_after, 0.0)

            # --- delisting -----------------------------------------------------
            # The exit happens AT the delisting price, so the marked-down price has
            # to reach `exec_price` and not only the mark. Applying it to the mark
            # alone closes the position at the last good price and records no loss
            # at all -- which is precisely the error this convention exists to
            # avoid, and it would have been invisible in the equity curve.
            settle_price = close.copy()
            if delisting_now.any():
                marked_down = np.nan_to_num(close, nan=0.0) * (1.0 + config.delisting_return)
                settle_price = np.where(delisting_now, marked_down, close)
                exec_price = np.where(delisting_now, marked_down, exec_price)
                shares_after = np.where(delisting_now, 0.0, shares_after)

            traded = shares_after - shares_before
            trade_notional = np.abs(traded) * np.nan_to_num(exec_price, nan=0.0)

            # --- trade costs ---------------------------------------------------
            if trade_notional.any():
                spread_bps = panel.spread_bps[bar] / 2.0
                impact_bps = self.costs.impact.cost_bps_array(
                    trade_notional,
                    panel.adv_notional[bar],
                    panel.volatility_daily[bar],
                    config.execution_horizon_days,
                )
                bar_spread = float(np.dot(trade_notional, spread_bps) * BPS)
                bar_impact = float(np.dot(trade_notional, impact_bps) * BPS)
                bar_commission = float(trade_notional.sum() * config.commission_bps * BPS)
            else:
                bar_spread = bar_impact = bar_commission = 0.0

            costs_total = bar_spread + bar_impact + bar_commission
            holding_total = borrow + financing

            flows = ledger.settle(
                bar=bar,
                shares_after=shares_after,
                exec_price=exec_price,
                previous_close=previous_close,
                close=settle_price,
                dividend_per_share=panel.dividend[bar],
                trade_costs=costs_total,
                holding_costs=holding_total,
            )

            equity[bar] = ledger.equity
            gross_pnl[bar] = flows["gross_pnl"]
            trade_cost[bar] = costs_total
            holding_cost[bar] = holding_total
            spread_cost[bar] = bar_spread
            impact_cost[bar] = bar_impact
            commission_cost[bar] = bar_commission
            borrow_cost[bar] = borrow
            financing_cost[bar] = financing
            traded_notional[bar] = float(trade_notional.sum())
            values_after = ledger.position_values(settle_price)
            gross_exposure[bar] = float(np.abs(values_after).sum())
            net_exposure[bar] = float(values_after.sum())
            n_positions[bar] = int((shares_after != 0).sum())
            if ledger.equity > 0:
                weights_held[bar] = values_after / ledger.equity

            if forced_exit.any():
                log.debug("backtest.stale_exit", bar=bar, positions=int(forced_exit.sum()))

            previous_close = np.where(np.isnan(settle_price), previous_close, settle_price)

            if ledger.equity <= 0:
                log.error(
                    "backtest.ruin",
                    bar=bar,
                    date=str(panel.dates[bar]),
                    note="equity reached zero; the remaining bars are not simulated",
                )
                equity[bar:] = ledger.equity
                break

        return BacktestResult.build(
            name=name,
            family=family or name,
            record_trial=record_trial,
            panel=panel,
            config=config,
            equity=equity,
            gross_pnl=gross_pnl,
            trade_cost=trade_cost,
            holding_cost=holding_cost,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            commission_cost=commission_cost,
            borrow_cost=borrow_cost,
            financing_cost=financing_cost,
            traded_notional=traded_notional,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            n_positions=n_positions,
            weights=weights_held,
            reconciliation=ledger.report,
        )


def _day_fractions(dates: tuple) -> np.ndarray:  # type: ignore[type-arg]
    """Calendar days between consecutive bars.

    Calendar, not trading, days: borrow and financing accrue over a weekend and
    over every holiday. Charging 252 days a year instead of 365 understates
    financing by nearly a third, which for a levered carry strategy is the whole
    edge.
    """
    stamps = np.array([d.timestamp() for d in dates], dtype=np.float64)
    gaps = np.diff(stamps) / 86400.0
    return np.concatenate([[0.0], gaps])
