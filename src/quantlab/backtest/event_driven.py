"""The event-driven engine.

The vectorised engine in :mod:`quantlab.backtest.vectorised` computes a bar's
position from the target weights alone. That is the right design for research
throughput and it structurally cannot express a path-dependent rule: a stop-loss
depends on what a position has done since it was opened, which the weights do not
know.

This engine walks the same bars with the same accounting, but resolves an ordered
list of :class:`~quantlab.backtest.events.Event` inside each one and lets
:class:`Rule` objects modify the target before it is traded.

**The equivalence guarantee.** With no rules attached, this engine must produce
the same equity curve as the vectorised engine, to floating-point tolerance. That
is asserted by a test, and it is the whole basis for trusting either of them: two
independent implementations of the same convention agreeing is evidence the
convention is implemented; a divergence is a bug in one of them rather than a
difference of opinion. It also gives a strategy a free diagnostic. Run it both
ways, and if the numbers move once a stop is attached, the size of that move is
the part of the result that depends on the stop rather than on the signal.

**What it is not.** With daily bars there is no intraday path, so this is not a
higher-fidelity simulation of execution -- a stop still fills at a price the panel
supplies. What it adds is sequencing and path dependence, not resolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import polars as pl

from quantlab.backtest.accounting import Ledger
from quantlab.backtest.events import EventKind, PortfolioState, event_queue
from quantlab.backtest.results import BacktestResult
from quantlab.backtest.vectorised import DAYS_PER_YEAR, BacktestConfig, _day_fractions
from quantlab.costs.base import BPS
from quantlab.costs.model import TransactionCostModel
from quantlab.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.backtest.panel import Panel

__all__ = ["EventDrivenBacktest", "Rule", "StopLossRule", "TradingHaltRule"]

log = get_logger("quantlab.backtest.event_driven")


class Rule(ABC):
    """A path-dependent constraint on the book.

    Rules see :class:`PortfolioState`, which carries the book and this bar's
    prices and nothing from later bars. They return a target in **shares**, so a
    rule that wants a position closed returns zero for it rather than trying to
    express that as a weight of a quantity that is about to change.
    """

    #: Reported on the result, so a tearsheet can say which rules were active.
    name: ClassVar[str]

    @abstractmethod
    def apply(self, state: PortfolioState, target_shares: np.ndarray) -> np.ndarray:
        """Return the target this rule will allow, given the book as it stands."""

    def triggered(self, state: PortfolioState) -> np.ndarray:
        """Instruments this rule acts on this bar. Used only for event ordering."""
        del state
        return np.zeros(0, dtype=int)


class StopLossRule(Rule):
    """Close a position that has lost more than a threshold since it was opened.

    The rule the vectorised engine cannot express, because the trigger depends on
    the entry price rather than on the current target.

    Measured from the **peak since entry** by default, matching
    :class:`~quantlab.risk.controls.StopLossPolicy`. Measured from the entry price
    instead, a position that has run up must give all of it back before the stop
    fires, so the control switches itself off exactly on the long-held trend
    positions it is usually cited as helping.
    """

    name: ClassVar[str] = "stop_loss"

    def __init__(self, threshold: float = -0.20, cooldown_bars: int = 5) -> None:
        if threshold >= 0:
            raise ValueError("threshold is a negative fraction")
        if cooldown_bars < 0:
            raise ValueError("cooldown_bars must be non-negative")
        self.threshold = threshold
        self.cooldown_bars = cooldown_bars

    def triggered(self, state: PortfolioState) -> np.ndarray:
        profit = state.profit_since_entry()
        # NaN compares False, so a flat instrument never triggers.
        breached = np.nan_to_num(profit, nan=0.0) <= self.threshold
        return np.flatnonzero(breached & (state.shares != 0))

    def apply(self, state: PortfolioState, target_shares: np.ndarray) -> np.ndarray:
        out = target_shares.copy()
        out[self.triggered(state)] = 0.0
        # A cooling instrument stays flat however attractive the target says it is.
        if state.cooldown.size:
            out[state.cooldown > 0] = 0.0
        return out


class TradingHaltRule(Rule):
    """Refuse to trade an instrument whose price has not moved for N bars.

    A price that repeats exactly is usually a stale print rather than a quiet
    market, and sizing into one means trading at a price nobody is quoting.
    """

    name: ClassVar[str] = "trading_halt"

    def __init__(self, bars: int = 3) -> None:
        if bars < 1:
            raise ValueError("bars must be positive")
        self.bars = bars
        self._repeats: np.ndarray | None = None
        self._last: np.ndarray | None = None

    def apply(self, state: PortfolioState, target_shares: np.ndarray) -> np.ndarray:
        if self._last is None:
            self._repeats = np.zeros_like(state.close, dtype=int)
            self._last = state.close.copy()
            return target_shares

        assert self._repeats is not None  # noqa: S101 - set together with _last
        same = np.isclose(state.close, self._last, rtol=0.0, atol=1e-12)
        self._repeats = np.where(same, self._repeats + 1, 0)
        self._last = state.close.copy()

        halted = self._repeats >= self.bars
        out = target_shares.copy()
        out[halted & (state.shares == 0)] = 0.0
        return out


class EventDrivenBacktest:
    """Bar-by-bar simulation with ordered intra-bar events and path-dependent rules."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        costs: TransactionCostModel | None = None,
        *,
        rules: list[Rule] | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.costs = costs or TransactionCostModel(use_asset_class_defaults=False)
        self.rules = list(rules or [])

    def run(
        self,
        panel: Panel,
        weights: pl.DataFrame,
        *,
        name: str = "event-driven",
        family: str = "",
        record_trial: bool = True,
    ) -> BacktestResult:
        config = self.config
        if panel.timing is not config.execution:
            raise ValueError(
                f"the panel was built for {panel.timing.value} execution but the run "
                f"is configured for {config.execution.value}."
            )

        targets, is_rebalance = panel.align_weights(weights)
        n_bars, n_symbols = panel.shape

        lag = config.signal_lag_bars
        if lag:
            targets = np.vstack([np.zeros((lag, n_symbols)), targets[:-lag]])
            is_rebalance = np.concatenate([np.zeros(lag, dtype=bool), is_rebalance[:-lag]])

        ledger = Ledger(n_symbols=n_symbols, initial_equity=config.initial_equity)
        day_fractions = _day_fractions(panel.dates)
        columns = _blank_columns(n_bars)
        weights_held = np.zeros((n_bars, n_symbols), dtype=np.float64)

        entry_price = np.full(n_symbols, np.nan)
        peak_price = np.full(n_symbols, np.nan)
        bars_held = np.zeros(n_symbols, dtype=int)
        cooldown = np.zeros(n_symbols, dtype=int)

        previous_close = panel.close[0].copy()
        columns["equity"][0] = ledger.equity
        delisting_bar = np.where(panel.delisted, panel.last_observed, -1)
        stops_fired = 0

        for bar in range(1, n_bars):
            close = panel.close[bar]
            exec_price = panel.exec_price[bar].copy()
            tradable = panel.tradable[bar]
            shares_before = ledger.shares

            state = PortfolioState(
                bar=bar,
                when=panel.dates[bar],
                shares=shares_before,
                exec_price=exec_price,
                close=close,
                tradable=tradable,
                equity=ledger.equity,
                # Against the peak since entry, not the entry price itself.
                entry_price=np.where(shares_before > 0, peak_price, entry_price),
                bars_held=bars_held,
                cooldown=cooldown,
            )

            triggered = sorted({int(i) for rule in self.rules for i in rule.triggered(state)})
            queue = event_queue(
                panel.dates[bar],
                bar,
                delisting=np.flatnonzero(delisting_bar == bar).tolist(),
                stale=np.flatnonzero((shares_before != 0) & ~tradable).tolist(),
                stops=triggered,
                rebalance=bool(is_rebalance[bar]),
            )
            kinds = {event.kind for event in queue}

            # --- holding costs, on the position carried into the bar ----------
            days = day_fractions[bar]
            values_before = shares_before * np.nan_to_num(previous_close, nan=0.0)
            short_notional = float(np.abs(values_before[values_before < 0]).sum())
            gross_before = float(np.abs(values_before).sum())
            borrow = short_notional * config.borrow_bps_annual * BPS * days / DAYS_PER_YEAR
            borrowed = max(0.0, gross_before - ledger.equity)
            financing = borrowed * config.financing_bps_annual * BPS * days / DAYS_PER_YEAR

            # --- rebalance ----------------------------------------------------
            if EventKind.REBALANCE in kinds:
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

            # --- rules, in the order they were supplied ------------------------
            before_rules = shares_after.copy()
            for rule in self.rules:
                shares_after = rule.apply(state, shares_after)
            stopped = np.flatnonzero((before_rules != 0) & (shares_after == 0))
            if stopped.size:
                stops_fired += int(stopped.size)
                cooldown[stopped] = max(
                    (getattr(r, "cooldown_bars", 0) for r in self.rules), default=0
                )

            shares_after = np.where(tradable, shares_after, 0.0)

            # --- delisting ------------------------------------------------------
            settle_price = close.copy()
            if EventKind.DELISTING in kinds:
                delisting_now = delisting_bar == bar
                marked_down = np.nan_to_num(close, nan=0.0) * (1.0 + config.delisting_return)
                settle_price = np.where(delisting_now, marked_down, close)
                exec_price = np.where(delisting_now, marked_down, exec_price)
                shares_after = np.where(delisting_now, 0.0, shares_after)

            traded = shares_after - shares_before
            trade_notional = np.abs(traded) * np.nan_to_num(exec_price, nan=0.0)

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

            # --- position bookkeeping the rules depend on ----------------------
            opened = (shares_after != 0) & (shares_before == 0)
            closed = (shares_after == 0) & (shares_before != 0)
            entry_price = np.where(opened, exec_price, entry_price)
            entry_price = np.where(closed, np.nan, entry_price)
            peak_price = np.where(opened, exec_price, np.fmax(peak_price, settle_price))
            peak_price = np.where(closed, np.nan, peak_price)
            bars_held = np.where(shares_after != 0, bars_held + 1, 0)
            cooldown = np.maximum(cooldown - 1, 0)

            _record(
                columns,
                bar,
                equity=ledger.equity,
                gross_pnl=flows["gross_pnl"],
                spread=bar_spread,
                impact=bar_impact,
                commission=bar_commission,
                borrow=borrow,
                financing=financing,
                traded=float(trade_notional.sum()),
            )
            values_after = ledger.position_values(settle_price)
            columns["gross_exposure"][bar] = float(np.abs(values_after).sum())
            columns["net_exposure"][bar] = float(values_after.sum())
            columns["n_positions"][bar] = int((shares_after != 0).sum())
            if ledger.equity > 0:
                weights_held[bar] = values_after / ledger.equity

            previous_close = np.where(np.isnan(settle_price), previous_close, settle_price)

            if ledger.equity <= 0:
                log.error("backtest.ruin", bar=bar, date=str(panel.dates[bar]), engine="event")
                break

        if stops_fired:
            log.info(
                "backtest.rules_fired",
                stops=stops_fired,
                rules=[rule.name for rule in self.rules],
                note="a stopped position does not participate in the bounce",
            )

        return BacktestResult.build(
            name=name,
            family=family,
            panel=panel,
            config=config,
            record_trial=record_trial,
            weights=weights_held,
            reconciliation=ledger.report,
            **columns,
        )


def _blank_columns(n_bars: int) -> dict[str, np.ndarray]:
    names = (
        "equity",
        "gross_pnl",
        "trade_cost",
        "holding_cost",
        "spread_cost",
        "impact_cost",
        "commission_cost",
        "borrow_cost",
        "financing_cost",
        "traded_notional",
        "gross_exposure",
        "net_exposure",
    )
    columns = {name: np.zeros(n_bars, dtype=np.float64) for name in names}
    columns["n_positions"] = np.zeros(n_bars, dtype=np.int32)
    return columns


def _record(
    columns: dict[str, np.ndarray],
    bar: int,
    *,
    equity: float,
    gross_pnl: float,
    spread: float,
    impact: float,
    commission: float,
    borrow: float,
    financing: float,
    traded: float,
) -> None:
    columns["equity"][bar] = equity
    columns["gross_pnl"][bar] = gross_pnl
    columns["spread_cost"][bar] = spread
    columns["impact_cost"][bar] = impact
    columns["commission_cost"][bar] = commission
    columns["borrow_cost"][bar] = borrow
    columns["financing_cost"][bar] = financing
    columns["trade_cost"][bar] = spread + impact + commission
    columns["holding_cost"][bar] = borrow + financing
    columns["traded_notional"][bar] = traded
