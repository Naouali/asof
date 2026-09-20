"""One paper-trading cycle.

Read a point-in-time snapshot, compute the signal, turn scores into weights,
size them against the current book, and fill the difference through the paper
broker at the backtest's own costs.

**The cycle is idempotent per as-of date.** Running twice for one day would
double the trades and put a step in the equity curve that nothing explains, so a
repeat is refused rather than absorbed. Re-running after a failure is safe.

**Nothing here forecasts.** The loop reads the snapshot at ``as_of`` and no
later, exactly as the backtest does, through the same :class:`Snapshot` guard.
That is the point of running paper at all: if the paper book and the backtest
diverge, the difference is execution and staleness rather than a different view
of the data.

**What paper trading is evidence of, and what it is not.** It is evidence that
the pipeline runs, that the signal produces positions on data it has not seen,
and that the costs are roughly what the backtest assumed. It is *not* evidence
that the strategy works: a few months of paper is far too short to distinguish a
Sharpe of 0.5 from zero (see :mod:`quantlab.paper.decay`), and no amount of it
fixes a signal that was overfitted before it started.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from quantlab.data.catalogue import AssetClass
from quantlab.logging import get_logger
from quantlab.paper.broker import Broker, PaperBroker, TargetOrder
from quantlab.paper.state import CycleRecord, PaperState, PositionRecord
from quantlab.signals.base import Signal, SignalUnavailableError

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from quantlab.data.pit import Snapshot
    from quantlab.data.store import Store

__all__ = ["PaperTradingLoop", "scores_to_weights"]

log = get_logger("quantlab.paper.loop")

#: Liquidity statistics are computed over this trailing window, matching the
#: backtest's default so the two price the same order the same way.
LIQUIDITY_WINDOW = 63


def scores_to_weights(scores: pl.DataFrame, *, gross: float = 1.0) -> dict[str, float]:
    """Dollar-neutral weights from cross-sectional scores, scaled to ``gross``.

    Demeaned then normalised by the sum of absolute deviations, so the book is
    neutral by construction and its gross exposure is exactly what was asked
    for. A signal with no dispersion produces no position rather than an
    arbitrary one -- if every instrument scores the same, the signal has no view
    and manufacturing one from rounding error is not a strategy.
    """
    if scores.height == 0:
        return {}
    values = scores["score"].to_numpy().astype(float)
    finite = np.isfinite(values)
    if not finite.any():
        return {}

    deviations = np.where(finite, values - values[finite].mean(), 0.0)
    scale = float(np.abs(deviations).sum())
    if scale <= 0:
        return {}
    weights = deviations / scale * gross
    return {
        str(symbol): float(weight)
        for symbol, weight in zip(scores["symbol"], weights, strict=True)
        if weight != 0.0
    }


@dataclass(slots=True)
class CycleOutcome:
    """What one cycle did, for the caller that has to report it."""

    record: CycleRecord
    skipped: bool = False
    reason: str = ""


class PaperTradingLoop:
    """Runs one strategy's paper book forward, one cycle at a time."""

    def __init__(
        self,
        signal: Signal,
        broker: Broker,
        state: PaperState,
        *,
        symbols: Sequence[str],
        gross_exposure: float = 1.0,
        asset_class: AssetClass = AssetClass.EQUITY,
    ) -> None:
        self.signal = signal
        self.broker = broker
        self.state = state
        self.symbols = list(symbols)
        self.gross_exposure = gross_exposure
        self.asset_class = asset_class

    def run_once(self, store: Store, as_of: dt.datetime) -> CycleOutcome:
        if self.state.has_run_for(as_of):
            return CycleOutcome(
                record=_empty_record(self.state.strategy, as_of),
                skipped=True,
                reason=(
                    f"a cycle for {as_of.isoformat()} is already recorded. Running "
                    "it again would double the trades and put a step in the equity "
                    "curve that nothing explains."
                ),
            )

        snapshot = store.as_of(as_of)
        try:
            scores = self.signal.compute(snapshot, self.symbols)
        except SignalUnavailableError as exc:
            return CycleOutcome(
                record=_empty_record(self.state.strategy, as_of),
                skipped=True,
                reason=f"signal unavailable: {exc}",
            )

        liquidity = self._liquidity(snapshot)
        marks = {symbol: stats["close"] for symbol, stats in liquidity.items()}
        equity = self.broker.equity(marks)
        targets = scores_to_weights(scores, gross=self.gross_exposure)

        orders: list[TargetOrder] = []
        unfilled: list[dict[str, Any]] = []
        for symbol, weight in targets.items():
            stats = liquidity.get(symbol)
            if stats is None or not np.isfinite(stats["close"]) or stats["close"] <= 0:
                # A target with no tradable price is a position the strategy is
                # not actually running. Recorded, because silence here looks
                # identical to the signal choosing to stay flat.
                unfilled.append({"symbol": symbol, "weight": weight, "reason": "no usable price"})
                continue
            orders.append(
                TargetOrder(
                    symbol=symbol,
                    target_shares=weight * equity / stats["close"],
                    reference_price=stats["close"],
                    adv_notional=stats["adv_notional"],
                    volatility_daily=stats["volatility_daily"],
                    spread_bps=stats["spread_bps"],
                    asset_class=self.asset_class,
                )
            )

        # Anything held but no longer wanted is closed, not left to drift.
        for symbol in self.broker.positions():
            if symbol not in targets and symbol in liquidity:
                stats = liquidity[symbol]
                orders.append(
                    TargetOrder(
                        symbol=symbol,
                        target_shares=0.0,
                        reference_price=stats["close"],
                        adv_notional=stats["adv_notional"],
                        volatility_daily=stats["volatility_daily"],
                        spread_bps=stats["spread_bps"],
                        asset_class=self.asset_class,
                    )
                )

        fills = self.broker.submit(orders, as_of)
        record = self._record(as_of, marks, fills, unfilled)
        self.state.append(record)
        return CycleOutcome(record=record)

    # ------------------------------------------------------------------ inputs --
    def _liquidity(self, snapshot: Snapshot) -> dict[str, dict[str, float]]:
        """Trailing liquidity statistics, built exactly as the backtest builds them."""
        from quantlab.backtest.prepare import panel_frame

        bars = snapshot.ohlcv_daily(symbols=self.symbols)
        if bars.height == 0:
            return {}
        keep = [c for c in ("symbol", "as_of", "close", "open", "volume") if c in bars.columns]
        frame = panel_frame(bars.select(keep), window=LIQUIDITY_WINDOW, spread_bps=None)
        latest = frame.sort("as_of").group_by("symbol").last()
        return {
            str(row["symbol"]): {
                "close": float(row["close"]),
                "adv_notional": float(row["adv_notional"]),
                "volatility_daily": float(row["volatility_daily"]),
                "spread_bps": float(row["spread_bps"]),
            }
            for row in latest.iter_rows(named=True)
            if row["close"] is not None
        }

    def _record(
        self,
        as_of: dt.datetime,
        marks: dict[str, float],
        fills: list[Any],
        unfilled: list[dict[str, Any]],
    ) -> CycleRecord:
        equity = self.broker.equity(marks)
        held = self.broker.positions()
        positions = [
            PositionRecord(
                symbol=symbol,
                shares=shares,
                mark=marks.get(symbol, float("nan")),
                value=shares * marks.get(symbol, 0.0),
                weight=(shares * marks.get(symbol, 0.0) / equity) if equity else 0.0,
            )
            for symbol, shares in sorted(held.items())
        ]
        values = [p.value for p in positions]
        cash = getattr(self.broker, "cash", equity - sum(values))

        return CycleRecord(
            strategy=self.state.strategy,
            as_of=as_of.isoformat(),
            ran_at=dt.datetime.now(tz=dt.UTC).isoformat(),
            equity=equity,
            cash=float(cash),
            gross_exposure=float(sum(abs(v) for v in values) / equity) if equity else 0.0,
            net_exposure=float(sum(values) / equity) if equity else 0.0,
            positions=positions,
            fills=[
                {
                    "symbol": f.symbol,
                    "shares": f.shares,
                    "reference_price": f.reference_price,
                    "fill_price": f.fill_price,
                    "cost": f.cost,
                    "slippage_bps": f.slippage_bps,
                }
                for f in fills
            ],
            traded_notional=float(sum(f.notional for f in fills)),
            costs_paid=float(sum(f.cost for f in fills)),
            unfilled=unfilled,
        )


def _empty_record(strategy: str, as_of: dt.datetime) -> CycleRecord:
    return CycleRecord(
        strategy=strategy,
        as_of=as_of.isoformat(),
        ran_at=dt.datetime.now(tz=dt.UTC).isoformat(),
        equity=0.0,
        cash=0.0,
        gross_exposure=0.0,
        net_exposure=0.0,
    )


def new_paper_broker(initial_equity: float) -> PaperBroker:
    """A fresh book, with the backtest's cost model attached."""
    return PaperBroker(cash=initial_equity)
