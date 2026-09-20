"""Backtest engines: vectorised research, event-driven execution, paper trading.

Milestone 4 delivers the vectorised engine. All three engines share the invariants
in spec section 6.4, and they are enforced rather than assumed:

* No forward-filling beyond a configured, documented staleness limit.
* An explicit rebalance-time convention -- signal on the close of T, traded at the
  open of T+1 by default. Trading at the signal's own close is available only with
  an explicit acknowledgement, and stamps the result as contaminated.
* Delisting returns applied when an instrument vanishes mid-sample.
* Position and cash accounting reconciled every bar by two independent routes,
  with a breach stopping the run.

Typical use::

    from quantlab.backtest import BacktestConfig, Panel, VectorisedBacktest

    panel = Panel.from_frame(prices)
    result = VectorisedBacktest(BacktestConfig()).run(panel, weights)
    print(result.summary())
"""

from __future__ import annotations

from quantlab.backtest.accounting import AccountingError, Ledger, ReconciliationReport
from quantlab.backtest.conventions import (
    DEFAULT_DELISTING_RETURN,
    ExecutionTiming,
    RebalanceFrequency,
    StalenessPolicy,
)
from quantlab.backtest.panel import Panel
from quantlab.backtest.results import BacktestResult, PerformanceStats
from quantlab.backtest.vectorised import BacktestConfig, VectorisedBacktest

__all__ = [
    "DEFAULT_DELISTING_RETURN",
    "AccountingError",
    "BacktestConfig",
    "BacktestResult",
    "ExecutionTiming",
    "Ledger",
    "Panel",
    "PerformanceStats",
    "RebalanceFrequency",
    "ReconciliationReport",
    "StalenessPolicy",
    "VectorisedBacktest",
]
