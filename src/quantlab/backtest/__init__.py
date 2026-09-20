"""Backtest engines: vectorised research, event-driven execution, paper trading.

Milestones 4, 10 and 11. All three share the same invariants: no forward-fill
beyond a documented staleness limit, an explicit rebalance-time convention,
delisting returns applied, and cash/position accounting reconciled to the cent
every bar under assertion (spec section 6.4).
"""

from __future__ import annotations

__all__: list[str] = []
