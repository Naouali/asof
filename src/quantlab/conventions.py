"""Vocabulary shared across layers.

A handful of terms -- rebalance frequency chief among them -- are used by signals,
by portfolio construction and by the backtest engine alike. They live here, outside
the layer stack, because putting them in any one layer forces the others to import
upward.

The architecture rule is that layers depend downward only, and a test walks the
import graph to enforce it. That test is what surfaced this: ``RebalanceFrequency``
started life in the backtest engine because that is where it was first needed, and
every signal declaring its own cadence then had to reach up into the engine to say
so.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["RebalanceFrequency"]


class RebalanceFrequency(str, Enum):
    """How often target weights are refreshed.

    Between rebalances, *shares* are held constant and weights drift with returns.
    Holding weights constant instead would imply trading every single day to
    maintain them -- free in a backtest, expensive in reality, and the difference
    is exactly the turnover this platform exists to charge for.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @property
    def approximate_per_year(self) -> float:
        return {
            RebalanceFrequency.DAILY: 252.0,
            RebalanceFrequency.WEEKLY: 52.0,
            RebalanceFrequency.MONTHLY: 12.0,
            RebalanceFrequency.QUARTERLY: 4.0,
        }[self]
