"""Validation: the module that saves the user money.

Milestone 5. Purged k-fold with embargo, combinatorial purged cross-validation,
walk-forward, deflated Sharpe with an AUTOMATIC trial counter, probability of
backtest overfitting, minimum backtest length, and empirical-Bayes luck adjustment
across the whole signal library (spec section 7).

The red-team suite in tests/ ships deliberately broken strategies. If the framework
passes a known-fake strategy, the framework is broken.
"""

from __future__ import annotations

__all__: list[str] = []
