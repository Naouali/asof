"""Signal library.

A signal returns cross-sectional scores or time-series positions. It never returns
weights -- portfolio construction owns weights (spec section 4). Every signal
declares its required datasets, rebalance frequency, expected turnover, academic
reference and known failure modes.

Build order is by evidence quality, not interest: Tier 1 complete and correct
(Milestone 6) before Tier 2 (Milestone 10) before Tier 3.
"""

from __future__ import annotations

__all__: list[str] = []
