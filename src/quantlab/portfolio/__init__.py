"""Portfolio construction: sizing, volatility targeting, optimisation.

Milestone 7. Turnover penalties live inside the optimiser, not as a post-hoc
filter: a Garleanu-Pedersen style dynamic programme trades partially toward a
slowly-moving aim portfolio (spec section 8).
"""

from __future__ import annotations

__all__: list[str] = []
