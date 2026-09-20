"""Portfolio construction: sizing, volatility targeting, optimisation.

Signals produce scores; this produces weights. Keeping them apart is what lets two
signals be combined at all -- a signal that sizes its own positions has taken over
risk management, and two of them cannot both be right.

The rule that shapes this package (spec section 8): **turnover penalties live
inside the optimiser, not as a post-hoc filter.** A filter -- rebalance monthly,
or skip small trades -- discards information without asking what it was worth.
:mod:`~quantlab.portfolio.dynamic` asks the right question instead: given that the
signal will decay anyway, how much of the way toward it is worth paying to travel?

    from quantlab.portfolio import Constraints, ledoit_wolf_covariance, risk_parity

    covariance = ledoit_wolf_covariance(returns)
    allocation = risk_parity(covariance.matrix)
"""

from __future__ import annotations

from quantlab.portfolio.constraints import Constraints, GroupLimit, build_groups
from quantlab.portfolio.covariance import (
    CovarianceEstimate,
    correlation_from_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    matrix_condition_number,
    sample_covariance,
)
from quantlab.portfolio.dynamic import (
    DynamicTrade,
    TradingPolicy,
    markowitz_portfolio,
)
from quantlab.portfolio.optimise import (
    Allocation,
    equal_weight,
    inverse_volatility,
    mean_variance,
    risk_contributions,
    risk_parity,
)
from quantlab.portfolio.weights import (
    cross_sectional_long_short,
    time_series_weights,
    volatility_target,
)

__all__ = [
    "Allocation",
    "Constraints",
    "CovarianceEstimate",
    "DynamicTrade",
    "GroupLimit",
    "TradingPolicy",
    "build_groups",
    "correlation_from_covariance",
    "cross_sectional_long_short",
    "equal_weight",
    "ewma_covariance",
    "inverse_volatility",
    "ledoit_wolf_covariance",
    "markowitz_portfolio",
    "matrix_condition_number",
    "mean_variance",
    "risk_contributions",
    "risk_parity",
    "sample_covariance",
    "time_series_weights",
    "volatility_target",
]
