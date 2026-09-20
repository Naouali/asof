"""Factor risk model, exposure attribution and risk controls.

This package exists to answer one question about every signal (spec section 8):
**is it secretly just beta, or size, or a sector bet?** That is not "does this make
money" but "would a passive position have made the same money more cheaply", and
it is the question a strategy's author is least able to ask themselves.

    from quantlab.risk import attribute

    print(attribute(returns, factors, factor_names=("MKT", "SMB", "HML", "MOM")).verdict())

Also here: drawdown controls and instrument-level stops. Both are risk management,
never alpha -- a drawdown control that improves backtested returns is almost
certainly fitted to the particular drawdowns in the sample.
"""

from __future__ import annotations

from quantlab.risk.controls import DrawdownControl, StopLossPolicy, drawdown_series
from quantlab.risk.factor_model import (
    FactorAttribution,
    PCARiskModel,
    attribute,
    newey_west_covariance,
    pca_risk_model,
)

__all__ = [
    "DrawdownControl",
    "FactorAttribution",
    "PCARiskModel",
    "StopLossPolicy",
    "attribute",
    "drawdown_series",
    "newey_west_covariance",
    "pca_risk_model",
]
