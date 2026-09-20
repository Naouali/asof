"""Paper trading: the loop, the broker seam, and the decay monitor.

Spec section 1 puts a hard boundary here. This package simulates execution and
leaves a clean interface where a live adapter would attach; it does not route
orders anywhere, and that is by instruction rather than by omission. See
:mod:`quantlab.paper.broker` for where such an adapter would go and what would
genuinely change if one were written.
"""

from __future__ import annotations

from quantlab.paper.broker import (
    Broker,
    Fill,
    LiveBrokerNotImplementedError,
    PaperBroker,
    TargetOrder,
)
from quantlab.paper.decay import DecayReport, assess_decay, sharpe_standard_error
from quantlab.paper.loop import PaperTradingLoop, scores_to_weights
from quantlab.paper.state import CycleRecord, PaperState, PositionRecord

__all__ = [
    "Broker",
    "CycleRecord",
    "DecayReport",
    "Fill",
    "LiveBrokerNotImplementedError",
    "PaperBroker",
    "PaperState",
    "PaperTradingLoop",
    "PositionRecord",
    "TargetOrder",
    "assess_decay",
    "scores_to_weights",
    "sharpe_standard_error",
]
