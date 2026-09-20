"""Signal library.

A signal returns cross-sectional scores or time-series positions. It never returns
weights -- portfolio construction owns weights (spec section 4). Every signal
declares its required datasets, rebalance frequency, expected turnover, academic
reference and, unusually, **how it is known to fail**.

Build order is by evidence quality, not by interest: Tier 1 complete and correct
before Tier 2 before Tier 3. Tier 3 signals are implemented anyway, because the
platform should let you re-test a decayed effect on current data -- with the prior
that it decayed attached to every result.

    from quantlab.signals import get_signal, SIGNAL_REGISTRY

    momentum = get_signal("trend.time_series_momentum")()
    scores = momentum.compute(snapshot, symbols)
"""

from __future__ import annotations

from quantlab.signals.base import (
    SIGNAL_REGISTRY,
    EvidenceGrade,
    Signal,
    SignalOutput,
    SignalSpec,
    SignalUnavailableError,
    get_signal,
    register_signal,
)
from quantlab.signals.loader import load_all_signals

__all__ = [
    "SIGNAL_REGISTRY",
    "EvidenceGrade",
    "Signal",
    "SignalOutput",
    "SignalSpec",
    "SignalUnavailableError",
    "get_signal",
    "load_all_signals",
    "register_signal",
]

load_all_signals()
