"""Import every signal module so the registry is populated.

Modules are listed explicitly rather than scanned, so adding a signal is a
deliberate act and an import error surfaces as an import error rather than as a
mysteriously absent signal.
"""

from __future__ import annotations

import importlib

__all__ = ["SIGNAL_MODULES", "load_all_signals"]

SIGNAL_MODULES: tuple[str, ...] = (
    "quantlab.signals.positioning.hedger_pressure",
    "quantlab.signals.volatility.term_structure",
    "quantlab.signals.crypto.carry",
    "quantlab.signals.equity.profitability",
    "quantlab.signals.futures.basis",
    "quantlab.signals.futures.trend",
    "quantlab.signals.fx.carry",
    "quantlab.signals.rates.carry",
)


def load_all_signals() -> None:
    """Import every signal module. Idempotent."""
    for module in SIGNAL_MODULES:
        importlib.import_module(module)
