"""QuantLab -- a multi-asset quantitative research and backtesting platform.

The platform exists to tell the truth about whether a signal works. Every module
here is built to make optimistic results *harder* to produce: point-in-time data
access, square-root market impact, purged cross-validation and deflated Sharpe
ratios are defaults, not opt-ins.

Layering (dependencies point downward only):

    cli / dashboard / reporting
        -> backtest, validation, risk
            -> portfolio, signals, costs
                -> universe
                    -> data
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
