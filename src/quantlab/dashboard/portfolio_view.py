"""Portfolio construction for the dashboard.

*How do these go together?* is a different question from *does this signal work*,
and it has a more reliable answer: correlation is estimable from a few hundred
observations in a way that expected return simply is not.

**Weights are the least interesting output here.** Risk contributions are what a
book is actually exposed to, and an equal-weighted portfolio of correlated assets
routinely puts most of its risk in one place while looking perfectly diversified
on the weights. Both are returned, always, and the UI shows them side by side.

**No expected returns are estimated.** `mean_variance` is offered with a flat
prior of zero, which makes it a minimum-variance solver. Sample means are noisy
enough that optimising on them reliably produces a worse portfolio than equal
weighting, and presenting an unstable estimate as a view is how a construction
tool starts doing harm.

This is a data-intelligence view. Nothing here places an order, sizes a real
book, or connects to a venue.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.config import get_settings
from quantlab.logging import get_logger

__all__ = ["build_portfolio"]

log = get_logger("quantlab.dashboard.portfolio")

METHODS = ("equal", "inverse-vol", "risk-parity", "min-variance")


def build_portfolio(
    symbols: list[str],
    *,
    as_of: str,
    method: str = "risk-parity",
    lookback: int = 504,
    max_position: float = 0.35,
) -> dict[str, Any]:
    """Construct a book and report what it is actually exposed to."""
    from quantlab.data.store import Store
    from quantlab.portfolio import (
        Constraints,
        correlation_from_covariance,
        equal_weight,
        inverse_volatility,
        ledoit_wolf_covariance,
        mean_variance,
        risk_contributions,
        risk_parity,
    )

    if method not in METHODS:
        return {"ok": False, "error": f"unknown method {method!r}; known: {', '.join(METHODS)}"}
    if len(symbols) < 2:
        return {"ok": False, "error": "a portfolio needs at least two instruments"}

    snapshot = Store(get_settings().layout).as_of(as_of)
    bars = snapshot.ohlcv_daily(symbols=symbols)
    if bars.height == 0:
        return {"ok": False, "error": f"no daily bars for {', '.join(symbols)} at {as_of}"}

    price = "adj_close" if "adj_close" in bars.columns else "close"
    wide = bars.sort("as_of").pivot(index="as_of", on="symbol", values=price).tail(lookback + 1)
    names = [c for c in wide.columns if c != "as_of"]
    complete = [c for c in names if wide[c].null_count() == 0]
    dropped = sorted(set(names) - set(complete))
    if len(complete) < 2:
        return {
            "ok": False,
            "error": "fewer than two instruments have a complete history over the window",
        }

    matrix = wide.select(complete).to_numpy()
    returns = np.diff(matrix, axis=0) / matrix[:-1]
    estimate = ledoit_wolf_covariance(returns)
    covariance = estimate.matrix

    try:
        if method == "equal":
            allocation = equal_weight(len(complete))
        elif method == "inverse-vol":
            allocation = inverse_volatility(covariance)
        elif method == "risk-parity":
            allocation = risk_parity(covariance)
        else:
            limits = Constraints(max_position=max_position, min_position=0.0, net_range=(1.0, 1.0))
            limits.check_feasible(len(complete))
            allocation = mean_variance(
                np.zeros(len(complete)), covariance, risk_aversion=5.0, constraints=limits
            )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    weights = allocation.weights
    contributions = risk_contributions(weights, covariance)
    total_risk = float(contributions.sum())
    shares = contributions / total_risk if total_risk > 0 else contributions

    correlation = correlation_from_covariance(covariance)
    off = correlation[~np.eye(len(complete), dtype=bool)]
    average_correlation = float(off.mean())
    # Independent bets under equicorrelation. An honest lower bound: a universe
    # with genuine sector structure diversifies somewhat better than this.
    n = len(complete)
    independent = n / (1 + (n - 1) * average_correlation) if average_correlation > -1 else n

    portfolio_vol = float(np.sqrt(weights @ covariance @ weights) * np.sqrt(252))
    volatilities = estimate.volatilities()

    return {
        "ok": True,
        "method": allocation.method,
        "as_of": as_of,
        "dropped": dropped,
        "positions": [
            {
                "symbol": complete[i],
                "weight": float(weights[i]),
                "volatility": float(volatilities[i]),
                "risk_share": float(shares[i]),
            }
            for i in range(len(complete))
        ],
        "correlation": {
            "symbols": complete,
            "matrix": [[float(v) for v in row] for row in correlation],
        },
        "portfolio": {
            "volatility": portfolio_vol,
            "average_correlation": average_correlation,
            "independent_bets": float(independent),
            "effective_positions": float(allocation.effective_positions),
            "instruments": n,
            "observations": int(returns.shape[0]),
            "condition_number": float(estimate.condition),
            "shrinkage": float(estimate.shrinkage),
            "converged": bool(allocation.converged),
        },
        "note": (
            "Weights are the less interesting half. Risk share is what the book is "
            "actually exposed to, and an equal-weighted portfolio of correlated "
            "assets routinely puts most of its risk in one place while looking "
            "diversified on the weights. No expected returns are estimated: "
            "min-variance uses a flat prior, because sample means are noisy enough "
            "that optimising on them reliably beats nothing."
        ),
    }


def portfolio_history(
    symbols: list[str], weights: list[float], as_of: str, lookback: int = 504
) -> dict[str, Any]:
    """Backward-looking value of holding these weights, rebalanced daily.

    Explicitly historical and explicitly costless -- it answers "how would these
    have moved together", not "what would this have returned". A rebalanced
    weight vector applied to past prices with no trading costs is not a backtest
    and is not recorded as a trial.
    """
    from quantlab.data.store import Store

    snapshot = Store(get_settings().layout).as_of(as_of)
    bars = snapshot.ohlcv_daily(symbols=symbols)
    if bars.height == 0:
        return {"ok": False, "error": "no bars"}

    price = "adj_close" if "adj_close" in bars.columns else "close"
    wide = bars.sort("as_of").pivot(index="as_of", on="symbol", values=price).tail(lookback + 1)
    present = [s for s in symbols if s in wide.columns]
    frame = wide.select(["as_of", *present]).drop_nulls()
    if frame.height < 2:
        return {"ok": False, "error": "not enough overlapping history"}

    matrix = frame.select(present).to_numpy()
    returns = np.diff(matrix, axis=0) / matrix[:-1]
    weight_vector = np.array([weights[symbols.index(s)] for s in present], dtype=float)
    scale = np.abs(weight_vector).sum()
    if scale > 0:
        weight_vector = weight_vector / scale

    portfolio = returns @ weight_vector
    curve = np.concatenate([[1.0], np.cumprod(1.0 + portfolio)])
    dates = frame["as_of"].to_list()
    step = max(1, len(curve) // 600)
    return {
        "ok": True,
        "points": [
            {"t": str(dates[i])[:10], "value": float(curve[i])} for i in range(0, len(curve), step)
        ],
        "note": "Costless and rebalanced daily. How they moved together, not a backtest.",
    }
