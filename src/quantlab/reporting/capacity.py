"""Deriving a capacity estimate from a backtest that has already run.

Spec section 13 forbids a strategy result without a capacity estimate, and the
easiest way to comply badly is to attach one built from invented inputs. The
numbers here come from the run instead: alpha and turnover from its own P&L, and
liquidity from the panel it traded.

Two choices matter and both understate capacity, which is the direction to err.

**Alpha is measured gross, and exactly once.** Capacity solves for the size at
which trading costs consume the edge, so the edge entering that calculation must
be the one before costs -- the realised return with the whole cost drag added
back. Holding costs are then handed to the model separately, because they do not
scale with size the way impact does, and they must not also be netted out of the
gross figure: doing both charges the strategy twice.

**Alpha is geometric, not arithmetic.** It is derived from CAGR, which is lower
than the mean return by roughly half the variance -- about 50 bp a year at 10%
volatility. Costs are arithmetic drags, so the strictly consistent comparison
would use the arithmetic mean; CAGR understates the edge and therefore
understates capacity, which is the direction spec section 14 asks for when a
choice is genuinely ambiguous.

**Liquidity is weighted by what is traded, not by what is held.** A strategy that
rebalances a few names heavily and holds the rest has its capacity set by the
names it actually trades. Weighting by holdings spreads the trading across
positions that are never touched and overstates capacity.

A related trap, and the reason rebalances are counted from the curve rather than
from the position dates: held weights drift with prices on every bar, so a run
that trades twelve times a year has a non-zero weight change on all 252. Counting
those as rebalances tells the capacity model the strategy makes 252 small trades
instead of 12 large ones, and because impact is concave in size, that understates
impact and overstates capacity. On the example ETF run the difference is 248
rebalances a year against a true 12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from quantlab.costs import CapacityModel, UniverseLiquidity
from quantlab.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.backtest.panel import Panel
    from quantlab.backtest.results import BacktestResult
    from quantlab.costs.capacity import CapacityResult

__all__ = ["capacity_for", "traded_liquidity"]

log = get_logger("quantlab.reporting.capacity")

BPS = 1e-4


def traded_liquidity(result: BacktestResult, panel: Panel) -> UniverseLiquidity:
    """The traded universe's liquidity, weighted by how much each name is traded.

    Liquidity statistics are averaged over the bars on which the name was
    tradable, so a symbol that listed halfway through the sample is described by
    the period it existed rather than by a mean over bars it did not.
    """
    weights = result.positions
    if weights.height == 0:
        raise ValueError(
            "the run holds no positions, so there is no traded universe and no capacity to estimate"
        )

    duplicates = weights.height - weights.select("as_of", "symbol").n_unique()
    if duplicates:
        raise ValueError(
            f"{duplicates} position row(s) repeat a (date, symbol) pair. Pivoting "
            "them would need an aggregation, and any choice of one would silently "
            "invent a weight the engine never held."
        )

    wide = weights.pivot(index="as_of", on="symbol", values="weight").fill_null(0.0).sort("as_of")
    symbols = [c for c in wide.columns if c != "as_of"]
    held = wide.select(symbols).to_numpy()
    traded = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, held.shape[1])))).sum(axis=0)

    if traded.sum() <= 0:
        raise ValueError("the run never traded, so its capacity is unbounded and meaningless")

    index = panel.symbol_index()
    missing = [s for s in symbols if s not in index]
    if missing:
        raise ValueError(f"positions reference symbols absent from the panel: {missing}")

    columns = [index[s] for s in symbols]
    tradable = panel.tradable[:, columns]

    def average(matrix: np.ndarray) -> np.ndarray:
        values = np.where(tradable, matrix[:, columns], np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(values, axis=0)

    frame = pl.DataFrame(
        {
            "symbol": symbols,
            "weight": traded / traded.sum(),
            "adv_notional": average(panel.adv_notional),
            "volatility_daily": average(panel.volatility_daily),
            "spread_bps": average(panel.spread_bps),
        }
    ).drop_nulls()

    dropped = len(symbols) - frame.height
    if dropped:
        log.warning(
            "reporting.capacity.dropped_symbols",
            dropped=dropped,
            note="no liquidity statistics on any tradable bar",
        )
        frame = frame.with_columns(weight=pl.col("weight") / pl.col("weight").sum())
    return UniverseLiquidity(frame)


def capacity_for(
    result: BacktestResult,
    panel: Panel,
    *,
    horizon_days: float = 1.0,
    model: CapacityModel | None = None,
) -> CapacityResult:
    """Break-even AUM for a run, from that run's own alpha, turnover and liquidity.

    Holding costs -- borrow and financing -- are passed through separately rather
    than folded into the alpha, because they do not scale with size the way impact
    does: they are a constant drag that shifts the whole curve down, and mixing
    them into the per-rebalance alpha would make them look like a trading cost the
    strategy could avoid by trading less.
    """
    stats = result.stats
    if stats.years <= 0:
        raise ValueError("a run with no elapsed time has no capacity")

    rebalances_per_year = _rebalances_per_year(result, stats.years)
    gross_alpha_annual_bps = stats.cagr / BPS + stats.cost_drag_bps_annual

    decomposition = result.cost_decomposition_bps_annual()
    holding = decomposition.get("borrow", 0.0) + decomposition.get("financing", 0.0)
    # `holding` is passed to the model separately and applied there exactly once.
    # It must therefore stay inside the gross figure above, which is the return
    # before *every* cost: subtracting it here as well charged the strategy twice
    # and reported a net alpha of -143 bp/yr against a realised CAGR of -68.

    capacity = (model or CapacityModel()).solve(
        traded_liquidity(result, panel),
        gross_alpha_bps_per_rebalance=gross_alpha_annual_bps / rebalances_per_year,
        turnover_per_rebalance=stats.turnover_annual / rebalances_per_year,
        rebalances_per_year=rebalances_per_year,
        horizon_days=horizon_days,
        holding_cost_bps_annual=holding,
    )
    log.info(
        "reporting.capacity",
        name=result.name,
        gross_alpha_bps_annual=round(gross_alpha_annual_bps, 1),
        holding_bps_annual=round(holding, 1),
        rebalances_per_year=round(rebalances_per_year, 1),
        break_even_aum=capacity.break_even_aum,
    )
    return capacity


def _trading_bars(result: BacktestResult) -> np.ndarray | None:
    """The bars on which the strategy actually traded, from the equity curve."""
    if "traded_notional" not in result.curve.columns:
        return None
    traded: np.ndarray = result.curve.filter(pl.col("traded_notional") > 0)["as_of"].to_numpy()
    return traded


def _rebalances_per_year(result: BacktestResult, years: float) -> float:
    """How often the strategy actually traded, counted rather than assumed.

    Counted from bars with non-zero traded notional, never from the number of
    dates on which a position was held: held weights drift with prices every bar,
    so the second count is the bar count and has nothing to do with trading.
    """
    trading_bars = _trading_bars(result)
    if trading_bars is None:
        raise ValueError(
            "the equity curve has no traded_notional column, so the rebalance "
            "frequency cannot be measured. Guessing it would silently change the "
            "capacity estimate by an order of magnitude."
        )
    if len(trading_bars) < 2:
        raise ValueError(
            f"the run traded on {len(trading_bars)} bar(s), which is too few to "
            "measure a rebalance frequency"
        )
    return max(1.0, len(trading_bars) / years)
