"""Turning a point-in-time snapshot into a panel the engine can trade.

The lake stores what sources publish: prices and volumes. The cost model needs
what sources do not publish: average daily value, return volatility, and a spread.
Deriving them is where a backtest most easily acquires look-ahead bias, because the
obvious implementations all use the full sample.

Every statistic here is therefore **trailing only**: a rolling window ending at the
bar it describes, with a minimum-periods requirement, and the early bars where that
requirement is unmet are dropped rather than back-filled from the future. A full-
sample volatility estimate would tell a 2008 backtest how volatile 2008 turned out
to be, and would make every volatility-scaled position exactly right in hindsight.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from quantlab.costs.spread import AbdiRanaldoSpread
from quantlab.logging import get_logger

__all__ = ["DEFAULT_SPREAD_BPS", "DEFAULT_WINDOW", "panel_frame", "rebalance_dates"]

log = get_logger("quantlab.backtest.prepare")

#: Trailing window for both liquidity statistics. A quarter is long enough for the
#: estimates to be stable and short enough to react to a regime change.
DEFAULT_WINDOW = 63

#: Fallback spread where none can be measured or estimated. Deliberately wide: an
#: instrument whose spread cannot be established is not an instrument whose spread
#: is zero. Overriding it downward should require evidence.
DEFAULT_SPREAD_BPS = 20.0


def panel_frame(
    bars: pl.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    spread_bps: float | dict[str, float] | None = None,
    estimate_spreads: bool = False,
    min_bars: int | None = None,
) -> pl.DataFrame:
    """Derive the liquidity inputs the cost model needs, trailing only.

    ``bars`` is a long frame with ``symbol``, ``as_of``, ``close`` and ``volume``,
    and optionally ``open``. It should come from a
    :class:`~quantlab.data.pit.Snapshot`, so that what it contains was knowable at
    the as-of date; this function cannot check that and does not try.

    ``spread_bps`` supplies a spread directly -- one number, or one per symbol.
    ``estimate_spreads`` instead runs the Abdi-Ranaldo estimator per symbol, which
    is a **last resort**: on liquid, volatile instruments those estimators measure
    daily volatility rather than spread, sometimes by several orders of magnitude
    (see docs/LIMITATIONS.md). Prefer observed quotes, then a documented
    assumption, then estimation.
    """
    required = {"symbol", "as_of", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars frame is missing {sorted(missing)}")
    if "volume" not in bars.columns and "quote_volume" not in bars.columns:
        raise ValueError(
            "bars frame has neither `volume` nor `quote_volume`, so average daily "
            "value cannot be computed and no capacity-aware cost is possible. "
            "Supply one, or supply `adv_notional` directly."
        )

    minimum = min_bars if min_bars is not None else max(2, window // 3)
    frame = bars.sort("symbol", "as_of")

    # Quote volume is already a currency amount; share volume needs the price.
    if "quote_volume" in frame.columns:
        frame = frame.with_columns(pl.col("quote_volume").alias("_value"))
    else:
        frame = frame.with_columns((pl.col("close") * pl.col("volume")).alias("_value"))

    frame = frame.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("_log_return")
    ).with_columns(
        # Trailing windows, ending at the bar they describe. `rolling_*` in polars
        # is backward-looking, which is the property this whole module depends on.
        pl.col("_value")
        .rolling_mean(window_size=window, min_samples=minimum)
        .over("symbol")
        .alias("adv_notional"),
        pl.col("_log_return")
        .rolling_std(window_size=window, min_samples=minimum)
        .over("symbol")
        .alias("volatility_daily"),
    )

    before = frame.height
    frame = frame.filter(
        pl.col("adv_notional").is_not_null()
        & pl.col("volatility_daily").is_not_null()
        & (pl.col("adv_notional") > 0)
        & (pl.col("volatility_daily") > 0)
    )
    if frame.height < before:
        log.info(
            "backtest.prepare.warmup_dropped",
            rows=before - frame.height,
            window=window,
            min_bars=minimum,
            reason="trailing statistics need history; the alternative is to borrow it "
            "from the future",
        )

    frame = frame.with_columns(
        _spread_column(frame, spread_bps=spread_bps, estimate_spreads=estimate_spreads)
    )

    keep = ["symbol", "as_of", "close", "adv_notional", "volatility_daily", "spread_bps"]
    for optional in ("open", "dividend"):
        if optional in frame.columns:
            keep.append(optional)
    return frame.select(keep).sort("as_of", "symbol")


def _spread_column(
    frame: pl.DataFrame,
    *,
    spread_bps: float | dict[str, float] | None,
    estimate_spreads: bool,
) -> pl.Expr:
    if isinstance(spread_bps, int | float):
        return pl.lit(float(spread_bps)).alias("spread_bps")

    if isinstance(spread_bps, dict):
        unknown = set(frame["symbol"].unique().to_list()) - set(spread_bps)
        if unknown:
            log.warning(
                "backtest.prepare.spread_missing",
                symbols=len(unknown),
                fallback_bps=DEFAULT_SPREAD_BPS,
                note="an instrument whose spread is unknown is not one whose spread is zero",
            )
        mapping = pl.DataFrame(
            {
                "symbol": list(spread_bps.keys()),
                "_spread": [float(v) for v in spread_bps.values()],
            }
        )
        return (
            pl.col("symbol")
            .replace_strict(
                mapping["symbol"].to_list(),
                mapping["_spread"].to_list(),
                default=DEFAULT_SPREAD_BPS,
            )
            .alias("spread_bps")
        )

    if not estimate_spreads:
        return pl.lit(DEFAULT_SPREAD_BPS).alias("spread_bps")

    if not {"high", "low"} <= set(frame.columns):
        raise ValueError(
            "estimating spreads needs `high` and `low` columns. Without them, supply "
            "spread_bps explicitly rather than letting the engine assume a number."
        )

    estimates: dict[str, float] = {}
    for symbol, group in frame.group_by("symbol"):
        key = symbol[0] if isinstance(symbol, tuple) else symbol
        try:
            estimate = AbdiRanaldoSpread().estimate(
                group["close"].to_numpy(), group["high"].to_numpy(), group["low"].to_numpy()
            )
        except ValueError as exc:
            log.warning(
                "backtest.prepare.spread_estimate_failed",
                symbol=str(key),
                error=str(exc)[:120],
                fallback_bps=DEFAULT_SPREAD_BPS,
            )
            estimates[str(key)] = DEFAULT_SPREAD_BPS
            continue
        if not estimate.above_resolution:
            log.warning(
                "backtest.prepare.spread_below_resolution",
                symbol=str(key),
                estimate_bps=round(estimate.raw_bps, 3),
                floor_bps=round(estimate.resolution_floor_bps or 0.0, 3),
                note="the estimate is inside its own noise; it measures volatility, not spread",
            )
        estimates[str(key)] = estimate.spread_bps

    return (
        pl.col("symbol")
        .replace_strict(
            list(estimates.keys()), list(estimates.values()), default=DEFAULT_SPREAD_BPS
        )
        .alias("spread_bps")
    )


def rebalance_dates(
    dates: list[dt.datetime], *, every: int = 21, warmup: int = 0
) -> list[dt.datetime]:
    """Every ``n``-th bar after a warm-up, as a rebalance calendar.

    Bar-count based rather than calendar based, so a panel's own trading days
    define the schedule and a market holiday cannot silently skip a rebalance.
    """
    if every < 1:
        raise ValueError("every must be at least 1")
    return [
        date
        for index, date in enumerate(dates)
        if index >= warmup and (index - warmup) % every == 0
    ]
