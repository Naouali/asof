"""Turning signal scores into portfolio weights.

Signals produce scores; this produces weights. Keeping them apart is what lets two
signals be combined at all -- a signal that sizes its own positions has taken over
risk management, and two of them cannot both be right.

**These are the simplest constructions that work, not the ones Milestone 7 will
build.** There is no covariance estimate, no turnover penalty inside the
optimisation, no exposure constraint. What is here is enough to take a Tier 1
signal end to end and see what it costs; the Gârleanu-Pedersen style dynamic
programme that trades toward a slowly-moving aim portfolio is Milestone 7's job,
and it will change these numbers.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quantlab.logging import get_logger

__all__ = [
    "cross_sectional_long_short",
    "time_series_weights",
    "volatility_target",
]

log = get_logger("quantlab.portfolio.weights")


def cross_sectional_long_short(
    scores: pl.DataFrame, *, quantile: float = 0.25, gross: float = 1.0
) -> pl.DataFrame:
    """Equal-weight the top and bottom quantile, dollar neutral.

    Deliberately crude: it uses only the *ranking*, which is all a cross-sectional
    score claims to carry. Weighting by score magnitude would read information into
    a number that does not have it, and the extra concentration is rarely paid for.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    if gross <= 0:
        raise ValueError("gross must be positive")

    out: list[pl.DataFrame] = []
    for (as_of,), group in scores.group_by("as_of", maintain_order=True):
        ranked = group.sort("score")
        count = ranked.height
        k = max(1, int(count * quantile))
        if count < 2 * k:
            # Too few names to form both legs without overlapping them.
            continue

        shorts = ranked.head(k)
        longs = ranked.tail(k)
        weight = gross / 2.0 / k
        out.append(
            pl.DataFrame(
                {
                    "symbol": [*longs["symbol"].to_list(), *shorts["symbol"].to_list()],
                    "as_of": [as_of] * (2 * k),
                    "weight": [weight] * k + [-weight] * k,
                }
            )
        )

    if not out:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8(),
                "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
                "weight": pl.Float64(),
            }
        )
    return pl.concat(out).sort("as_of", "symbol")


def time_series_weights(
    scores: pl.DataFrame, *, max_gross: float = 1.0, scale: float = 1.0
) -> pl.DataFrame:
    """Scale time-series positions to a gross exposure cap.

    A time-series signal's score is already a position: each instrument is judged
    against its own history, so an all-long book is a legitimate output and is what
    trend following produces in a sustained move. All this does is cap the total.
    """
    if max_gross <= 0:
        raise ValueError("max_gross must be positive")

    out: list[pl.DataFrame] = []
    for (as_of,), group in scores.group_by("as_of", maintain_order=True):
        raw = group["score"].to_numpy() * scale
        gross = float(np.abs(raw).sum())
        if gross <= 0:
            continue
        # Scale down only. Scaling *up* a weak signal to hit a gross target turns
        # "no view" into a full position, which is how a flat month becomes a loss.
        weights = raw * min(1.0, max_gross / gross)
        out.append(
            pl.DataFrame(
                {
                    "symbol": group["symbol"].to_list(),
                    "as_of": [as_of] * group.height,
                    "weight": weights,
                }
            )
        )

    if not out:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8(),
                "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
                "weight": pl.Float64(),
            }
        )
    return pl.concat(out).sort("as_of", "symbol")


def volatility_target(
    weights: pl.DataFrame,
    realised_volatility: pl.DataFrame,
    *,
    target_annual: float = 0.10,
    max_leverage: float = 3.0,
) -> pl.DataFrame:
    """Scale the whole book toward a volatility target.

    Spec section 4 is careful about this one: implement volatility targeting as
    **risk management, not as alpha**. The factor-level "volatility-managed alpha"
    result is contested and largely does not survive costs, so the claim made here
    is only the modest one -- that a book held at roughly constant volatility is
    easier to size, to lever and to live through than one whose risk wanders.

    ``realised_volatility`` carries ``as_of`` and ``volatility`` (annualised,
    trailing). Leverage is capped, because a volatility target with no cap becomes
    a leverage machine in quiet markets -- which is precisely when quiet ends.
    """
    if target_annual <= 0:
        raise ValueError("target_annual must be positive")
    if max_leverage < 1:
        raise ValueError("max_leverage must be at least 1")

    scaled = (
        weights.join(realised_volatility, on="as_of", how="inner")
        .with_columns(
            multiplier=pl.when(pl.col("volatility") > 0)
            .then(
                pl.min_horizontal(
                    pl.lit(target_annual) / pl.col("volatility"), pl.lit(max_leverage)
                )
            )
            .otherwise(0.0)
        )
        .with_columns(weight=pl.col("weight") * pl.col("multiplier"))
    )

    capped = scaled.filter(pl.col("multiplier") >= max_leverage).height
    if capped:
        log.info(
            "portfolio.vol_target.capped",
            rows=capped,
            max_leverage=max_leverage,
            note="a volatility target with no cap becomes a leverage machine in quiet markets",
        )
    return scaled.select("symbol", "as_of", "weight").sort("as_of", "symbol")
