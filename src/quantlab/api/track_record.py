"""What following somebody would actually have done.

The app's whole claim is that a disclosure is only worth what it was worth on the
day it became public. This module is that claim turned into a number: for each
filer, take every purchase they disclosed, buy it at the close of the day the
world found out, hold it for a fixed span, and compare the result with holding an
index over the same days.

Four rules keep the answer honest, and each of them makes the numbers smaller:

*Only finished windows count.* A trade disclosed ten days ago has no 90-day
return yet. Asking for one means either waiting or peeking, so a trade whose
window has not closed by the as-of date is not measured at all.

*The entry is the disclosure, never the trade.* The filer's own price is not
available to anybody else. It is reported separately, as the gap between what
they got and what a reader could have got, which is the thing this app exists to
show.

*A benchmark, not a raw return.* A year in which everything rose makes every
filer look skilled. What is reported is the difference from the same money in an
index over exactly the same days.

*Sample size, dispersion, and the luck of the crowd.* One good trade is not a
record. Every figure carries how many trades it rests on and how widely they
scattered; a record whose interval straddles zero is called what it is. With a
hundred filers measured, the best of them will look good by chance, so the
distribution of everybody is reported beside the leader board -- the only honest
way to see whether anybody stands out from it.
"""

from __future__ import annotations

import datetime as dt
import random
import statistics as stats
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from quantlab.api.analytics import _closes
from quantlab.api.events import eastern_date
from quantlab.api.portfolios import OPTION_TYPES, _identity, _trades
from quantlab.api.queries import Lens

__all__ = ["BENCHMARK", "HORIZONS", "MIN_TRADES", "records"]

#: What the same money in the market would have done. An index ETF, because it is
#: a thing a reader could actually have bought on the day.
BENCHMARK = "SPY"
#: Holding spans offered, in calendar days.
HORIZONS = (30, 90, 180)
DEFAULT_HORIZON = 90
#: Below this many finished trades a filer is measured but not ranked: a mean of
#: four trades is an anecdote with a decimal point.
MIN_TRADES = 10
#: A gap this long between a trade and its disclosure is a typo in a date.
MAX_LAG_DAYS = 400
#: Shuffles used to ask how good the best filer would look if nobody had any skill.
SHUFFLES = 400
#: The seed is fixed so the same lake gives the same answer twice. A figure that
#: moved every time the page was opened would be worth nothing.
SEED = 20260921


@dataclass(frozen=True, slots=True)
class _Measured:
    """One purchase, held from its disclosure for the horizon."""

    actor_id: str
    actor: str
    role: str | None
    ticker: str
    traded_on: dt.date
    public_on: dt.date
    excess_pct: float
    own_excess_pct: float | None


def _excess(
    prices: tuple[list[dt.date], list[float]],
    bench: tuple[list[dt.date], list[float]],
    entry: dt.date,
    horizon: int,
) -> float | None:
    """The position's return less the benchmark's, over the same days.

    Both legs are read at the first close ON OR AFTER the entry day -- the first
    price anyone acting on the news could have paid -- and at the last close on or
    before the exit day.
    """
    exit_on = entry + dt.timedelta(days=horizon)

    def at_entry(series: tuple[list[dt.date], list[float]]) -> float | None:
        days, closes = series
        index = bisect_left(days, entry)
        return closes[index] if index < len(days) else None

    def at_exit(series: tuple[list[dt.date], list[float]]) -> float | None:
        days, closes = series
        index = bisect_right(days, exit_on)
        return closes[index - 1] if index else None

    bought, sold = at_entry(prices), at_exit(prices)
    base, benched = at_entry(bench), at_exit(bench)
    if not bought or not sold or not base or not benched:
        return None
    return (sold / bought - 1) * 100 - (benched / base - 1) * 100


def _spread(values: list[float]) -> dict[str, Any]:
    """Mean, its interval, median and hit rate for one filer's measured trades."""
    count = len(values)
    mean = stats.fmean(values)
    # The spread of the mean itself: with a handful of trades it is wide enough to
    # swallow the mean, and saying so is the point.
    error = 1.96 * stats.stdev(values) / count**0.5 if count > 1 else float("inf")
    return {
        "trades": count,
        "mean_excess_pct": round(mean, 2),
        "low_pct": round(mean - error, 2) if count > 1 else None,
        "high_pct": round(mean + error, 2) if count > 1 else None,
        "median_excess_pct": round(stats.median(values), 2),
        "beat_rate": round(sum(1 for value in values if value > 0) / count, 3),
        #: True when the interval does not contain zero: the one case in which the
        #: record says something the size of the sample can support.
        "distinguishable": count > 1 and (mean - error) * (mean + error) > 0,
    }


def records(
    lens: Lens, *, horizon: int = DEFAULT_HORIZON, min_trades: int = MIN_TRADES
) -> dict[str, Any]:
    """Every filer's measured record at following-distance, ranked."""
    horizon = horizon if horizon in HORIZONS else DEFAULT_HORIZON
    frame = _trades(lens)
    if frame.height == 0:
        return _empty(lens, horizon, min_trades)

    rows = [
        row
        for row in frame.iter_rows(named=True)
        if row["transaction_type"] == "purchase"
        and row["symbol"] != "NO_TICKER"
        and row["asset_type"] not in OPTION_TYPES
    ]
    tickers = {str(row["symbol"]) for row in rows}
    earliest = min((row["as_of"].date() for row in rows), default=lens.today)
    series = _closes(lens, tickers | {BENCHMARK}, earliest - dt.timedelta(days=10))
    bench = series.get(BENCHMARK)
    if bench is None:
        return _empty(lens, horizon, min_trades, why=f"the lake holds no {BENCHMARK} prices")

    measured: list[_Measured] = []
    unpriced = unfinished = 0
    for row in rows:
        ticker = str(row["symbol"])
        prices = series.get(ticker)
        if prices is None:
            unpriced += 1
            continue
        traded = row["as_of"].date()
        public = _public_date(row)
        if (public - traded).days > MAX_LAG_DAYS:
            continue
        # Peeking check: the window must have closed by the date being viewed.
        if public + dt.timedelta(days=horizon) > lens.today:
            unfinished += 1
            continue
        excess = _excess(prices, bench, public, horizon)
        if excess is None:
            unpriced += 1
            continue
        actor_id, actor, role = _identity(row)
        measured.append(
            _Measured(
                actor_id=actor_id,
                actor=actor,
                role=role,
                ticker=ticker,
                traded_on=traded,
                public_on=public,
                excess_pct=excess,
                own_excess_pct=_excess(prices, bench, traded, horizon),
            )
        )

    by_filer: dict[str, list[_Measured]] = defaultdict(list)
    for item in measured:
        by_filer[item.actor_id].append(item)

    people = []
    for actor_id, items in by_filer.items():
        spread = _spread([item.excess_pct for item in items])
        own = [item.own_excess_pct for item in items if item.own_excess_pct is not None]
        people.append(
            {
                "actor_id": actor_id,
                "actor": items[0].actor,
                "role": items[0].role,
                **spread,
                #: What the filer's own timing was worth, for the same trades. The
                #: gap between this and the follower's figure is the cost of the delay.
                "own_mean_excess_pct": round(stats.fmean(own), 2) if own else None,
                "first": min(item.public_on for item in items),
                "last": max(item.public_on for item in items),
                "ranked": spread["trades"] >= min_trades,
            }
        )
    people.sort(key=lambda row: (row["ranked"], row["mean_excess_pct"]), reverse=True)

    everyone = [item.excess_pct for item in measured]
    ranked = [person for person in people if person["ranked"]]
    luck = _luck(measured, min_trades)
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "horizon_days": horizon,
        "benchmark": BENCHMARK,
        "min_trades": min_trades,
        "measured": len(measured),
        "filers": len(by_filer),
        "ranked": len(ranked),
        "unpriced": unpriced,
        "unfinished": unfinished,
        "everyone": _spread(everyone) if everyone else None,
        "standouts": sum(1 for person in ranked if person["distinguishable"]),
        #: How often chance alone produces a leader this good, and how many filers
        #: would be expected to clear a 95% interval by luck at this sample size.
        "luck": luck,
        "expected_by_luck": round(len(ranked) * 0.05, 1),
        "people": people,
        "why_empty": None,
    }


def _luck(measured: list[_Measured], min_trades: int) -> dict[str, Any] | None:
    """How good the best filer would look if none of them had any skill.

    The same trades are dealt out again at random, keeping how many each filer
    made, and the best average is noted. Repeating that says how often a leader as
    good as the real one appears from nothing at all -- which is the only way to
    read a leader board of a hundred people without fooling yourself.
    """
    counts = Counter(item.actor_id for item in measured)
    sizes = [count for count in counts.values() if count >= min_trades]
    if not sizes or len(measured) < min_trades:
        return None

    real = max(
        stats.fmean([item.excess_pct for item in measured if item.actor_id == actor])
        for actor, count in counts.items()
        if count >= min_trades
    )
    pool = [item.excess_pct for item in measured]
    # Shuffling to measure luck, not to keep a secret.
    rng = random.Random(SEED)  # noqa: S311
    beaten = 0
    for _ in range(SHUFFLES):
        rng.shuffle(pool)
        cursor = 0
        best = -float("inf")
        for size in sizes:
            best = max(best, stats.fmean(pool[cursor : cursor + size]))
            cursor += size
        if best >= real:
            beaten += 1
    return {
        "best_mean_excess_pct": round(real, 2),
        "shuffles": SHUFFLES,
        #: The share of shuffles in which chance produced a leader at least this
        #: good. Near 1 means the leader board is noise.
        "as_good_by_chance": round(beaten / SHUFFLES, 3),
    }


def _public_date(row: dict[str, Any]) -> dt.date:
    return eastern_date(row["known_at"])


def _empty(lens: Lens, horizon: int, min_trades: int, why: str | None = None) -> dict[str, Any]:
    return {
        "as_of": lens.as_of,
        "today": lens.today,
        "horizon_days": horizon,
        "benchmark": BENCHMARK,
        "min_trades": min_trades,
        "measured": 0,
        "filers": 0,
        "ranked": 0,
        "unpriced": 0,
        "unfinished": 0,
        "everyone": None,
        "standouts": 0,
        "luck": None,
        "expected_by_luck": 0.0,
        "people": [],
        "why_empty": why or "no congressional trades have been ingested yet",
    }
