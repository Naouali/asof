"""Running a signal across a rebalance calendar.

One snapshot per rebalance date, which is slower than computing the whole panel at
once and is the entire point: a signal that sees a single snapshot cannot see past
its own as-of instant, whatever its author intended. Computing the panel in one
pass would be faster and would make look-ahead a matter of author discipline
rather than of structure.

The cost is real -- a monthly rebalance over twenty years is 240 snapshot reads --
and it is the price of the guarantee.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from quantlab.data.store import Store
from quantlab.logging import get_logger
from quantlab.signals.base import Signal, SignalUnavailableError

__all__ = ["SignalRun", "monthly_dates", "run_signal"]

log = get_logger("quantlab.signals.runner")


@dataclass(frozen=True, slots=True)
class SignalRun:
    """Scores over a rebalance calendar, with the gaps recorded."""

    signal_name: str
    scores: pl.DataFrame
    #: Dates the signal was asked for, whether or not it produced anything.
    requested: tuple[dt.datetime, ...]
    #: Dates it produced no scores at all -- warm-up, or a data gap.
    empty_dates: tuple[dt.datetime, ...]

    @property
    def coverage(self) -> float:
        if not self.requested:
            return 0.0
        return 1.0 - len(self.empty_dates) / len(self.requested)

    def describe(self) -> str:
        return (
            f"{self.signal_name}: {self.scores.height:,} scores over "
            f"{len(self.requested)} rebalance dates "
            f"({self.coverage:.0%} produced a cross-section)"
        )


def run_signal(
    signal: Signal,
    store: Store,
    *,
    symbols: Sequence[str],
    dates: Sequence[dt.datetime],
    skip_unavailable: bool = False,
) -> SignalRun:
    """Compute ``signal`` at each date, through a snapshot taken at that date.

    ``skip_unavailable`` tolerates a signal that cannot run on some dates -- a
    fundamentals signal before its first filing, say. It does **not** tolerate a
    signal that cannot run at all: if every date is unavailable the error is
    re-raised, because a run that produced nothing should not look like a run that
    produced zeros.
    """
    frames: list[pl.DataFrame] = []
    empty: list[dt.datetime] = []
    failures = 0

    for date in dates:
        snapshot = store.as_of(date)
        try:
            scores = signal.compute(snapshot, symbols)
        except SignalUnavailableError:
            failures += 1
            empty.append(date)
            if not skip_unavailable:
                raise
            continue

        if scores.height == 0:
            empty.append(date)
            continue
        frames.append(scores)

    if failures == len(dates) and dates:
        raise SignalUnavailableError(
            f"{signal.spec.name} could not run on any of {len(dates)} dates. Its "
            f"required datasets are {signal.spec.required_datasets}; the lake has "
            "none of them in this window."
        )

    combined = (
        pl.concat(frames).sort("as_of", "symbol")
        if frames
        else signal.empty(store.as_of(dates[-1] if dates else dt.datetime.now(dt.UTC)))
    )
    run = SignalRun(
        signal_name=signal.spec.name,
        scores=combined,
        requested=tuple(dates),
        empty_dates=tuple(empty),
    )
    log.info(
        "signals.run",
        signal=signal.spec.name,
        dates=len(dates),
        scores=combined.height,
        coverage=round(run.coverage, 3),
    )
    return run


def monthly_dates(
    available: Sequence[dt.datetime], *, every: int = 21, warmup_bars: int = 0
) -> list[dt.datetime]:
    """A rebalance calendar drawn from the bars that actually exist.

    Counted in bars rather than calendar months, so a market holiday cannot skip a
    rebalance and a signal is never asked for a date on which nothing traded.
    """
    if every < 1:
        raise ValueError("every must be at least 1")
    return [
        date
        for index, date in enumerate(available)
        if index >= warmup_bars and (index - warmup_bars) % every == 0
    ]
