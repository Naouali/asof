"""Intra-bar events, and the order they happen in.

The vectorised engine decides a bar's position in one expression. That is fast
and it cannot express anything path-dependent: a stop-loss depends on what the
position has done since it was opened, which is not a function of the target
weights alone. This module gives the event-driven engine an explicit ordering for
the things that can happen inside a single bar, so a rule that fires and a
rebalance that follows it compose in one defined way rather than in whichever way
the code happened to be written.

**What this does not buy.** With daily bars there is no intraday path, so an
event queue does not add execution fidelity: a stop still triggers at a price the
panel supplies, not at the price it would really have filled at somewhere inside
the day. What it adds is *sequencing* -- that a delisting is settled before a
rebalance sizes into the name, that a stop is evaluated on the position carried
into the bar rather than the one the rebalance is about to create. Those orderings
change results and are otherwise decided by accident. A genuinely event-driven
simulation needs intraday data, which no free source supplies (docs/LIMITATIONS.md).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

__all__ = ["Event", "EventKind", "PortfolioState", "event_queue"]


class EventKind(IntEnum):
    """What can happen inside a bar, ordered by when it is resolved.

    The values are the ordering, so ``sorted`` is the schedule. Each is placed
    where it is for a reason that changes results:

    ``DELISTING`` first, because an instrument that has vanished cannot be
    stopped out, rebalanced into, or marked -- it has to leave the book before
    anything else looks at it.

    ``STALE_EXIT`` next: an instrument with no fresh price is untradable, and a
    rule reading its last known price would be reading a price nobody could deal
    at.

    ``STOP`` before ``REBALANCE``, because a stop applies to the position carried
    *into* the bar. Evaluating it after the rebalance would test the new position
    against the old position's losses, which is a different and wrong rule.

    ``REBALANCE`` then ``MARK``: the book is valued after it is traded.
    """

    DELISTING = 0
    STALE_EXIT = 1
    STOP = 2
    REBALANCE = 3
    MARK = 4


@dataclass(frozen=True, slots=True)
class Event:
    """One thing happening to one book at one bar."""

    when: dt.datetime
    bar: int
    kind: EventKind
    #: Column indices this event concerns; empty means the whole book.
    symbols: tuple[int, ...] = ()

    def __lt__(self, other: Event) -> bool:
        return (self.when, self.kind) < (other.when, other.kind)


def event_queue(
    when: dt.datetime,
    bar: int,
    *,
    delisting: Sequence[int] = (),
    stale: Sequence[int] = (),
    stops: Sequence[int] = (),
    rebalance: bool = False,
) -> list[Event]:
    """The ordered events for one bar.

    ``MARK`` is always present: a bar with no activity still has to be valued, and
    omitting it would leave the equity curve with a hole on quiet days.
    """
    events = [Event(when, bar, EventKind.MARK)]
    if delisting:
        events.append(Event(when, bar, EventKind.DELISTING, tuple(delisting)))
    if stale:
        events.append(Event(when, bar, EventKind.STALE_EXIT, tuple(stale)))
    if stops:
        events.append(Event(when, bar, EventKind.STOP, tuple(stops)))
    if rebalance:
        events.append(Event(when, bar, EventKind.REBALANCE))
    return sorted(events)


@dataclass(slots=True)
class PortfolioState:
    """What a rule is allowed to see when it decides.

    Deliberately narrow. A rule gets the book as it stands, the prices of this
    bar, and the history of its own positions -- and nothing from later bars. The
    engine passes this object rather than the panel precisely so that a rule
    cannot reach forward: there is no bar index to index past.
    """

    bar: int
    when: dt.datetime
    #: Shares held coming into this bar.
    shares: np.ndarray
    #: Price this bar's trades fill at.
    exec_price: np.ndarray
    #: Price the book is marked at.
    close: np.ndarray
    #: True where the instrument has a price fresh enough to trade.
    tradable: np.ndarray
    equity: float
    #: Price at which each open position was entered, NaN where flat.
    entry_price: np.ndarray
    #: Bars each position has been open, zero where flat.
    bars_held: np.ndarray
    #: Bars remaining before a stopped instrument may be re-entered.
    cooldown: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def profit_since_entry(self) -> np.ndarray:
        """Fractional P&L of each open position since it was opened.

        Signed by the direction of the position, so a short that has fallen shows
        a profit. NaN where flat, so a rule cannot mistake "no position" for "no
        gain" -- and a comparison against NaN is False, which fails closed.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            move = np.where(
                np.isfinite(self.entry_price) & (self.entry_price > 0),
                self.close / self.entry_price - 1.0,
                np.nan,
            )
        return np.sign(self.shares) * move
