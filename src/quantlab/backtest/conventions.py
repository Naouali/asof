"""Rebalance timing, staleness and delisting.

Three conventions decide whether a backtest is a measurement or a fiction, and all
three are the kind of thing that is easy to get wrong silently:

**When the signal is acted on.** A signal computed from the close of T cannot be
traded at that same close -- the close is not knowable until it has happened. The
default is therefore to trade at the open of T+1. Trading at the signal's own close
is available, because the red-team suite needs a strategy that cheats, but it must
be asked for by name.

**How long stale data may be carried.** Forward-filling a price indefinitely turns
a halted or delisted instrument into a position that never moves and never loses.
The limit is configured and enforced, not assumed.

**What happens when an instrument disappears.** Free data simply stops serving a
delisted symbol. Treating its last observed price as a final settlement assumes the
holder got out whole, which for a performance-related delisting is badly wrong in a
direction that flatters long books.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quantlab.conventions import RebalanceFrequency
from quantlab.logging import get_logger

# RebalanceFrequency is re-exported for callers that already import it from here;
# it lives in quantlab.conventions because signals and portfolio construction
# need it too, and neither may import upward into the engine.
__all__ = [
    "DEFAULT_DELISTING_RETURN",
    "ExecutionTiming",
    "RebalanceFrequency",
    "StalenessPolicy",
]

log = get_logger("quantlab.backtest.conventions")

#: Return applied on the final bar of an instrument that disappears mid-sample.
#: The classic estimate for performance-related delistings is around -30%, and the
#: sign matters in both directions: omitting it overstates long-leg returns, and
#: understates what a short position would actually have earned. Free data supplies
#: no delisting returns at all, so this is an assumption that every equity result
#: inherits -- which is why it is a named constant rather than a literal.
DEFAULT_DELISTING_RETURN = -0.30


class ExecutionTiming(str, Enum):
    """When a signal computed on the close of T is actually traded."""

    NEXT_OPEN = "next_open"
    """Trade at the open of T+1. The default, and the most defensible with daily
    data: the signal uses information available at T's close, and the open of the
    following session is the first price anyone could actually transact at."""

    NEXT_CLOSE = "next_close"
    """Trade at the close of T+1. Realistic for a strategy that works orders
    through the day, and it is what a VWAP execution approximates -- though with
    free data there is no intraday path, so a true VWAP cannot be simulated."""

    SAME_CLOSE = "same_close"
    """**Look-ahead.** Trade at the very close that produced the signal.

    This is not a convention, it is a bug, and it is available only because the
    red-team suite in Milestone 5 needs a strategy that cheats in a known way.
    Selecting it requires ``acknowledge_look_ahead=True`` and stamps the result as
    contaminated, so a number produced this way can never be mistaken for one that
    was not."""


@dataclass(frozen=True, slots=True)
class StalenessPolicy:
    """How long a price may be carried forward before the position is closed.

    Forward-filling without a limit is one of the quieter ways to manufacture
    performance: a halted instrument holds its last price forever, so a position in
    it has zero volatility and zero drawdown, and a risk-parity weighting scheme
    will happily allocate more to it precisely because it looks calm.
    """

    #: Bars a price may be reused before the instrument is treated as untradable.
    max_bars: int = 5
    #: When true, a position in a stale instrument is liquidated at its last good
    #: price. When false, the whole run fails instead.
    liquidate_on_breach: bool = True

    def __post_init__(self) -> None:
        if self.max_bars < 0:
            raise ValueError("max_bars must be non-negative")

    @classmethod
    def strict(cls) -> StalenessPolicy:
        """No forward-filling at all. A missing price closes the position."""
        return cls(max_bars=0)

    def describe(self) -> str:
        if self.max_bars == 0:
            return "no forward-filling; a missing price closes the position"
        return (
            f"prices forward-filled up to {self.max_bars} bars, then the position "
            f"is {'liquidated' if self.liquidate_on_breach else 'an error'}"
        )
