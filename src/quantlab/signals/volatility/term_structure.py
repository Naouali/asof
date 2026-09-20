"""VIX term structure.

The slope of the implied-volatility curve. In calm markets the curve is upward
sloping -- three-month implied volatility above one-month -- because demand for
longer-dated protection is steady while near-term realised volatility is low. In
a selloff the front end spikes and the curve inverts.

Trading the slope is a way of harvesting the variance risk premium: selling
near-term volatility when the curve is steep and standing aside when it inverts.

**Tier 2, and the reasons are the interesting part.**

*The premium is real and the payoff is not symmetric.* Selling volatility earns a
small positive return most of the time and occasionally loses a decade of it in a
week. The Sharpe ratio of a short-volatility strategy is a poor description of
it, and this platform's deflated Sharpe does not fix that: it corrects for how
hard you searched, not for a return distribution whose left tail is the whole
story. Read the skew and kurtosis on the tearsheet, not the ratio.

*The signal is not the trade.* VIX spot cannot be bought, so any implementation
goes through futures, options or ETPs, each with a roll cost that this signal
does not model and that has historically consumed much of the premium. A backtest
that treats the index level as tradeable is not describing anything that could
have been done.

*It is crowded, and the crowding is the mechanism.* The 2018 "Volmageddon"
episode was the short-volatility trade unwinding into itself. A signal whose
counterparties all hold the same position is one whose worst days are correlated
with everyone else's worst days.

**What the inversion rule actually did**, measured on the two episodes that
destroyed short-volatility strategies:

===========  ==========================  ==================================
Episode      Position going in           When it went flat
===========  ==========================  ==================================
Feb 2018     -0.62 on 1 February         2 February, slope 0.984 -- the day
                                         before VIX tripled from 17 to 37
Feb 2020     -0.57 on 18 February        24 February, having scaled down
                                         through the 20th and 21st
===========  ==========================  ==================================

It de-risked before the worst day in both cases, and in both cases only after
taking the first leg: VIX had already run 13.5 to 17.3 before the 2018 exit
triggered. This is a de-risking rule, not protection against a gap. It works when
the curve inverts *before* the crash rather than *with* it, and there is no
guarantee of that ordering. The exits above are also measured at the close, while
the platform's default convention trades the next open -- and VIX gapped
overnight on both occasions, so a realistic implementation is worse than these
numbers.

The signal is deliberately a **time-series position**, not a cross-sectional
score: there is one volatility curve, and ranking it against other instruments
would be ranking it against things it has no relationship with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import polars as pl

from quantlab.conventions import RebalanceFrequency
from quantlab.data.catalogue import AssetClass
from quantlab.logging import get_logger
from quantlab.signals.base import (
    EvidenceGrade,
    Signal,
    SignalOutput,
    SignalSpec,
    SignalUnavailableError,
    register_signal,
)

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.data.pit import Snapshot

__all__ = ["CONTANGO_FLOOR", "VixTermStructure", "term_structure_slope"]

log = get_logger("quantlab.signals.volatility.term_structure")

#: Below this ratio the curve is flat or inverted and the position is zero. Not
#: a tuned parameter: it is the point at which the premium being harvested stops
#: existing, and holding through an inversion is how a short-volatility strategy
#: turns a bad week into a terminal one.
CONTANGO_FLOOR = 1.0

#: The slope at which the position is fully on. A ratio of 1.15 is roughly the
#: median of the calm-market distribution, so this scales in over the normal
#: range rather than switching on at an arbitrary threshold.
FULL_POSITION_SLOPE = 1.15


def term_structure_slope(front: np.ndarray, back: np.ndarray) -> np.ndarray:
    """``back / front`` -- above one is contango, below is backwardation."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((front > 0) & (back > 0), back / front, np.nan)


@register_signal
class VixTermStructure(Signal):
    """Short volatility when the curve is steep; flat when it inverts."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="volatility.vix_term_structure",
        asset_class=AssetClass.OPTIONS,
        tier=2,
        output=SignalOutput.TIME_SERIES_POSITION,
        required_datasets=("series_observations",),
        rebalance=RebalanceFrequency.DAILY,
        expected_turnover_annual=8.0,
        evidence=EvidenceGrade.MIXED,
        reference=(
            "Carr & Wu (2009), 'Variance Risk Premiums'; Bollerslev, Tauchen & "
            "Zhou (2009), 'Expected Stock Returns and Variance Risk Premia'; "
            "Johnson (2017), 'Risk Premia and the VIX Term Structure'"
        ),
        known_failure_modes=(
            "The payoff is not symmetric: selling volatility earns a little most "
            "of the time and occasionally loses years of it in a week, so a Sharpe "
            "ratio is a poor description of the strategy and deflation does not "
            "fix that -- it corrects for search intensity, not for a left tail. "
            "VIX spot is not investable, so any implementation pays a roll cost "
            "through futures, options or ETPs that this signal does not model and "
            "that has historically consumed much of the premium. The trade is "
            "crowded, and the crowding is the mechanism: February 2018 was the "
            "short-volatility position unwinding into itself. Finally, the curve "
            "inverts fastest exactly when a position in it is largest."
        ),
        warmup_days=0,
        notes=(
            "A time-series position, not a cross-sectional score: there is one "
            "volatility curve, and ranking it against unrelated instruments would "
            "be meaningless. The position is capped at 1.0 -- short volatility "
            "with leverage is how the tail becomes terminal."
        ),
    )

    def __init__(
        self,
        *,
        front: str = "VIX",
        back: str = "VIX3M",
        max_position: float = 1.0,
    ) -> None:
        if max_position <= 0:
            raise ValueError("max_position must be positive")
        self.front = front
        self.back = back
        self.max_position = max_position

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        del symbols  # the curve is the universe
        series = snapshot.series(symbols=[self.front, self.back])
        if series.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name} needs {self.front} and {self.back} and the lake "
                f"has neither at {snapshot.as_of:%Y-%m-%d}. Ingest them with "
                "`quantlab data ingest -f cboe.series_observations`."
            )

        wide = (
            series.sort("as_of")
            .pivot(index="as_of", on="symbol", values="value")
            .drop_nulls()
            .sort("as_of")
        )
        missing = [s for s in (self.front, self.back) if s not in wide.columns]
        if missing or wide.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name}: no date carries both {self.front} and "
                f"{self.back}. {self.back} begins later than {self.front}, so an "
                "early snapshot legitimately has only one of them."
            )

        slope = term_structure_slope(wide[self.front].to_numpy(), wide[self.back].to_numpy())
        latest = float(slope[-1])
        if not np.isfinite(latest):
            raise SignalUnavailableError(
                f"{self.spec.name}: the latest curve is not computable at "
                f"{snapshot.as_of:%Y-%m-%d}."
            )

        # Negative: a steep curve is a SHORT volatility position. The premium is
        # paid to whoever sells the insurance, and getting this backwards buys
        # insurance continuously -- a slow, confident loss punctuated by one good
        # week.
        scaled = (latest - CONTANGO_FLOOR) / (FULL_POSITION_SLOPE - CONTANGO_FLOOR)
        position = -float(np.clip(scaled, 0.0, 1.0)) * self.max_position

        if latest <= CONTANGO_FLOOR:
            log.info(
                "signals.vix_term_structure.inverted",
                as_of=snapshot.as_of.isoformat(),
                slope=round(latest, 4),
                note="curve is flat or backwardated; the premium is not there to harvest",
            )
        return self.finalise(snapshot, {self.front: position})
