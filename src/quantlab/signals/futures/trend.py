"""Time-series momentum, also called trend following.

Spec section 4 calls this *"the single most robust effect in the literature"*, and
it is the reason Tier 1 is built first. An instrument that has risen over the past
year tends to keep rising for a while; the effect appears across equity indices,
bonds, currencies and commodities, over a century of data, and it survived
publication, which almost nothing else does.

Construction, following Moskowitz, Ooi and Pedersen:

* Several lookbacks -- roughly one, three and twelve months -- because the effect
  exists at several horizons and blending them is far more stable than picking
  one. Picking one is also how a parameter sweep starts.
* Each lookback's return is divided by the volatility it was earned through, so a
  quiet instrument's 5% move counts for more than a violent one's. Without this the
  signal is dominated by whatever is most volatile rather than what is most trending.
* The blend is a score, not a position. Volatility targeting belongs to portfolio
  construction, which can see the whole book; a signal that sizes itself has
  quietly taken over risk management.

This is a **time-series** signal: each instrument is judged against its own
history, not against the cross-section. An all-long or all-short book is a
legitimate output, and is exactly what trend following does in a sustained move.
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
    register_signal,
)

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.data.pit import Snapshot

__all__ = ["DEFAULT_LOOKBACKS", "TimeSeriesMomentum"]

log = get_logger("quantlab.signals.trend")

#: One, three and twelve months in trading days. Three horizons rather than one,
#: because the choice of a single lookback is both unstable and the first step of
#: a parameter search nobody counts.
DEFAULT_LOOKBACKS: tuple[int, ...] = (21, 63, 252)

#: Trailing window for the volatility each lookback's return is scaled by.
DEFAULT_VOL_WINDOW = 63

#: Scores are clipped to this many standard deviations. A trend three sigma strong
#: is not three times as reliable as one sigma; without the clip a single
#: instrument's dislocation dominates the whole book.
CLIP = 2.0


@register_signal
class TimeSeriesMomentum(Signal):
    """Volatility-scaled trend, blended across lookbacks."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="trend.time_series_momentum",
        asset_class=AssetClass.FUTURES,
        tier=1,
        output=SignalOutput.TIME_SERIES_POSITION,
        required_datasets=("ohlcv_daily",),
        rebalance=RebalanceFrequency.MONTHLY,
        expected_turnover_annual=2.0,
        evidence=EvidenceGrade.STRONG,
        reference=(
            "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE; "
            "Hurst, Ooi & Pedersen (2017), 'A Century of Evidence on Trend-Following'"
        ),
        known_failure_modes=(
            "Loses money in choppy, mean-reverting markets and in sharp reversals "
            "from a sustained trend -- 2009 and 2020 are the canonical examples, "
            "where the positions accumulated over months were wrong within days. "
            "Returns have positive skew but long flat periods, so it is abandoned "
            "at the worst time. Crowding has compressed the effect since roughly "
            "2010 without eliminating it. Requires a genuinely diversified set of "
            "instruments: on a handful of correlated markets it is one bet."
        ),
        warmup_days=252 + DEFAULT_VOL_WINDOW,
        notes=(
            "The volatility scaling here is part of the SIGNAL -- comparing a move "
            "to the volatility it was earned through. Portfolio-level volatility "
            "targeting is a separate overlay and lives in quantlab.portfolio."
        ),
    )

    def __init__(
        self,
        lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
        *,
        vol_window: int = DEFAULT_VOL_WINDOW,
        clip: float = CLIP,
    ) -> None:
        if not lookbacks:
            raise ValueError("at least one lookback is required")
        if min(lookbacks) < 2:
            raise ValueError("lookbacks must be at least two bars")
        if vol_window < 2:
            raise ValueError("vol_window must be at least two bars")
        self.lookbacks = tuple(int(x) for x in lookbacks)
        self.vol_window = int(vol_window)
        self.clip = float(clip)

    @property
    def warmup_days(self) -> int:
        return max(self.lookbacks) + self.vol_window

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        bars = snapshot.ohlcv_daily(symbols=list(symbols))
        if bars.height == 0:
            raise_unavailable(self.spec.name, snapshot)

        scores: dict[str, float] = {}
        for (symbol,), group in bars.group_by("symbol", maintain_order=True):
            closes = group.sort("as_of")["close"].to_numpy()
            score = self.score_series(closes)
            if score is not None:
                scores[str(symbol)] = score

        if not scores:
            log.info(
                "signals.trend.no_history",
                as_of=snapshot.as_of.isoformat(),
                symbols=len(symbols),
                warmup_days=self.warmup_days,
                note="every instrument is still inside its warm-up window",
            )
        return self.finalise(snapshot, scores)

    def score_series(self, closes: np.ndarray) -> float | None:
        """Blended, volatility-scaled trend for one price series.

        Separated from :meth:`compute` so it can be tested directly against a
        hand-computed answer, without a lake or a snapshot in the way.
        """
        closes = np.asarray(closes, dtype=float)
        closes = closes[np.isfinite(closes) & (closes > 0)]
        if len(closes) < self.warmup_days:
            return None

        log_returns = np.diff(np.log(closes))
        volatility = float(np.std(log_returns[-self.vol_window :], ddof=1))
        if volatility <= 0:
            # A series with no variation has no trend to measure, and dividing by
            # it would produce an infinite score rather than an obvious error.
            return None

        components = []
        for lookback in self.lookbacks:
            if len(closes) <= lookback:
                continue
            move = float(np.log(closes[-1] / closes[-1 - lookback]))
            # Scale by the volatility the move was earned through, so the number is
            # comparable across instruments and across horizons.
            components.append(move / (volatility * np.sqrt(lookback)))

        if not components:
            return None
        return float(np.clip(np.mean(components), -self.clip, self.clip))


def raise_unavailable(name: str, snapshot: Snapshot) -> None:
    from quantlab.signals.base import SignalUnavailableError

    raise SignalUnavailableError(
        f"{name}: the snapshot at {snapshot.as_of:%Y-%m-%d} holds no daily bars for "
        "these instruments. Ingest them, or narrow the universe -- an empty score "
        "frame would look like a signal with no view rather than like missing data."
    )
