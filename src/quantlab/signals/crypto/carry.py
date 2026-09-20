"""Perpetual futures funding carry.

A perpetual swap has no expiry, so it is tethered to spot by a periodic payment
between longs and shorts. When the rate is positive, longs pay shorts; when it is
negative, shorts pay longs. Holding the side that receives is a carry trade, and
it is the cleanest carry available anywhere in the free-data universe: the
payments are observed, complete, and published without a key.

The sign convention is the thing to get right. This signal scores the **expected
carry from holding a long position**, so it is the *negative* of the funding rate:
persistently positive funding makes a long expensive and a short paid. Getting
this backwards produces a strategy that systematically takes the losing side of
every carry, which backtests as a consistent, confident, slow loss.

Carry is not free money. It is compensation for a risk, and in crypto the risk is
concentrated exactly where the carry is: funding is most extreme when positioning
is most crowded, and a liquidation cascade moves the price against the crowded side
while making the position hardest to exit.
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

__all__ = ["PerpetualFundingCarry"]

log = get_logger("quantlab.signals.crypto.carry")

HOURS_PER_YEAR = 24 * 365
DEFAULT_LOOKBACK_DAYS = 30

#: Funding this extreme is usually a dislocation rather than a carry opportunity:
#: 200% annualised means the market is paying almost anything to stay positioned,
#: which is the moment before it stops. Scores are clipped here.
CLIP_ANNUAL = 2.0


@register_signal
class PerpetualFundingCarry(Signal):
    """Expected annualised carry from holding a perpetual long."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="carry.crypto_perp_funding",
        asset_class=AssetClass.CRYPTO,
        tier=1,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("funding_rate",),
        rebalance=RebalanceFrequency.WEEKLY,
        expected_turnover_annual=6.0,
        evidence=EvidenceGrade.STRONG,
        reference=(
            "Koijen, Moskowitz, Pedersen & Vrugt (2018), 'Carry', JFE -- the "
            "cross-asset framing; perpetual funding is its crypto instance"
        ),
        known_failure_modes=(
            "Carry is compensation for risk, and in crypto the risk sits exactly "
            "where the carry does: funding is most extreme when positioning is most "
            "crowded, and a liquidation cascade moves price against the crowded side "
            "while making the position hardest to exit. Returns are strongly "
            "negatively skewed -- many small gains, occasional large losses -- which "
            "flatters every Sharpe ratio computed on them. The funding formula, cap "
            "and interval have all changed over time and differ by venue, so a "
            "backtest spanning a regime change is comparing different instruments. "
            "Shorting the perp to harvest positive funding also carries borrow and "
            "exchange counterparty risk that no return series shows."
        ),
        warmup_days=DEFAULT_LOOKBACK_DAYS,
        notes=(
            "Scores the carry from a LONG position, i.e. the negative of funding. "
            "A positive score means being long is paid."
        ),
    )

    def __init__(
        self, lookback_days: int = DEFAULT_LOOKBACK_DAYS, *, clip_annual: float = CLIP_ANNUAL
    ) -> None:
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1")
        self.lookback_days = int(lookback_days)
        self.clip_annual = float(clip_annual)

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        import datetime as dt

        start = snapshot.as_of - dt.timedelta(days=self.lookback_days)
        funding = snapshot.funding(symbols=list(symbols), start=start)
        if funding.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name}: no funding observations in the "
                f"{self.lookback_days} days to {snapshot.as_of:%Y-%m-%d}. Ingest "
                "binance.funding_rate, or drop this signal -- an empty score frame "
                "would look like a signal with no view rather than like missing data."
            )

        scores: dict[str, float] = {}
        for (symbol,), group in funding.group_by("symbol", maintain_order=True):
            score = self.score_series(
                group["funding_rate"].to_numpy(), group["interval_hours"].to_numpy()
            )
            if score is not None:
                scores[str(symbol)] = score
        return self.finalise(snapshot, scores)

    def score_series(self, funding_rates: np.ndarray, interval_hours: np.ndarray) -> float | None:
        """Annualised carry from a long, for one instrument.

        The interval is measured rather than assumed: it has changed over time and
        differs by contract, and annualising an eight-hour rate as though it were
        daily overstates the carry threefold.
        """
        rates = np.asarray(funding_rates, dtype=float)
        intervals = np.asarray(interval_hours, dtype=float)
        usable = np.isfinite(rates) & np.isfinite(intervals) & (intervals > 0)
        if not usable.any():
            return None

        rates, intervals = rates[usable], intervals[usable]
        periods_per_year = HOURS_PER_YEAR / float(np.median(intervals))
        annualised_funding = float(np.mean(rates)) * periods_per_year

        # Negated: this is the carry earned by a LONG. Positive funding means the
        # long pays, so it scores negatively.
        return float(np.clip(-annualised_funding, -self.clip_annual, self.clip_annual))
