"""FX carry, from the interest rate differential.

Under covered interest parity the forward premium equals the interest rate
differential, so the carry from holding a currency is the difference between its
rate and the funding currency's. No forward quotes needed -- which matters,
because free FX data has none.

**Tier 3 in spirit, Tier 1 in construction.** The spec places FX carry in Tier 3
and is right to: post-publication out-of-sample performance collapsed badly, and
the strategy's defining feature is that it works for years and then loses a decade
of accrual in a month. It is implemented here because the machinery is shared with
the other carry signals and because the platform should let you re-test a decayed
effect on current data -- with the prior attached, which is what
``EvidenceGrade.DECAYED`` is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from quantlab.conventions import RebalanceFrequency
from quantlab.data.catalogue import AssetClass
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

__all__ = ["FxCarry", "forward_implied_carry"]


def forward_implied_carry(foreign_rate: float, domestic_rate: float) -> float:
    """Annualised carry from being long the foreign currency. Decimals, not percent."""
    if max(abs(foreign_rate), abs(domestic_rate)) > 1.0:
        raise ValueError("rates look like percentages; pass decimals")
    return foreign_rate - domestic_rate


@register_signal
class FxCarry(Signal):
    """Cross-sectional FX carry from rate differentials."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="carry.fx",
        asset_class=AssetClass.FX,
        tier=3,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("series_observations",),
        rebalance=RebalanceFrequency.MONTHLY,
        expected_turnover_annual=2.0,
        evidence=EvidenceGrade.DECAYED,
        reference=(
            "Lustig, Roussanov & Verdelhan (2011), 'Common Risk Factors in Currency "
            "Markets'; the post-publication decay is documented in the replication "
            "literature"
        ),
        known_failure_modes=(
            "THE PUBLISHED EFFECT LARGELY STOPPED WORKING AFTER PUBLICATION. What "
            "remains has the classic carry shape: years of steady accrual erased in "
            "weeks, with extreme negative skew that flatters every Sharpe computed "
            "on it. High-yielding currencies are high-yielding because they are "
            "risky, and the risk arrives all at once. Free FX data is a single daily "
            "ECB fix with no bid/ask, so transaction costs here are assumed rather "
            "than measured -- and carry strategies are exactly the kind whose edge "
            "is the same order of magnitude as their spread."
        ),
        warmup_days=0,
        notes=(
            "Tier 3. Implemented so the effect can be re-tested on current data, "
            "with the evidence that it decayed attached to every result."
        ),
    )

    def __init__(self, rate_series: dict[str, str] | None = None, base: str = "DFF") -> None:
        #: Currency to the FRED series carrying its policy rate. Extend deliberately.
        self.rate_series = rate_series or {}
        self.base = base

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        del symbols
        if not self.rate_series:
            raise SignalUnavailableError(
                f"{self.spec.name}: no currency rate series configured. Free sources "
                "cover US rates (FRED) and ECB rates, but a cross-sectional FX carry "
                "needs a policy rate per currency, which the free catalogue does not "
                "supply for most of them. Configure `rate_series` explicitly rather "
                "than letting the signal trade a universe of two."
            )

        wanted = [self.base, *self.rate_series.values()]
        series = snapshot.series(symbols=wanted)
        if series.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name} needs {wanted}; the lake has none at "
                f"{snapshot.as_of:%Y-%m-%d}. Set QUANTLAB_FRED_API_KEY and ingest."
            )

        latest = series.sort("as_of").group_by("symbol").agg(pl.col("value").last())
        values = dict(zip(latest["symbol"], latest["value"], strict=True))
        if self.base not in values:
            raise SignalUnavailableError(f"{self.spec.name}: missing base rate {self.base}")

        domestic = values[self.base] / 100.0
        scores = {
            currency: forward_implied_carry(values[name] / 100.0, domestic)
            for currency, name in self.rate_series.items()
            if name in values
        }
        return self.finalise(snapshot, scores)
