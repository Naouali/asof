"""Rates carry and roll-down.

The expected return of holding a bond, absent any view on rates, has two parts:
the **carry** you earn from the yield itself, and the **roll-down** you earn as the
bond ages into a lower point on an upward-sloping curve and is repriced there.

    expected return ≈ y_n + D · (y_n − y_{n−1})

Both are observable from the curve alone, which is what makes rates carry the
cleanest carry in any asset class: no estimation, no forecast, just today's curve.

Needs two points on the yield curve, which means FRED and a free key. Without one
the signal declares itself and refuses to run.
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

__all__ = ["RatesCarry", "carry_and_rolldown"]

#: FRED constant-maturity series, by tenor in years. These are never revised, so
#: they are safe to read from FRED rather than needing ALFRED vintages.
TENOR_SERIES: dict[float, str] = {
    0.25: "DGS3MO",
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    5.0: "DGS5",
    10.0: "DGS10",
    30.0: "DGS30",
}


def carry_and_rolldown(
    yield_long: float, yield_short: float, tenor_years: float, *, shorter_years: float
) -> float:
    """Annualised expected return from holding the longer bond for a year.

    ``yield_long`` is the yield at ``tenor_years``; ``yield_short`` the yield at
    ``shorter_years``. Both as decimals -- 0.045 for 4.5%, not 4.5. Passing
    percentages produces an answer a hundred times too large, which in a carry
    signal looks like an extraordinary opportunity rather than a unit error.

    Duration is approximated by the remaining tenor, which is exact for a zero and
    close enough for a par bond at moderate yields. A signal that ranks curves
    against each other does not need more precision than that; one that sizes a
    position from the number does.
    """
    if tenor_years <= shorter_years:
        raise ValueError("tenor_years must exceed shorter_years to roll down")
    if max(abs(yield_long), abs(yield_short)) > 1.0:
        raise ValueError(
            f"yields look like percentages ({yield_long}, {yield_short}); pass "
            "decimals. A hundredfold error here reads as an extraordinary "
            "opportunity rather than as a unit mistake."
        )
    duration = tenor_years - 1.0
    return yield_long + duration * (yield_long - yield_short)


@register_signal
class RatesCarry(Signal):
    """Carry plus roll-down across the curve."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="carry.rates",
        asset_class=AssetClass.RATES,
        tier=1,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("series_observations",),
        rebalance=RebalanceFrequency.MONTHLY,
        expected_turnover_annual=1.5,
        evidence=EvidenceGrade.STRONG,
        reference="Koijen, Moskowitz, Pedersen & Vrugt (2018), 'Carry', JFE",
        known_failure_modes=(
            "Carry is largest when the curve is steepest, which is usually when the "
            "market expects rates to rise -- so the signal is systematically long "
            "duration into hiking cycles, and 2022 is the recent demonstration of "
            "what that costs. Returns are negatively skewed: years of steady accrual "
            "interrupted by a rapid repricing. The roll-down term assumes the curve "
            "stays put, which is exactly the assumption that fails when it matters. "
            "Duration approximated by tenor overstates it for high-coupon bonds."
        ),
        warmup_days=0,
        notes="Needs a free FRED key. Without one the signal refuses to run.",
    )

    def __init__(self, tenor_years: float = 10.0, shorter_years: float = 2.0) -> None:
        for tenor in (tenor_years, shorter_years):
            if tenor not in TENOR_SERIES:
                raise ValueError(
                    f"no FRED series for a {tenor}-year tenor; known tenors: {sorted(TENOR_SERIES)}"
                )
        self.tenor_years = tenor_years
        self.shorter_years = shorter_years

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        del symbols  # the curve is the universe here, not a list of instruments
        wanted = [TENOR_SERIES[self.tenor_years], TENOR_SERIES[self.shorter_years]]
        series = snapshot.series(symbols=wanted)
        if series.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name} needs {wanted} and the lake has neither at "
                f"{snapshot.as_of:%Y-%m-%d}. Set QUANTLAB_FRED_API_KEY -- the key is "
                "free -- and run `quantlab data ingest -f fred.series_observations`."
            )

        latest = series.sort("as_of").group_by("symbol").agg(pl.col("value").last())
        values = dict(zip(latest["symbol"], latest["value"], strict=True))
        missing = [name for name in wanted if name not in values]
        if missing:
            raise SignalUnavailableError(f"{self.spec.name}: missing series {missing}")

        # FRED publishes constant-maturity yields as percentages.
        score = carry_and_rolldown(
            values[wanted[0]] / 100.0,
            values[wanted[1]] / 100.0,
            self.tenor_years,
            shorter_years=self.shorter_years,
        )
        return self.finalise(snapshot, {f"UST{self.tenor_years:g}Y": score})
