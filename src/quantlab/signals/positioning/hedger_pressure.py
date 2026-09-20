"""Commercial hedger pressure.

The oldest positioning signal in the futures literature. Commercial traders --
producers, merchants, processors -- hold futures to hedge physical exposure, and
they hedge whether or not the price is attractive. Speculators take the other
side and, the theory goes, are paid a risk premium for doing so. Hedging pressure
is the attempt to measure that premium directly: when commercials are unusually
short a market, speculators are unusually long, and the premium being paid to
them should be unusually large.

**Tier 2, and the reasons are specific.** Keynes and Hicks give it a mechanism
and Bessembinder (1992) and De Roon, Nijman and Veld (2000) find it in the data,
which is more than most signals have. But:

*It is weekly.* Fifty-two observations a year caps how much any parameter fitted
to it can be trusted, and caps the Sharpe that is achievable even if the effect
is real.

*The categories are administrative, not economic.* "Commercial" is a
self-reported CFTC classification, and a swap dealer hedging an index position is
filed alongside a farmer hedging a crop. The disaggregated report splits them
from 2006; before that they are one number and the signal is measuring a mixture.

*Reclassifications create level shifts that look exactly like signal.* The CFTC
has moved traders between categories, and a position series that jumps because
someone was re-filed is indistinguishable in the data from one that jumps because
someone traded. This is why the score is normalised within a trailing window
rather than against the full history: a z-score against a mean that includes a
reclassification is measuring the reclassification.

*It is published with a three-day lag*, which the data layer already encodes --
``known_at`` is the Friday release, not the Tuesday snapshot. A backtest that
reads the Tuesday number on Tuesday will find this signal works much better than
it does, and that is the single most common error in published COT research.
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

__all__ = ["DEFAULT_LOOKBACK_WEEKS", "CommercialHedgerPressure", "hedger_pressure_index"]

log = get_logger("quantlab.signals.positioning.hedger_pressure")

#: Weeks of history the z-score is computed over. Three years: long enough for a
#: position to be judged unusual, short enough that a reclassification ages out
#: of the window rather than distorting it for ever.
DEFAULT_LOOKBACK_WEEKS = 156

#: Below this many observations the z-score is noise wearing a number.
MIN_OBSERVATIONS = 52


def hedger_pressure_index(long: np.ndarray, short: np.ndarray) -> np.ndarray:
    """Net commercial position, scaled by the size of the commercial book.

    ``(long − short) / (long + short)``, bounded in [−1, 1]. Scaling by the
    commercials' own total rather than by open interest is deliberate: open
    interest moves with speculative activity too, so dividing by it makes the
    index move when the *other* side trades, which is not what the signal claims
    to measure.
    """
    total = long + short
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, (long - short) / total, np.nan)


@register_signal
class CommercialHedgerPressure(Signal):
    """Cross-sectional score from how unusually hedged each market is."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="positioning.hedger_pressure",
        asset_class=AssetClass.FUTURES,
        tier=2,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("positioning",),
        rebalance=RebalanceFrequency.WEEKLY,
        expected_turnover_annual=3.0,
        evidence=EvidenceGrade.MIXED,
        reference=(
            "Keynes (1930), 'A Treatise on Money'; Hicks (1939), 'Value and "
            "Capital'; Bessembinder (1992), 'Systematic Risk, Hedging Pressure, "
            "and Risk Premiums in Futures Markets'; De Roon, Nijman & Veld "
            "(2000), 'Hedging Pressure Effects in Futures Markets'"
        ),
        known_failure_modes=(
            "Weekly data caps the achievable Sharpe however real the effect is: "
            "fifty-two observations a year is a small sample for anything fitted "
            "to it. The CFTC's 'commercial' category is an administrative "
            "self-classification, so a swap dealer hedging an index position sits "
            "alongside a farmer hedging a crop, and before the 2006 disaggregated "
            "report they are one number. Reclassifications move traders between "
            "categories and produce level shifts indistinguishable from real "
            "position changes, which is why the score is normalised within a "
            "trailing window rather than against full history. The three-day "
            "publication lag is the single most common source of look-ahead in "
            "published COT research: reading Tuesday's number on Tuesday makes "
            "this signal look far better than it is."
        ),
        warmup_days=MIN_OBSERVATIONS * 7,
        notes=(
            "Reads the legacy report by default, because it is the only one with "
            "history before 2006. Pass report='disaggregated' for the split that "
            "separates swap dealers from physical hedgers, at the cost of two "
            "decades of sample."
        ),
    )

    def __init__(
        self,
        *,
        report: str = "legacy",
        category: str = "commercial",
        lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    ) -> None:
        if lookback_weeks < MIN_OBSERVATIONS:
            raise ValueError(
                f"lookback_weeks must be at least {MIN_OBSERVATIONS}; a z-score over "
                "fewer than a year of weekly observations is noise wearing a number"
            )
        self.report = report
        self.category = category
        self.lookback_weeks = lookback_weeks

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        frame = snapshot.frame("positioning", symbols=list(symbols))
        if frame.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name} needs CFTC positioning and the lake has none at "
                f"{snapshot.as_of:%Y-%m-%d}. Ingest it with "
                "`quantlab data ingest -f cftc_cot.positioning`. Note that the "
                "report is published on the Friday after its Tuesday snapshot, so "
                "a snapshot taken mid-week legitimately sees nothing new."
            )

        wanted = frame.filter(
            (pl.col("report") == self.report)
            & (pl.col("category") == self.category)
            & pl.col("measure").is_in(["long", "short"])
        )
        if wanted.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name}: the lake holds positioning but nothing for "
                f"report={self.report!r} category={self.category!r}. The legacy "
                "report has commercial/noncommercial/nonreportable; the "
                "disaggregated one splits those and has no 'commercial' at all."
            )

        wide = (
            wanted.pivot(index=["symbol", "as_of"], on="measure", values="value")
            .drop_nulls(["long", "short"])
            .sort("symbol", "as_of")
        )

        scores: dict[str, float] = {}
        thin: list[str] = []
        for symbol, group in wide.group_by("symbol", maintain_order=True):
            history = group.tail(self.lookback_weeks)
            if history.height < MIN_OBSERVATIONS:
                thin.append(str(symbol[0]))
                continue

            index = hedger_pressure_index(history["long"].to_numpy(), history["short"].to_numpy())
            index = index[np.isfinite(index)]
            if index.size < MIN_OBSERVATIONS:
                thin.append(str(symbol[0]))
                continue

            spread = float(index.std(ddof=1))
            if spread <= 0:
                thin.append(str(symbol[0]))
                continue

            # Commercials short is speculators long, and the premium accrues to
            # the speculator. A LOW hedger-pressure index is therefore a LONG
            # signal, so the z-score is negated. Getting this backwards produces a
            # strategy that consistently pays the risk premium instead of earning
            # it, and it backtests as a confident, slow loss.
            z = (float(index[-1]) - float(index.mean())) / spread
            scores[str(symbol[0])] = -z

        if thin:
            log.info(
                "signals.hedger_pressure.thin_history",
                symbols=thin,
                required=MIN_OBSERVATIONS,
                note="a z-score over less than a year of weekly data is not a z-score",
            )
        if not scores:
            raise SignalUnavailableError(
                f"{self.spec.name}: no instrument has {MIN_OBSERVATIONS} weeks of "
                f"{self.report}/{self.category} positioning by "
                f"{snapshot.as_of:%Y-%m-%d}."
            )
        return self.finalise(snapshot, scores)
