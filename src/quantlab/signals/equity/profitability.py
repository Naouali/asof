"""Equity profitability and quality.

Spec section 4 puts these in Tier 1 for a blunt reason: they were *"among the only
survivors in the post-2005 large-cap universe."* Most published equity anomalies
stopped working after they were published, or only ever worked in microcaps that
nobody could trade at size. Profitability is one of the few that did neither.

Three constructions, in increasing order of how well they have held up:

**Gross profitability** (Novy-Marx). ``(revenue − cost of goods sold) / total
assets``. Deliberately measured at the top of the income statement, before the
accounting choices that management has discretion over.

**Operating profitability** (Fama-French five-factor). Subtracts SG&A and interest
and scales by book equity. Closer to what an investor actually receives, and
noisier for it.

**Cash-based operating profitability** (Ball, Gerakos, Linnainmaa, Nikolaev).
Operating profitability with accruals removed. The strongest of the three, and the
reason is instructive: accruals are the part of earnings management controls, so
removing them removes the part most likely to be managed.

All three need as-filed fundamentals, which among free sources means SEC EDGAR and
nothing else. Until that fetcher lands this signal declares itself, validates its
own arithmetic, and refuses to run.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
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

__all__ = [
    "DEFAULT_MAX_STALENESS",
    "EquityProfitability",
    "ProfitabilityMeasure",
    "profitability",
    "select_annual",
]

log = get_logger("quantlab.signals.equity.profitability")


class ProfitabilityMeasure(str, Enum):
    GROSS = "gross"
    OPERATING = "operating"
    CASH_BASED_OPERATING = "cash_based_operating"


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """The line items a profitability measure needs, for one company-period.

    Missing items are ``None`` or NaN, never zero. A missing SG&A treated as zero
    makes a company look more profitable than it is, and the filers with missing
    line items are systematically the small, badly reported ones -- so the error is
    not random noise, it is a size tilt.
    """

    revenue: float | None
    cost_of_goods_sold: float | None
    total_assets: float | None
    sg_and_a: float | None = None
    interest_expense: float | None = None
    book_equity: float | None = None
    accruals: float | None = None


def profitability(data: Fundamentals, measure: ProfitabilityMeasure) -> float | None:
    """Compute one profitability ratio, or ``None`` when an input is missing.

    Returns ``None`` rather than substituting zero for a missing line item. A
    missing SG&A treated as zero makes a company look more profitable than it is,
    and the companies with missing line items are systematically the small, badly
    reported ones -- so the error is not random, it is a size tilt.
    """
    if not _present(data.total_assets) or float(data.total_assets or 0.0) <= 0:
        return None
    if not _present(data.revenue) or not _present(data.cost_of_goods_sold):
        return None

    revenue = float(data.revenue or 0.0)
    cogs = float(data.cost_of_goods_sold or 0.0)
    assets = float(data.total_assets or 0.0)

    if measure is ProfitabilityMeasure.GROSS:
        return (revenue - cogs) / assets

    if (
        not _present(data.sg_and_a)
        or not _present(data.book_equity)
        or float(data.book_equity or 0.0) <= 0
    ):
        return None
    equity = float(data.book_equity or 0.0)
    interest = float(data.interest_expense or 0.0) if _present(data.interest_expense) else 0.0
    operating = revenue - cogs - float(data.sg_and_a or 0.0) - interest

    if measure is ProfitabilityMeasure.OPERATING:
        return operating / equity

    if not _present(data.accruals):
        return None
    return (operating - float(data.accruals or 0.0)) / equity


#: Flows -- measured over a window -- and stocks, measured at an instant. Mixing
#: the two spans is the failure this selector exists to prevent.
FLOW_METRICS = frozenset(
    {"revenue", "cost_of_goods_sold", "sg_and_a", "interest_expense", "accruals"}
)
STOCK_METRICS = frozenset({"total_assets", "book_equity"})

#: How stale an annual figure may be before it is dropped. Eighteen months allows
#: a company that files its 10-K three months after year end to remain scoreable
#: for a full year afterwards; beyond that the number describes a different
#: business. Spec section 13 forbids forward-filling fundamentals without a
#: documented limit, and this is the limit.
DEFAULT_MAX_STALENESS = dt.timedelta(days=548)


def select_annual(
    frame: pl.DataFrame, *, as_of: dt.datetime, max_staleness: dt.timedelta
) -> pl.DataFrame:
    """One value per (symbol, metric), on a consistent annual basis.

    Three mistakes are avoided here, and the first two were live before EDGAR
    made the spans visible.

    *Spans are not mixed.* EDGAR reports the same metric over 3-, 6-, 9- and
    12-month windows in the same filing. Taking the most recently filed value per
    metric picked nine-month revenue, a single quarter's SG&A and an instantaneous
    balance sheet, then divided one by the other. For Apple that understated gross
    profitability by 16%, and operating profitability -- a quarter's costs against
    nine months of revenue -- was not a ratio of anything.

    *Flows come from the annual figure, stocks from the latest instant.* A
    year of revenue belongs against the assets that produced it.

    *Stale figures are dropped rather than carried.* Without a limit, a company
    that stopped reporting a line item keeps its last value for ever and scores
    on it; Apple's interest expense was last filed in 2023.
    """
    if frame.height == 0:
        return frame

    basis = (
        pl.when(pl.col("metric").is_in(list(STOCK_METRICS)))
        .then(pl.lit("instant"))
        .otherwise(pl.lit("FY"))
    )
    on_basis = frame.filter(pl.col("fiscal_period") == basis)

    fresh = on_basis.filter(pl.col("as_of") >= as_of - max_staleness)
    return fresh.sort("as_of", "known_at").group_by("symbol", "metric").agg(pl.col("value").last())


def _present(value: float | None) -> bool:
    """A line item is present only if it is a real number, not NaN.

    Zero is a legitimate value -- a company really can report no interest expense
    -- so it must be distinguished from absent, which is what NaN records.
    """
    return value is not None and not np.isnan(value)


#: EDGAR XBRL tags, per line item. Several are listed because tagging is
#: inconsistent across filers and eras -- the normalisation this implies is the
#: main source of error in any free fundamentals pipeline, and it is why coverage
#: has to be reported per metric rather than assumed.
XBRL_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ),
    "cost_of_goods_sold": ("CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"),
    "total_assets": ("Assets",),
    "sg_and_a": (
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ),
    "interest_expense": ("InterestExpense", "InterestIncomeExpenseNet"),
    "book_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "accruals": ("IncreaseDecreaseInAccountsReceivable",),
}


@register_signal
class EquityProfitability(Signal):
    """Cross-sectional profitability score."""

    spec: ClassVar[SignalSpec] = SignalSpec(
        name="equity.profitability",
        asset_class=AssetClass.EQUITY,
        tier=1,
        output=SignalOutput.CROSS_SECTIONAL_SCORE,
        required_datasets=("fundamentals",),
        rebalance=RebalanceFrequency.QUARTERLY,
        expected_turnover_annual=0.6,
        evidence=EvidenceGrade.STRONG,
        reference=(
            "Novy-Marx (2013), 'The Other Side of Value'; Fama & French (2015), "
            "'A Five-Factor Asset Pricing Model'; Ball, Gerakos, Linnainmaa & "
            "Nikolaev (2016), 'Accruals, Cash Flows, and Operating Profitability'"
        ),
        known_failure_modes=(
            "Profitable firms are expensive, so the long leg carries a persistent "
            "negative value tilt and the two factors offset each other in exactly "
            "the periods each is tested. The signal is slow-moving, so most of its "
            "measured alpha comes from a handful of rebalances and the effective "
            "sample is far smaller than the bar count suggests. XBRL tagging is "
            "inconsistent before roughly 2012 and among small filers, so early "
            "coverage is both thin and biased toward large, well-reported "
            "companies. Missing line items correlate with size, so dropping them is "
            "itself a size tilt rather than a random loss of data."
        ),
        warmup_days=0,
        notes=(
            "Needs as-filed fundamentals, which among free sources means SEC EDGAR "
            "and nothing else. That fetcher lands in Milestone 9; until then this "
            "signal refuses to run rather than returning an empty cross-section."
        ),
    )

    def __init__(
        self,
        measure: ProfitabilityMeasure = ProfitabilityMeasure.GROSS,
        *,
        max_staleness: dt.timedelta = DEFAULT_MAX_STALENESS,
    ) -> None:
        self.measure = measure
        self.max_staleness = max_staleness

    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        frame = snapshot.frame("fundamentals", symbols=list(symbols))
        if frame.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name} needs as-filed fundamentals and the lake has none "
                f"at {snapshot.as_of:%Y-%m-%d}. Among free sources only SEC EDGAR "
                "supplies them with a filed date; that fetcher lands in Milestone 9. "
                "Returning an empty cross-section here would look like a signal with "
                "no view rather than like missing data."
            )

        selected = select_annual(frame, as_of=snapshot.as_of, max_staleness=self.max_staleness)
        if selected.height == 0:
            raise SignalUnavailableError(
                f"{self.spec.name}: the lake holds fundamentals at "
                f"{snapshot.as_of:%Y-%m-%d} but none within "
                f"{self.max_staleness.days} days of it. Scoring on older figures "
                "would rank companies on how recently they filed."
            )
        wide = selected.pivot(index="symbol", on="metric", values="value")

        scores: dict[str, float] = {}
        for row in wide.iter_rows(named=True):
            value = profitability(
                Fundamentals(
                    revenue=row.get("revenue"),
                    cost_of_goods_sold=row.get("cost_of_goods_sold"),
                    total_assets=row.get("total_assets"),
                    sg_and_a=row.get("sg_and_a"),
                    interest_expense=row.get("interest_expense"),
                    book_equity=row.get("book_equity"),
                    accruals=row.get("accruals"),
                ),
                self.measure,
            )
            if value is not None and np.isfinite(value):
                scores[str(row["symbol"])] = value

        coverage = len(scores) / max(1, len(symbols))
        if coverage < 0.5:
            log.warning(
                "signals.profitability.thin_coverage",
                as_of=snapshot.as_of.isoformat(),
                coverage=round(coverage, 3),
                measure=self.measure.value,
                note="missing line items correlate with size, so this is a tilt, not noise",
            )
        return self.finalise(snapshot, scores)
