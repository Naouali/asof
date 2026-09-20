"""The content of a tearsheet, as structured rows.

Every renderer builds from these, so the text, Markdown, HTML and JSON forms of a
tearsheet cannot drift apart and quietly disagree about what the strategy did.

A panel is a title, a list of ``(label, value, note)`` rows, and a severity. The
severity is what drives ordering: a tearsheet puts what is wrong at the top,
because a warning printed under the equity curve is a warning nobody read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.backtest.results import BacktestResult
    from quantlab.costs.capacity import CapacityResult
    from quantlab.validation.report import ValidationReport

__all__ = [
    "Panel",
    "Row",
    "Severity",
    "capacity_panel",
    "cost_panel",
    "data_quality_panel",
    "exposure_panel",
    "performance_panel",
    "validation_panel",
]


class Severity(IntEnum):
    """How loudly a panel needs to be read.

    Ordered so that ``sorted`` puts the worst first. This is the mechanism that
    stops a tearsheet burying its own caveats.
    """

    DISQUALIFYING = 0
    WARNING = 1
    INFORMATIONAL = 2


@dataclass(frozen=True, slots=True)
class Row:
    label: str
    value: str
    #: Why the number is what it is, or what it does not mean.
    note: str = ""


@dataclass(frozen=True, slots=True)
class Panel:
    title: str
    rows: tuple[Row, ...]
    severity: Severity = Severity.INFORMATIONAL
    #: Free text shown under the rows. Used for the things a table cannot say.
    footer: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "severity": self.severity.name.lower(),
            "rows": [{"label": r.label, "value": r.value, "note": r.note} for r in self.rows],
            "footer": self.footer,
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------- panels --
def performance_panel(result: BacktestResult) -> Panel:
    """What the strategy did, with the Sharpe never shown bare.

    The deflated figure leads. The undeflated one is present because the
    difference between them is the whole lesson, but it is labelled as the upper
    bound it is rather than as a result.
    """
    stats = result.stats
    evidence = result.evidence
    rows = [
        Row("Period", f"{stats.years:.1f} years", f"{stats.bars:,} bars"),
        Row("CAGR", f"{stats.cagr:+.2%}"),
        Row("Volatility", f"{stats.volatility_annual:.2%}", "annualised"),
        Row(
            "Max drawdown",
            f"{stats.max_drawdown:.2%}",
            f"{stats.max_drawdown_days} days to recover",
        ),
    ]
    warnings: list[str] = []
    severity = Severity.INFORMATIONAL

    if evidence is None:  # pragma: no cover - build() refuses this case
        rows.append(Row("Sharpe", f"{stats.sharpe_undeflated:.2f}", "UNDEFLATED -- not a result"))
    else:
        rows += [
            Row(
                "Deflated Sharpe",
                f"{evidence.deflated:.3f}",
                f"probability the result is not the best of {evidence.trials} tries",
            ),
            Row(
                "Sharpe, after haircuts",
                f"{evidence.haircut_annual:.2f}",
                "selection and post-publication decay applied",
            ),
            Row(
                "Sharpe, undeflated",
                f"{stats.sharpe_undeflated:.2f} net / {stats.sharpe_gross_undeflated:.2f} gross",
                "upper bound, not an estimate",
            ),
            Row("Trials in family", f"{evidence.trials}", f"family '{result.family}'"),
        ]
        if not evidence.survives_deflation:
            severity = Severity.DISQUALIFYING
            warnings.append(
                f"The deflated Sharpe of {evidence.deflated:.3f} is below 0.95: after "
                f"{evidence.trials} trial(s) this result is not distinguishable from "
                "the best of the search. Nothing below is evidence that it works."
            )

    rows += [
        Row("Hit rate", f"{stats.hit_rate:.1%}", "share of bars with positive P&L"),
        Row(
            "Skew / excess kurtosis",
            f"{stats.skewness:+.2f} / {stats.excess_kurtosis:+.2f}",
            "negative skew and fat tails earn a larger deflation haircut",
        ),
    ]

    if result.contaminated:
        severity = Severity.DISQUALIFYING
        warnings.append(
            "CONTAMINATED: this run used look-ahead execution, trading at the close "
            "that produced the signal. It is a diagnostic, never a finding."
        )

    return Panel("Performance", tuple(rows), severity, warnings=tuple(warnings))


def cost_panel(result: BacktestResult) -> Panel:
    """The cost decomposition, because a single net number is not a diagnostic.

    A strategy killed by spread needs a slower rebalance; one killed by impact
    needs less size; one killed by borrow needs a different short book. The total
    tells you none of that.
    """
    stats = result.stats
    decomposition = result.cost_decomposition_bps_annual()
    rows = [
        Row("Turnover", f"{stats.turnover_annual:.1%}/yr"),
        Row("Total cost drag", f"{stats.cost_drag_bps_annual:.0f} bp/yr"),
    ]
    rows += [
        Row(f"  {component}", f"{value:.0f} bp/yr")
        for component, value in sorted(decomposition.items(), key=lambda kv: -kv[1])
        if abs(value) > 0.05
    ]

    warnings: list[str] = []
    severity = Severity.INFORMATIONAL
    gross, net = stats.sharpe_gross_undeflated, stats.sharpe_undeflated
    if gross > 0:
        retained = net / gross
        rows.append(
            Row(
                "Sharpe retained after costs",
                f"{retained:.0%}",
                f"{net:.2f} of {gross:.2f} gross",
            )
        )
        if retained < 0.5:
            severity = Severity.WARNING
            warnings.append(
                f"Costs consume {1 - retained:.0%} of the gross Sharpe. A result this "
                "cost-sensitive is also sensitive to every cost parameter, none of "
                "which is calibrated to your fills."
            )
    return Panel("Costs", tuple(rows), severity, warnings=tuple(warnings))


def capacity_panel(capacity: CapacityResult) -> Panel:
    """Break-even AUM. Spec section 13 forbids a strategy result without one."""
    rows = [
        Row("Gross alpha", f"{capacity.gross_alpha_bps_annual:.0f} bp/yr", "before costs"),
        Row(
            "Net alpha at zero size",
            f"{capacity.net_alpha_bps_at_zero:.0f} bp/yr",
            "what survives at a size small enough to be costless",
        ),
    ]
    warnings: list[str] = []

    if not capacity.is_viable or capacity.break_even_aum is None:
        return Panel(
            "Capacity",
            (*rows, Row("Break-even AUM", "none", capacity.note)),
            Severity.DISQUALIFYING,
            warnings=(
                f"There is no AUM at which this strategy makes money after costs. {capacity.note}",
            ),
        )

    severity = Severity.INFORMATIONAL
    rows += [
        Row(
            "Break-even AUM",
            f"${capacity.break_even_aum / 1e6:,.1f}m",
            "where net alpha reaches zero",
        ),
        Row("Cost at break-even", f"{capacity.cost_bps_annual:.0f} bp/yr"),
        Row("Binding constraint", capacity.binding_constraint),
        Row(
            "Max single-name participation",
            f"{capacity.max_participation:.2%} of ADV",
            "at the break-even size",
        ),
    ]
    if capacity.extrapolated:
        severity = Severity.WARNING
        warnings.append(
            f"The break-even size requires {capacity.max_participation:.1%} of a "
            "name's daily volume, beyond the range the square-root law was "
            "calibrated on, where it is known to understate impact. This capacity "
            "is an upper bound, not an estimate."
        )
    if capacity.break_even_aum < 25e6:
        severity = min(severity, Severity.WARNING)
        warnings.append(
            f"Break-even AUM of ${capacity.break_even_aum / 1e6:,.1f}m is small "
            "enough that fixed costs -- data, infrastructure, a salary -- plausibly "
            "exceed the gross alpha. Capacity is not the same as viability."
        )
    return Panel(
        "Capacity", tuple(rows), severity, footer=capacity.summary(), warnings=tuple(warnings)
    )


def validation_panel(report: ValidationReport | None) -> Panel:
    """Out-of-sample evidence, or its absence stated plainly."""
    if report is None:
        return Panel(
            "Validation",
            (Row("Cross-validation", "not run"),),
            Severity.WARNING,
            warnings=(
                "No purged cross-validation, no CPCV path dispersion and no PBO were "
                "supplied. The performance above is in-sample and the deflated Sharpe "
                "corrects only for how hard the parameters were searched -- not for "
                "the choice of universe, period or signal construction.",
            ),
        )

    rows = [
        Row(
            "Sample length",
            f"{report.years_available:.1f}y available / "
            f"{report.minimum_backtest_years:.1f}y needed",
            "minimum backtest length for this Sharpe and trial count",
        )
    ]
    warnings: list[str] = []
    severity = Severity.INFORMATIONAL

    if not report.sample_is_long_enough:
        severity = Severity.DISQUALIFYING
        warnings.append(
            f"The sample is {report.years_available:.1f} years where "
            f"{report.minimum_backtest_years:.1f} are needed to distinguish this "
            "Sharpe from the best of the search. The result is not yet testable."
        )

    dispersion = report.path_dispersion
    if dispersion is not None:
        low, median, high = dispersion
        rows.append(
            Row(
                "CPCV path Sharpe",
                f"{median:.2f} [{low:.2f}, {high:.2f}]",
                "median and 5th-95th percentile across reassembled paths",
            )
        )
        if low < 0:
            severity = min(severity, Severity.WARNING)
            warnings.append(
                f"The 5th percentile CPCV path loses money (Sharpe {low:.2f}). The "
                "headline figure is one draw from a distribution that includes "
                "outcomes you would have abandoned."
            )

    if report.pbo is not None:
        rows.append(
            Row(
                "PBO",
                f"{report.pbo:.1%}",
                "probability the in-sample best is below median out-of-sample",
            )
        )
        if report.pbo > 0.5:
            severity = Severity.DISQUALIFYING
            warnings.append(
                f"Probability of backtest overfitting is {report.pbo:.0%}. The "
                "configuration that looked best in-sample is more likely than not to "
                "be below median out-of-sample: the search found noise."
            )

    return Panel("Validation", tuple(rows), severity, warnings=tuple(warnings))


def exposure_panel(result: BacktestResult) -> Panel:
    stats = result.stats
    return Panel(
        "Exposure",
        (
            Row("Average gross", f"{stats.average_gross_exposure:.2f}x"),
            Row("Average net", f"{stats.average_net_exposure:+.2f}x"),
            Row("Average positions", f"{stats.average_positions:.0f}"),
        ),
    )


def data_quality_panel(result: BacktestResult, caveats: tuple[str, ...] = ()) -> Panel:
    """What the inputs were, and what is known to be wrong with them.

    The caveats come from the source catalogue rather than from prose, so a source
    whose limitations are recorded once appears on every tearsheet that used it.
    """
    quality = result.data_quality
    rows = [
        Row("Instruments", f"{quality['symbols']:,}", f"{quality['bars']:,} bars"),
        Row("Execution", str(quality["execution"]), f"{quality['signal_lag_bars']} bar signal lag"),
        Row("Staleness policy", str(quality["staleness_policy"])),
        Row("Forward-filled cells", f"{quality['forward_filled_cells']:,}"),
    ]
    warnings: list[str] = []
    severity = Severity.INFORMATIONAL

    delisted = int(quality["delisted_symbols"])
    if delisted:
        rows.append(
            Row(
                "Delisted instruments",
                f"{delisted}",
                f"{quality['delisting_return_applied']:+.0%} applied on exit",
            )
        )

    total_cells = max(1, int(quality["bars"]) * int(quality["symbols"]))
    filled_share = int(quality["forward_filled_cells"]) / total_cells
    if filled_share > 0.05:
        severity = Severity.WARNING
        warnings.append(
            f"{filled_share:.1%} of the panel was filled forward from an earlier "
            "bar. A forward-filled price is a price nobody could have traded at, "
            "and it flatters both volatility and turnover."
        )

    warnings.extend(caveats)
    if caveats:
        severity = min(severity, Severity.WARNING)

    return Panel("Data quality", tuple(rows), severity, warnings=tuple(warnings))


def reconciliation_panel(result: BacktestResult) -> Panel:
    """Whether the books balanced. A failure here invalidates everything above."""
    report = result.reconciliation
    balanced = getattr(report, "balanced", True)
    return Panel(
        "Accounting",
        (Row("Reconciliation", "balanced" if balanced else "FAILED", report.describe()),),
        Severity.INFORMATIONAL if balanced else Severity.DISQUALIFYING,
        warnings=()
        if balanced
        else (
            "The two independent routes to the ending equity disagree. Every number "
            "on this tearsheet is derived from a ledger that does not balance.",
        ),
    )
