"""Tearsheets.

The governing rule of this platform is that it exists to tell you the truth about
whether a signal works, not to produce impressive-looking equity curves. A
tearsheet is where that rule is easiest to break, so two of spec section 13's
prohibitions are enforced here structurally rather than by convention:

* **No Sharpe without its deflated value and trial count.** A result whose run was
  not recorded as a trial cannot be rendered. There is no flag to override it.
* **No strategy result without a capacity estimate.** ``build`` requires one.

Both are constructor arguments rather than optional decorations, so the failure
mode is a tearsheet that will not render rather than a tearsheet that renders
without the inconvenient parts.

The third rule is ordering. Panels are sorted by severity, so a strategy that
fails deflation says so above its equity curve rather than beneath it. A caveat
printed under the chart is a caveat nobody read.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quantlab.logging import get_logger
from quantlab.reporting.panels import (
    Panel,
    Severity,
    capacity_panel,
    cost_panel,
    data_quality_panel,
    exposure_panel,
    performance_panel,
    reconciliation_panel,
    validation_panel,
)

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.backtest.results import BacktestResult
    from quantlab.costs.capacity import CapacityResult
    from quantlab.validation.report import ValidationReport

__all__ = ["Tearsheet", "TearsheetError"]

log = get_logger("quantlab.reporting.tearsheet")


class TearsheetError(RuntimeError):
    """A tearsheet was asked to report something it is not allowed to report."""


@dataclass(frozen=True, slots=True)
class Tearsheet:
    """One strategy's results, assembled and ordered by how badly they read."""

    name: str
    generated_at: dt.datetime
    panels: tuple[Panel, ...]
    result: BacktestResult
    capacity: CapacityResult
    validation: ValidationReport | None

    # ------------------------------------------------------------------ build --
    @classmethod
    def build(
        cls,
        result: BacktestResult,
        *,
        capacity: CapacityResult,
        validation: ValidationReport | None = None,
        caveats: tuple[str, ...] = (),
        generated_at: dt.datetime | None = None,
    ) -> Tearsheet:
        """Assemble a tearsheet, or refuse to.

        ``caveats`` are the source-catalogue limitations for whatever data the run
        used; :func:`quantlab.reporting.caveats_for` collects them.
        """
        if result.evidence is None:
            raise TearsheetError(
                f"{result.name} has no Sharpe evidence attached, so its Sharpe ratio "
                "cannot be reported (spec section 13: never a Sharpe without its "
                "deflated value and trial count). The run was not recorded as a "
                "trial -- re-run it with record_trial=True. Opting out of the trial "
                "count is opting out of the tearsheet."
            )

        panels = [
            performance_panel(result),
            capacity_panel(capacity),
            validation_panel(validation),
            cost_panel(result),
            exposure_panel(result),
            data_quality_panel(result, caveats),
            reconciliation_panel(result),
        ]
        # Worst first. Severity is the ordering key and ties keep their order, so
        # the layout is deterministic.
        panels.sort(key=lambda panel: panel.severity)

        sheet = cls(
            name=result.name,
            generated_at=generated_at or dt.datetime.now(tz=dt.UTC),
            panels=tuple(panels),
            result=result,
            capacity=capacity,
            validation=validation,
        )
        log.info(
            "reporting.tearsheet",
            name=result.name,
            deflated=round(result.evidence.deflated, 4),
            trials=result.evidence.trials,
            break_even_aum=capacity.break_even_aum,
            disqualifying=len(sheet.disqualifying),
            warnings=len(sheet.warnings),
        )
        return sheet

    # ------------------------------------------------------------------ views --
    @property
    def warnings(self) -> tuple[str, ...]:
        """Every warning from every panel, worst panel first."""
        return tuple(warning for panel in self.panels for warning in panel.warnings)

    @property
    def disqualifying(self) -> tuple[str, ...]:
        return tuple(
            warning
            for panel in self.panels
            if panel.severity is Severity.DISQUALIFYING
            for warning in panel.warnings
        )

    @property
    def is_disqualified(self) -> bool:
        """Whether anything here forbids reading the result as evidence."""
        return bool(self.disqualifying)

    def verdict(self) -> str:
        """The bottom line, in the order a reader should take it.

        Deliberately not a score. A single number summarising whether a strategy
        works is the thing this whole platform is arguing against.
        """
        evidence = self.result.evidence
        assert evidence is not None  # noqa: S101 - build() guarantees it

        if self.is_disqualified:
            reasons = "\n".join(f"  - {reason}" for reason in self.disqualifying)
            return f"NOT EVIDENCE that {self.name} works:\n{reasons}"

        if not self.capacity.is_viable:  # pragma: no cover - a disqualifying panel
            return f"{self.name} has no viable capacity."

        lines = [
            f"{self.name} survives deflation at {evidence.deflated:.3f} over "
            f"{evidence.trials} trial(s), with a Sharpe of "
            f"{evidence.haircut_annual:.2f} after haircuts and break-even AUM of "
            f"${self.capacity.break_even_aum / 1e6:,.1f}m."
            if self.capacity.break_even_aum is not None
            else f"{self.name} survives deflation at {evidence.deflated:.3f}.",
        ]
        if self.warnings:
            lines.append(
                f"{len(self.warnings)} caveat(s) apply; none is disqualifying, and "
                "each is stated above rather than summarised away."
            )
        if self.validation is None:
            lines.append(
                "No out-of-sample validation was supplied, so this is an in-sample "
                "result that has been corrected for search intensity and nothing else."
            )
        return " ".join(lines)

    def as_dict(self) -> dict[str, Any]:
        """The whole tearsheet as data, for the dashboard and for storage."""
        evidence = self.result.evidence
        assert evidence is not None  # noqa: S101 - build() guarantees it
        return {
            "name": self.name,
            "generated_at": self.generated_at.isoformat(),
            "verdict": self.verdict(),
            "is_disqualified": self.is_disqualified,
            "headline": {
                "deflated_sharpe": evidence.deflated,
                "sharpe_after_haircuts": evidence.haircut_annual,
                "sharpe_undeflated_net": self.result.stats.sharpe_undeflated,
                "trials": evidence.trials,
                "family": self.result.family,
                "cagr": self.result.stats.cagr,
                "max_drawdown": self.result.stats.max_drawdown,
                "break_even_aum": self.capacity.break_even_aum,
            },
            "panels": [panel.as_dict() for panel in self.panels],
            "warnings": list(self.warnings),
            "stats": self.result.stats.as_dict(),
        }
