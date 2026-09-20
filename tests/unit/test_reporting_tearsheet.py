"""Tearsheet assembly.

The tests are mostly about what a tearsheet *refuses* to do. Spec section 13's
prohibitions are enforced structurally here rather than by convention, so the
important cases are the ones where a report would be misleading and the module
declines to produce it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import pytest

from quantlab.reporting import Severity, Tearsheet, TearsheetError
from quantlab.reporting.panels import (
    capacity_panel,
    cost_panel,
    data_quality_panel,
    performance_panel,
    validation_panel,
)


# ----------------------------------------------------------------------- doubles --
class FakeEvidence:
    def __init__(self, *, deflated: float = 0.99, trials: int = 3) -> None:
        self.deflated = deflated
        self.trials = trials
        self.gross_annual = 1.1
        self.net_annual = 0.9
        self.haircut_annual = 0.5

    @property
    def survives_deflation(self) -> bool:
        return self.deflated >= 0.95


class FakeStats:
    years = 10.0
    bars = 2520
    cagr = 0.09
    volatility_annual = 0.11
    sharpe_undeflated = 0.9
    sharpe_gross_undeflated = 1.1
    max_drawdown = -0.18
    max_drawdown_days = 240
    turnover_annual = 1.4
    cost_drag_bps_annual = 60.0
    skewness = -0.3
    excess_kurtosis = 2.0
    hit_rate = 0.53
    average_gross_exposure = 1.0
    average_net_exposure = 0.0
    average_positions = 12.0

    def as_dict(self) -> dict[str, float]:
        return {"cagr": self.cagr}


class FakeReconciliation:
    balanced = True

    def describe(self) -> str:
        return "2,519 bars reconciled"


class FakeResult:
    def __init__(self, **overrides: Any) -> None:
        self.name = "fake-strategy"
        self.family = "fake"
        self.stats = FakeStats()
        self.evidence: Any = FakeEvidence()
        self.reconciliation: Any = FakeReconciliation()
        self.contaminated = False
        self.data_quality = {
            "bars": 2520,
            "symbols": 12,
            "forward_filled_cells": 0,
            "staleness_policy": "5 bars then liquidate",
            "delisted_symbols": 0,
            "delisting_return_applied": -0.3,
            "execution": "next_open",
            "signal_lag_bars": 1,
        }
        n = 2520
        equity = 1e7 * np.cumprod(1.0 + np.full(n, 0.0004))
        self.curve = pl.DataFrame(
            {
                "as_of": [
                    dt.datetime(2015, 1, 1, tzinfo=dt.UTC) + dt.timedelta(days=i) for i in range(n)
                ],
                "equity": equity,
                "drawdown": np.zeros(n),
                "spread_cost": np.full(n, 40.0),
                "impact_cost": np.full(n, 20.0),
                "commission_cost": np.full(n, 5.0),
                "borrow_cost": np.full(n, 10.0),
                "financing_cost": np.full(n, 5.0),
            }
        )
        self.positions = pl.DataFrame({"as_of": [], "symbol": [], "weight": []})
        for key, value in overrides.items():
            setattr(self, key, value)

    @property
    def equity_curve(self) -> pl.DataFrame:
        return self.curve.select("as_of", "equity", "drawdown")

    def cost_decomposition_bps_annual(self) -> dict[str, float]:
        return {"spread": 25.0, "impact": 18.0, "commission": 4.0, "borrow": 10.0, "financing": 3.0}


class FakeCapacity:
    def __init__(self, *, break_even_aum: float | None = 400e6, extrapolated: bool = False) -> None:
        self.break_even_aum = break_even_aum
        self.gross_alpha_bps_annual = 150.0
        self.cost_bps_annual = 150.0
        self.net_alpha_bps_at_zero = 140.0
        self.binding_constraint = "impact"
        self.max_participation = 0.03
        self.extrapolated = extrapolated
        self.note = "the signal does not pay"

    @property
    def is_viable(self) -> bool:
        return self.break_even_aum is not None and self.break_even_aum > 1e6

    def summary(self) -> str:
        return "break-even AUM $400.0m"


def sheet(**overrides: Any) -> Tearsheet:
    capacity = overrides.pop("capacity", FakeCapacity())
    validation = overrides.pop("validation", None)
    caveats = overrides.pop("caveats", ())
    return Tearsheet.build(
        FakeResult(**overrides),  # type: ignore[arg-type]
        capacity=capacity,  # type: ignore[arg-type]
        validation=validation,
        caveats=caveats,
    )


# ------------------------------------------------------------------- the refusals --
def test_a_result_without_a_trial_count_cannot_be_reported() -> None:
    """Spec section 13: never a Sharpe without its deflated value and trial
    count. There is deliberately no flag to override this."""
    with pytest.raises(TearsheetError, match="deflated value and trial count"):
        sheet(evidence=None)


def test_the_refusal_says_how_to_fix_it() -> None:
    with pytest.raises(TearsheetError, match="record_trial=True"):
        sheet(evidence=None)


def test_capacity_is_a_required_argument() -> None:
    """Spec section 13 again: no strategy result without a capacity estimate.
    Making it a keyword argument with no default is the enforcement."""
    with pytest.raises(TypeError):
        Tearsheet.build(FakeResult())  # type: ignore[arg-type, call-arg]


# --------------------------------------------------------------------- ordering --
def test_panels_are_ordered_worst_first() -> None:
    """A caveat printed under the equity curve is a caveat nobody read."""
    built = sheet(evidence=FakeEvidence(deflated=0.2, trials=90))
    severities = [panel.severity for panel in built.panels]

    assert severities == sorted(severities)
    assert built.panels[0].title == "Performance"
    assert built.panels[0].severity is Severity.DISQUALIFYING


def test_a_clean_result_still_puts_the_missing_validation_above_the_details() -> None:
    built = sheet()
    titles = [p.title for p in built.panels]
    assert titles.index("Validation") < titles.index("Exposure")


# ---------------------------------------------------------------------- verdicts --
def test_a_failing_deflation_is_not_evidence() -> None:
    built = sheet(evidence=FakeEvidence(deflated=0.31, trials=64))

    assert built.is_disqualified
    assert built.verdict().startswith("NOT EVIDENCE")
    assert "64 trial(s)" in built.verdict()


def test_a_surviving_result_says_what_it_survived() -> None:
    built = sheet(validation=None)

    assert not built.is_disqualified
    verdict = built.verdict()
    assert "survives deflation" in verdict
    assert "break-even AUM" in verdict
    # And still says what was not tested.
    assert "in-sample" in verdict


def test_an_unviable_capacity_disqualifies_the_result() -> None:
    built = sheet(capacity=FakeCapacity(break_even_aum=None))

    assert built.is_disqualified
    assert any("no AUM at which" in w for w in built.disqualifying)


def test_contamination_disqualifies_whatever_the_numbers_say() -> None:
    built = sheet(contaminated=True)

    assert built.is_disqualified
    assert any("CONTAMINATED" in w for w in built.disqualifying)
    assert "diagnostic" in " ".join(built.disqualifying)


def test_a_failed_reconciliation_disqualifies_everything_above_it() -> None:
    class Broken:
        balanced = False

        def describe(self) -> str:
            return "worst discrepancy 12.40"

    built = sheet(reconciliation=Broken())
    assert built.is_disqualified
    assert any("does not balance" in w for w in built.disqualifying)


# ------------------------------------------------------------------------ panels --
def test_the_undeflated_sharpe_is_never_shown_bare() -> None:
    """It appears, because the gap between it and the deflated figure is the
    lesson, but it is labelled as the upper bound it is."""
    panel = performance_panel(FakeResult())  # type: ignore[arg-type]
    row = next(r for r in panel.rows if r.label == "Sharpe, undeflated")
    assert "upper bound" in row.note
    assert panel.rows.index(row) > panel.rows.index(
        next(r for r in panel.rows if r.label == "Deflated Sharpe")
    )


def test_costs_are_decomposed_not_summarised() -> None:
    """A strategy killed by spread needs a slower rebalance; one killed by borrow
    needs a different short book. The total tells you neither."""
    panel = cost_panel(FakeResult())  # type: ignore[arg-type]
    labels = [r.label.strip() for r in panel.rows]
    for component in ("spread", "impact", "commission", "borrow"):
        assert component in labels


def test_a_strategy_that_loses_most_of_its_sharpe_to_costs_is_flagged() -> None:
    result = FakeResult()
    result.stats.sharpe_undeflated = 0.3  # type: ignore[misc]
    panel = cost_panel(result)  # type: ignore[arg-type]

    assert panel.severity is Severity.WARNING
    assert any("cost-sensitive" in w for w in panel.warnings)


def test_an_extrapolated_capacity_is_labelled_an_upper_bound() -> None:
    panel = capacity_panel(FakeCapacity(extrapolated=True))  # type: ignore[arg-type]
    assert panel.severity is Severity.WARNING
    assert any("upper bound" in w for w in panel.warnings)


def test_a_tiny_break_even_warns_that_capacity_is_not_viability() -> None:
    panel = capacity_panel(FakeCapacity(break_even_aum=8e6))  # type: ignore[arg-type]
    assert any("fixed costs" in w for w in panel.warnings)


def test_missing_validation_is_stated_not_omitted() -> None:
    panel = validation_panel(None)
    assert panel.severity is Severity.WARNING
    assert any("in-sample" in w for w in panel.warnings)


def test_heavy_forward_filling_is_flagged() -> None:
    result = FakeResult()
    result.data_quality["forward_filled_cells"] = 5_000  # of 2520 * 12
    panel = data_quality_panel(result)  # type: ignore[arg-type]

    assert panel.severity is Severity.WARNING
    assert any("nobody could have traded at" in w for w in panel.warnings)


def test_source_caveats_reach_the_panel() -> None:
    panel = data_quality_panel(
        FakeResult(),  # type: ignore[arg-type]
        caveats=("Yahoo: delisted tickers disappear entirely",),
    )
    assert any("delisted tickers" in w for w in panel.warnings)
    assert panel.severity is Severity.WARNING


# -------------------------------------------------------------------- as_dict --
def test_the_dictionary_headline_leads_with_the_deflated_figure() -> None:
    data = sheet().as_dict()

    assert data["headline"]["deflated_sharpe"] == pytest.approx(0.99)
    assert data["headline"]["trials"] == 3
    assert data["is_disqualified"] is False
    assert data["panels"][0]["severity"] in {"disqualifying", "warning", "informational"}


def test_every_warning_survives_into_the_dictionary() -> None:
    built = sheet(evidence=FakeEvidence(deflated=0.1, trials=40))
    assert len(built.as_dict()["warnings"]) == len(built.warnings)
    assert built.as_dict()["verdict"].startswith("NOT EVIDENCE")


def test_the_generated_timestamp_is_recorded() -> None:
    moment = dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.UTC)
    built = Tearsheet.build(
        FakeResult(),  # type: ignore[arg-type]
        capacity=FakeCapacity(),  # type: ignore[arg-type]
        generated_at=moment,
    )
    assert built.generated_at == moment
    assert "2026-01-02" in built.as_dict()["generated_at"]


def test_severity_orders_worst_first() -> None:
    assert sorted(Severity) == [
        Severity.DISQUALIFYING,
        Severity.WARNING,
        Severity.INFORMATIONAL,
    ]


def test_replace_keeps_the_sheet_frozen() -> None:
    built = sheet()
    assert replace(built, name="other").name == "other"
    with pytest.raises(AttributeError):
        built.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------- the validation panel --
class FakeValidation:
    def __init__(
        self,
        *,
        years_available: float = 12.0,
        minimum_backtest_years: float = 6.0,
        path_sharpes: np.ndarray | None = None,
        pbo: float | None = None,
    ) -> None:
        self.years_available = years_available
        self.minimum_backtest_years = minimum_backtest_years
        self.path_sharpes = path_sharpes
        self.pbo = pbo

    @property
    def sample_is_long_enough(self) -> bool:
        return self.years_available >= self.minimum_backtest_years

    @property
    def path_dispersion(self) -> tuple[float, float, float] | None:
        if self.path_sharpes is None or len(self.path_sharpes) < 3:
            return None
        low, median, high = np.percentile(self.path_sharpes, [5, 50, 95])
        return float(low), float(median), float(high)


def test_a_validated_result_reports_its_sample_length() -> None:
    panel = validation_panel(FakeValidation())  # type: ignore[arg-type]
    assert panel.severity is Severity.INFORMATIONAL
    assert any("12.0y available" in r.value for r in panel.rows)
    assert panel.warnings == ()


def test_a_sample_too_short_to_test_is_disqualifying() -> None:
    panel = validation_panel(  # type: ignore[arg-type]
        FakeValidation(years_available=3.0, minimum_backtest_years=11.0)
    )
    assert panel.severity is Severity.DISQUALIFYING
    assert any("not yet testable" in w for w in panel.warnings)


def test_cpcv_dispersion_is_reported_as_a_range() -> None:
    panel = validation_panel(  # type: ignore[arg-type]
        FakeValidation(path_sharpes=np.linspace(0.4, 1.6, 40))
    )
    row = next(r for r in panel.rows if r.label == "CPCV path Sharpe")
    assert "[" in row.value and "]" in row.value
    assert panel.severity is Severity.INFORMATIONAL


def test_a_losing_fifth_percentile_path_is_flagged() -> None:
    """The headline is one draw from a distribution. If the bad draws lose money,
    that belongs on the tearsheet rather than in the appendix."""
    panel = validation_panel(  # type: ignore[arg-type]
        FakeValidation(path_sharpes=np.linspace(-0.9, 2.0, 40))
    )
    assert panel.severity is Severity.WARNING
    assert any("loses money" in w for w in panel.warnings)


def test_a_high_pbo_is_disqualifying() -> None:
    panel = validation_panel(FakeValidation(pbo=0.72))  # type: ignore[arg-type]
    assert panel.severity is Severity.DISQUALIFYING
    assert any("found noise" in w for w in panel.warnings)


def test_a_low_pbo_is_reported_without_alarm() -> None:
    panel = validation_panel(FakeValidation(pbo=0.08))  # type: ignore[arg-type]
    assert panel.severity is Severity.INFORMATIONAL
    assert any(r.label == "PBO" for r in panel.rows)


def test_a_disqualifying_pbo_outranks_a_warning_about_paths() -> None:
    panel = validation_panel(  # type: ignore[arg-type]
        FakeValidation(path_sharpes=np.linspace(-0.9, 2.0, 40), pbo=0.8)
    )
    assert panel.severity is Severity.DISQUALIFYING
    assert len(panel.warnings) == 2


def test_validation_reaches_the_verdict_and_the_sheet() -> None:
    built = sheet(validation=FakeValidation(pbo=0.9))
    assert built.is_disqualified
    assert any("found noise" in w for w in built.disqualifying)
    # And a validated sheet no longer says it was never validated.
    clean = sheet(validation=FakeValidation())
    assert "No out-of-sample validation" not in clean.verdict()


def test_delisted_instruments_are_reported() -> None:
    result = FakeResult()
    result.data_quality["delisted_symbols"] = 4
    panel = data_quality_panel(result)  # type: ignore[arg-type]
    row = next(r for r in panel.rows if r.label == "Delisted instruments")
    assert row.value == "4"
    assert "-30%" in row.note
