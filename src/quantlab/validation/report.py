"""The validation report: everything a Sharpe ratio needs before it can be quoted.

One entry point, :func:`validate_strategy`, which takes a return series and the
trial registry and returns the full picture -- probabilistic and deflated Sharpe,
the trial count that produced it, the haircuts, and a verdict in words.

The verdict matters as much as the numbers. "Deflated Sharpe 0.31" invites the
reader to decide 0.31 sounds fine; "this result is what the luckiest of 47 tries
would look like even with no edge" does not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quantlab.validation.registry import TrialRegistry, TrialSet
from quantlab.validation.splitting import CombinatorialPurgedCV
from quantlab.validation.statistics import (
    HaircutSchedule,
    SharpeEvidence,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)

__all__ = ["ValidationReport", "cpcv_path_returns", "validate_strategy"]

BARS_PER_YEAR = 252.0


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """What the validation module concluded, and why."""

    name: str
    family: str
    evidence: SharpeEvidence
    trials: TrialSet
    minimum_backtest_years: float
    years_available: float
    #: Sharpe on each reassembled CPCV path, when one was run.
    path_sharpes: np.ndarray | None = None
    #: Probability of backtest overfitting, when a parameter sweep was supplied.
    pbo: float | None = None
    pbo_logits: np.ndarray | None = None

    @property
    def sample_is_long_enough(self) -> bool:
        return self.years_available >= self.minimum_backtest_years

    @property
    def path_dispersion(self) -> tuple[float, float, float] | None:
        """``(5th percentile, median, 95th percentile)`` of the CPCV path Sharpes."""
        if self.path_sharpes is None or len(self.path_sharpes) < 3:
            return None
        return (
            float(np.percentile(self.path_sharpes, 5)),
            float(np.median(self.path_sharpes)),
            float(np.percentile(self.path_sharpes, 95)),
        )

    @property
    def passes(self) -> bool:
        """Whether every check the report ran was cleared.

        Deliberately conjunctive. A result that survives deflation but fails PBO
        has not been validated -- it has passed one test out of two.
        """
        if not self.evidence.survives_deflation:
            return False
        if not self.sample_is_long_enough:
            return False
        if self.pbo is not None and self.pbo > 0.5:
            return False
        dispersion = self.path_dispersion
        return not (dispersion is not None and dispersion[0] <= 0)

    def failures(self) -> list[str]:
        """Every check that did not clear, in plain words."""
        out: list[str] = []
        if not self.evidence.survives_deflation:
            out.append(
                f"Deflated Sharpe {self.evidence.deflated:.3f} is below 0.95. After "
                f"{self.trials.count} trial(s), this result is consistent with having "
                "found the luckiest member of the search rather than a real edge."
            )
        if not self.sample_is_long_enough:
            out.append(
                f"The sample is {self.years_available:.1f} years, but "
                f"{self.minimum_backtest_years:.1f} years would be needed before a "
                f"Sharpe of {self.evidence.net_annual:.2f} could be distinguished from "
                f"the best of {self.trials.count} tries."
            )
        if self.pbo is not None and self.pbo > 0.5:
            out.append(
                f"Probability of backtest overfitting is {self.pbo:.0%}. The "
                "configuration that looked best in-sample lands below median "
                "out-of-sample more often than not, so the selection rule is worse "
                "than choosing at random."
            )
        dispersion = self.path_dispersion
        if dispersion is not None and dispersion[0] <= 0:
            out.append(
                f"The worst CPCV paths are negative (5th percentile "
                f"{dispersion[0]:.2f}). The headline Sharpe depends on which slice of "
                "history you happened to test on."
            )
        return out

    def describe(self) -> str:
        lines = [
            f"{self.name} [family: {self.family}]",
            "  " + self.evidence.describe().replace("\n", "\n  "),
            f"  sample {self.years_available:.1f}y; "
            f"{self.minimum_backtest_years:.1f}y needed for {self.trials.count} trial(s)",
        ]
        dispersion = self.path_dispersion
        if dispersion is not None:
            lines.append(
                f"  CPCV paths ({0 if self.path_sharpes is None else len(self.path_sharpes)}): "
                f"p5 {dispersion[0]:.2f} / median {dispersion[1]:.2f} / "
                f"p95 {dispersion[2]:.2f}"
            )
        if self.pbo is not None:
            lines.append(f"  probability of backtest overfitting: {self.pbo:.0%}")

        failures = self.failures()
        if failures:
            lines.append("  VERDICT: does not validate")
            lines.extend(f"    - {failure}" for failure in failures)
        else:
            lines.append(
                "  VERDICT: clears every check run. That is not proof of an edge -- "
                "only the paper-trading loop is out-of-sample in the way that counts."
            )
        return "\n".join(lines)


def cpcv_path_returns(
    n_observations: int,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    cv: CombinatorialPurgedCV,
    *,
    label_end: np.ndarray | None = None,
) -> np.ndarray:
    """Out-of-sample returns along each reassembled CPCV path.

    ``fit_predict(train_indices, test_indices)`` fits on the training indices and
    returns the strategy's returns over the test indices. The refitting is the
    point: running CPCV over a *fixed* return series produces identical paths,
    because nothing about the strategy depends on which split it is in. If your
    paths come back identical, the cross-validation is measuring nothing and the
    honest response is to say so rather than to report the spread as evidence.

    Returns an array of shape ``(n_paths, n_observations)``.
    """
    cache: dict[tuple[int, ...], np.ndarray] = {}
    paths = cv.paths(n_observations, label_end=label_end)
    out = np.full((len(paths), n_observations), np.nan, dtype=float)

    for path_index, path in enumerate(paths):
        for segment in path:
            key = segment.split.test_groups
            if key not in cache:
                cache[key] = np.asarray(
                    fit_predict(segment.split.train, segment.split.test), dtype=float
                )
            predicted = cache[key]
            if len(predicted) != len(segment.split.test):
                raise ValueError(
                    f"fit_predict returned {len(predicted)} values for "
                    f"{len(segment.split.test)} test observations"
                )
            # Take only this path's own group from the split's test set; a split
            # spans several groups and they belong to different paths.
            positions = np.searchsorted(segment.split.test, segment.indices)
            out[path_index, segment.indices] = predicted[positions]

    return out


def validate_strategy(
    net_returns: np.ndarray,
    *,
    name: str,
    family: str,
    registry: TrialRegistry,
    bars_per_year: float = BARS_PER_YEAR,
    gross_returns: np.ndarray | None = None,
    sweep_returns: np.ndarray | None = None,
    path_returns: np.ndarray | None = None,
    haircuts: HaircutSchedule | None = None,
) -> ValidationReport:
    """Assemble the full validation picture for one strategy.

    ``net_returns`` is the strategy's per-bar return series after costs.
    ``sweep_returns`` is ``(observations, configurations)`` for the parameter
    search, if one was run -- without it, PBO cannot be computed, because PBO
    measures the *selection*, not the winner.
    """
    returns = np.asarray(net_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 3:
        raise ValueError("need at least three return observations to validate anything")

    schedule = haircuts or HaircutSchedule()
    trials = registry.family(family)
    # A family with nothing recorded still represents one trial: this one.
    trial_count = max(1, trials.count)
    observations = len(returns)
    years = observations / bars_per_year

    per_bar = _sharpe(returns)
    net_annual = per_bar * math.sqrt(bars_per_year)
    gross_annual = (
        _sharpe(np.asarray(gross_returns, dtype=float)) * math.sqrt(bars_per_year)
        if gross_returns is not None
        else net_annual
    )
    skewness = _moment(returns, 3)
    excess_kurtosis = _moment(returns, 4) - 3.0
    variance = trials.sharpe_variance(observations)

    evidence = SharpeEvidence(
        gross_annual=gross_annual,
        net_annual=net_annual,
        haircut_annual=schedule.apply(net_annual),
        observations=observations,
        trials=trial_count,
        sharpe_variance=variance,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        probabilistic=probabilistic_sharpe_ratio(
            per_bar,
            observations=observations,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
        ),
        deflated=deflated_sharpe_ratio(
            per_bar,
            observations=observations,
            trials=trial_count,
            sharpe_variance=variance,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
        ),
        expected_max_from_search=expected_max_sharpe(trial_count, variance),
        haircuts=schedule,
    )

    pbo: float | None = None
    logits: np.ndarray | None = None
    if sweep_returns is not None:
        matrix = np.asarray(sweep_returns, dtype=float)
        if matrix.shape[1] >= 2 and matrix.shape[0] >= 32:
            pbo, logits = probability_of_backtest_overfitting(matrix)

    path_sharpes: np.ndarray | None = None
    if path_returns is not None:
        paths = np.asarray(path_returns, dtype=float)
        path_sharpes = np.array(
            [_sharpe(row[np.isfinite(row)]) * math.sqrt(bars_per_year) for row in paths]
        )

    return ValidationReport(
        name=name,
        family=family,
        evidence=evidence,
        trials=trials,
        minimum_backtest_years=minimum_backtest_length(max(abs(net_annual), 1e-6), trial_count),
        years_available=years,
        path_sharpes=path_sharpes,
        pbo=pbo,
        pbo_logits=logits,
    )


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    return float(np.mean(returns) / deviation) if deviation > 0 else 0.0


def _moment(returns: np.ndarray, order: int) -> float:
    if len(returns) < 3:
        return 0.0 if order == 3 else 3.0
    deviation = float(np.std(returns, ddof=0))
    if deviation == 0:
        return 0.0 if order == 3 else 3.0
    centred = returns - np.mean(returns)
    return float(np.mean(centred**order) / deviation**order)
