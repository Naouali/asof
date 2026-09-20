"""The automatic trial counter.

Spec section 7: *"Manual honesty about trial counts does not work."* These tests
pin the properties that make the count honest despite the operator.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from quantlab.validation.luck import luck_adjust, t_statistic
from quantlab.validation.registry import (
    Trial,
    TrialRegistry,
    config_fingerprint,
)


def trial(family: str = "momentum", **overrides: object) -> Trial:
    base: dict[str, object] = {
        "family": family,
        "config_hash": config_fingerprint(lookback=250),
        "name": "mom-250",
        "sharpe_annual": 0.8,
        "observations": 2520,
    }
    base.update(overrides)
    return Trial(**base)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# Fingerprints
# ----------------------------------------------------------------------------------
def test_fingerprints_are_order_independent() -> None:
    assert config_fingerprint(a=1, b=2) == config_fingerprint(b=2, a=1)


def test_changing_any_parameter_changes_the_fingerprint() -> None:
    assert config_fingerprint(lookback=250) != config_fingerprint(lookback=251)


def test_floating_point_noise_does_not_create_a_new_trial() -> None:
    """Two runs differing in the sixteenth decimal are the same trial. Treating
    them as two would let arithmetic noise inflate -- or reset -- a trial count."""
    assert config_fingerprint(y=0.5) == config_fingerprint(y=0.5 + 1e-15)


def test_nested_structures_and_enums_fingerprint_stably() -> None:
    from quantlab.backtest.conventions import ExecutionTiming

    first = config_fingerprint(
        engine={"execution": ExecutionTiming.NEXT_OPEN, "equity": 1e7},
        symbols=["A", "B"],
    )
    second = config_fingerprint(
        symbols=["A", "B"],
        engine={"equity": 1e7, "execution": ExecutionTiming.NEXT_OPEN},
    )
    assert first == second


def test_paths_and_dates_fingerprint_without_exploding(tmp_path: Path) -> None:
    assert config_fingerprint(path=tmp_path / "weights.parquet", day=dt.date(2024, 1, 1))


# ----------------------------------------------------------------------------------
# Counting
# ----------------------------------------------------------------------------------
def test_an_empty_registry_reports_no_trials(tmp_path: Path) -> None:
    assert TrialRegistry(tmp_path).family("nothing").count == 0


def test_a_parameter_sweep_counts_as_many_trials(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path)
    for lookback in (60, 120, 250, 500):
        registry.record(
            trial(config_hash=config_fingerprint(lookback=lookback), name=f"mom-{lookback}")
        )
    assert registry.family("momentum").count == 4


def test_re_running_an_identical_configuration_is_not_a_new_trial(tmp_path: Path) -> None:
    """The search did not widen, so the count should not grow."""
    registry = TrialRegistry(tmp_path)
    assert registry.record(trial())
    assert not registry.record(trial(name="same-config-different-label"))
    assert registry.family("momentum").count == 1


def test_families_are_counted_separately(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path)
    registry.record(trial(family="momentum"))
    registry.record(trial(family="carry", config_hash=config_fingerprint(tenor=3)))
    assert registry.family("momentum").count == 1
    assert registry.family("carry").count == 1
    assert registry.families() == ["carry", "momentum"]


def test_the_record_survives_a_new_registry_object(tmp_path: Path) -> None:
    """Trial counts that reset when the process restarts would be worse than
    useless: they would be most flattering exactly when a long search had ended."""
    TrialRegistry(tmp_path).record(trial())
    assert TrialRegistry(tmp_path).family("momentum").count == 1


def test_a_corrupt_line_does_not_silently_reduce_the_count(tmp_path: Path) -> None:
    """A dropped trial makes every deflated Sharpe more flattering, so it is warned
    about rather than absorbed."""
    registry = TrialRegistry(tmp_path)
    registry.record(trial())
    registry.path.write_text(registry.path.read_text() + "{not json\n")
    assert registry.family("momentum").count == 1


def test_the_best_trial_is_identified(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path)
    for lookback, sharpe in ((60, 0.3), (250, 1.4), (500, 0.9)):
        registry.record(
            trial(
                config_hash=config_fingerprint(lookback=lookback),
                name=f"mom-{lookback}",
                sharpe_annual=sharpe,
            )
        )
    best = registry.family("momentum").best
    assert best is not None and best.name == "mom-250"


# ----------------------------------------------------------------------------------
# Sharpe variance
# ----------------------------------------------------------------------------------
def test_a_single_trial_has_no_dispersion(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path)
    registry.record(trial())
    assert registry.family("momentum").sharpe_variance() == 0.0


def test_variance_is_floored_at_the_sampling_variance(tmp_path: Path) -> None:
    """Without the floor, a sweep of near-identical configurations shows almost no
    dispersion, the expected maximum collapses, and forty trials get deflated as
    though they were one -- which is the search pattern most likely to overfit."""
    registry = TrialRegistry(tmp_path)
    for index in range(10):
        registry.record(trial(config_hash=config_fingerprint(variant=index), sharpe_annual=0.8))
    observations = 2520
    assert registry.family("momentum").sharpe_variance(observations) == pytest.approx(
        1.0 / observations
    )


def test_a_genuinely_wide_search_reports_its_real_dispersion(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path)
    for index, sharpe in enumerate((-1.0, 0.0, 1.0, 2.0)):
        registry.record(trial(config_hash=config_fingerprint(variant=index), sharpe_annual=sharpe))
    assert registry.family("momentum").sharpe_variance(2520) > 1.0 / 2520


def test_summary_covers_every_family(tmp_path: Path) -> None:
    registry = TrialRegistry(tmp_path)
    registry.record(trial(family="a"))
    registry.record(trial(family="b", config_hash=config_fingerprint(x=1)))
    assert {s.family for s in registry.summary()} == {"a", "b"}
    assert "distinct configuration" in registry.family("a").describe()


# ----------------------------------------------------------------------------------
# Luck adjustment across the library
# ----------------------------------------------------------------------------------
def test_pure_noise_shrinks_to_nothing() -> None:
    """Spec section 7: if Var(t) is near 1, the dispersion is exactly what two
    hundred dart-throwing monkeys would produce."""
    import numpy as np

    rng = np.random.default_rng(0)
    adjustment = luck_adjust(rng.normal(0, 1, 300))
    assert adjustment.variance == pytest.approx(1.0, abs=0.2)
    assert adjustment.shrinkage < 0.15
    assert adjustment.survivors == ()
    assert "nothing has been found" in adjustment.verdict


def test_real_dispersion_survives_shrinkage() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, 300)
    values[:15] += 5.0
    adjustment = luck_adjust({f"sig{i}": v for i, v in enumerate(values)})
    assert adjustment.variance > 1.5
    assert adjustment.shrinkage > 0.3
    assert len(adjustment.survivors) > 0


def test_shrinkage_is_one_minus_the_inverse_variance() -> None:
    import numpy as np

    rng = np.random.default_rng(2)
    adjustment = luck_adjust(rng.normal(0, 2, 200))
    assert adjustment.shrinkage == pytest.approx(1.0 - 1.0 / adjustment.variance)
    assert adjustment.shrunk_t == pytest.approx(adjustment.t_statistics * adjustment.shrinkage)


def test_a_library_too_small_to_judge_is_refused() -> None:
    with pytest.raises(ValueError, match="tells you about your sample size"):
        luck_adjust([1.0, 2.0, 3.0])


def test_a_small_library_is_flagged_as_unreliable() -> None:
    import numpy as np

    rng = np.random.default_rng(3)
    adjustment = luck_adjust(rng.normal(0, 2, 8))
    assert not adjustment.reliable
    assert "indicative" in adjustment.describe()


def test_t_statistic_from_sharpe_and_years() -> None:
    """A Sharpe of 1.0 over four years is t = 2.0, the conventional threshold --
    and a useful reminder of how little four good-looking years establishes."""
    assert t_statistic(1.0, 4.0) == pytest.approx(2.0)


def test_ranked_output_is_ordered_by_shrunk_strength() -> None:
    import numpy as np

    rng = np.random.default_rng(4)
    values = rng.normal(0, 2, 50)
    ranked = luck_adjust({f"s{i}": v for i, v in enumerate(values)}).ranked()
    magnitudes = [abs(row[2]) for row in ranked]
    assert magnitudes == sorted(magnitudes, reverse=True)
