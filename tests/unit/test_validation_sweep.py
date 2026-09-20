"""Sweeping a signal across instruments.

The sweep exists because the question it answers -- does this work everywhere or
only on some names -- always looks like yes when read off a table. These tests
are mostly about the two ways it could quietly stop protecting against that:
counting one trial instead of many, and letting a noisy cell inflate the variance
every other cell is judged against.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.validation.sweep import (
    MIN_YEARS,
    SweepResult,
    cell_returns_from_prices,
    sweep_returns,
    time_series_positions,
)


def noise(n: int = 2520, seed: int = 0, mean: float = 0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, 0.01, n)


# ------------------------------------------------------------------ the null --
def test_pure_noise_survives_nothing() -> None:
    """Fourteen cells of nothing produce a best cell that looks like something.
    That best cell is the whole reason this module exists."""
    cells = {f"N{i}": noise(seed=i) for i in range(14)}
    result = sweep_returns(cells, signal="noise")

    assert result.survivors == ()
    best = result.best_raw
    assert best is not None
    # Not that the shrunk t is zero: with fourteen cells Var(t) is itself noisy
    # and will sometimes exceed 1 by chance, which the module warns about. The
    # invariant that matters is that nothing clears the bar.
    assert abs(best.shrunk_t) < 2.0


@pytest.mark.slow
def test_the_correction_is_what_stops_noise_being_reported() -> None:
    """The number that matters when a sweep is run repeatedly.

    Two hundred sweeps of fourteen pure-noise cells each -- 2,800 cells of
    nothing. Read naively at |t| >= 2, roughly half of those sweeps hand you a
    "winner", because that is what testing fourteen things does. After the
    dispersion correction, almost none do.

    The correction is not conservatism for its own sake. It is the difference
    between a search that reports a finding every other time it is run and one
    that reports almost none.
    """
    naive = corrected = 0
    for trial in range(200):
        cells = {f"N{i}": noise(seed=trial * 100 + i) for i in range(14)}
        result = sweep_returns(cells, signal="noise")
        naive += any(abs(c.t_statistic) >= 2.0 for c in result.cells)
        corrected += len(result.survivors)

    assert naive > 40, f"only {naive}/200 naive sweeps flagged noise; fixture too quiet"
    assert corrected <= 3, f"{corrected} noise cells survived the correction"
    assert corrected < naive / 10, (
        f"the correction must be worth more than a small factor: "
        f"{naive} naive vs {corrected} corrected"
    )


def test_the_verdict_names_the_winner_and_dismisses_it() -> None:
    result = sweep_returns({f"N{i}": noise(seed=i) for i in range(12)}, signal="noise")

    verdict = result.verdict()
    assert "Nothing survives" in verdict
    assert "maximum of 12 noisy draws" in verdict
    assert "an instrument the signal suits" in verdict, "reads as careless otherwise"


def test_every_cell_counts_as_a_trial() -> None:
    """Sweeping fourteen instruments is fourteen trials. Quoting the winner as
    though it were the only thing tried is the error the whole platform exists
    to prevent."""
    result = sweep_returns({f"N{i}": noise(seed=i) for i in range(14)}, signal="noise")
    assert result.trials == 14


# ------------------------------------------------------- real heterogeneity --
def test_a_genuine_edge_in_one_cell_survives() -> None:
    """The correction has to be able to pass something, or it is not a test --
    it is a policy of always saying no."""
    cells = {f"N{i}": noise(seed=i) for i in range(13)}
    # One cell with a real, large edge: Sharpe of about 1.5 over ten years.
    cells["REAL"] = np.random.default_rng(99).normal(1.5 / np.sqrt(252) * 0.01, 0.01, 2520)

    result = sweep_returns(cells, signal="mixed")

    assert result.adjustment.variance > 1.0, "a real cell widens the dispersion"
    assert "REAL" in [c.name for c in result.survivors]


def test_shrinkage_is_applied_to_the_survivor_too() -> None:
    """Surviving is not the same as being believed at face value. The reported
    t is the shrunk one, always below the raw one."""
    cells = {f"N{i}": noise(seed=i) for i in range(13)}
    cells["REAL"] = np.random.default_rng(99).normal(1.5 / np.sqrt(252) * 0.01, 0.01, 2520)

    result = sweep_returns(cells, signal="mixed")
    real = next(c for c in result.cells if c.name == "REAL")

    assert abs(real.shrunk_t) < abs(real.t_statistic)


# ----------------------------------------------------------------- guards --
def test_one_cell_is_not_a_sweep() -> None:
    with pytest.raises(ValueError, match="at least two cells"):
        sweep_returns({"ONLY": noise()}, signal="x")


def test_a_thin_cell_cannot_survive_and_does_not_set_the_variance() -> None:
    """A t-statistic over two years is mostly noise. Letting one into the
    library inflates the variance every other cell is judged against, which
    makes the correction weaker exactly when the evidence is thinnest."""
    cells = {f"N{i}": noise(seed=i) for i in range(10)}
    cells["THIN"] = np.random.default_rng(7).normal(0.004, 0.01, 200)  # under a year

    result = sweep_returns(cells, signal="mixed")
    thin = next(c for c in result.cells if c.name == "THIN")

    assert thin.thin
    assert not thin.survives
    assert "THIN" not in result.adjustment.names, "excluded from the dispersion"
    assert thin.years < MIN_YEARS


def test_too_few_judgeable_cells_is_refused() -> None:
    cells = {name: np.random.default_rng(i).normal(0.0, 0.01, 200) for i, name in enumerate("AB")}
    with pytest.raises(ValueError, match="mostly noise"):
        sweep_returns(cells, signal="x")


def test_a_constant_cell_is_thin_not_infinite() -> None:
    """Zero variance would divide by zero and report an infinite Sharpe."""
    cells = {f"N{i}": noise(seed=i) for i in range(10)}
    cells["FLAT"] = np.zeros(2520)

    result = sweep_returns(cells, signal="mixed")
    assert next(c for c in result.cells if c.name == "FLAT").thin


# ------------------------------------------------------------- positions --
def test_the_position_is_lagged() -> None:
    """A position taken on the close that produced the signal earns that day's
    return for free, and on autocorrelated data that one bar is most of the
    apparent edge."""
    prices = np.cumprod(1 + np.full(400, 0.001)) * 100
    position = time_series_positions(prices, lookback=252, lag=1)

    assert np.isnan(position[252]), "the first computable bar is not yet tradeable"
    assert position[253] == 1.0


def test_a_rising_series_is_long_and_a_falling_one_short() -> None:
    up = np.cumprod(1 + np.full(400, 0.001)) * 100
    down = np.cumprod(1 - np.full(400, 0.001)) * 100

    assert time_series_positions(up)[-1] == 1.0
    assert time_series_positions(down)[-1] == -1.0


def test_a_short_lookback_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two bars"):
        time_series_positions(np.ones(10), lookback=1)


def test_costs_are_charged_on_every_flip() -> None:
    """A sweep run gross ranks instruments by how much they trend AND by how
    cheap they are to trade, and those are not the same thing."""
    rng = np.random.default_rng(5)
    choppy = 100 * np.cumprod(1 + rng.normal(0.0, 0.02, 1500))

    free = cell_returns_from_prices({"X": choppy}, cost_bps_per_turn=0.0)["X"]
    charged = cell_returns_from_prices({"X": choppy}, cost_bps_per_turn=10.0)["X"]

    assert charged.sum() < free.sum()


def test_the_result_renders_as_a_table() -> None:
    result = sweep_returns({f"N{i}": noise(seed=i) for i in range(10)}, signal="noise")
    frame = result.to_frame()

    assert set(frame.columns) >= {"cell", "sharpe", "t_statistic", "shrunk_t", "survives"}
    assert frame.height == result.trials
    assert isinstance(result, SweepResult)
    assert "Var(t)" in result.describe()


# ------------------------------------------------------------ parameter grids --
def test_the_grid_is_the_trial_count() -> None:
    """Six parameter sets is six trials, not one, however much the write-up
    afterwards says 'we used a 252-day lookback'."""
    from quantlab.validation.sweep import parameter_grid

    grid = parameter_grid(lookback=[63, 126, 252], skip=[0, 21])
    assert len(grid) == 6
    assert {"lookback": 252, "skip": 21} in grid


def test_an_empty_grid_is_a_single_default_cell() -> None:
    from quantlab.validation.sweep import parameter_grid

    assert parameter_grid() == [{}]


def test_cells_are_labelled_stably() -> None:
    from quantlab.validation.sweep import describe_cell

    assert describe_cell({"b": 2, "a": 1}) == "a=1 b=2", "sorted, so runs compare"
    assert describe_cell({}) == "default"


# ------------------------------------------------------ dependence vs no edge --
def test_near_identical_cells_are_not_reported_as_a_failed_test() -> None:
    """The subtlety that makes the whole correction easy to misread.

    The null assumes cells are INDEPENDENT -- under it Var(t) is 1. Correlated
    cells move together and drive it toward 0. So a very low variance does not
    mean 'tested many ways and failed'; it means the cells were nearly the same
    experiment. A seven-value parameter sweep on real data came out at 0.01.
    """
    base = noise(seed=1)
    # Seven cells that are the same series with a whisker of difference.
    cells = {
        f"p{i}": base + np.random.default_rng(100 + i).normal(0, 1e-6, len(base)) for i in range(7)
    }
    result = sweep_returns(cells, signal="x", dimension="parameter set")

    assert result.adjustment.variance < 0.5
    verdict = result.verdict()
    assert "INDEPENDENT" in verdict
    assert "no independent information" in verdict


def test_genuinely_independent_cells_get_the_plain_verdict() -> None:
    """The caveat must not fire on every sweep, or it becomes noise."""
    cells = {f"N{i}": noise(seed=500 + i) for i in range(14)}
    result = sweep_returns(cells, signal="x")

    if result.adjustment.variance >= 0.5:
        assert "INDEPENDENT" not in result.verdict()
