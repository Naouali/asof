"""Cross-validation that does not leak, tested for the leak."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantlab.validation.splitting import (
    CombinatorialPurgedCV,
    PurgedKFold,
    WalkForward,
)

N = 1000


# ----------------------------------------------------------------------------------
# The leak itself
# ----------------------------------------------------------------------------------
def test_train_and_test_never_overlap() -> None:
    for split in PurgedKFold(5, horizon=20).split(N):
        assert not set(split.train.tolist()) & set(split.test.tolist())


def test_purging_removes_exactly_the_overlapping_labels() -> None:
    """With a fixed horizon, the number purged around an interior fold is the
    horizon itself: the observations whose label window runs into the test set."""
    splits = list(PurgedKFold(5, horizon=20, embargo_pct=0.0).split(N))
    interior = splits[2]
    assert interior.purged == 20


def test_a_zero_horizon_purges_nothing() -> None:
    """Non-overlapping labels are the one case where plain k-fold is safe."""
    for split in PurgedKFold(5, horizon=0, embargo_pct=0.0).split(N):
        assert split.purged == 0


def test_a_longer_horizon_purges_more() -> None:
    counts = [
        sum(s.purged for s in PurgedKFold(5, horizon=h, embargo_pct=0.0).split(N))
        for h in (0, 10, 50, 100)
    ]
    assert counts == sorted(counts)


def test_the_embargo_removes_observations_after_the_test_window() -> None:
    """Serial correlation means a training point just after the test set still
    carries information from it, even with no label overlap."""
    without = next(iter(PurgedKFold(5, horizon=0, embargo_pct=0.0).split(N)))
    with_embargo = next(iter(PurgedKFold(5, horizon=0, embargo_pct=0.05).split(N)))
    assert with_embargo.embargoed == 50
    assert with_embargo.train_size < without.train_size


def test_folds_are_contiguous_in_time_never_shuffled() -> None:
    """Shuffling would let a model train on the future to predict the past, which
    scores beautifully and means nothing."""
    for split in PurgedKFold(5).split(N):
        assert np.array_equal(split.test, np.arange(split.test.min(), split.test.max() + 1))


def test_explicit_label_ends_are_honoured() -> None:
    """Labels of varying length -- a triple-barrier scheme, say -- need per-
    observation end times rather than one horizon."""
    ends = np.minimum(np.arange(N) + np.arange(N) % 50, N - 1)
    splits = list(PurgedKFold(5, embargo_pct=0.0).split(N, label_end=ends))
    assert sum(s.purged for s in splits) > 0


def test_a_label_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="cannot end before"):
        list(PurgedKFold(3).split(100, label_end=np.zeros(100, dtype=int)))


def test_a_horizon_that_consumes_everything_fails_loudly() -> None:
    """If purging leaves no training data, the experiment cannot be validated and
    saying so beats returning an empty fold."""
    with pytest.raises(ValueError, match="no training data left"):
        list(PurgedKFold(2, horizon=N).split(N))


# ----------------------------------------------------------------------------------
# Combinatorial purged CV
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(("groups", "test_groups"), [(6, 2), (8, 2), (8, 3), (10, 2)])
def test_path_count_matches_the_combinatorial_formula(groups: int, test_groups: int) -> None:
    cv = CombinatorialPurgedCV(n_groups=groups, test_groups=test_groups)
    assert cv.n_splits == math.comb(groups, test_groups)
    assert cv.n_paths == cv.n_splits * test_groups // groups


def test_every_path_covers_the_sample_exactly_once() -> None:
    """The property that makes a path a backtest: no observation tested twice, none
    missed. Taking a split's whole test set rather than this path's own group would
    duplicate observations and inflate every path statistic."""
    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, horizon=20)
    for path in cv.paths(N):
        covered = np.concatenate([segment.indices for segment in path])
        assert len(covered) == N
        assert len(np.unique(covered)) == N


def test_non_contiguous_test_groups_do_not_purge_the_whole_sample() -> None:
    """The bug this cost: combinatorial CV routinely picks the first and last
    groups together, and purging over the span between them deletes almost all the
    training data -- silently, as an inexplicable failure much later."""
    cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, horizon=20)
    splits = {s.test_groups: s for s in cv.split(N)}
    far_apart = splits[(0, 5)]
    assert far_apart.train_size > N * 0.5
    assert far_apart.removed_fraction < 0.2


def test_cpcv_never_leaks() -> None:
    cv = CombinatorialPurgedCV(n_groups=8, test_groups=2, horizon=30, embargo_pct=0.02)
    for split in cv.split(N):
        assert not set(split.train.tolist()) & set(split.test.tolist())


def test_cpcv_rejects_degenerate_configurations() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        CombinatorialPurgedCV(n_groups=2)
    with pytest.raises(ValueError, match="fewer than n_groups"):
        CombinatorialPurgedCV(n_groups=5, test_groups=5)


# ----------------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------------
def test_walk_forward_only_ever_trains_on_the_past() -> None:
    for split in WalkForward(train_size=250, test_size=125).split(N):
        assert split.train.max() < split.test.min()


def test_expanding_windows_grow_and_rolling_ones_do_not() -> None:
    expanding = [s.train_size for s in WalkForward(train_size=250, test_size=125).split(N)]
    rolling = [
        s.train_size for s in WalkForward(train_size=250, test_size=125, expanding=False).split(N)
    ]
    assert expanding == sorted(expanding)
    assert expanding[-1] > expanding[0]
    assert max(rolling) <= 250


def test_walk_forward_purges_labels_that_run_into_the_test_window() -> None:
    """The only leak available in walk-forward, and it is entirely silent."""
    splits = list(WalkForward(train_size=250, test_size=125, horizon=20).split(N))
    assert splits[0].purged == 20


def test_walk_forward_covers_the_sample_after_its_warm_up() -> None:
    splits = list(WalkForward(train_size=250, test_size=125).split(N))
    covered = np.concatenate([s.test for s in splits])
    assert covered.min() == 250
    assert covered.max() == N - 1
    assert len(np.unique(covered)) == len(covered)


def test_a_training_window_longer_than_the_sample_fails_loudly() -> None:
    with pytest.raises(ValueError, match="leave nothing to test"):
        list(WalkForward(train_size=N, test_size=10).split(N))


# ----------------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------------
def test_splits_report_what_purging_cost() -> None:
    """If purging removes most of the training set, the label horizon is too long
    for the sample and no amount of cross-validation will rescue the experiment."""
    # An interior fold: the first has no training data before it to purge.
    split = list(PurgedKFold(5, horizon=100, embargo_pct=0.02).split(N))[2]
    assert split.purged > 0
    assert 0 < split.removed_fraction < 1
