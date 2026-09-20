"""Cross-validation that does not leak.

Standard k-fold assumes observations are exchangeable. Financial observations are
not: a label computed over the next twenty days overlaps the twenty days after it,
so an observation in the training set can share almost all of its outcome with one
in the test set. The model then "predicts" something it was effectively shown, and
the out-of-sample score is fiction.

Two corrections, both from López de Prado:

**Purging** removes training observations whose label window overlaps the test
window at all. This is the correction that matters most and the one most often
skipped, because without it nothing appears to be wrong.

**Embargo** additionally removes training observations immediately *after* the test
window. Serial correlation in features means a training point just after the test
set still carries information from it, even with no label overlap.

:class:`CombinatorialPurgedCV` goes further and produces a *distribution* of
out-of-sample paths rather than a single number. One backtest path gives you one
Sharpe and no sense of its sampling error; sixty paths tell you whether the result
is robust or whether you happened to test on a kind quarter.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from quantlab.logging import get_logger

__all__ = [
    "CombinatorialPurgedCV",
    "PathSegment",
    "PurgedKFold",
    "Split",
    "WalkForward",
]

log = get_logger("quantlab.validation.splitting")


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test division, with the cost of purging recorded.

    ``purged`` and ``embargoed`` are kept because they are a diagnostic: if purging
    removes most of the training set, the label horizon is long relative to the
    sample and no amount of cross-validation will rescue the experiment.
    """

    train: np.ndarray
    test: np.ndarray
    purged: int
    embargoed: int
    #: Which test group(s) this split used, for CPCV path reconstruction.
    test_groups: tuple[int, ...] = ()

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @property
    def removed_fraction(self) -> float:
        total = self.train_size + self.purged + self.embargoed
        return (self.purged + self.embargoed) / total if total else 0.0


def _label_ends(n: int, label_end: Sequence[int] | np.ndarray | None, horizon: int) -> np.ndarray:
    """Index at which each observation's label is finally known."""
    if label_end is not None:
        ends = np.asarray(label_end, dtype=np.int64)
        if len(ends) != n:
            raise ValueError(f"label_end has {len(ends)} entries for {n} observations")
        if np.any(ends < np.arange(n)):
            raise ValueError("a label cannot end before the observation that generates it")
        return ends
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    return np.minimum(np.arange(n) + horizon, n - 1)


def _contiguous_blocks(indices: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted indices into ``(start, end)`` runs of consecutive values."""
    if len(indices) == 0:
        return []
    ordered = np.sort(indices)
    breaks = np.flatnonzero(np.diff(ordered) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(ordered) - 1]])
    return [(int(ordered[a]), int(ordered[b])) for a, b in zip(starts, ends, strict=True)]


def _purge_and_embargo(
    n: int,
    test_indices: np.ndarray,
    ends: np.ndarray,
    embargo: int,
) -> tuple[np.ndarray, int, int]:
    """Training indices left after purging overlaps and applying the embargo.

    Purging is done **per contiguous test block**, not over the span from the first
    test index to the last. Combinatorial cross-validation regularly picks test
    groups that are far apart -- the first and the last, say -- and treating that
    as one window purges the entire sample between them, which silently deletes
    almost all the training data and would show up only as an inexplicable failure.
    """
    candidates = np.setdiff1d(np.arange(n), test_indices, assume_unique=False)
    overlaps = np.zeros(len(candidates), dtype=bool)
    embargo_zone = np.zeros(len(candidates), dtype=bool)

    for test_start, test_end in _contiguous_blocks(test_indices):
        # An observation leaks if it starts before this block ends and its label is
        # still running when the block starts: the two share an outcome.
        overlaps |= (candidates <= test_end) & (ends[candidates] >= test_start)
        if embargo > 0:
            embargo_zone |= (candidates > test_end) & (candidates <= test_end + embargo)

    purged = int(overlaps.sum())
    embargoed = int((embargo_zone & ~overlaps).sum())
    keep = ~(overlaps | embargo_zone)
    return candidates[keep], purged, embargoed


@dataclass(frozen=True, slots=True)
class PathSegment:
    """One group's out-of-sample slice within a reassembled CPCV path."""

    split: Split
    group: int
    indices: np.ndarray


class PurgedKFold:
    """K-fold over time, with purging and an embargo.

    Folds are contiguous blocks in time order, never shuffled: shuffling would let
    a model train on the future to predict the past, which scores beautifully and
    means nothing.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        horizon: int = 0,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def split(
        self, n: int, *, label_end: Sequence[int] | np.ndarray | None = None
    ) -> Iterator[Split]:
        if n < self.n_splits * 2:
            raise ValueError(f"{n} observations cannot be split into {self.n_splits} usable folds")
        ends = _label_ends(n, label_end, self.horizon)
        embargo = int(n * self.embargo_pct)
        bounds = np.array_split(np.arange(n), self.n_splits)

        for fold, test_indices in enumerate(bounds):
            train, purged, embargoed = _purge_and_embargo(n, test_indices, ends, embargo)
            if len(train) == 0:
                raise ValueError(
                    f"fold {fold} has no training data left after purging. The label "
                    "horizon is too long relative to the sample -- shorten it, or "
                    "accept that this experiment cannot be validated."
                )
            yield Split(
                train=train,
                test=test_indices,
                purged=purged,
                embargoed=embargoed,
                test_groups=(fold,),
            )


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation.

    Splits the sample into ``n_groups`` blocks and tests on every combination of
    ``test_groups`` of them, purging and embargoing as usual. Each observation
    therefore appears out-of-sample many times, in different company, and the
    splits reassemble into ``n_paths`` distinct backtest paths.

    The point is the distribution. A single walk-forward gives one number with no
    sense of its sampling error; a spread of paths shows whether a Sharpe of 1.2 is
    "reliably around 1.2" or "anywhere from -0.3 to 2.5 depending on which quarter
    you happened to test on".
    """

    def __init__(
        self,
        n_groups: int = 6,
        test_groups: int = 2,
        *,
        horizon: int = 0,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_groups < 3:
            raise ValueError("n_groups must be at least 3")
        if not 1 <= test_groups < n_groups:
            raise ValueError("test_groups must be at least 1 and fewer than n_groups")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_groups = n_groups
        self.test_groups = test_groups
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.test_groups)

    @property
    def n_paths(self) -> int:
        """Distinct out-of-sample paths the splits reassemble into."""
        return self.n_splits * self.test_groups // self.n_groups

    def split(
        self, n: int, *, label_end: Sequence[int] | np.ndarray | None = None
    ) -> Iterator[Split]:
        if n < self.n_groups * 2:
            raise ValueError(f"{n} observations cannot be split into {self.n_groups} usable groups")
        ends = _label_ends(n, label_end, self.horizon)
        embargo = int(n * self.embargo_pct)
        groups = np.array_split(np.arange(n), self.n_groups)

        for chosen in combinations(range(self.n_groups), self.test_groups):
            test_indices = np.concatenate([groups[index] for index in chosen])
            train, purged, embargoed = _purge_and_embargo(n, test_indices, ends, embargo)
            if len(train) == 0:
                raise ValueError(f"test groups {chosen} leave no training data after purging")
            yield Split(
                train=train,
                test=np.sort(test_indices),
                purged=purged,
                embargoed=embargoed,
                test_groups=chosen,
            )

    def paths(
        self, n: int, *, label_end: Sequence[int] | np.ndarray | None = None
    ) -> list[list[PathSegment]]:
        """Reassemble the splits into complete, non-overlapping backtest paths.

        Each group appears in exactly ``n_paths`` splits, so the *j*-th split
        containing a group supplies that group's out-of-sample segment on path *j*.
        Every path then covers the whole sample exactly once.

        The subtlety worth stating: a split tests on several groups at once, and
        those groups can be assigned to *different* paths. A path therefore takes
        only its own group's slice of a split's test set, never the whole thing.
        Taking the whole test set duplicates observations and inflates every path
        statistic -- which looks like a longer sample rather than like a bug.
        """
        groups = np.array_split(np.arange(n), self.n_groups)
        by_group: dict[int, list[Split]] = {index: [] for index in range(self.n_groups)}
        for split in self.split(n, label_end=label_end):
            for group in split.test_groups:
                by_group[group].append(split)

        assembled: list[list[PathSegment]] = []
        for path in range(self.n_paths):
            assembled.append(
                [
                    PathSegment(split=by_group[group][path], group=group, indices=groups[group])
                    for group in range(self.n_groups)
                ]
            )
        return assembled


class WalkForward:
    """Sequential out-of-sample testing: fit on the past, test on what came next.

    The most intuitive validation and the weakest, because it produces exactly one
    path. It is here because it is what an actual deployment looks like, and a
    strategy that cannot survive it is not worth cross-validating -- but a strategy
    that *does* survive it has cleared a single sample, not a distribution.

    ``expanding`` keeps all history in the training window; ``rolling`` keeps a
    fixed length, which adapts faster to regime change and discards the evidence
    that a regime changed.
    """

    def __init__(
        self,
        *,
        train_size: int,
        test_size: int,
        expanding: bool = True,
        horizon: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if train_size < 1 or test_size < 1:
            raise ValueError("train_size and test_size must be positive")
        self.train_size = train_size
        self.test_size = test_size
        self.expanding = expanding
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def split(
        self, n: int, *, label_end: Sequence[int] | np.ndarray | None = None
    ) -> Iterator[Split]:
        if n <= self.train_size:
            raise ValueError(
                f"{n} observations leave nothing to test after a {self.train_size}-bar "
                "training window"
            )
        ends = _label_ends(n, label_end, self.horizon)
        embargo = int(n * self.embargo_pct)

        start = self.train_size
        fold = 0
        while start < n:
            stop = min(start + self.test_size, n)
            test_indices = np.arange(start, stop)
            train_start = 0 if self.expanding else max(0, start - self.train_size)
            candidates = np.arange(train_start, start)

            # Purge from the training window anything whose label runs into the test
            # window. In walk-forward this is the only leak available, and it is
            # entirely silent.
            overlaps = ends[candidates] >= start
            purged = int(overlaps.sum())
            train = candidates[~overlaps]
            if len(train) == 0:
                raise ValueError(f"walk-forward fold {fold} has no training data after purging")

            yield Split(
                train=train,
                test=test_indices,
                purged=purged,
                embargoed=embargo,
                test_groups=(fold,),
            )
            start = stop
            fold += 1
