"""The signal contract.

Every signal declares what it needs, what it costs to trade, where it comes from,
and -- unusually -- **how it is known to fail**. That last field is not decoration.
A signal library accumulates entries faster than anyone can remember the caveats
attached to each, and the tearsheet that quotes a Sharpe should be able to quote
the failure modes beside it.

Two rules shape the interface:

**Signals never return weights.** They return a score or a position, and portfolio
construction owns the conversion to weights (spec section 4). A signal that sizes
its own positions has quietly taken over risk management, and two such signals
cannot be combined without one of them being wrong.

**Signals read only through a point-in-time snapshot.** ``compute`` receives a
:class:`~quantlab.data.pit.Snapshot`, so a signal physically cannot see data that
was not knowable on its as-of date. That is the whole reason the snapshot exists,
and it is why the signal interface takes one rather than a DataFrame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import polars as pl

from quantlab.conventions import RebalanceFrequency
from quantlab.data.catalogue import AssetClass
from quantlab.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.data.pit import Snapshot

__all__ = [
    "SIGNAL_REGISTRY",
    "EvidenceGrade",
    "Signal",
    "SignalOutput",
    "SignalSpec",
    "SignalUnavailableError",
    "get_signal",
    "register_signal",
]

log = get_logger("quantlab.signals")


class SignalUnavailableError(RuntimeError):
    """A signal cannot run because the data it needs is not in the lake.

    Raised rather than returning empty scores. An empty cross-section looks
    exactly like a signal with no view, and a strategy built on one would trade
    nothing while appearing to work.
    """


class SignalOutput(str, Enum):
    """What the numbers a signal returns actually mean."""

    CROSS_SECTIONAL_SCORE = "cross_sectional_score"
    """Comparable *within* a date, across instruments. Only the ranking is
    meaningful; the level is not. Used by long/short constructions."""

    TIME_SERIES_POSITION = "time_series_position"
    """Comparable *across* dates for one instrument. Roughly a standardised
    position in [-1, 1], where zero means flat. Used by trend following, where an
    instrument's own history is the reference and other instruments are irrelevant."""


class EvidenceGrade(str, Enum):
    """How much the literature actually supports a signal.

    Spec section 4 asks the platform to be argumentative: a Tier 3 signal must
    surface the prior evidence that it is decayed or dead, not merely be
    implemented and left to speak for itself.
    """

    STRONG = "strong"
    """Replicated widely, survives costs, still works out of sample. Tier 1."""

    MIXED = "mixed"
    """Real but fragile: sensitive to construction, crowded, or costly. Tier 2."""

    DECAYED = "decayed"
    """The published effect has weakened or disappeared since publication. Tier 3.
    Implemented so it can be re-tested on current data, with the prior attached."""


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Everything a reader of a signal's Sharpe ratio should know about it."""

    name: str
    asset_class: AssetClass
    #: 1, 2 or 3 -- the spec's build priority, which is evidence quality, not interest.
    tier: int
    output: SignalOutput
    #: Canonical datasets this signal reads, as named in quantlab.data.schemas.
    required_datasets: tuple[str, ...]
    rebalance: RebalanceFrequency
    #: One-way turnover per year, as a fraction of the book. An estimate, and the
    #: single most useful number for guessing whether a signal survives costs
    #: before any of it is built.
    expected_turnover_annual: float
    evidence: EvidenceGrade
    reference: str
    #: How this signal is known to go wrong. Never empty.
    known_failure_modes: str
    #: Trailing history the signal needs before it can produce a score.
    warmup_days: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.tier <= 3:
            raise ValueError("tier must be 1, 2 or 3")
        if not self.required_datasets:
            raise ValueError(f"{self.name}: a signal must declare the data it reads")
        if len(self.known_failure_modes) < 40:
            raise ValueError(
                f"{self.name}: known_failure_modes must say something substantive. "
                "Every signal in the literature has a documented way of failing; if "
                "you cannot name this one's, you do not understand it well enough "
                "to trade it."
            )
        if self.expected_turnover_annual < 0:
            raise ValueError("expected_turnover_annual must be non-negative")

    def describe(self) -> str:
        return (
            f"{self.name} (tier {self.tier}, {self.evidence.value} evidence)\n"
            f"  asset class    {self.asset_class.value}\n"
            f"  output         {self.output.value}\n"
            f"  datasets       {', '.join(self.required_datasets)}\n"
            f"  rebalance      {self.rebalance.value}, ~{self.expected_turnover_annual:.0%} "
            f"turnover a year\n"
            f"  reference      {self.reference}\n"
            f"  fails when     {self.known_failure_modes}"
        )


SIGNAL_REGISTRY: dict[str, type[Signal]] = {}


def register_signal(cls: type[Signal]) -> type[Signal]:
    """Class decorator adding a signal to the registry."""
    name = cls.spec.name
    if name in SIGNAL_REGISTRY:
        raise RuntimeError(f"duplicate signal name {name!r}")
    SIGNAL_REGISTRY[name] = cls
    return cls


def get_signal(name: str) -> type[Signal]:
    try:
        return SIGNAL_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(SIGNAL_REGISTRY))
        raise KeyError(f"unknown signal {name!r}; known signals: {known}") from None


class Signal(ABC):
    """Base class for one signal."""

    spec: ClassVar[SignalSpec]

    #: Columns every signal returns.
    OUTPUT_COLUMNS: ClassVar[tuple[str, ...]] = ("symbol", "as_of", "score")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.spec.name!r}, tier={self.spec.tier})"

    # ---------------------------------------------------------------- compute --
    @abstractmethod
    def compute(self, snapshot: Snapshot, symbols: Sequence[str]) -> pl.DataFrame:
        """Scores for ``symbols`` as of the snapshot's instant.

        Returns ``symbol``, ``as_of`` and ``score``. ``as_of`` is the snapshot's
        instant, not the date of the data used -- the score is a statement about
        what was knowable then.

        Raises :class:`SignalUnavailableError` when the data is not there. An empty
        frame means "no view on any of these instruments", which is a different
        claim and must not be produced by a missing dataset.
        """

    # ---------------------------------------------------------------- helpers --
    def empty(self, snapshot: Snapshot) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8(),
                "as_of": pl.Datetime(time_unit="us", time_zone="UTC"),
                "score": pl.Float64(),
            }
        ).with_columns(pl.lit(snapshot.as_of).alias("as_of"))

    def finalise(self, snapshot: Snapshot, scores: dict[str, float]) -> pl.DataFrame:
        """Assemble a score mapping into the canonical output frame."""
        usable = {
            symbol: value
            for symbol, value in scores.items()
            if value is not None and value == value  # drops NaN
        }
        if not usable:
            return self.empty(snapshot)
        return pl.DataFrame(
            {
                "symbol": list(usable.keys()),
                "as_of": [snapshot.as_of] * len(usable),
                "score": [float(v) for v in usable.values()],
            }
        ).sort("symbol")

    def require_datasets(self, snapshot: Snapshot) -> None:
        """Fail loudly when a required dataset has nothing in it at this date."""
        missing = []
        for dataset in self.spec.required_datasets:
            if snapshot.frame(dataset).height == 0:
                missing.append(dataset)
        if missing:
            raise SignalUnavailableError(
                f"{self.spec.name} needs {missing} and the lake has none at "
                f"{snapshot.as_of:%Y-%m-%d}. Ingest it, or drop this signal from the "
                "run -- it cannot produce a score and returning an empty one would "
                "look like a signal with no view."
            )
