"""Paper-trading state, on disk.

Append-only. Every cycle writes a new record and nothing rewrites an old one, so
the history of what the book *thought* at each point survives — which is the only
way to tell later whether a decision was bad or merely unlucky.

Kept as JSONL under the data root rather than in Postgres. The compose stack has
a database and this deliberately does not use it: the platform has to run from a
clone with no services, and a paper book that cannot be inspected without a
running container is one nobody inspects.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quantlab.logging import get_logger

__all__ = ["CycleRecord", "PaperState", "PositionRecord"]

log = get_logger("quantlab.paper.state")


@dataclass(frozen=True, slots=True)
class PositionRecord:
    symbol: str
    shares: float
    mark: float
    value: float
    weight: float


@dataclass(frozen=True, slots=True)
class CycleRecord:
    """One paper-trading cycle, as it stood when it ran."""

    strategy: str
    #: The point-in-time instant the signals were computed at.
    as_of: str
    #: When the cycle actually ran. Not the same thing, and the gap is the
    #: staleness of the decision.
    ran_at: str
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    positions: list[PositionRecord] = field(default_factory=list)
    #: Orders as filled, with the cost each one paid.
    fills: list[dict[str, Any]] = field(default_factory=list)
    traded_notional: float = 0.0
    costs_paid: float = 0.0
    #: Instruments the signal wanted but the book could not reach, and why. Kept
    #: because a target that is never filled is a strategy that is not being run,
    #: and it would otherwise look like a strategy that chose not to trade.
    unfilled: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperState:
    """The append-only record of one paper book."""

    def __init__(self, path: Path, strategy: str) -> None:
        self.path = path
        self.strategy = strategy
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_strategy(cls, state_dir: Path, strategy: str) -> PaperState:
        safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in strategy)
        return cls(state_dir / "paper" / f"{safe}.jsonl", strategy)

    # ------------------------------------------------------------------ write --
    def append(self, record: CycleRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict(), separators=(",", ":")) + "\n")
        log.info(
            "paper.cycle",
            strategy=self.strategy,
            as_of=record.as_of,
            equity=round(record.equity, 2),
            positions=len(record.positions),
            traded=round(record.traded_notional, 2),
            costs=round(record.costs_paid, 2),
        )

    # ------------------------------------------------------------------- read --
    def history(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def latest(self) -> dict[str, Any] | None:
        records = self.history()
        return records[-1] if records else None

    def equity_curve(self) -> tuple[list[dt.datetime], list[float]]:
        dates, equity = [], []
        for record in self.history():
            dates.append(dt.datetime.fromisoformat(record["as_of"]))
            equity.append(float(record["equity"]))
        return dates, equity

    def periods_per_year(self) -> float:
        """Observations a year, inferred from the cycles actually recorded.

        Not assumed to be 252. A book rebalanced fortnightly has about 26
        observations a year, and annualising its Sharpe by the square root of 252
        overstates it threefold -- which would turn "too early to tell" into a
        number someone acts on.
        """
        dates, _ = self.equity_curve()
        if len(dates) < 3:
            return 252.0
        gaps = [
            (later - earlier).total_seconds() / 86400.0
            for earlier, later in itertools.pairwise(dates)
        ]
        spacing = sorted(gaps)[len(gaps) // 2]
        if spacing <= 0:
            return 252.0
        # Calendar days between cycles, converted to trading periods a year.
        return max(1.0, 365.25 / spacing)

    def has_run_for(self, as_of: dt.datetime) -> bool:
        """Whether a cycle already exists for this instant.

        Running twice for one date would double-count the trades and produce an
        equity curve with a step nobody can explain.
        """
        stamp = as_of.isoformat()
        return any(record["as_of"] == stamp for record in self.history())
