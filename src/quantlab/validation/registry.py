"""The automatic trial counter.

Spec section 7 is explicit about why this is machinery and not a convention:
*"Manual honesty about trial counts does not work."* Nobody remembers that they
tried forty lookbacks last Tuesday, and the deflated Sharpe is only as honest as
the number fed into it. So the platform counts.

Every backtest run records itself here: the signal family, a fingerprint of the
configuration, and the moments the deflation needs. The count that reaches the
deflated Sharpe is the number of **distinct configurations** ever run against that
family -- re-running an identical config is not a new trial, but changing a single
parameter is.

The registry is append-only JSONL in the state directory, for the same reason the
lake is append-only: a count you can quietly revise downward is not a count. It is
deliberately awkward to reduce. Deleting the file resets your trial count to zero,
and the only person that fools is you.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quantlab.logging import get_logger

__all__ = ["Trial", "TrialRegistry", "TrialSet", "config_fingerprint"]

log = get_logger("quantlab.validation.registry")

REGISTRY_FILENAME = "trials.jsonl"
BARS_PER_YEAR = 252.0


def config_fingerprint(**parameters: Any) -> str:
    """Stable hash of a run's configuration.

    Canonical JSON with sorted keys, so the same configuration always produces the
    same fingerprint regardless of dict ordering. Floats are rounded to twelve
    significant digits: two runs differing in the sixteenth decimal of a
    floating-point parameter are the same trial, and treating them as two would let
    a trial count be inflated -- or, worse, reset -- by arithmetic noise.
    """

    def canonical(value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return str(value)
            return float(f"{value:.12g}")
        if isinstance(value, dict):
            return {str(k): canonical(v) for k, v in sorted(value.items())}
        if isinstance(value, list | tuple):
            return [canonical(v) for v in value]
        if isinstance(value, Path | dt.date | dt.datetime):
            return str(value)
        if hasattr(value, "value") and hasattr(value, "name"):  # an Enum
            return str(value.value)
        return value

    payload = json.dumps(canonical(parameters), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Trial:
    """One configuration, run once against one signal family."""

    family: str
    config_hash: str
    name: str
    #: Annualised, for readability. Deflation uses the per-observation value.
    sharpe_annual: float
    observations: int
    skewness: float = 0.0
    excess_kurtosis: float = 0.0
    recorded_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sharpe_per_observation(self) -> float:
        return self.sharpe_annual / math.sqrt(BARS_PER_YEAR)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "config_hash": self.config_hash,
            "name": self.name,
            "sharpe_annual": self.sharpe_annual,
            "observations": self.observations,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "recorded_at": self.recorded_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trial:
        return cls(
            family=str(payload["family"]),
            config_hash=str(payload["config_hash"]),
            name=str(payload.get("name", "")),
            sharpe_annual=float(payload["sharpe_annual"]),
            observations=int(payload["observations"]),
            skewness=float(payload.get("skewness", 0.0)),
            excess_kurtosis=float(payload.get("excess_kurtosis", 0.0)),
            recorded_at=dt.datetime.fromisoformat(payload["recorded_at"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class TrialSet:
    """Every distinct configuration tried against one family."""

    family: str
    trials: tuple[Trial, ...]

    @property
    def count(self) -> int:
        return len(self.trials)

    @property
    def best(self) -> Trial | None:
        return max(self.trials, key=lambda t: t.sharpe_annual) if self.trials else None

    def sharpe_variance(self, observations: int | None = None) -> float:
        """Variance of the per-observation Sharpe ratios across trials.

        Floored at ``1/T``, the sampling variance of a Sharpe estimate under the
        null. Without that floor, a sweep of near-identical configurations shows
        almost no dispersion, the expected maximum collapses to nearly zero, and
        forty trials get deflated as though they were one -- which is precisely the
        search pattern most likely to overfit.
        """
        if self.count < 2:
            return 0.0
        values = np.array([t.sharpe_per_observation for t in self.trials], dtype=float)
        observed = float(np.var(values, ddof=1))
        sample_length = observations or int(np.median([t.observations for t in self.trials]))
        floor = 1.0 / sample_length if sample_length > 0 else 0.0
        return max(observed, floor)

    def describe(self) -> str:
        if not self.trials:
            return f"{self.family}: no trials recorded"
        best = self.best
        if best is None:  # pragma: no cover - unreachable given the guard above
            return f"{self.family}: no trials recorded"
        return (
            f"{self.family}: {self.count} distinct configuration(s) tried, "
            f"best annualised Sharpe {best.sharpe_annual:.2f} ({best.name})"
        )


class TrialRegistry:
    """Append-only record of every configuration ever backtested."""

    def __init__(self, state_dir: Path) -> None:
        self.path = Path(state_dir) / REGISTRY_FILENAME

    # ----------------------------------------------------------------- write --
    def record(self, trial: Trial) -> bool:
        """Record a trial. Returns ``True`` if it was new to its family.

        Re-running an identical configuration is not a new trial -- the search did
        not widen -- so it is recorded once and counted once.
        """
        existing = {t.config_hash for t in self.family(trial.family).trials}
        if trial.config_hash in existing:
            log.debug(
                "validation.trial_already_seen",
                family=trial.family,
                config_hash=trial.config_hash,
            )
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trial.as_dict()) + "\n")
        log.info(
            "validation.trial_recorded",
            family=trial.family,
            config_hash=trial.config_hash,
            trials_in_family=len(existing) + 1,
        )
        return True

    # ------------------------------------------------------------------ read --
    def all_trials(self) -> list[Trial]:
        if not self.path.exists():
            return []
        out: list[Trial] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Trial.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                # A malformed line must not silently reduce the trial count, which
                # would make every deflated Sharpe more flattering.
                log.warning("validation.trial_unreadable", error=str(exc)[:120])
        return out

    def family(self, name: str) -> TrialSet:
        """Every distinct configuration recorded against one family.

        Deduplicated by fingerprint, keeping the first record, so the count is of
        the search's *width* rather than of how many times it was re-run.
        """
        seen: dict[str, Trial] = {}
        for trial in self.all_trials():
            if trial.family == name and trial.config_hash not in seen:
                seen[trial.config_hash] = trial
        return TrialSet(family=name, trials=tuple(seen.values()))

    def families(self) -> list[str]:
        return sorted({trial.family for trial in self.all_trials()})

    def summary(self) -> list[TrialSet]:
        return [self.family(name) for name in self.families()]
