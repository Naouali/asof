"""Position, exposure and group constraints.

A mean-variance optimiser handed real estimates will, left alone, put ninety per
cent of the book in three names and lever the result four times. It is not
malfunctioning: it is taking the inputs at face value, and the inputs do not
deserve it. Constraints are the mechanism for saying so.

Group constraints -- sector, country, asset class -- matter more than they look.
A long/short book that is dollar neutral and sector-concentrated is a sector bet
wearing a hedge, and the exposure attribution in :mod:`quantlab.risk` exists to
catch exactly that after the fact. Constraining it beforehand is cheaper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Constraints", "GroupLimit", "build_groups"]


@dataclass(frozen=True, slots=True)
class GroupLimit:
    """A cap on the net or gross exposure of a set of positions."""

    name: str
    #: Indices of the members within the weight vector.
    members: tuple[int, ...]
    max_net: float | None = None
    max_gross: float | None = None

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError(f"group {self.name!r} has no members")
        if self.max_net is None and self.max_gross is None:
            raise ValueError(f"group {self.name!r} constrains nothing")


@dataclass(frozen=True, slots=True)
class Constraints:
    """What the optimiser is allowed to do."""

    #: Largest absolute weight in any one position.
    max_position: float | None = 0.10
    #: Smallest allowed weight. ``0.0`` makes the book long only.
    min_position: float | None = None
    #: Cap on the sum of absolute weights. The leverage limit.
    max_gross: float | None = 2.0
    #: Bounds on the sum of signed weights. ``(0, 0)`` forces dollar neutrality.
    net_range: tuple[float, float] | None = None
    #: Sector, country or asset-class caps.
    groups: tuple[GroupLimit, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.max_position is not None and self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.max_gross is not None and self.max_gross <= 0:
            raise ValueError("max_gross must be positive")
        if (
            self.min_position is not None
            and self.max_position is not None
            and self.min_position > self.max_position
        ):
            raise ValueError("min_position cannot exceed max_position")
        if self.net_range is not None and self.net_range[0] > self.net_range[1]:
            raise ValueError("net_range must be (low, high)")

    @classmethod
    def long_only(cls, max_position: float = 0.10) -> Constraints:
        return cls(max_position=max_position, min_position=0.0, max_gross=1.0, net_range=(1.0, 1.0))

    @classmethod
    def dollar_neutral(cls, max_position: float = 0.05, max_gross: float = 2.0) -> Constraints:
        return cls(max_position=max_position, max_gross=max_gross, net_range=(0.0, 0.0))

    @classmethod
    def unconstrained(cls) -> Constraints:
        """No limits at all.

        Provided so a tearsheet can show what the optimiser wanted before being
        told it could not have it -- the gap between the two is usually the most
        informative thing about a mean-variance result.
        """
        return cls(max_position=None, min_position=None, max_gross=None, net_range=None)

    def _gross_is_implied(self) -> bool:
        """Whether the net constraint already bounds gross exposure.

        True for a long-only book: with every weight non-negative, the sum of
        weights *is* the sum of absolute weights.
        """
        if self.min_position is None or self.min_position < 0:
            return False
        if self.net_range is None or self.max_gross is None:
            return False
        return self.net_range[1] <= self.max_gross + 1e-12

    # ------------------------------------------------------------------ apply --
    def bounds(self, n_assets: int) -> list[tuple[float | None, float | None]]:
        low = self.min_position
        high = self.max_position
        if high is not None and low is None:
            low = -high
        return [(low, high)] * n_assets

    def clip(self, weights: np.ndarray) -> np.ndarray:
        """Force a weight vector inside the box bounds. Used as a starting point."""
        low, high = self.bounds(len(weights))[0]
        clipped: np.ndarray = np.clip(np.asarray(weights, dtype=float), low, high)
        return clipped

    def scipy_constraints(self, n_assets: int) -> list[dict[str, Any]]:
        """Inequality and equality constraints in the form SLSQP expects."""
        out: list[dict[str, Any]] = []

        if self.max_gross is not None and not self._gross_is_implied():
            # |w|.sum() is non-smooth at zero, so it is only worth imposing when it
            # can actually bind. In a long-only book the net constraint already
            # bounds gross exactly, and imposing both makes SLSQP report the
            # perfectly feasible problem as "inequality constraints incompatible".
            out.append(
                {
                    "type": "ineq",
                    "fun": lambda w, cap=self.max_gross: cap - np.abs(w).sum(),
                }
            )

        if self.net_range is not None:
            low, high = self.net_range
            if low == high:
                out.append({"type": "eq", "fun": lambda w, t=low: w.sum() - t})
            else:
                out.append({"type": "ineq", "fun": lambda w, t=low: w.sum() - t})
                out.append({"type": "ineq", "fun": lambda w, t=high: t - w.sum()})

        for group in self.groups:
            members = np.array(group.members, dtype=int)
            if np.any(members >= n_assets):
                raise ValueError(
                    f"group {group.name!r} references asset {int(members.max())} "
                    f"in a universe of {n_assets}"
                )
            if group.max_net is not None:
                out.append(
                    {
                        "type": "ineq",
                        "fun": lambda w, m=members, cap=group.max_net: cap - abs(w[m].sum()),
                    }
                )
            if group.max_gross is not None:
                out.append(
                    {
                        "type": "ineq",
                        "fun": lambda w, m=members, cap=group.max_gross: cap - np.abs(w[m]).sum(),
                    }
                )
        return out

    def check_feasible(self, n_assets: int) -> None:
        """Raise if no weight vector can satisfy these constraints at this size.

        The trap this exists for: ``Constraints.long_only()`` caps each position
        at 10%, so on a universe of eight names the largest possible book is 80%
        of capital while the net constraint demands 100%. The optimiser reports
        "inequality constraints incompatible" and returns weights that violate the
        constraints -- which look entirely plausible.

        Checked before optimising, so an impossible problem is named rather than
        silently answered.
        """
        if n_assets < 1:
            raise ValueError("need at least one asset")

        low, high = self.bounds(n_assets)[0]
        if low is not None and high is not None and low > high:
            raise ValueError(f"position bounds [{low}, {high}] are empty")

        reachable_low = (low or -np.inf) * n_assets
        reachable_high = (high or np.inf) * n_assets

        if self.net_range is not None:
            target_low, target_high = self.net_range
            if target_low > reachable_high + 1e-12:
                raise ValueError(
                    f"net exposure of {target_low:.2f} is unreachable: {n_assets} "
                    f"positions capped at {high:.2f} sum to at most "
                    f"{reachable_high:.2f}. Widen max_position, or use a larger "
                    "universe."
                )
            if target_high < reachable_low - 1e-12:
                raise ValueError(
                    f"net exposure of {target_high:.2f} is unreachable: {n_assets} "
                    f"positions floored at {low:.2f} sum to at least "
                    f"{reachable_low:.2f}."
                )

        if (
            self.max_gross is not None
            and self.net_range is not None
            and abs(self.net_range[0]) > self.max_gross + 1e-12
        ):
            raise ValueError(
                f"net exposure of {self.net_range[0]:.2f} cannot fit inside a "
                f"gross limit of {self.max_gross:.2f}"
            )

    # ----------------------------------------------------------------- report --
    def violations(self, weights: np.ndarray) -> list[str]:
        """Every constraint this weight vector breaks, in plain words.

        Numerical optimisers satisfy constraints to a tolerance, not exactly, so a
        small breach is expected. A large one means the problem was infeasible and
        the optimiser gave up somewhere that looks like an answer.
        """
        weights = np.asarray(weights, dtype=float)
        out: list[str] = []
        tolerance = 1e-6

        if self.max_position is not None:
            worst = float(np.abs(weights).max())
            if worst > self.max_position + tolerance:
                out.append(f"largest position {worst:.3f} exceeds {self.max_position:.3f}")
        if self.min_position is not None:
            smallest = float(weights.min())
            if smallest < self.min_position - tolerance:
                out.append(f"smallest position {smallest:.3f} below {self.min_position:.3f}")
        if self.max_gross is not None:
            gross = float(np.abs(weights).sum())
            if gross > self.max_gross + tolerance:
                out.append(f"gross exposure {gross:.3f} exceeds {self.max_gross:.3f}")
        if self.net_range is not None:
            net = float(weights.sum())
            low, high = self.net_range
            if not (low - tolerance <= net <= high + tolerance):
                out.append(f"net exposure {net:.3f} outside [{low:.3f}, {high:.3f}]")

        for group in self.groups:
            members = np.array(group.members, dtype=int)
            if group.max_net is not None:
                net = abs(float(weights[members].sum()))
                if net > group.max_net + tolerance:
                    out.append(f"group {group.name!r} net {net:.3f} exceeds {group.max_net:.3f}")
            if group.max_gross is not None:
                gross = float(np.abs(weights[members]).sum())
                if gross > group.max_gross + tolerance:
                    out.append(
                        f"group {group.name!r} gross {gross:.3f} exceeds {group.max_gross:.3f}"
                    )
        return out

    def describe(self) -> str:
        parts: list[str] = []
        if self.max_position is not None:
            parts.append(f"position <= {self.max_position:.1%}")
        if self.min_position is not None:
            parts.append(f"position >= {self.min_position:.1%}")
        if self.max_gross is not None:
            parts.append(f"gross <= {self.max_gross:.1f}x")
        if self.net_range is not None:
            low, high = self.net_range
            parts.append("dollar neutral" if low == high == 0 else f"net in [{low}, {high}]")
        for group in self.groups:
            parts.append(f"{group.name} capped")
        return "; ".join(parts) if parts else "unconstrained"


def build_groups(labels: Sequence[str], **caps: float) -> tuple[GroupLimit, ...]:
    """Group limits from a per-asset label, e.g. a sector or country column."""
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(str(label), []).append(index)
    return tuple(
        GroupLimit(
            name=name,
            members=tuple(members),
            max_net=caps.get("max_net"),
            max_gross=caps.get("max_gross"),
        )
        for name, members in sorted(groups.items())
    )
