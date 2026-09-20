"""Saved backtest results, so strategies can be compared without re-running them.

A backtest is expensive and its output is small. Storing the curve and the
statistics when a run finishes turns strategy comparison from a thirty-second
wait into a file read, which is the difference between a dashboard someone opens
and one they do not.

**What is stored is the whole honest result, not the headline.** The deflated
Sharpe, the trial count, the capacity verdict and every warning travel with the
curve. A comparison view that showed Sharpe ratios side by side without them
would be the single most misleading screen this platform could produce -- three
strategies ranked by a number that none of them earned.

Results are keyed by run name and overwritten, because a re-run of the same
configuration supersedes its predecessor. The trial registry, which must never
forget, is a different file with different rules.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quantlab.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.reporting.tearsheet import Tearsheet

__all__ = ["ResultsStore", "SavedRun"]

log = get_logger("quantlab.reporting.results_store")

#: Points kept in a stored curve. A chart cannot resolve more, and a dashboard
#: that ships four thousand points per strategy is a dashboard that stalls.
CURVE_POINTS = 600


class SavedRun(dict[str, Any]):
    """One stored result. A plain mapping, so it serialises without ceremony."""

    @property
    def name(self) -> str:
        return str(self.get("name", ""))

    @property
    def deflated(self) -> float:
        return float(self.get("headline", {}).get("deflated_sharpe", 0.0))

    @property
    def is_disqualified(self) -> bool:
        return bool(self.get("is_disqualified", True))


class ResultsStore:
    """Backtest results on disk, one JSON file per run name."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name)
        return self.directory / f"{safe}.json"

    # ------------------------------------------------------------------ write --
    def save(self, sheet: Tearsheet, *, config_path: str | None = None) -> Path:
        """Store a rendered tearsheet's data, with a downsampled equity curve."""
        payload = sheet.as_dict()
        curve = sheet.result.curve.select("as_of", "equity", "drawdown")
        step = max(1, curve.height // CURVE_POINTS)

        payload["curve"] = [
            {
                "t": str(row["as_of"])[:10],
                "equity": float(row["equity"]),
                "drawdown": float(row["drawdown"]),
            }
            for row in curve[::step].iter_rows(named=True)
        ]
        payload["config_path"] = config_path
        payload["saved_at"] = dt.datetime.now(tz=dt.UTC).isoformat()
        payload["costs"] = sheet.result.cost_decomposition_bps_annual()

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(sheet.name)
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        log.info(
            "reporting.result_saved",
            name=sheet.name,
            path=str(path),
            points=len(payload["curve"]),
        )
        return path

    # ------------------------------------------------------------------- read --
    def load(self, name: str) -> SavedRun | None:
        path = self._path(name)
        if not path.exists():
            return None
        return SavedRun(json.loads(path.read_text(encoding="utf-8")))

    def all(self) -> list[SavedRun]:
        """Every stored run, worst-verdict first.

        Ordered so a disqualified strategy cannot sit quietly below the fold
        while a reader compares the ones above it.
        """
        if not self.directory.exists():
            return []
        runs: list[SavedRun] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                runs.append(SavedRun(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError) as exc:  # pragma: no cover - corrupt file
                log.warning("reporting.result_unreadable", path=str(path), error=str(exc))
        return sorted(runs, key=lambda r: (not r.is_disqualified, r.deflated), reverse=False)

    def summary(self) -> list[dict[str, Any]]:
        """Comparison rows: enough to rank strategies, never the Sharpe alone."""
        rows = []
        for run in self.all():
            head = run.get("headline", {})
            stats = run.get("stats", {})
            rows.append(
                {
                    "name": run.name,
                    "deflated_sharpe": head.get("deflated_sharpe"),
                    "sharpe_after_haircuts": head.get("sharpe_after_haircuts"),
                    "sharpe_undeflated_net": head.get("sharpe_undeflated_net"),
                    "trials": head.get("trials"),
                    "family": head.get("family"),
                    "cagr": head.get("cagr"),
                    "max_drawdown": head.get("max_drawdown"),
                    "volatility": stats.get("volatility_annual"),
                    "turnover": stats.get("turnover_annual"),
                    "cost_drag_bps": stats.get("cost_drag_bps_annual"),
                    "years": stats.get("years"),
                    "break_even_aum": head.get("break_even_aum"),
                    "is_disqualified": run.is_disqualified,
                    "warnings": len(run.get("warnings", [])),
                    "saved_at": run.get("saved_at"),
                }
            )
        return rows
