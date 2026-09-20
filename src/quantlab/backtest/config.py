"""Strategy run configuration.

A run is defined by a YAML file so that it is a committed artefact rather than a
set of arguments someone remembers typing. Spec section 1 requires a given commit
plus a given data snapshot to reproduce a result exactly; that is only meaningful
if the run's parameters are part of the commit.

The ``as_of`` field is the one that matters most. It fixes the point-in-time
snapshot the backtest reads, so re-running the same config months later reads the
same data -- including the same *vintage* of any restated series -- rather than
whatever the lake has learned since.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quantlab.backtest.conventions import ExecutionTiming, StalenessPolicy
from quantlab.backtest.vectorised import BacktestConfig
from quantlab.costs.impact import ImpactParams, SquareRootImpact
from quantlab.costs.model import TransactionCostModel

__all__ = ["RunConfig", "load_run_config"]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One reproducible backtest run."""

    name: str
    #: Point-in-time snapshot date. Everything the run reads was knowable then.
    as_of: dt.date
    symbols: tuple[str, ...]
    start: dt.date
    #: Path to a weights file (parquet or csv) with symbol, as_of, weight.
    weights_path: Path
    dataset: str = "ohlcv_daily"
    source: str | None = None
    engine: BacktestConfig = field(default_factory=BacktestConfig)
    impact: ImpactParams = field(default_factory=ImpactParams)
    spread_bps: float | None = None
    estimate_spreads: bool = False
    liquidity_window: int = 63

    def cost_model(self) -> TransactionCostModel:
        return TransactionCostModel(
            impact=SquareRootImpact(self.impact), use_asset_class_defaults=False
        )


def load_run_config(path: Path) -> RunConfig:
    """Parse a run config, failing loudly on anything malformed."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    for required in ("name", "as_of", "symbols", "weights"):
        if required not in raw:
            raise ValueError(f"{path}: missing required key `{required}`")

    engine_raw = dict(raw.get("engine") or {})
    execution = ExecutionTiming(engine_raw.pop("execution", ExecutionTiming.NEXT_OPEN.value))
    staleness = StalenessPolicy(max_bars=int(engine_raw.pop("max_staleness_bars", 5)))
    engine = BacktestConfig(execution=execution, staleness=staleness, **engine_raw)

    impact_raw = dict(raw.get("impact") or {})
    weights_path = (path.parent / str(raw["weights"])).resolve()

    return RunConfig(
        name=str(raw["name"]),
        as_of=_as_date(raw["as_of"], f"{path}: as_of"),
        symbols=tuple(str(s) for s in raw["symbols"]),
        start=_as_date(raw.get("start", "1990-01-01"), f"{path}: start"),
        weights_path=weights_path,
        dataset=str(raw.get("dataset", "ohlcv_daily")),
        source=str(raw["source"]) if raw.get("source") else None,
        engine=engine,
        impact=ImpactParams(**impact_raw),
        spread_bps=float(raw["spread_bps"]) if "spread_bps" in raw else None,
        estimate_spreads=bool(raw.get("estimate_spreads", False)),
        liquidity_window=int(raw.get("liquidity_window", 63)),
    )


def _as_date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{where}: {value!r} is not an ISO date (YYYY-MM-DD)") from exc
