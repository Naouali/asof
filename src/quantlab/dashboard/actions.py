"""Running the platform from the browser.

The read-only page reports what the installation *is*. This module lets it run
things: compute a signal, sweep it, back one up with a full backtest.

**A UI changes the statistics, and the platform has to account for that.** The
cost of trying one more idea drops from typing a command to clicking a button,
so many more ideas get tried — and every one of them is a trial that the next
Sharpe ratio has to be deflated against. Nothing here bypasses the trial
registry, every backtest records itself exactly as the CLI's does, and the page
shows the running trial count beside the results rather than at the bottom of a
different screen. A search that is easy to run is a search that needs its count
displayed more prominently, not less.

**Nothing here routes an order.** Spec section 1: research and paper trading
only. There is no endpoint that could.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from quantlab.config import get_settings
from quantlab.logging import get_logger

__all__ = ["run_backtest", "run_signal_scores", "run_sweep", "trial_summary"]

log = get_logger("quantlab.dashboard.actions")

#: A browser request that runs longer than this is a bad experience and usually
#: a mistake in the request rather than honest work. Sweeps over the default
#: universe finish well inside it.
MAX_CELLS = 40


def _store() -> Any:
    from quantlab.data.store import Store

    return Store(get_settings().layout)


def available_signals() -> list[dict[str, Any]]:
    """Every registered signal, with what a caller needs to choose one."""
    from quantlab.signals import SIGNAL_REGISTRY, load_all_signals

    load_all_signals()
    out = []
    for name, cls in sorted(SIGNAL_REGISTRY.items()):
        spec = cls.spec
        out.append(
            {
                "name": name,
                "tier": spec.tier,
                "asset_class": spec.asset_class.value,
                "evidence": spec.evidence.value,
                "output": spec.output.value,
                "turnover": spec.expected_turnover_annual,
                "datasets": list(spec.required_datasets),
                "fails_when": spec.known_failure_modes,
                "reference": spec.reference,
            }
        )
    return out


def run_signal_scores(name: str, as_of: str, symbols: list[str] | None) -> dict[str, Any]:
    """Today's scores for one signal, through a point-in-time snapshot."""
    from quantlab.signals import get_signal
    from quantlab.signals.base import SignalUnavailableError

    try:
        signal_class = get_signal(name)
    except KeyError as exc:
        # A browser must not get a traceback for a typo. The registry's message
        # already lists what does exist, which is the useful half.
        return {"ok": False, "error": str(exc).strip('"')}

    store = _store()
    universe = symbols or sorted(store.symbols(signal_class.spec.required_datasets[0]))
    if not universe:
        return {"ok": False, "error": "no instruments in the lake for this signal's datasets"}

    try:
        scores = signal_class().compute(store.as_of(as_of), universe)
    except SignalUnavailableError as exc:
        # Not an error page: a signal refusing to run is a designed outcome and
        # the reason is the useful part.
        return {"ok": False, "unavailable": True, "error": str(exc)}

    pairs = [
        (str(symbol), float(value))
        for symbol, value in zip(scores["symbol"], scores["score"], strict=True)
    ]
    rows = [
        {"symbol": symbol, "score": value}
        for symbol, value in sorted(pairs, key=lambda pair: -abs(pair[1]))
    ]
    return {
        "ok": True,
        "signal": name,
        "as_of": as_of,
        "output": signal_class.spec.output.value,
        "scores": rows,
        "note": "Scores, never weights. Portfolio construction owns sizing.",
    }


def run_sweep(
    signal: str | None,
    as_of: str,
    symbols: list[str] | None,
    parameters: dict[str, list[Any]] | None,
    lookback: int = 252,
) -> dict[str, Any]:
    """Sweep across instruments, or across a signal's own parameters."""
    from quantlab.validation.sweep import (
        cell_returns_from_prices,
        parameter_grid,
        sweep_returns,
        sweep_signal_parameters,
    )

    # The size of the search is a property of the request, not of the lake, so
    # it is checked before anything is loaded. Otherwise an empty lake reports a
    # data problem for what is really an oversized query.
    grid = parameter_grid(**(parameters or {})) if (signal and parameters) else [{}]
    if len(grid) > MAX_CELLS:
        return {
            "ok": False,
            "error": (
                f"{len(grid)} parameter sets is more than this page will run "
                f"({MAX_CELLS}). That is not only about waiting: {len(grid)} "
                "cells is a search of that size, and the deflated Sharpe of "
                "the best one would be judged against all of them. Use the "
                "CLI if you mean it."
            ),
        }

    store = _store()
    universe = symbols or sorted(store.symbols("ohlcv_daily"))
    if len(universe) < 2:
        return {"ok": False, "error": "a sweep needs at least two instruments"}

    try:
        if signal:
            result = sweep_signal_parameters(
                signal, grid, store=store, symbols=universe, as_of=as_of
            )
        else:
            bars = store.as_of(as_of).ohlcv_daily(symbols=universe)
            if bars.height == 0:
                return {"ok": False, "error": f"no bars for those instruments at {as_of}"}
            wide = (
                bars.sort("as_of")
                .pivot(index="as_of", on="symbol", values="adj_close")
                .drop_nulls()
            )
            prices = {c: wide[c].to_numpy() for c in wide.columns if c != "as_of"}
            result = sweep_returns(
                cell_returns_from_prices(prices, lookback=lookback),
                signal=f"trend({lookback})",
                dimension="instrument",
            )
    except (KeyError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "signal": result.signal,
        "dimension": result.dimension,
        "trials": result.trials,
        "variance": result.adjustment.variance,
        "shrinkage": result.adjustment.shrinkage,
        "verdict": result.verdict(),
        "survivors": [c.name for c in result.survivors],
        "cells": [
            {
                "name": c.name,
                "sharpe": c.sharpe,
                "t": c.t_statistic,
                "shrunk_t": c.shrunk_t,
                "survives": c.survives,
                "thin": c.thin,
                "years": c.years,
            }
            for c in sorted(result.cells, key=lambda c: -abs(c.t_statistic))
        ],
    }


def run_backtest(config_path: str) -> dict[str, Any]:
    """Run a backtest config and return its tearsheet, refusals included."""
    from pathlib import Path

    from quantlab.reporting import Tearsheet, TearsheetError, caveats_for, sources_behind
    from quantlab.reporting.capacity import capacity_for

    path = Path(config_path)
    if not path.exists():
        return {"ok": False, "error": f"no run config at {path}"}

    try:
        from quantlab.cli import _load_and_run

        run, panel, result = _load_and_run(path)
        capacity = capacity_for(result, panel)
        keys = [run.source] if run.source else list(sources_behind([run.dataset]))
        sheet = Tearsheet.build(result, capacity=capacity, caveats=caveats_for(keys))
    except (TearsheetError, ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}
    except SystemExit as exc:  # typer.Exit from the shared loader
        return {"ok": False, "error": f"the run could not start ({exc})"}

    data = sheet.as_dict()
    curve = result.curve.select("as_of", "equity")
    step = max(1, curve.height // 400)
    data["curve"] = [
        {"t": str(row["as_of"])[:10], "equity": float(row["equity"])}
        for row in curve[::step].iter_rows(named=True)
    ]
    return {"ok": True, **data}


def saved_strategies() -> dict[str, Any]:
    """Every stored backtest, for the comparison view.

    Returns the honest columns alongside the flattering one. A screen that
    ranked three strategies by Sharpe without the trial count and the
    disqualification beside it would be the most misleading thing this platform
    could render.
    """
    from quantlab.reporting.results_store import ResultsStore

    store = ResultsStore(get_settings().layout.runs / "results")
    return {"strategies": store.summary()}


def saved_strategy(name: str) -> dict[str, Any]:
    """One stored backtest in full: curve, panels, warnings."""
    from quantlab.reporting.results_store import ResultsStore

    run = ResultsStore(get_settings().layout.runs / "results").load(name)
    if run is None:
        return {"ok": False, "error": f"no saved result for {name!r}"}
    return {"ok": True, **run}


def lake_summary() -> dict[str, Any]:
    """What is actually in the lake, per dataset."""
    store = _store()
    try:
        stats = store.stats()
    except (OSError, ValueError) as exc:  # pragma: no cover - empty lake
        log.warning("dashboard.lake_unreadable", error=str(exc))
        return {"datasets": []}
    return {
        "datasets": [
            {
                "dataset": row.dataset,
                "source": row.source,
                "asset_class": row.asset_class,
                "rows": row.rows,
                "symbols": row.symbols,
                "megabytes": round(row.bytes / 1e6, 1),
                "earliest": str(row.first_as_of)[:10],
                "latest": str(row.last_as_of)[:10],
            }
            for row in stats
        ]
    }


def trial_summary() -> dict[str, Any]:
    """The running trial count, which a browser makes it much easier to inflate."""
    from quantlab.validation.registry import TrialRegistry

    try:
        sets = TrialRegistry(get_settings().layout.state).summary()
    except (OSError, ValueError) as exc:  # pragma: no cover - unreadable registry
        log.warning("dashboard.trials_unreadable", error=str(exc))
        return {"families": [], "total": 0}

    rows: list[dict[str, Any]] = []
    total = 0
    for group in sets:
        if not group.count:
            continue
        total += group.count
        rows.append(
            {
                "family": group.family,
                "trials": group.count,
                "best_sharpe": group.best.sharpe_annual if group.best else 0.0,
            }
        )
    return {"families": rows, "total": total}


def lake_universe() -> list[str]:
    try:
        return sorted(_store().symbols("ohlcv_daily"))
    except (OSError, ValueError):  # pragma: no cover - empty lake
        return []


def today() -> str:
    return dt.datetime.now(tz=dt.UTC).date().isoformat()
