"""Run configs: parsed strictly, because a run is meant to be reproducible."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from quantlab.backtest.config import load_run_config
from quantlab.backtest.conventions import ExecutionTiming

MINIMAL = """
name: test-run
as_of: "2024-06-28"
symbols: [AAPL, MSFT]
weights: w.parquet
"""


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(body)
    return path


def test_shipped_example_parses(repo_root: Path) -> None:
    config = load_run_config(repo_root / "configs" / "backtest_example.yaml")
    assert config.name == "etf-momentum-12-1"
    assert config.engine.execution is ExecutionTiming.NEXT_OPEN
    assert len(config.symbols) == 14


def test_minimal_config_gets_sensible_defaults(tmp_path: Path) -> None:
    config = load_run_config(write(tmp_path, MINIMAL))
    assert config.as_of == dt.date(2024, 6, 28)
    assert config.dataset == "ohlcv_daily"
    assert config.engine.execution is ExecutionTiming.NEXT_OPEN
    assert config.engine.staleness.max_bars == 5
    assert config.liquidity_window == 63


def test_weights_path_resolves_relative_to_the_config(tmp_path: Path) -> None:
    """So a config and its weights move together, and a run started from another
    directory reads the same file."""
    config = load_run_config(write(tmp_path, MINIMAL))
    assert config.weights_path == (tmp_path / "w.parquet").resolve()


@pytest.mark.parametrize("key", ["name", "as_of", "symbols", "weights"])
def test_missing_required_keys_are_named(tmp_path: Path, key: str) -> None:
    body = "\n".join(line for line in MINIMAL.strip().splitlines() if not line.startswith(key))
    with pytest.raises(ValueError, match=f"missing required key `{key}`"):
        load_run_config(write(tmp_path, body))


def test_a_non_iso_date_is_rejected(tmp_path: Path) -> None:
    body = MINIMAL.replace('"2024-06-28"', "last tuesday")
    with pytest.raises(ValueError, match="not an ISO date"):
        load_run_config(write(tmp_path, body))


def test_engine_and_impact_settings_come_through(tmp_path: Path) -> None:
    body = (
        MINIMAL
        + """
spread_bps: 4.5
liquidity_window: 40
engine:
  initial_equity: 5000000
  execution: next_close
  max_staleness_bars: 2
  borrow_bps_annual: 240
impact:
  y: 0.8
  delta: 0.6
"""
    )
    config = load_run_config(write(tmp_path, body))
    assert config.engine.initial_equity == 5_000_000
    assert config.engine.execution is ExecutionTiming.NEXT_CLOSE
    assert config.engine.staleness.max_bars == 2
    assert config.engine.borrow_bps_annual == 240
    assert config.impact.y == 0.8
    assert config.spread_bps == 4.5
    assert config.liquidity_window == 40
    assert config.cost_model().impact.params.delta == 0.6


def test_look_ahead_still_needs_acknowledging_from_a_config(tmp_path: Path) -> None:
    """The guard lives on BacktestConfig, so it cannot be bypassed by putting the
    setting in a file instead of in code."""
    body = MINIMAL + "\nengine:\n  execution: same_close\n"
    with pytest.raises(ValueError, match="look-ahead bias"):
        load_run_config(write(tmp_path, body))


def test_an_unknown_execution_timing_is_rejected(tmp_path: Path) -> None:
    body = MINIMAL + "\nengine:\n  execution: whenever\n"
    with pytest.raises(ValueError):
        load_run_config(write(tmp_path, body))
