#!/usr/bin/env python
"""Generate the golden-file fixture for the backtest regression test.

Spec section 11: a fixed strategy on a fixed data snapshot must produce identical
results across commits. This script builds that fixed snapshot -- deterministically,
from a seeded generator, so it needs no network and no lake -- runs the backtest,
and records the result.

**Regenerate deliberately, never to make a test pass.** A change in this fixture is
a change in what the engine computes. If it moves, the question is whether the new
number is more correct than the old one, and the answer belongs in the commit
message.

    python scripts/gen_backtest_golden.py           # write the fixture
    python scripts/gen_backtest_golden.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantlab.backtest import (  # noqa: E402
    BacktestConfig,
    BacktestResult,
    ExecutionTiming,
    Panel,
    StalenessPolicy,
    VectorisedBacktest,
)
from quantlab.costs.impact import ImpactParams, SquareRootImpact  # noqa: E402
from quantlab.costs.model import TransactionCostModel  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
CURVE_PATH = FIXTURES / "golden_backtest_curve.parquet"
STATS_PATH = FIXTURES / "golden_backtest_stats.json"

SEED = 20240101
N_BARS = 504  # two years of trading days
N_SYMBOLS = 25
START = dt.datetime(2020, 1, 2, tzinfo=dt.UTC)


def build_panel() -> Panel:
    """A deterministic panel with a gap and a delisting, so the fixture exercises
    the awkward paths and not only the happy one."""
    rng = np.random.default_rng(SEED)
    dates = [START + dt.timedelta(days=int(i * 365 / 252)) for i in range(N_BARS)]
    symbols = [f"SYM{i:02d}" for i in range(N_SYMBOLS)]

    drift = rng.normal(0.0003, 0.0002, N_SYMBOLS)
    volatility = rng.uniform(0.010, 0.035, N_SYMBOLS)
    shocks = rng.normal(0.0, 1.0, (N_BARS, N_SYMBOLS))
    closes = 100.0 * np.exp(np.cumsum(drift + volatility * shocks, axis=0))
    opens = closes * (1.0 + rng.normal(0.0, 0.001, (N_BARS, N_SYMBOLS)))
    adv = np.exp(rng.uniform(np.log(5e6), np.log(5e8), N_SYMBOLS))
    spreads = rng.uniform(2.0, 25.0, N_SYMBOLS)

    rows: list[dict[str, object]] = []
    for bar, date in enumerate(dates):
        for index, symbol in enumerate(symbols):
            # SYM07 goes dark for three bars; SYM13 delists two thirds of the way in.
            if index == 7 and 100 <= bar < 103:
                continue
            if index == 13 and bar >= 340:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "as_of": date,
                    "close": float(closes[bar, index]),
                    "open": float(opens[bar, index]),
                    "adv_notional": float(adv[index]),
                    "volatility_daily": float(volatility[index]),
                    "spread_bps": float(spreads[index]),
                }
            )
    return Panel.from_frame(
        pl.DataFrame(rows),
        timing=ExecutionTiming.NEXT_OPEN,
        staleness=StalenessPolicy(max_bars=5),
    )


def build_weights(panel: Panel) -> pl.DataFrame:
    """Monthly cross-sectional reversal, long/short, dollar neutral.

    Chosen because it trades enough to make costs matter and is completely
    specified by the panel -- no signal library required, so the fixture stays
    valid as Milestone 6 lands.
    """
    rows: list[dict[str, object]] = []
    lookback = 21
    for bar in range(lookback, panel.n_bars, 21):
        window = panel.close[bar] / panel.close[bar - lookback] - 1.0
        usable = np.isfinite(window) & panel.tradable[bar]
        if usable.sum() < 6:
            continue
        scores = np.where(usable, -window, np.nan)
        ranked = np.argsort(np.where(usable, scores, np.inf))
        longs, shorts = ranked[: int(usable.sum()) // 4], ranked[-(int(usable.sum()) // 4) :]
        for index in longs:
            rows.append(
                {
                    "symbol": panel.symbols[index],
                    "as_of": panel.dates[bar],
                    "weight": 0.5 / len(longs),
                }
            )
        for index in shorts:
            rows.append(
                {
                    "symbol": panel.symbols[index],
                    "as_of": panel.dates[bar],
                    "weight": -0.5 / len(shorts),
                }
            )
    return pl.DataFrame(rows)


def run() -> BacktestResult:
    panel = build_panel()
    weights = build_weights(panel)
    engine = VectorisedBacktest(
        BacktestConfig(
            initial_equity=10_000_000.0,
            execution=ExecutionTiming.NEXT_OPEN,
            staleness=StalenessPolicy(max_bars=5),
            delisting_return=-0.30,
            borrow_bps_annual=180.0,
            financing_bps_annual=550.0,
            commission_bps=0.5,
        ),
        TransactionCostModel(
            impact=SquareRootImpact(ImpactParams(y=0.5, delta=0.5, permanent_fraction=0.5)),
            use_asset_class_defaults=False,
        ),
    )
    # Not recorded as a trial: this fixture is re-run on every test invocation and
    # would otherwise inflate the operator's own trial count with a strategy they
    # never searched over.
    return engine.run(panel, weights, name="golden-reversal", record_trial=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the fixture is stale")
    args = parser.parse_args()

    result = run()
    stats = {k: round(v, 12) for k, v in result.stats.as_dict().items()}

    if args.check:
        if not CURVE_PATH.exists() or not STATS_PATH.exists():
            print("golden fixture is missing; run scripts/gen_backtest_golden.py")
            return 1
        stored = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if stored != stats:
            print("golden fixture is stale. The engine now computes different numbers.")
            for key, value in stats.items():
                if stored.get(key) != value:
                    print(f"  {key}: {stored.get(key)} -> {value}")
            return 1
        return 0

    FIXTURES.mkdir(parents=True, exist_ok=True)
    result.curve.write_parquet(CURVE_PATH, compression="zstd", statistics=True)
    STATS_PATH.write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {CURVE_PATH.relative_to(REPO)} ({result.curve.height} bars)")
    print(f"wrote {STATS_PATH.relative_to(REPO)}")
    print()
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
