# QuantLab

A multi-asset quantitative research and backtesting platform built entirely on free
data sources.

**This system exists to tell you the truth about whether a signal works, not to
produce impressive-looking equity curves.** A backtest that is easy to write but
silently leaks future information is worse than no backtest at all, because it
destroys capital with confidence.

That principle is enforced structurally, not by discipline:

| Concern | How it is enforced |
| --- | --- |
| Look-ahead bias | Signals read data only through a point-in-time snapshot that raises on any row learned after the as-of date |
| Survivorship bias | Universes are built from historical constituent lists; every symbol observed is retained forever |
| Transaction costs | Square-root market impact is the default; flat basis-point costs are not available as a default anywhere |
| Overfitting | Deflated Sharpe with an **automatic** trial counter, probability of backtest overfitting, and combinatorial purged cross-validation |
| Capacity | Every strategy result carries a break-even AUM; a strategy without one is not finished |
| Data quality | Every source's known biases are structured data, rendered into every tearsheet |

Research and paper trading only. There is no live order routing, by design.

## Quick start

Docker is the only prerequisite.

```bash
git clone <this repo> && cd quantlab
cp .env.example .env     # optional: the stack runs unchanged, with zero API keys
make up                  # builds images, starts the stack, waits for health, runs doctor
```

Then:

- dashboard — http://127.0.0.1:8080
- Jupyter — http://127.0.0.1:8888

```bash
make doctor        # what this installation can and cannot do
make catalogue     # every data source, its availability, its known biases
make test
make down
```

`make help` lists every target.

## Running with no API keys

The stack starts and runs with none. Keyless sources cover US Treasury and ECB
rates, Frankfurter FX, Stooq and Yahoo prices, **SEC EDGAR point-in-time
fundamentals**, the Ken French and Open Source Asset Pricing factor libraries, CFTC
positioning, CBOE volatility and every crypto venue — enough for the complete Tier 1
signal set.

Adding a free FRED key unlocks macro and rates work, including the ALFRED vintage
archive, which is the only way to build a macro signal without look-ahead bias.
`quantlab doctor` names every unavailable source and the environment variable that
would enable it.

## Layout

```
src/quantlab/
  data/        sources, ingest, parquet lake + DuckDB, trading calendars, point-in-time
  universe/    investable universe construction
  signals/     signal library, by asset class, tiered by evidence quality
  costs/       square-root impact, spread, financing, borrow, capacity
  portfolio/   sizing, vol targeting, turnover-aware optimisation
  backtest/    vectorised research engine, event-driven engine, paper trading
  validation/  CPCV, deflated Sharpe, PBO, empirical-Bayes luck adjustment
  risk/        factor risk model and exposure attribution
  reporting/   tearsheets and dashboard data
```

Layers depend downward only. Notebooks are for exploring; anything that produces a
number you might act on lives in `src/` with a test.

## Documentation

- [docs/DATA_CATALOGUE.md](docs/DATA_CATALOGUE.md) — every source and every caveat
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) — every conservative choice made under ambiguity
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what this platform cannot tell you
- [docs/MILESTONES.md](docs/MILESTONES.md) — build order and current status

## Status

Milestone 1 of 12 complete: repository skeleton, Docker stack, Makefile, CI, data
source catalogue. Commands whose implementation lands in a later milestone exit
non-zero naming that milestone, rather than returning an empty result that could be
mistaken for a successful run.
