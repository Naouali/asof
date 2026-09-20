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
make ingest        # pull data into the lake (works with zero API keys)
make status        # what the lake holds and how stale it is
make test
make down
```

## Querying the lake

```bash
# Inspection: sees everything, including data nobody could have known at the time.
quantlab data query "select symbol, count(*) from ohlcv_daily group by 1"

# Research: only what was knowable on 2020-03-16, enforced in a sandbox that
# cannot reach the lake files at all.
quantlab data query --as-of 2020-03-16 -d ohlcv_daily \
  "select symbol, max(as_of) from ohlcv_daily group by 1"
```

In Python:

```python
from quantlab.data.store import Store

snapshot = Store().as_of("2020-03-16")
bars = snapshot.ohlcv_daily(symbols=["SPY"])     # nothing after 2020-03-16
snapshot.ohlcv_daily(end="2020-06-01")           # raises LookAheadError
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

## Costing a trade

```bash
# Decomposed, with what a flat model would have claimed
quantlab costs estimate -n 200e6 --adv 8e9 --vol 0.018 --spread 1.2 --compare-flat 10

# The number every finished strategy needs
quantlab costs capacity --alpha 25 --turnover 0.4 --rebalances 12 --names 200
```

There is deliberately **no flat basis-point cost model available as a default**.
Flat costs are roughly right for the small trades a researcher tests on and wildly
optimistic at deployment size, which makes every strategy look scalable and every
capacity estimate infinite.

## Running a backtest

```bash
quantlab backtest --config configs/backtest_example.yaml
```

The config's `as_of` fixes the point-in-time snapshot the run reads, so re-running
months later sees the same data — and the same vintage of any restated series —
rather than whatever the lake has learned since.

What the engine enforces rather than assumes: signal on the close of T traded at
the open of T+1; forward-fill capped at a configured limit; a delisting return
applied when an instrument vanishes; calendar-day financing accrual; and
cash-and-position accounting reconciled every bar by two independent routes, with
a breach stopping the run.

A 30-year, 3,000-name backtest with full costs runs in about 7 seconds.

## The signal library

```bash
quantlab signals list                    # tiered by evidence, not by interest
quantlab signals show carry.fx           # reference, and how it is known to fail
quantlab signals run trend.time_series_momentum --as-of 2026-09-18
```

Every signal declares its datasets, cadence, expected turnover, academic reference
and **how it is known to fail** — the last is validated, on the grounds that if you
cannot name how a signal goes wrong you do not understand it well enough to trade
it. Signals return scores, never weights; portfolio construction owns those.

Signals whose data the free catalogue cannot supply are implemented and **refuse to
run**, naming what is missing. An empty cross-section looks exactly like a signal
with no view.

## Building a portfolio

Signals produce scores; this turns them into a book, and reports the risk each
position actually carries rather than only its weight.

```bash
quantlab portfolio covariance -s SPY -s QQQ -s TLT -s GLD --as-of 2026-09-18
quantlab portfolio build -s SPY -s QQQ -s TLT -s GLD --as-of 2026-09-18 -m risk-parity
```

An equal-weighted book of correlated assets routinely puts most of its risk in one
place while looking perfectly diversified on the weights, so `build` prints both.
Covariance is Ledoit-Wolf shrunk by default, with the intensity derived rather than
chosen. `--method mean-variance` uses a **flat prior of zero** for expected returns,
never the sample mean: sample means are noisy enough that optimising on them
reliably produces a worse portfolio than equal weighting.

Constraints that cannot be satisfied raise instead of returning something
plausible. `Constraints.long_only()` caps a position at 10%, so on eight names the
largest possible book is 80% of capital against a net target of 100% — SLSQP
answers that with a weight vector that looks fine and violates the constraints.

## Is it secretly just beta?

```bash
quantlab risk attribute -s IWM --as-of 2026-09-18     # against Ken French factors
quantlab risk pca -s SPY -s TLT -s GLD --as-of 2026-09-18   # when none are available
```

Standard errors are Newey-West. Strategy returns are autocorrelated, and OLS
standard errors on autocorrelated data come out too small — inflating every
t-statistic toward finding alpha that is not there.

Validated against instruments whose answer is known: SPY attributes to a market
beta of 0.993 with alpha of +0.02% a year (t = 0.04); IWM loads +0.93 on the
small-cap factor because it *is* the small-cap index; TLT's market beta is −0.002.
Getting SPY's alpha to zero took fixing two real bugs — see
[docs/MILESTONES.md](docs/MILESTONES.md).

## Reporting a result

```bash
quantlab report --config configs/backtest_example.yaml           # to the terminal
quantlab report -c configs/backtest_example.yaml -f html -o s.html
```

The tearsheet **refuses to render** a Sharpe ratio without its deflated value and
trial count, and refuses to render at all without a capacity estimate. Both are
constructor arguments with no default and no override flag, so the failure mode is
a report that will not build rather than one that quietly omits the inconvenient
parts. Panels are ordered worst-first: a strategy that fails deflation says so
above its equity curve. `report` exits non-zero when the result is disqualified,
because the exit code is the only part of a tearsheet a script reads.

The example ETF momentum run opens with:

```
NOT EVIDENCE that etf-momentum-12-1 works:
  - The deflated Sharpe of 0.470 is below 0.95: after 1 trial(s) this result is
    not distinguishable from the best of the search.
  - There is no AUM at which this strategy makes money after costs. Spread,
    commission and holding costs exceed the gross edge at any size; this is not
    a capacity problem, the signal does not pay.
```

That is a real result on real data, and printing it is the point.

## What the literature already tried

```bash
quantlab validate anomalies --t-stat 2.4
```

Chen & Zimmermann catalogued every published cross-sectional equity predictor they
could find, with the t-statistic the original paper reported. Of 212 predictors the
median is **4.0** and only **2.7%** fall below |t| = 2 — a distribution truncated
exactly where journals stop accepting papers. The signals that were tried and
abandoned are absent by construction, so it is a *lower bound* on how hard the
space has been searched.

That is the context a t-statistic needs before it means anything, and it is why
this platform will not print a Sharpe without a trial count.

## Validating a result

Every backtest records itself as a trial, and the count deflates the Sharpe it
reports. There is no way to search quietly and still quote a deflated number.

```bash
quantlab validate trials              # what the platform has counted
quantlab validate trials momentum     # the individual configurations
quantlab validate library             # empirical-Bayes shrinkage across families
quantlab validate sharpe 1.8 --years 10 --trials 300   # deflate a published number
```

A Sharpe of 1.5 over five years is worth 1.000 after one trial, 0.155 after two
hundred, and 0.003 after ten thousand. That is the entire point.

**No statistic here detects survivorship bias.** The red-team suite asserts it, so
that a clean validation report is never mistaken for evidence of a clean universe.

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

Milestones 1–5 of 12 complete: repository skeleton, Docker stack and CI; the data
layer — parquet lake, point-in-time snapshots, trading calendars and eight fetchers
across Yahoo, Binance and FRED/ALFRED; the cost models — square-root impact,
Almgren-Chriss scheduling, spread estimation with its bias problem documented,
financing and borrow, and the capacity calculator; the vectorised backtest engine
with its accounting identities enforced every bar; the validation module — purged
cross-validation, deflated Sharpe with an automatic trial counter, PBO,
empirical-Bayes shrinkage, and a red-team suite of deliberately broken strategies;
and the Tier 1 signal library with its Ken French benchmark.

Commands whose implementation lands in a later milestone exit non-zero naming that
milestone, rather than returning an empty result that could be mistaken for a
successful run. See [docs/MILESTONES.md](docs/MILESTONES.md) for what is built and
what is still unverified.
