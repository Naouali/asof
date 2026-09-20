# Limitations

What this platform cannot tell you. Read this before you act on any number it
produces.

This document is deliberately blunt. A research platform that oversells itself is
the mechanism by which backtests destroy capital.

## 1. The data is free, and you get what you pay for

### Equity survivorship bias cannot be fully solved

Without paid CRSP, the universe of US equities that free sources will serve is
biased toward companies that still exist. Delisted names — the ones that went to
zero — are partly or entirely missing.

Mitigations in place: universes are built from historical index constituent and SEC
filer lists rather than from tickers that exist today; every symbol observed is
retained forever; academic factor datasets are preferred where the bias would
dominate; equity tearsheets carry an explicit residual-bias warning.

**Residual bias remains, and it inflates long-leg returns and deflates short-leg
losses.** Treat cross-sectional equity results from self-built universes as an
upper bound.

### Restated fundamentals and macro data

SEC EDGAR gives genuine as-filed fundamentals — this is the platform's strongest
free dataset. Almost everything else does not. FRED serves the latest vintage of
every series; EIA and USDA revise without publishing an archive. For those, the
platform uses its own `ingested_at` as the vintage, which only works from the day
collection starts. **History before the first ingest is restated**, and any signal
built on it is optimistic by an unknown amount.

### No free historical options data

There is none. The snapshot collector accumulates chains from the day it first
runs. **No options backtest before that date is possible.** Any result claiming one
is fabricated.

### Futures term structure is barely available

Carry and basis-momentum need at least two points on the curve. Free continuous
front-month series use an opaque roll rule and carry no second contract. The only
free route to a real curve is exchange settlement bulletins, whose history starts
when you start collecting and whose terms of service may forbid automated access.

### FX has no intraday path and no spreads

ECB reference rates are a single daily fix, not a tradable quote. FX research is
restricted to daily-or-slower signals, and FX transaction costs are *assumed*
rather than measured.

## 2. Costs are modelled, not observed

The square-root impact law is one of the most robust regularities in market
microstructure, but its calibration constant is fitted to institutional flow that
this platform cannot see. The defaults are deliberately punitive. **A strategy that
survives them may still be unprofitable; a strategy that does not survive them is
almost certainly unprofitable.** The second inference is much stronger than the
first.

Short borrow fees, where data is unavailable, are assumed punitive — enough to
erase most published anomaly alphas. That is a feature.

## 3. Validation bounds overfitting, it does not eliminate it

Deflated Sharpe, PBO and combinatorial purged cross-validation correct for trials
the platform can *count*. They cannot correct for:

- Trials you ran elsewhere, or in your head, before writing the config
- The decades of published research that shaped which signals you consider at all
- Your choice of universe, sample period and rebalance convention

The automatic trial counter addresses the first of these only within this
installation. **The literature's selection bias is not correctable by any statistic.**

A useful working prior: **live Sharpe ≈ 0.5 × backtest Sharpe**, and the platform
displays the haircuts that get you there.

## 4. The only honest out-of-sample test is forward time

Everything else — walk-forward, purged CV, held-out samples — reuses data you have
already seen. The paper-trading loop is the single most valuable component in this
system, and it is the slowest to give you an answer. There is no shortcut.

## 5. Execution realism has a ceiling

The event-driven engine models order types, partial fills, slippage, latency and
rejects, but it is calibrated on free data. For crypto, where full order books are
free, this is genuinely good. For equities, futures and FX, it is an approximation
whose error you cannot measure without paid tick data.

## 6. No live trading

There is no order routing, by design. The gap between a paper-trading loop and live
execution — queue position, partial fills at your actual broker, borrow
availability on the day, operational failure — is large and is not simulated here.

## 7. What "reproducible" means here

A given commit plus a given *data snapshot* produces identical results. It does not
mean re-running ingest reproduces the snapshot: Yahoo restates adjusted closes,
EDGAR receives amended filings, and exchanges delist symbols. **Snapshot before you
research, and keep the snapshot.**
