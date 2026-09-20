# Limitations

What this platform cannot tell you. Read this before you act on any number it
produces.

This document is deliberately blunt. A research platform that oversells itself is
the mechanism by which backtests destroy capital.

## 1. The data is free, and you get what you pay for

### Stooq is blocked, so Yahoo's adjustments are unverified

As of 2026-09-20 Stooq serves a JavaScript proof-of-work anti-bot interstitial
instead of CSV. QuantLab does not solve anti-bot challenges, so that source fails
loudly and is disabled in the shipped ingest plan.

The consequence is not merely "one fewer source". Stooq was the **independent
cross-check** on Yahoo's undocumented adjustment methodology. Without it, every
equity price in this system comes from one unofficial API whose adjustments nobody
outside Yahoo can verify, and a systematic error in those adjustments would be
invisible to this platform.

### Yahoo restates prices for splits that have not happened yet

A May 2014 Apple close comes back as $21.12. The real figure was $591.48. Yahoo has
divided it by the 7:1 split that June **and** by the 4:1 split six years later.

Returns are unaffected, which is why most signals are safe. But every price-*level*
signal is wrong and nothing raises: penny-stock filters, nominal price momentum,
round-number effects, and any dollar-volume threshold computed from price times
volume. `adj_close` is worse — it is recomputed whenever Yahoo reprocesses a
corporate action, so a backtest re-run months later can produce different numbers
from identical code.

### Yahoo sometimes serves only 20 years of history, silently

Observed repeatedly on 2026-09-20: the identical SPY request returned 5462 bars
from 2005-01-03 on one call and 5030 bars from 2006-09-20 on another minutes later.
No error either time.

It cannot be detected from the response. When Yahoo truncates, it also reports
`firstTradeDate` as the start of the truncated range, so the payload is internally
consistent and every in-payload check passes. Requesting history in eight-year
windows helps sometimes and not always.

Two things limit the damage, and neither eliminates it:

- The lake is **append-only**, so a later truncated fetch never destroys history
  you already have, and ingest warns (`ingest.coverage_regression`) when a fetch
  returns less than the lake holds.
- On a **first** ingest there is nothing to compare against, so the loss is
  invisible.

**Practical advice: after your first ingest, check `quantlab data status` against
the span you expect, and re-run until it matches.** A backtest over 2006-2026
instead of 2005-2026 excludes the financial crisis, which is not a detail.

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

Four specific limits worth knowing:

- **Nothing here is calibrated to your fills.** Y, δ and the permanent fraction
  are literature defaults. The right way to use this module is to replace them
  with numbers fitted to your own executions; until then every cost is an
  informed guess.
- **Beyond ~10% of ADV the law is extrapolation**, and it understates there. Any
  capacity number that requires trading more than that is labelled an upper bound,
  and should be read as "certainly no more than this", not "about this".
- **Low-frequency spread estimators fail outright on liquid, volatile
  instruments.** This is worse than the upward bias the literature describes, and
  it is measured, not theorised. On 993 days of real Binance daily bars
  (2026-09-20), against a true quoted BTCUSDT spread of **0.001 bp**:

  | Estimator | Estimate | Error |
  | --- | --- | --- |
  | Roll | 167 bp | ×167,000 |
  | Corwin-Schultz | 106 bp | ×106,000 |
  | Abdi-Ranaldo | 62 bp | ×62,000 |

  Abdi-Ranaldo was *undefined* for seven of the eight majors tested. These
  estimators are not biased here — they are measuring daily volatility instead of
  spread. The statistical reason is that a daily-bar estimator cannot resolve a
  spread below roughly `σ/√n`, which for BTC over 993 days is about 8 bp, while
  the true spread is four orders of magnitude smaller. Every estimate now reports
  that resolution floor.

  **Never estimate a spread you can observe.** Crypto books are free; the platform
  warns when a crypto instrument is costed from an estimate. For equities and
  futures, where free quotes do not exist, estimated spreads are used, are marked
  `spread_source="estimated"`, and are not trustworthy in absolute terms — treat
  them as an upper bound on a liquid name and read the capacity number
  accordingly. On simulated data with a 20 bp spread (where the spread is a
  meaningful fraction of daily variation) the same estimators came out +379%,
  +250% and −6% respectively, which is the regime the literature describes and the
  reason Abdi-Ranaldo is the default.
- **Short borrow is assumed, not known.** 15 bp/month by default. A long/short
  equity strategy's net alpha is roughly linear in this number, so it is the first
  thing to replace with real data.

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

## 7. Capacity is a ceiling, not a plan

The break-even AUM assumes the impact model is right, the ADV forecast is right,
and that trading is spread across the universe in proportion to weights. All three
degrade in the direction of *less* capacity:

- Equal-weighted homogeneous universes overstate capacity, because a real
  universe's thin names cost disproportionately more. On a 50/50 book of a $4.9bn
  and a $100m name, costing name-by-name gives 2.6× the cost that the combined ADV
  would suggest.
- ADV is measured over a trailing window and is itself a forecast. It collapses
  precisely when you most want to trade.
- Crowding is not modelled at all. If others hold the same position, your exit is
  correlated with theirs and impact is worse than any single-trader model says.

## 8. The point-in-time guarantee has a hard edge

`Snapshot` makes look-ahead bias structurally unavailable **for data that is in the
lake**. It cannot help with the two harder cases:

- **Sources that revise without publishing vintages** (EIA, USDA, CoinGecko market
  caps). For these, `known_at` is our own download time, so history before the day
  you started collecting is restated and the platform says so rather than
  pretending otherwise.
- **Your own knowledge.** The snapshot does not know which symbols you chose to
  ingest, and you chose them knowing which ones did well. A universe of fourteen
  ETFs that still exist in 2026 is a survivorship-biased universe no matter how
  correct the timestamps are.

## 9. What "reproducible" means here

A given commit plus a given *data snapshot* produces identical results. It does not
mean re-running ingest reproduces the snapshot: Yahoo restates adjusted closes,
EDGAR receives amended filings, and exchanges delist symbols. **Snapshot before you
research, and keep the snapshot.**
