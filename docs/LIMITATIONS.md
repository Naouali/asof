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

### No statistic here detects survivorship bias

This is worth stating on its own, because a clean validation report invites exactly
the wrong inference. A strategy run on a universe assembled from instruments that
happened to survive produces a real return series from a real edge — the edge of
having excluded the failures in advance. There is nothing statistically anomalous
about it, so the probabilistic Sharpe, the deflation and PBO all **pass** it.

The red-team suite asserts this rather than hiding it
(`test_survivorship_bias_is_invisible_to_every_statistic_here`). Survivorship is
caught in the data layer — historical constituent lists, retained delisted tickers,
the delisting count on every backtest result — or it is not caught at all.

### Deleting the trial registry resets your trial count

It lives in a JSONL file in the state directory and is deliberately awkward to
reduce, but it is a file. The only person fooled by emptying it is you.

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

## 7. Most of the Tier 1 signal set cannot run on free data

Six Tier 1 signals are implemented. **Two of them can actually run.**

| Signal | Blocked by |
| --- | --- |
| `carry.rates` | Needs a free FRED key, which is a five-minute fix. |
| `equity.profitability` | Needs as-filed fundamentals — SEC EDGAR, Milestone 9. |
| `carry.commodity_basis` | Needs a second point on the futures curve, which free data does not supply at all. |
| `carry.fx` | Needs a policy rate per currency; the free catalogue covers two. |

Each refuses to run and says which. None returns an empty cross-section, because
an empty cross-section looks exactly like a signal with no view and a strategy
built on one trades nothing while appearing to work.

The consequence for the platform as a whole: **cross-asset diversification, which
is what makes trend following and carry work, is largely unavailable.** Trend on
fourteen correlated ETFs is one bet, and the 0.14 net Sharpe it produces reflects
that rather than reflecting the effect.

## 8. What the vectorised engine does not model

It is a *research* engine. It answers "was there an edge here, net of costs", and
it deliberately does not answer "could this have been executed":

- **Fills are assumed.** Every order fills completely, at the configured price,
  with cost charged as a deduction. There are no partial fills, no rejects, no
  queue position and no latency. The event-driven engine (Milestone 10) is what
  tests whether a vectorised result survives realistic execution, and a strategy
  going to paper trading must clear it first.
- **Splits are not handled.** The engine consumes an already-split-adjusted price
  series and requires dividends on the same basis. Mixing bases misstates every
  total return by the split factor, and nothing detects it.
- **Borrow availability is not modelled.** A short is assumed to be borrowable at
  the configured fee, always. Recalls and forced buy-ins are real costs that appear
  nowhere.
- **Cash earns nothing and there is no margin call.** A strategy that would have
  been liquidated by its broker runs to the end here.

## 9. Capacity is a ceiling, not a plan

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

## 10. The point-in-time guarantee has a hard edge

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

## 11. What "reproducible" means here

A given commit plus a given *data snapshot* produces identical results. It does not
mean re-running ingest reproduces the snapshot: Yahoo restates adjusted closes,
EDGAR receives amended filings, and exchanges delist symbols. **Snapshot before you
research, and keep the snapshot.**

## Portfolio construction and risk (Milestone 7)

**`adj_close` is restated, so the research commands are not point-in-time.**
`quantlab portfolio covariance`, `portfolio build`, `risk attribute` and `risk pca`
compute total returns from `adj_close`, which carries dividends. They have to:
price returns drop the dividend yield from every observation, and in an
attribution that missing yield comes back as alpha — SPY attributes to −1.45% a
year (t = −2.3) on price returns against +0.02% (t = 0.04) on total returns.

The cost is that the provider rewrites the entire `adj_close` history each time a
dividend is paid. A snapshot taken today therefore does **not** reproduce the
series as it stood a year ago, and these four commands are consequently
*ex-post* measurements of realised exposure over a fixed historical window, not
point-in-time research inputs. They must not be used to generate a trading signal.
The backtest engine does not use `adj_close`; it applies corporate actions on
their own dates against raw closes, and that path remains point-in-time.

**Attribution is limited to the factors that are free.** Ken French daily
Mkt-RF/SMB/HML/RF and momentum, and nothing else. There is no quality factor, no
betting-against-beta, no free daily international or sector factor set. A
strategy whose exposure is to something not on that list will attribute to
residual alpha by default — the residual is "what these five factors do not
explain", which is not the same thing as skill. The daily factor files also lag
the price data by roughly two months, so the most recent window is unattributable.

**Statistical factors are not exposures.** `risk pca` will always find structure;
that is what PCA does. Decomposing the correlation matrix keeps the first
component from collapsing onto the highest-variance asset, but it does not make
any component mean anything. Only the first is reliably interpretable, and only
as "the common factor". The rest have no names and naming them is how a risk
model becomes a story.

**No capacity estimate is attached to a portfolio.** `portfolio build` reports
weights, risk contributions and portfolio volatility. It does not cost the trades
needed to reach those weights from a current book, so it cannot tell you what the
portfolio is worth after impact or what size it survives to. Spec section 13
forbids reporting a *strategy result* without a capacity estimate; a portfolio
construction command is not a strategy result, but the gap is real and the
backtest path is where capacity is enforced.

**The controls are tested for mechanism, not calibrated.** The drawdown
thresholds (−10% soft, −25% hard) and stop threshold (−20%, 5-bar cooldown) are
round numbers, not fitted values, and deliberately so: a control fitted to the
drawdowns in a sample is fitted to the sample. Neither has been validated against
a live book, and neither should be expected to improve returns.

## Tearsheets (Milestone 8)

**A tearsheet that renders is not a tearsheet that passed.** The refusals
enforce two specific things — a trial count and a capacity estimate — and nothing
more. A result can render cleanly and still be worthless for reasons the module
cannot see: a universe assembled from today's survivors, a signal whose
construction encoded a fact nobody knew at the time, a cost model calibrated to
literature rather than to your fills. `is_disqualified` means *these* checks
failed, never that the remaining ones passed.

**The deflated Sharpe corrects for search intensity alone.** It counts
configurations tried inside this platform. It cannot count the ideas discarded
before anything was run, the datasets chosen because they looked promising, or
the twenty years of published research that already picked this signal out for
you. The trial count is a floor on how hard the space was searched, not an
estimate of it.

**Capacity inherits every cost assumption.** The break-even AUM is only as good
as the impact coefficients, the assumed spread, and the borrow rate — none of
which is calibrated to real fills, and the last of which is a 180 bp/yr default
that dominates the short leg of the example run. Treat the number as an order of
magnitude, and read the binding constraint rather than the AUM.

**Caveats are a superset when the source is unknown.** A run that did not record
which source it read gets the caveats of every source that could have supplied
its dataset. That errs toward showing a limitation that did not apply rather than
hiding one that did, but it means the list is not evidence about what was
actually used.

**No benchmark comparison.** A tearsheet reports the strategy and nothing else.
It does not tell you whether a passive position would have done the same thing
more cheaply — `quantlab risk attribute` answers that, and the two are not yet
joined up. Reading a tearsheet in isolation can make a levered beta position look
like a strategy.

**The chart is deliberately minimal.** One log-scale equity line and a drawdown
band, downsampled to 600 points. There is no rolling Sharpe, no monthly return
table, no underwater plot with dates. Those are useful and they are also where
a reader's attention goes instead of to the deflation statistics, which is why
the first version does not have them.

## Positioning and fundamentals (Milestone 9)

**COT release dates are derived, not observed.** No CFTC payload carries one, and
the CFTC publishes no machine-readable release calendar — only the prose rule and
a warning that "holidays can change the COT release schedule". The derivation
here is the rule plus federal holiday arithmetic, rolled forward. It will be
wrong for any ad-hoc delay the CFTC did not announce in a form this code can
read, and the 2018–19 government shutdown is the known case: report dates run
through it unbroken while publication was suspended for weeks. A COT signal
backtested across that winter sees those reports about six weeks before they
actually appeared. There is no free machine-readable record of the catch-up
schedule, so this is not corrected — it is disclosed.

**COT positioning is weekly, and that caps what it can support.** Fifty-two
observations a year is a small sample for any parameter you fit, and the
categories are reclassified from time to time, which creates level shifts that
look exactly like signal. Normalise within a regime, not across one.

**EDGAR covers US registrants only.** No foreign private issuer that files 20-F
without XBRL, no ETF, no company that has deregistered. A universe built from
what EDGAR returns today is a universe of current filers, so the survivorship
problem is displaced rather than solved: the filings of a delisted company remain
in EDGAR, but you have to know its CIK to ask, and the ticker mapping only lists
live registrants.

**XBRL coverage is biased by sector, not random.** Measured over five large
filers: revenue, total assets, book equity and interest expense present for all
five; cost of goods sold for three. JPMorgan has none because banks do not report
one. Dropping filers with missing line items therefore drops financials, and a
profitability screen built this way is implicitly a screen against banks.

**Tagging practice changed around 2012**, and small filers tag inconsistently
throughout. A long backtest on EDGAR fundamentals is running on a materially
different data-generating process before and after that boundary, and the early
period is both thinner and tilted toward large, well-reported companies.

**Only one XBRL tag per metric is read.** Where a filer reports a concept under a
tag not in the precedence list, that metric is simply absent for them rather than
approximated from a near-neighbour. That is deliberate — the alternative is
silently summing things that are not the same — but it means coverage is a
function of the tag list, which is short.

**The fundamentals universe is fifteen companies.** Each company's facts payload
is several megabytes, so the default is small enough to ingest politely. Nothing
here has been validated at the scale a real cross-sectional equity strategy needs.

## Volatility indices and the anomaly catalogue (Milestone 9)

**VIX is not investable, and nothing in the lake stops you forgetting that.** It
is stored as an index level rather than a bar, which prevents the backtest engine
treating it as tradeable, but a signal can still read it and a portfolio can still
be sized against it. Every tradeable expression — futures, options, the ETPs —
carries a roll cost that spot does not, and over any long horizon they have
underperformed the index by a wide margin. A backtest that "holds VIX" is not a
strategy anyone could have run.

**The VIX series spans a methodology change.** CBOE moved to the model-free
variance-swap calculation on 2003-09-22. The pre-2003 history served today is a
back-cast under the new method; the index actually published then was VXO,
computed from OEX implied volatilities. A fetch spanning that date logs a warning
and the data is still one column, so a long-horizon study has to decide
deliberately whether to use it.

**The anomaly catalogue is a snapshot of one team's replication effort.** Chen
and Zimmermann's assessments are judgements, not measurements, and the 212
predictors are the ones they could find and code — not every anomaly ever
published, and certainly not every anomaly ever tried. The published t-statistic
is what the original paper reported, so it inherits whatever that paper's
specification choices were.

**The published t-stat distribution understates the search.** Median 4.0, with
2.7% below |t| = 2. That is not evidence the field is finding real effects; it is
evidence journals do not publish t-statistics below 2. The signals that were
tried and abandoned are absent by construction, so the distribution is a lower
bound on how hard the space has been searched — which means the multiple-testing
correction it motivates is also a lower bound.

**The catalogue stops in 2016.** Nothing published since is in it, and the
post-2016 literature is where the replication debate has been most active.

**OSAP's distribution channel is fragile by the source's own choice.** The files
live on Google Drive behind ids that change on re-upload, with no versioned URL,
no content hash and no API. The id is pinned and the payload shape is checked, so
a change fails loudly — but it will fail, and when it does the fix is a manual
re-pin rather than anything automatic.

## Option chains (Milestone 9)

**This dataset has no history and never will have.** No free source sells
historical option chains. The collector's first run is 2026-09-20, and nothing
before that date exists or can be obtained. Any options research is therefore
limited to the window since collection began, and that window only grows if the
job keeps running. A gap in it is permanent.

**A snapshot is not a path.** One observation a day, from a delayed file. Realised
gamma P&L, intraday hedging error, anything that needs the path between two
closes — none of it can be reconstructed from this however long it accumulates.

**Quote quality degrades sharply away from the money.** Measured on one SPX
chain of 20,236 quotes with open interest:

| moneyness | implied vol (median) | relative spread (median) |
| --- | --- | --- |
| near the money (0.9–1.1×) | 0.14 | 1.2% |
| below 0.5× spot | 0.59, max **4.49** | 7.1% |
| above 1.5× spot | 0.15 | 16.3% |

A 449% implied volatility on a deep in-the-money call is not a volatility, it is
an artefact of inverting a price that is essentially intrinsic value. Use the
near-the-money surface; treat the wings as indicative at best.

**The greeks are CBOE's.** Computed with a model, dividend assumption and rate
curve none of which is published with the numbers. They are stored because a free
field is worth keeping, and they should be recomputed before anything is traded
on them.

## EIA (Milestone 9)

**Not point-in-time, and unfixably so.** EIA revises weekly inventories and
monthly production and serves only the current value. There is no ALFRED
equivalent — the number as first published is not archived anywhere, by EIA or by
anyone else. Every row is therefore restated, a backtest reading it has
look-ahead, and nothing in this codebase can remove it. The mitigations are to
lag the series well past the revision window, or to treat any result built on it
as an upper bound. A warning is logged on every fetch so this cannot be forgotten
quietly.

**Not run against the live API.** No free EIA key was configured, so the parser
is tested against a fixture hand-built from EIA's documented v2 envelope. That
asserts it handles the documented shape, not the actual one — the same debt FRED
carried until a key arrived. The release-date arithmetic, which is where the
look-ahead risk lives, needs no key and is fully tested.

## The event-driven engine (Milestone 10)

**It is not a higher-fidelity simulation.** With daily bars there is no intraday
path, so a stop still fills at a price the panel supplies rather than at the price
it would really have filled at somewhere inside the day. What the engine adds is
sequencing and path dependence. A genuinely event-driven simulation needs intraday
data, which no free source supplies at any useful history.

**It is slower, and that is structural.** A Python bar loop with rule dispatch
cannot match a vectorised numpy pass. Use it when a rule depends on the path and
the vectorised engine for everything else — they agree exactly when no rule is
attached, so there is nothing lost by choosing on speed.

**A rule that improves a backtest is suspicious.** M7 said this about drawdown
controls and it is more true here: a stop that raises the Sharpe of a losing
strategy is almost always fitted to that sample's particular drawdowns. The
honest use of the two engines is to measure how much of a result depends on the
rule, not to search for the rule that makes the result look best — and every such
search is a trial the deflated Sharpe should be counting.

## Tier 2 signals (Milestone 10)

**Hedging pressure is weekly, and that is a hard ceiling.** Fifty-two observations
a year caps the achievable Sharpe however real the effect is, and caps how much
any parameter fitted to it can be trusted. The "commercial" category is an
administrative self-classification, so before the 2006 disaggregated report a
swap dealer hedging an index sits in the same number as a farmer hedging a crop.

**The VIX term-structure signal is not the trade.** VIX spot is not investable.
Any implementation goes through futures, options or ETPs, each with a roll cost
this signal does not model and which has historically consumed much of the
premium. A backtest treating the index level as tradeable is not describing
anything anyone could have done.

**Its inversion rule is de-risking, not protection.** Measured on the two
episodes that destroyed short-volatility strategies: it was −0.62 short on 1
February 2018 and went flat on the 2nd, the day before VIX tripled — but only
after VIX had already run from 13.5 to 17.3. In February 2020 it scaled out over
four days. It works when the curve inverts *before* the crash rather than *with*
it, and nothing guarantees that ordering. Those exits are also measured at the
close while the platform trades the next open, and VIX gapped overnight on both
occasions, so a realistic implementation is worse than the figures above.

**And the Sharpe ratio is the wrong statistic for it.** Selling volatility earns
a little most of the time and occasionally loses years of it in a week. The
deflated Sharpe corrects for how hard you searched, not for a return distribution
whose left tail is the entire story. Read the skew and excess kurtosis on the
tearsheet.

## Paper trading (Milestone 11)

**Paper trading is not evidence that a strategy works.** It is evidence that the
pipeline runs, that the signal produces positions on data it has not seen, and
that realised costs are roughly what the backtest assumed. Those are worth
having and they are not the same claim. No amount of paper fixes a signal that
was overfitted before it started, and the sample is always far too short: the
real ETF book here would need about **ten years** of fortnightly cycles to
distinguish its live Sharpe from the backtest's expectation.

**The simulator is deterministic and a real venue is not.** Every order fills
completely, at a price derived from the panel and the cost model, immediately.
A real broker partially fills, rejects, and moves while the order is worked. The
gap between those is not modelled anywhere in this platform and is not small.

**Costs are the backtest's model, not measured fills.** Using the same model on
both sides is the right call — it removes a spurious difference — but it means
paper trading cannot validate the cost model. If the impact coefficients are
wrong, they are wrong identically in both places and the agreement proves
nothing about either.

**The decay test assumes IID returns.** The standard error
`sqrt((1 + S²/2)/T)` is the usual asymptotic form. Real returns are
autocorrelated and fat-tailed, both of which widen it, so the test is
*optimistic*: a difference it calls insignificant is comfortably insignificant,
but one it calls significant deserves more scepticism than the t-statistic
suggests.

**No live execution, by instruction.** Spec section 1 puts it outside v1. The
`Broker` abstract class is where an adapter would attach and
`quantlab/paper/broker.py` documents what would change — none of it hard to
write, all of it easy to write wrongly.
