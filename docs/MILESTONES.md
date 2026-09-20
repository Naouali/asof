# Build order and status

Milestones are completed in order; none starts before the previous one is tested
and committed.

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Skeleton + Docker: repo structure, compose stack, Makefile, CI | **complete** |
| 2 | Data layer: store, PIT machinery, calendars, FRED + Stooq/Yahoo + Binance | **complete** |
| 3 | Cost models: square-root impact, spread, financing, capacity calculator | **complete** |
| 4 | Vectorised backtest engine with enforced accounting identities | **complete** |
| 5 | Validation: CPCV, DSR with automatic trial counting, PBO, empirical-Bayes shrinkage | **complete** |
| 6 | Tier 1 signals: time-series momentum, cross-asset carry, equity profitability | **complete** |
| 7 | Portfolio construction: vol targeting, risk parity, turnover-aware optimisation, factor risk model | **complete** |
| 8 | Reporting: tearsheets with capacity and validation statistics | **complete** |
| 9 | Remaining data sources: SEC EDGAR, CFTC COT, EIA, academic factors, options collector | **complete** |
| 10 | Event-driven engine and Tier 2 signals | **complete** |
| 11 | Paper trading loop, dashboard, decay monitor | **complete** |
| 12 | Documentation: data catalogue, how-to-add-a-signal, LIMITATIONS.md | **complete** |

## Milestone 1 — what was delivered

**Repository and packaging**
- Monorepo with downward-only layer boundaries, enforced by a test that walks the
  import graph.
- `pyproject.toml` with ranged direct dependencies and a committed `uv.lock`
  pinning all 176 packages. CI runs `uv lock --check`.
- ruff (lint + format) and mypy in `--strict` mode, both clean.

**Docker**
- Four Dockerfiles over a shared multi-stage base: `base` (runtime + a `dev` stage
  carrying test tooling), `worker`, `dashboard`, `research`.
- Multi-architecture by construction; a CI job builds the base image for
  `linux/amd64` and `linux/arm64`.
- Non-root user, `tini` as PID 1, healthchecks on every service, resource limits on
  every service, named volume for the data lake.
- Loopback-only port publishing; Postgres not published at all.

**Operations**
- `Makefile` as the single entrypoint. `make up` builds, starts, waits for health,
  and runs `doctor`.
- APScheduler-based scheduler driven by `configs/schedule.yaml`, with clean SIGTERM
  shutdown and per-job subprocess isolation.
- Heartbeat-based liveness for the worker and scheduler.
- `.env.example` documenting every setting; the stack runs with the file untouched.

**Data catalogue (groundwork for Milestone 2)**
- 29 sources registered as structured data with 65 recorded caveats, a point-in-time
  quality classification, per-source ingest throttles, and licence notes.
- `docs/DATA_CATALOGUE.md` is generated from the registry; CI fails if it is stale.
- Surfaced by `quantlab data catalogue`, by `/api/sources`, and on the dashboard.

**Tests** — 89 passing, covering the catalogue's invariants, config and secret
handling, heartbeats, health checks, the logging chain, schedule parsing and job
execution, the CLI, the dashboard, repository structure (layering, the pandas ban,
the no-broker-SDK rule) and the compose stack's operational guarantees. Integration
tests that need a Docker daemon are marked and skipped without one.

## Milestone 2 — what was delivered

**Storage**
- Parquet lake, Hive-partitioned `source/dataset/asset_class/year`, queried
  in-process by DuckDB. Append-only: a restatement is a new row with a later
  `known_at`, and the superseded value stays on disk.
- Six canonical dataset schemas, strictly validated on write. Every row carries
  `as_of`, `known_at` and `ingested_at` as timezone-aware UTC microseconds.
- Partition filenames are content-addressed, so re-ingesting identical data is a
  no-op and the incremental overlap window costs nothing.

**Point-in-time**
- `store.as_of(date)` returns a `Snapshot` enforcing the contract four ways:
  query filtering, argument guards that raise rather than truncate, a
  post-condition check on every returned frame, and a DuckDB sandbox built with
  `enable_external_access=false` that cannot reach the lake files at all.
- 24 leakage tests attack the contract from each of those angles, including a
  monkeypatched regression that removes the filter and asserts the post-condition
  catches it.

**Calendars** — session closes in UTC per venue, DST- and half-day-correct.
Unknown venues raise instead of defaulting to a US calendar.

**Sources** — eight fetchers behind one interface:
`yahoo.ohlcv_daily`, `yahoo.corporate_actions`, `binance.ohlcv_bars`,
`binance.funding_rate`, `binance.instruments`, `fred.series_observations`,
`alfred.series_observations`, `stooq.ohlcv_daily` (blocked, fails loudly).

**Ingest** — plan-driven, resumable from the lake without a cursor file, with
per-source token-bucket rate limiting, deterministic backoff, Binance weight-header
throttling, an offline mode that refuses sockets, and a JSONL run registry.

**CLI** — `data ingest`, `data status`, `data fetchers`, `data query` (with
`--as-of` running inside the point-in-time sandbox).

**Tests** — 251 unit tests, all offline: a transport-level block fails any test
that opens a socket. Fixtures are real recorded payloads; provenance is documented
in `tests/fixtures/README.md`.

## Milestone 3 — what was delivered

**Impact** — the square-root law `Y·σ·(Q/ADV)^δ`, split into permanent and
temporary, with only half the permanent charged (it accrues while the order is
worked). Per-asset-class calibrations, each with its rationale in code. Beyond
~10% of ADV the result is flagged as extrapolation.

**Execution** — Almgren-Chriss optimal scheduling, dimensionally consistent in
participation units, whose risk-neutral solution reproduces the square-root law to
within the discretisation error (asserted to converge). Efficient frontier of
cost against certainty. A power-law propagator model for the event-driven engine,
producing impact *paths* rather than averages.

**Spread** — observed quotes where available, plus Roll, Corwin-Schultz and
Abdi-Ranaldo. Every estimator reports `coverage` and a **resolution floor**, and
every one raises rather than clamping an undefined result to zero. The bias
correction is explicit machinery with provenance, defaulting to no shrinkage and
saying so.

Validating these against real Binance data produced the milestone's most
important finding: with a true BTCUSDT spread of 0.001 bp, Roll returned 167 bp
and Corwin-Schultz 106 bp, and Abdi-Ranaldo was undefined for seven of eight
majors. On liquid volatile instruments these estimators measure daily volatility,
not spread. Instruments now carry a `spread_source` provenance field and the
model warns when a crypto spread is estimated rather than observed. See
docs/LIMITATIONS.md.

**Financing** — punitive default borrow (15 bp/month, flagged as assumed), margin
financing on the levered portion only, zero short rebate by default, signed
perpetual funding, and futures roll as two crossings per roll.

**Capacity** — break-even AUM by numerical solve, checked against a closed form to
1.5e-15, with the curve for the tearsheet. Verified to scale with alpha squared.
A strategy whose costs beat its edge at any size reports *no* capacity rather than
a small one.

**CLI** — `quantlab costs estimate` (decomposed, with `--compare-flat` to show
what a flat model would claim) and `quantlab costs capacity`.

**Tests** — 426 total, 92% coverage on `costs`. Assertions are algebraic
identities or empirical regularities, not pinned outputs: the 4×-size-for-2×-impact
rule, `Q^1.5` currency scaling, alpha-squared capacity, scalar/vectorised
agreement, and estimator bias measured against a simulated known spread.

## Milestone 4 — what was delivered

**The engine.** Daily/weekly/monthly rebalancing over a dense panel with full
square-root costs, financing and borrow. The cross-section is vectorised; the time
axis is a loop, because equity at bar *t* depends on bar *t−1* and costs depend on
absolute trade size — a recursion that cannot be collapsed into a cumulative
product without approximating the cost model.

**Accounting.** Every bar is reconciled by two independent routes: a cash ledger
and a P&L statement. They share no arithmetic, so a discrepancy is a bug rather
than a rounding artefact, and a breach stops the run. Verified to balance exactly
across a 30-year, 3,000-name run.

**Conventions, enforced rather than assumed.** Signal on the close of T traded at
the open of T+1; forward-fill capped at a configured limit with liquidation beyond
it; a −30% delisting return applied at the *execution* price; calendar-day
financing accrual; and a refusal to run a panel whose execution convention
disagrees with the config's.

**Throughput.** The spec's target is a 30-year, 3,000-name backtest with full
costs in under ten seconds. Measured: **7.4s** (4.6s panel + 2.9s run). Two earlier
implementations missed it — pivoting each column separately took 111 seconds, and a
composite (date, symbol) join took 8 seconds of the budget on string hashing. The
shipped version maps to integer indices and scatters.

**Reproducibility.** A golden-file fixture generated from a seeded panel that
deliberately contains a data gap and a delisting, pinned bar-for-bar and
statistic-for-statistic. Regenerated only by hand, never to make a test pass.

**End to end.** `quantlab backtest --config` reads market data through a
point-in-time snapshot fixed by the config's `as_of`, derives trailing-only
liquidity statistics, and runs. On the real ingested ETF lake a 12-1 momentum
strategy came out at 0.20 gross Sharpe and **0.09 net** — with 84 of the 111 bp/yr
cost drag being *borrow*, which is the spec's own prediction about anomaly alphas
arriving on the first strategy the platform ever ran.

**Tests** — 90 in the backtest suite, 97% coverage. Includes Hypothesis property
tests asserting the accounting identity holds for arbitrary input sequences, and a
throughput test that fails if the ten-second budget is exceeded.

## Milestone 5 — what was delivered

**The automatic trial counter.** Every backtest records itself; the count of
distinct configurations per family feeds the deflated Sharpe on the result. Spec
section 7 is explicit that this cannot rely on the operator remembering, so it does
not. Opting out of recording also opts out of getting a deflated number.

**Overfitting statistics.** Probabilistic Sharpe (corrects for the sample),
deflated Sharpe (for the search), PBO by combinatorially symmetric cross-validation
(for the selection rule), and minimum backtest length. Demonstrated: a Sharpe of
1.5 over five years goes from 1.000 at one trial to 0.155 at two hundred to 0.003
at ten thousand.

**Cross-validation that does not leak.** Purged k-fold with embargo, combinatorial
purged CV producing a distribution of paths, and walk-forward. A bug found here:
purging over the span from the first to the last test index deletes the entire
sample when the test groups are far apart, which combinatorial CV does routinely.

**Empirical-Bayes luck adjustment.** `shrinkage = 1 − 1/Var(t)` across the signal
library. On 200 pure-noise signals it returns 0.00 with the verdict *"the
dispersion of your results is what pure chance produces… nothing has been found."*

**The red-team suite** — 11 tests over four deliberately broken strategies. Three
are rejected. The fourth documents a limit: **survivorship bias is invisible to
every statistic in this module**, and the test asserts that the framework passes a
survivorship-biased strategy, so nobody mistakes a clean validation report for
evidence of a clean universe.

**Sharpe ratios are now never reported bare.** Every backtest summary carries the
deflated value and the trial count that produced it.

**Tests** — 101 in the validation suite, 92% coverage. Two unit bugs caught by
them: `minimum_backtest_length` was mixing per-observation and annualised Sharpe
units (understating the requirement by a factor of 252), and the CPCV path
reassembly double-counted observations when a split spanned groups belonging to
different paths.

## Milestone 6 — what was delivered

**The signal contract.** Every signal declares its datasets, cadence, expected
turnover, academic reference and — unusually — **how it is known to fail**. The
failure-modes field is validated: a terse one is a construction error, on the
grounds that if you cannot name how a signal goes wrong you do not understand it
well enough to trade it.

**Six signals**, tiered by evidence rather than interest:

| Signal | Tier | State |
| --- | --- | --- |
| `trend.time_series_momentum` | 1 | Runs. Blended 1/3/12-month, volatility-scaled per instrument. |
| `carry.crypto_perp_funding` | 1 | Runs. Funding modelled as the cash flow it is. |
| `carry.rates` | 1 | Implemented; needs a free FRED key. |
| `carry.commodity_basis` | 1 | Implemented; free data supplies no second contract (spec 3.4). |
| `equity.profitability` | 1 | Implemented; needs SEC EDGAR (Milestone 9). |
| `carry.fx` | 3 | Implemented and marked **decayed**, so it can be re-tested with the prior attached. |

**Ken French fetcher**, pulled forward from Milestone 9 because this milestone's
acceptance criterion is a benchmark and a benchmark you cannot run is not one.
46,060 daily factor observations back to 1990.

**The benchmark, honestly scoped.** Spec section 3.3 asks for >0.9 correlation with
UMD. That threshold applies to a comparable universe; an ETF factor against a
three-thousand-stock factor is a different portfolio, not a noisy version of one.
`Comparability` makes the distinction explicit. Measured: **+0.539** over 4,904
overlapping days — right sign, right magnitude, and the real test waits on a stock
universe.

**Results, end to end on real ingested data.** Both Tier 1 signals that can run
were taken through the full pipeline — snapshot, signal, weights, costs, ledger,
deflation:

- Trend on 14 ETFs: 0.22 gross Sharpe, **0.14 net**, deflated 0.738 — does not
  survive. The spec predicts this: *"on a handful of correlated markets it is one
  bet."*
- Crypto carry on 8 majors: 0.34 gross, **0.15 net**, deflated 0.647 — does not
  survive. 768 bp/yr of cost drag, dominated by impact at weekly turnover.

Neither is a finding. Both are the platform working.

**Tests** — 93 in the signal suite. A layering violation was caught by the
import-graph test and fixed by moving `RebalanceFrequency` out of the engine.

## Milestone 7 — what was delivered

**Covariance that can be inverted.** Sample, EWMA and Ledoit-Wolf shrinkage
against two targets. The shrinkage intensity is *derived*, not chosen — short
samples and wide universes shrink hard, long samples and narrow ones barely at
all, and neither is a setting anyone tunes. The constant-correlation target is
the default because the identity target is so wrong for a real universe that the
estimator correctly declines to use it.

**Four construction methods** — equal weight, inverse volatility, risk parity and
mean-variance — reporting risk contributions alongside weights, because an
equal-weighted book of correlated assets routinely puts most of its risk in one
place while looking perfectly diversified on the weights. `quantlab portfolio
build` prints both.

**Constraints that refuse impossible problems.** `Constraints.long_only()` caps a
position at 10%, so on eight names the largest possible book is 80% of capital
against a net target of 100%. SLSQP answers that with "inequality constraints
incompatible" and a plausible-looking weight vector that violates the
constraints; `check_feasible` raises instead, naming the arithmetic.

**Gârleanu-Pedersen dynamic trading**, solving for the aim portfolio and the
constant trade rate rather than rebalancing all the way to the target each period.

**Factor attribution with Newey-West standard errors**, answering the question the
spec sets for it: *is this signal secretly just beta?* Validated against
instruments whose answer is known:

| | market β | SMB | HML | R² | alpha |
| --- | --- | --- | --- | --- | --- |
| SPY | +0.993 | −0.113 | +0.014 | 99.3% | +0.02%/yr (t = 0.04) |
| IWM | +1.017 | **+0.926** | +0.295 | 97.5% | −1.55%/yr (t = −1.01) |
| QQQ | +1.166 | −0.121 | **−0.338** | 95.0% | +2.90%/yr (t = 1.21) |
| TLT | −0.002 | +0.147 | −0.108 | 3.8% | −8.44%/yr (t = −1.31) |
| GLD | +0.142 | +0.092 | −0.028 | 3.1% | +12.44%/yr (t = 1.52) |

IWM loads +0.93 on the small-cap factor because it *is* the small-cap index; QQQ
carries the growth tilt; TLT's market beta is zero to three decimal places; the
equity factors explain 3% of gold. SPY's alpha is zero, which is the single
strongest end-to-end check in the platform — it only came out that way after two
real bugs were fixed (see below).

**A PCA risk model** for when no factor returns exist, and **drawdown controls and
trailing stops**, both documented as risk management rather than alpha.

### Three bugs worth recording

**Price returns manufacture alpha.** Computing returns from `close` rather than
`adj_close` drops the dividend yield from every observation. Attributing SPY that
way produced −1.45% a year with t = −2.3 — comfortably "significant", entirely an
artefact, and equal to SPY's 1.57% yield. A short book would have shown the same
error as manufactured *positive* alpha. This is the failure mode the platform
exists to catch, and it was found by checking a number whose true value was known
rather than by reading the code.

**PCA on covariance finds the loudest asset, not the market.** The first component
of the twelve-ETF universe loaded +0.71 on a 41%-volatility oil fund and under
0.35 on everything else. Maximising explained variance is precisely what selects
for that. Decomposing the correlation matrix instead gives a first component
loading −0.31 to −0.38 on every equity and credit name and 0.09 on oil.

**A from-entry stop switches itself off.** Measuring the loss from the entry price
rather than the high-water mark means a position that has run up 300% must give
back all of it before the stop fires. Firing rates on a driftless random walk:
6 per thousand bars from entry, 122 trailing. The positions this affects most are
the long-held trend positions that stops are cited as helping.

### Measured, not asserted

* On a 40-asset, 80-observation factor panel, Ledoit-Wolf shrinkage toward the
  constant-correlation target removes **21%** of the sample covariance error.
  Toward the identity target it removes **none** — the shrunk estimate is about
  **6% worse** than not shrinking at all. The derived intensity is doing its job
  in both cases; the identity target is simply the wrong structure, and a target
  that is wrong enough cannot be rescued by shrinking the right amount toward it.
* A residual information ratio of 1.7 is detected **97%** of the time over six
  years; an IR of 0.83 — a strategy most desks would fund — about **half** the
  time; an IR of 0.42 rarely. The estimator is unbiased at every level. A single
  t-statistic near the threshold carries almost no information, which is why the
  platform reports a deflated Sharpe and trial count beside it.

## Milestone 8 — what was delivered

**A tearsheet that refuses to flatter.** Two of spec section 13's prohibitions
are enforced structurally rather than by convention: `Tearsheet.build` raises if
the run carries no trial count, and capacity is a keyword argument with no
default. There is no override flag for either, so the failure mode is a report
that will not render rather than one that renders without the inconvenient parts.

**Panels ordered by severity, worst first.** A strategy that fails deflation says
so above its equity curve. Ordering by topic would put the accounting
reconciliation above the reason the result is worthless.

**Four renderings from one source of content** — text, Markdown, HTML and a JSON
dictionary for the dashboard — so they cannot drift apart and disagree about what
the strategy did. The HTML is self-contained: no scripts, no fonts, no network,
21 KB including an inline SVG of the equity curve.

**Capacity derived from the run**, not supplied: alpha and turnover from its own
P&L, liquidity from the panel it traded, rebalance frequency counted from bars
that actually traded.

**Caveats from the catalogue.** A limitation recorded once when a source is
registered appears on every tearsheet built from that source's data. The example
run surfaces five Yahoo caveats, including the survivorship warning, without
anyone remembering to mention them.

### The example run, reported honestly

A 12-1 momentum book on fourteen ETFs, 2010-2026, is the first strategy the
platform has reported end to end. Its tearsheet opens:

> **NOT EVIDENCE that etf-momentum-12-1 works:**
> - The deflated Sharpe of 0.470 is below 0.95: after 1 trial(s) this result is
>   not distinguishable from the best of the search.
> - There is no AUM at which this strategy makes money after costs. Spread,
>   commission and holding costs exceed the gross edge at any size; this is not
>   a capacity problem, the signal does not pay.

CAGR −0.68%, Sharpe −0.02 net against 0.11 gross, turnover 1,020%/yr, and a cost
drag of 133 bp/yr of which **89 bp is borrow** — the platform's punitive
180 bp/yr default on the short leg. That is the result, and reporting it is the
point.

### Three bugs, all found by checking arithmetic against the run

**Holding costs charged twice.** The capacity link netted borrow and financing
out of the gross alpha *and* passed them to the capacity model, which applies
them again. Net alpha came back at −143 bp/yr against a realised CAGR of −68.
Gross alpha is now the realised return with the whole drag added back, exactly
once: −68.2 + 133.1 = +64.9 bp/yr.

**Every held bar counted as a rebalance.** Held weights drift with prices, so the
run's positions frame carried 4,122 distinct dates while only **197 bars actually
traded**. Counting the former told the capacity model the strategy made 248 small
trades a year rather than 12 large ones, and because impact is concave in size
that understates impact and **overstates** capacity. Rebalances are now counted
from non-zero traded notional.

**Downsampling could hide the trough.** The first SVG emitted all 4,182 points
and cost 96 KB. Bucketing fixed the size, but decimating the drawdown the same
way as the equity line would let a one-bar crash vanish from the chart — the most
flattering thing a downsample can do. Equity now takes the last value in each
bucket and drawdown takes the minimum, so the worst drawdown cannot be lost. A
test constructs a single-bar 60% crash in four thousand bars and asserts it
survives a 50-point render.

## Milestone 9 — what was delivered

Six sources, and the point-in-time contract is the work in every one of them.
None of these payloads carries the date its data became public, so each release
time had to be established from the publisher's own documentation and derived.

| Source | Rows | The lag that had to be derived |
| --- | --- | --- |
| CFTC COT | 290,646 | Tuesday snapshot, Friday 15:30 ET release |
| SEC EDGAR | 19,760 | filed date at EDGAR's 17:30 ET acceptance cutoff |
| CBOE indices | 27,311 | 16:15 ET settlement |
| OSAP catalogue | 331 | sample end vs publication, kept apart |
| Options chains | 20,236/day | CBOE's quote timestamp, not collection time |
| EIA | — | Wed/Thu 10:30 ET release (awaiting a key) |

US federal holiday rules were written out for this: the federal calendar is not
the exchange calendar, three of these publishers are federal agencies, and
pandas — which has the rules — is banned here.

**The options collector is enabled**, and that is a deliberate decision rather
than a default. It is the only dataset in the platform that cannot be
backfilled: no free source sells historical chains, so its history starts the day
it first runs and every day it is off is gone permanently. It reads CBOE delayed
quotes rather than the Yahoo endpoint the catalogue originally named, because
Yahoo now refuses unauthenticated callers and CBOE serves the exchange's own data
with greeks and no credentials.

**The anomaly catalogue** is the most useful thing here for the platform's actual
purpose. 212 published cross-sectional predictors with the t-statistic each
original paper reported: median 4.0, and only 2.7% below |t| = 2. That is not
evidence the field finds real effects, it is evidence of where journals stop
accepting papers, and it makes the distribution a lower bound on how hard the
space has been searched. `quantlab validate anomalies` places a result in it.

### Bugs found

- **YAML parsed contract codes as octal.** `002602` became 1410, Socrata answers
  an unknown code with an empty array rather than an error, and only codes whose
  digits are all 0-7 were affected — so gold survived and corn did not. It cost
  57% of the COT data silently: 185,402 rows against 290,646.
- **`federal_holidays(1994)` held a date from 1993.** A Saturday New Year's Day
  is observed on 31 December of the previous year, so it was filed under a year
  no lookup would search. Found by a sweep over 1986-2040.
- **The coverage-regression detector cried wolf.** It grouped by symbol alone, so
  one contract under three reports beginning 1992, 2006 and 2006 tripped it every
  ingest. It now groups by the schema key.
- **`fy` on an XBRL fact is the filing's fiscal year, not the fact's.** Apple's
  FY2023 revenue restated in the FY2025 10-K carries fy=2025.
- **equity.profitability mixed reporting periods.** It took the most recently
  *filed* value per metric, giving nine-month revenue, one quarter's SG&A and a
  three-year-stale interest expense over an instantaneous balance sheet. For
  Apple that understated gross profitability by 16%; across the universe it moved
  Walmart from last place to first.

### Measured, not assumed

- XBRL coverage over five large filers: revenue, total assets, book equity and
  interest expense present for all five, cost of goods sold for three. JPMorgan
  has none because banks do not report one — so dropping filers with missing line
  items drops financials, not a random sample.
- COT's accounting identity holds to **zero** on both sides across 142 weeks,
  which verifies the column mapping rather than assuming it.
- SPX option quote quality by moneyness: near the money, implied vol median 0.14
  and relative spread 1.2%; below half spot, implied vol reaches 4.49 and spreads
  7%; above 1.5× spot, spreads 16%.

## Milestone 10 — what was delivered

**A second backtest engine, and a reason to trust both.** The vectorised engine
computes a bar's position from the target weights alone, which is fast and
structurally cannot express a rule that depends on the path — a stop-loss
triggers on what a position has done since it was opened, which the weights do
not know. The event-driven engine resolves ordered intra-bar events and lets
rules modify the target before it trades.

With no rules attached the two engines agree **bit for bit**: maximum relative
equity difference 0.000e+00 across 400 bars and six instruments, identical
Sharpe, identical cost decomposition. That is the point of building it this way.
Two independent implementations of one convention agreeing is evidence the
convention is implemented; a divergence would be a bug in one of them rather than
a difference of opinion. It also gives a free diagnostic — attach a rule, and the
size of the change is the part of the result that depends on the rule rather than
on the signal.

What it is honestly not: a higher-fidelity execution simulation. With daily bars
there is no intraday path, so a stop still fills at a price the panel supplies.
What it adds is sequencing — a delisting settles before a rebalance sizes into
the name, a stop is evaluated on the position carried *into* the bar — and those
orderings change results and would otherwise be decided by accident.

**Two Tier 2 signals**, both built on data Milestone 9 delivered, and both with a
sign convention that reversed produces a strategy which pays a risk premium every
period instead of earning it.

`positioning.hedger_pressure` — commercial hedging pressure from the CFTC data.
Keynes, Hicks, Bessembinder, De Roon: hedgers are net short, speculators take the
other side and are compensated for it, so an unusually short commercial book is a
long signal. Normalised over a trailing 156 weeks rather than full history,
because the CFTC reclassifies traders between categories and a z-score against a
mean containing a reclassification is measuring the reclassification. Runs on the
real lake across eleven contracts.

`volatility.vix_term_structure` — short volatility when the curve is steep, flat
when it inverts. The scale-in bound of 1.15 is the observed median slope over
4,276 days rather than a fitted parameter.

### Measured on the episodes that mattered

The VIX signal's inversion rule, on the two events that destroyed
short-volatility strategies:

| | Position going in | Went flat |
| --- | --- | --- |
| Feb 2018 | −0.62 on 1 February | 2 February, the day before VIX tripled 17 → 37 |
| Feb 2020 | −0.57 on 18 February | 24 February, having scaled down from the 20th |

It de-risked before the worst day both times, and both times only after taking
the first leg — VIX had already run 13.5 to 17.3 before the 2018 exit fired. It
is a de-risking rule, not protection against a gap, and it works only when the
curve inverts *before* the crash rather than *with* it. The exits are also
measured at the close while the platform trades the next open, and VIX gapped
overnight on both occasions.

## Milestone 11 — what was delivered

**A paper-trading loop that charges itself properly.** Fills go through the same
`TransactionCostModel` object the backtest uses -- not a simplified version. A
paper book that filled at mid would beat its own backtest for no reason and the
difference would read as the strategy working. On the real ETF book the loop
pays 10.7 bp a side and holds a dollar-neutral 1.00x gross book across six
instruments.

The cycle is idempotent per as-of date: a repeat is refused rather than
absorbed, which makes re-running after a failure safe. State is append-only
JSONL under the data root rather than the compose stack's Postgres, because the
platform has to run from a clone with no services.

**The broker seam, and nothing behind it.** `Broker` is abstract, `PaperBroker`
is the only implementation, and a test asserts there is no second one. Spec
section 1 puts live execution outside v1 by instruction. What would genuinely
change if an adapter were written -- non-deterministic fills, the broker's record
becoming authoritative and needing reconciliation every cycle, a timed-out
submission that may already have reached the exchange -- is documented at the
seam rather than left as an exercise.

**A decay monitor that answers the question people skip.** Not "has it decayed"
but "can you yet tell". It compares against the *haircut* backtest Sharpe
(×0.53, Milestone 5's prior) rather than the raw number, because a strategy
delivering half its backtest is doing exactly what was predicted and comparing
against the printed figure would declare decay on every well-behaved strategy.

### A bug worth recording

The first real decay run reported a live Sharpe of **4.08** against an expected
0.64 -- a spectacular number, and wrong. The monitor annualised by the square
root of 252 while the book was rebalanced fortnightly. Inferring the frequency
from the recorded cycle dates instead gives 24 periods a year and a live Sharpe
of **1.27**, and changes the verdict from an implied triumph to the honest one:

> 8 observations is too few to conclude anything. The standard error of a Sharpe
> over this sample is 2.34 annualised, which is wider than most of the effects
> anyone is looking for.

253 observations -- about ten years at this cadence -- would be needed to
distinguish the result from expectation. That is the uncomfortable arithmetic the
monitor exists to state.

**Every CLI command is now implemented.** Through Milestones 1-10 the
unimplemented ones exited non-zero naming the milestone that would deliver them;
paper trading was the last. The test that policed them now asserts no stub
remains.

## Milestone 12 — what was delivered

Mostly restructuring and correction rather than new prose. `LIMITATIONS.md` had
grown a section per milestone, which is build order rather than reader order, so
it is now Part I — what constrains every result — and Part II — what constrains
each component. `DATA_CATALOGUE.md` is generated and was already current.

**`ADDING_A_SIGNAL.md`** is new: the full process, organised around the three
mistakes that do not look like mistakes and cost real time in this build. A
reversed sign, which backtests as a confident slow loss rather than as an error.
A mixed reporting basis, which divided nine-month revenue by an instantaneous
balance sheet. A timestamp join across datasets, which appears to work while
regressing one series against another shifted by every missing session.

### Two documentation bugs, and one of them was serious

**Section 5 was overselling the platform.** It claimed the event-driven engine
"models order types, partial fills, slippage, latency and rejects". Written in
Milestone 1 as a plan; the engine delivered in Milestone 10 models none of them.
The document whose entire purpose is to stop this platform overselling itself was
the thing doing the overselling, for eleven milestones. It now says what the
engine actually does — sequencing and path dependence, not fill fidelity — and a
test asserts the corrected wording stays.

**Section 7's signal inventory had drifted.** It said six Tier 1 signals were
implemented and two could run. By Milestone 11 there were eight signals across
three tiers and six of them ran — EDGAR, a FRED key and the Milestone 9 sources
had each unblocked one without the document noticing. A count that drifts is
worse than no count, because a reader will trust it, so a test now compares the
inventory against the signal registry and fails when they disagree.

### Final state

| | |
| --- | --- |
| Tests | 1,205 passing, 4 slow, 3 skipped without a Docker daemon |
| Coverage | 89.95% against an 85% gate |
| Source files | 98 |
| Signals | 8 implemented, 6 runnable on free data |
| Data sources | 29 catalogued, 24 with a working fetcher |
| Assumptions recorded | 142, each with the reason it was the conservative choice |

## Still unverified

**Docker.** The `make up` acceptance criterion has still **not** been executed: the
development machine has the Docker CLI but no daemon. The compose file, Dockerfiles
and Makefile are covered by static tests and by CI jobs written for them, but the
first real `docker compose up` will happen either in CI or on a machine with a
running daemon. Treat the Docker layer as unproven until one of those goes green.

**Cost calibration.** Every impact and financing parameter is a literature default,
not a number fitted to real fills. The models are correct; whether they are
*calibrated* for your execution is unknown and unknowable from free data. The
spread estimators have been validated against simulation, where the answer is
known, but not against live equity quotes, which are not free.

**EIA.** No free EIA key was configured, so the fetcher has never run against
the live API. Its parser is tested against a fixture built from EIA's documented
v2 envelope — which asserts it handles the documented shape, not the actual one.
The release-date arithmetic, where the look-ahead risk actually lives, needs no
key and is fully tested. This is the same debt FRED carried until a key arrived,
and it is paid off the same way.

**Live execution.** Out of scope by instruction (spec section 1), so the gap
between the paper loop and a real venue is not merely unverified, it is
unmodelled. See `quantlab/paper/broker.py`.

## Verified since

**FRED and ALFRED — now verified (2026-09-20).** A key was configured and both
fetchers were run against the live API: 115,314 FRED rows and 28,075 ALFRED rows.
The hand-constructed fixtures have been replaced with recorded payloads. The
parser needed no change, but one test did — it asserted exactly three vintages of
2020 Q1 because the invented fixture had three, where the live payload has nine.

The vintage machinery was verified end to end on real revision history. Of the
ALFRED observations in the lake, 2,614 carry more than one vintage and in every
one of those the value changed. Real GDP for 2020 Q2 reads:

| Snapshot taken | GDPC1 for 2020 Q2 |
| --- | --- |
| 2020-09-01 | 17,282.2 |
| 2021-06-01 | 17,302.5 |
| 2026-09-20 | 19,078.0 |

The first print was 17,205.8 and today's figure is 10.9% higher. A snapshot sees
only the vintage that existed at its instant, which is the entire purpose of the
`known_at` contract. (The 2023 jump is a benchmark re-basing rather than a data
revision, so levels are not comparable across it; growth rates are.)
