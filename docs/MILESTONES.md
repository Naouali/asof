# Build order and status

Milestones are completed in order; none starts before the previous one is tested
and committed.

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Skeleton + Docker: repo structure, compose stack, Makefile, CI | **complete** |
| 2 | Data layer: store, PIT machinery, calendars, FRED + Stooq/Yahoo + Binance | **complete** |
| 3 | Cost models: square-root impact, spread, financing, capacity calculator | **complete** |
| 4 | Vectorised backtest engine with enforced accounting identities | not started |
| 5 | Validation: CPCV, DSR with automatic trial counting, PBO, empirical-Bayes shrinkage | not started |
| 6 | Tier 1 signals: time-series momentum, cross-asset carry, equity profitability | not started |
| 7 | Portfolio construction: vol targeting, risk parity, turnover-aware optimisation, factor risk model | not started |
| 8 | Reporting: tearsheets with capacity and validation statistics | not started |
| 9 | Remaining data sources: SEC EDGAR, CFTC COT, EIA, academic factors, options collector | not started |
| 10 | Event-driven engine and Tier 2 signals | not started |
| 11 | Paper trading loop, dashboard, decay monitor | not started |
| 12 | Documentation: data catalogue, how-to-add-a-signal, LIMITATIONS.md | in progress |

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

## Not yet verified

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

**FRED and ALFRED.** No free FRED key was available, so neither fetcher has been
run against the live API. Their guard rails are tested, and their parsers are
tested against fixtures **hand-constructed from the published response shape**
rather than recorded — which asserts that the parser handles the documented shape,
not the actual one. Re-record both fixtures once a key is configured; this is
flagged in `tests/fixtures/README.md`.
