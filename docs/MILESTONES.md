# Build order and status

Milestones are completed in order; none starts before the previous one is tested
and committed.

| # | Milestone | Status |
| --- | --- | --- |
| 1 | Skeleton + Docker: repo structure, compose stack, Makefile, CI | **complete** |
| 2 | Data layer: store, PIT machinery, calendars, FRED + Stooq/Yahoo + Binance | not started |
| 3 | Cost models: square-root impact, spread, financing, capacity calculator | not started |
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

## Not yet verified

The `make up` acceptance criterion has **not** been executed: the development
machine has the Docker CLI but no daemon. The compose file, Dockerfiles and
Makefile are covered by static tests and by CI jobs written for it, but the first
real `docker compose up` will happen either in CI or on a machine with a running
daemon. Treat the Docker layer as unproven until one of those goes green.
