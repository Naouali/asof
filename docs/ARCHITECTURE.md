# Architecture

## Layering

Dependencies point downward only. A test walks the import graph of `src/quantlab`
and fails on any upward edge.

```
        cli / dashboard / reporting
                    |
        backtest  /  validation
                    |
        portfolio  /  risk
                    |
        costs  /  signals
                    |
        data  /  universe
```

`config`, `logging`, `paths`, `runtime` and `health` sit outside the layer stack:
they are leaf utilities anything may import.

## Why these choices

**Polars, not pandas, in the signal path.** pandas' implicit index makes
time-misalignment bugs easy to write and invisible to review: a join that silently
aligns a fundamental to the wrong date is look-ahead bias that produces a beautiful
equity curve. Polars has no implicit index, and its lazy engine makes the
computation graph inspectable. Enforced by a ruff banned-import rule and by a test
that ignores `# noqa`.

**DuckDB over parquet, not a database server.** Research queries are analytical
scans over columnar files. DuckDB does that in-process, at speed, with no service to
run, no schema migrations, and no network hop. Postgres exists only for
paper-trading state, the run registry and dashboard metadata — mutable,
transactional, small.

**Every row carries `as_of` and `ingested_at`.** `as_of` is the date the observation
refers to; `ingested_at` is when we learned it. Point-in-time queries filter on the
second. For sources that revise without publishing vintages, `ingested_at` is the
only vintage we have, which is why daily snapshotting starts as early as possible.

**Heartbeats, not PIDs, for health.** A deadlocked process keeps its PID.

**Unimplemented commands exit non-zero.** An empty result is indistinguishable from
a successful run that found nothing. That ambiguity is the failure mode the whole
platform exists to prevent, so it is not allowed in the CLI either.

## Containers

| Service | Image | Role |
| --- | --- | --- |
| `db` | `timescale/timescaledb` | Paper-trading state, run registry, dashboard metadata. **Never bulk market data.** |
| `worker` | `quantlab-worker` | Ingestion and backtest execution; the container `make shell` and `make backtest` run in. |
| `scheduler` | `quantlab-worker` | APScheduler driving `configs/schedule.yaml`; runs each job as a subprocess so a crash cannot take the scheduler down. |
| `dashboard` | `quantlab-dashboard` | FastAPI reporting UI. Local assets only — no CDN, so it works offline. |
| `research` | `quantlab-research` | Jupyter Lab with the repo's `src` mounted read-only. |

DuckDB is embedded, not a service.

The `quantlab-data` named volume holds the lake and survives `docker compose down`
and every image rebuild. Only `make clean-data` removes it, and it asks first.

## Where a live broker adapter would attach

There is none, by design. When one is eventually written, it attaches below
`backtest/paper.py`: the paper loop already produces *target positions* and *intended
trades* as data. A live adapter would consume that same interface and add order
placement, fill reconciliation and position synchronisation against the broker's
record of truth. Nothing above that boundary should need to change — and nothing
above it should ever import a broker SDK, which a test currently enforces.

## Where to go next

- [ADDING_A_SIGNAL.md](ADDING_A_SIGNAL.md) — the full process for a new signal,
  including the three mistakes that do not look like mistakes: a reversed sign,
  a mixed reporting basis, and a timestamp join that appears to work.
- [LIMITATIONS.md](LIMITATIONS.md) — what constrains every result, then what
  constrains each component.
- [ASSUMPTIONS.md](ASSUMPTIONS.md) — every choice made under ambiguity, and why
  it was the conservative one. Numbered, so a decision can be cited.
