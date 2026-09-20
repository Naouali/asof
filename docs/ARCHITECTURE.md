# Architecture

## Flow

```
configs/ingest.yaml          what to fetch
        |
  data/ingest.py             plan -> jobs -> date windows (full or incremental)
        |
  data/sources/*.py          one fetcher per external API, over data/http.py
        |
  data/schemas.py            every frame is validated against a canonical schema
        |
  data/store.py              append-only parquet lake, queried by embedded DuckDB
        |
  data/pit.py                point-in-time snapshots over the lake
```

```
  api/queries.py             a Lens: one snapshot, and the events derived from it
        |
  api/app.py                 read-only HTTP API, which also serves the built UI
        |
  ui/                        the interface (React + TypeScript), built to static files
```

`cli.py` drives it by hand; `scheduler.py` drives it on a cron from
`configs/schedule.yaml`. `data/catalogue.py` is pure metadata describing every
source, and is imported by all of the above.

`config`, `logging`, `paths`, `runtime` and `health` are leaf utilities anything
may import.

## Why these choices

**Polars, not pandas.** pandas' implicit index makes time-misalignment bugs easy to
write and invisible to review: a join that silently aligns an observation to the
wrong date looks fine. Polars has no implicit index. Enforced by a ruff
banned-import rule and by a test that ignores `# noqa`.

**DuckDB over parquet, not a database server.** Queries over market data are
analytical scans over columnar files. DuckDB does that in-process, at speed, with
no service to run, no schema migrations, and no network hop. The lake is plain
Hive-partitioned parquet, so anything else that reads parquet can read it too.

**Every row carries `as_of`, `known_at` and `ingested_at`.** `as_of` is the instant
the observation refers to; `known_at` is when it became knowable — the vintage or
publication instant for a source that revises, `as_of` for one that does not;
`ingested_at` is when we downloaded it. Point-in-time queries filter on `known_at`.
For sources that revise without publishing vintages, our own download time is the
only vintage there is, which is why recurring snapshots start as early as possible.

**Append-only.** Ingest never rewrites a file. A restatement arrives as a new row
with a later `known_at`; the superseded value stays on disk, which is what keeps
"what did we believe on date D" answerable.

**Incremental ingest has no cursor file.** The resume point is derived from the
lake itself — the newest `as_of` stored for that source and dataset — minus an
overlap window that catches late corrections. A cursor file is state that can
disagree with the data it describes.

**Failure is loud.** No fallback source, no empty frame on error, no swallowed
exception. An empty result is indistinguishable from a successful run that found
nothing, so a failed job exits non-zero instead.

**Offline mode is a hard refusal.** With `QUANTLAB_OFFLINE=true` a stray network
call raises, so anything reading the lake is provably reading the lake.

**Heartbeats, not PIDs, for health.** A deadlocked process keeps its PID.

**The app only reads.** No route writes to the lake, starts an ingest or reaches
the network, and the compose file mounts the lake read-only into it. Data arrives
by `quantlab data ingest`. That is what makes the app safe to leave running, and
it keeps one process responsible for what is in the lake.

**"As of" is decided once.** Every page is a pure function of a `Lens` -- one
point-in-time snapshot and the events derived from it -- so a query written later
cannot forget the date. A date means the END of that day in Washington, because
that is when the last thing filed on it became public. The one exception is
labelled: when the reader is in the past, the feed also returns what had already
happened and was not public yet, under its own key, and the interface draws it
only behind the timeline's curtain.

**One event shape.** A Form 4, a House report and a 13F become the same thing:
somebody did something on one day, and the world found out on another. The feed,
the ticker page and search never learn which filing a row came from.

**No UI kit, no CDN.** The interface is hand-written CSS and hand-drawn SVG, with
its fonts bundled. It looks like itself rather than like a component library, it
works offline, and it runs under a content-security policy that allows nothing
from anywhere else.

**No login yet, said plainly.** `/api/health` reports `authentication: none`,
`quantlab serve` binds to loopback and warns when asked for more, and compose
publishes to `127.0.0.1` only. A test asserts the binding.

## Containers

| Service | Image | Role |
| --- | --- | --- |
| `worker` | `quantlab-worker` | The container `make ingest` and `make shell` run in. |
| `scheduler` | `quantlab-worker` | APScheduler driving `configs/schedule.yaml`; runs each job as a subprocess so a crash cannot take the scheduler down. |
| `web` | `quantlab-web` | The app, on `127.0.0.1:8080`. Node builds the interface in a first stage and is not shipped. The lake is mounted read-only. |

DuckDB is embedded, not a service. There is no database container.

The `quantlab-data` named volume holds the lake and survives `docker compose down`
and every image rebuild. Only `make clean-data` removes it, and it asks first.

## Where to go next

- [DATA_CATALOGUE.md](DATA_CATALOGUE.md) — every source and every caveat.
- [LIMITATIONS.md](LIMITATIONS.md) — what this data cannot tell you.
- [ASSUMPTIONS.md](ASSUMPTIONS.md) — every choice made under ambiguity. Numbered,
  so a decision can be cited.
