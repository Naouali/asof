# Assumptions

Where a requirement was ambiguous, the **most conservative** reading was taken —
the one that understates strategy performance — and recorded here. Each entry names
the milestone in which it was made so it can be revisited with context.

Correct anything here that is wrong for your purposes; these are defaults, not
convictions.

## Milestone 1 — skeleton and Docker

| # | Ambiguity | Choice made | Why it is the conservative one |
| --- | --- | --- | --- |
| 1 | Python version. The spec says 3.11+. | Images pin **3.12**; `requires-python = ">=3.11,<3.13"`. | 3.12 has complete wheel coverage for the scientific stack on both architectures. The upper bound prevents a silent jump to a version some dependency has not been tested against. |
| 2 | Dependency pinning granularity. | Ranges in `pyproject.toml`, **exact pins for all 176 packages in a committed `uv.lock`**, installed with `uv sync --frozen`. | `--frozen` refuses to update the lock, so an image can never be built from a resolution that is not the committed one. CI runs `uv lock --check` so the two cannot drift. |
| 3 | What an unimplemented command should do. | Exit **2** with the milestone named. | The alternative — returning an empty result — is indistinguishable from "ran fine, found nothing", which is exactly the failure mode spec section 13 forbids. |
| 4 | Whether to enable scheduled jobs before their code exists. | All jobs disabled except `doctor`. | An enabled job that exits 2 every night trains the operator to ignore scheduler failures. |
| 5 | Default `User-Agent`. | A string containing the literal word `unconfigured`. | `doctor` catches it before the SEC does. A plausible-looking default would get silently blocked at the first EDGAR request. |
| 6 | Whether missing API keys should block startup. | No. Every key is optional; `doctor` reports what is unavailable and why. | Spec section 10 requires the stack to run with zero keys. |
| 7 | Where paper-trading state lives. | Postgres + TimescaleDB, **never bulk market data**. | Keeps the DuckDB-over-parquet research path free of a database server, per spec 3.8. A test asserts no other database engine appears in the compose file. |
| 8 | Dashboard and Jupyter authentication. | No auth; both bound to `127.0.0.1` only. | Auth that is not audited is worse than a loopback binding that is obvious. A test asserts every published port starts with `127.0.0.1:`. If you ever change that binding, add a token. |
| 9 | Postgres exposure. | Not published to the host at all. | The default password is `quantlab`. An exposed Postgres with a default password is a real liability, not a theoretical one. The override example shows how to publish it deliberately. |
| 10 | How container health is determined. | Heartbeat files whose freshness is asserted, not PID liveness. | A deadlocked worker keeps its PID. A heartbeat check catches the case that matters. |
| 11 | `PYTHONHASHSEED`. | Pinned to `0` in the images and in CI; `doctor` warns when it is unset. | Unseeded hash randomisation varies set and dict iteration order, which is enough to change floating-point reduction order and break byte-identical results. |
| 12 | Scheduler timezone. | UTC, with market-local conversions deferred to the trading calendars. | A DST shift in a local-time cron would silently move a job across a data release. |
| 13 | Whether the data catalogue is documentation or code. | Code (`src/quantlab/data/catalogue.py`); `docs/DATA_CATALOGUE.md` is generated from it and CI fails if it is stale. | A hand-maintained caveat that has drifted is worse than no caveat: it is a false assurance. |
| 14 | pandas. | Permitted only in the `research` extra for interop; **banned in `signals/`, `backtest/`, `costs/`, `portfolio/`** by a ruff rule and by a test that ignores `# noqa`. | Spec section 13. Polars' lack of an implicit index makes silent time-misalignment much harder to write. |
| 15 | Source rate limits where the provider publishes none. | A conservative self-imposed throttle per source, recorded in the catalogue (0.2–8 req/s). | Being throttled or IP-banned mid-ingest corrupts a dataset in ways that are hard to detect later. |
| 16 | Exchange settlement scraping (the only free route to a futures curve). | Catalogued, throttled to 0.5 req/s, and **disabled by default** pending a terms-of-service review. | Automated access may be prohibited. This one needs a human decision, not a default. |
