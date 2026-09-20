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

## Milestone 2 — data layer

| # | Ambiguity | Choice made | Why it is the conservative one |
| --- | --- | --- | --- |
| 17 | The spec's row contract is `as_of` + `ingested_at`. | Added a third column, **`known_at`**, and made it the point-in-time filter. `ingested_at` is kept for audit only. | Collapsing "when the world learned it" into "when we downloaded it" makes ALFRED vintages and SEC filed dates unusable — and those are the only genuine point-in-time data available for free. With one column, either historical macro backtests are impossible or they are dishonest. |
| 18 | What `as_of` means for a daily bar. | The **session close in UTC**, from the venue's trading calendar — not the session date. | A bar is not information until it has closed. A bare date makes a Tokyo close and a New York close on the "same date" look simultaneous when they are fourteen hours apart. |
| 19 | What a bare `date` means at a query boundary. | `start` widens to 00:00:00, `end` and the snapshot instant to 23:59:59.999999, all UTC. | If both widened to end-of-day, `scan(start=T, end=T)` would return nothing — a silently empty result, the exact failure this platform exists to prevent. End-of-day for the snapshot matches the spec's rebalance convention (signal on the close of T). |
| 20 | Whether an unknown exchange should default to a US calendar. | Raise `UnknownVenueError`. | Guessing a venue's hours misaligns every bar of that instrument by hours. It never raises and it always flatters. |
| 21 | Whether FRED may be used for any series. | **No.** Every series must be explicitly classified in `SERIES_POLICY` as never-revised (FRED, with a publication lag) or revised (ALFRED only). An unclassified series raises. | Pulling `GDPC1` from FRED yields a beautiful macro signal and no clue why it fails live. Forcing the classification makes the decision visible in a code review. |
| 22 | Yahoo's `adj_close` and raw prices. | Stored, and documented as restated. Total returns are to be built from `close` plus the `corporate_actions` dataset. | Yahoo restates prices for splits that happen *later* — a May 2014 Apple close is served divided by 28, covering a split six years after the bar. Returns are unaffected; price-level signals are wrong and nothing raises. |
| 23 | Stooq now serves a JavaScript proof-of-work anti-bot challenge instead of CSV. | **Do not solve it.** Detect the interstitial, fail loudly, disable the job in the shipped plan, and record the lost cross-check in LIMITATIONS.md. | It is an access control the operator deliberately deployed. Defeating it is a terms-of-service problem and gets the IP banned mid-ingest. Yahoo becomes the primary equity source and its adjustments are now unverified by an independent party — which is a real cost, stated rather than hidden. |
| 24 | How incremental ingest resumes. | From the newest `as_of` already in the lake, minus a per-job overlap window. No cursor file. | A cursor file is state that can disagree with the data it describes. The data cannot disagree with itself. The overlap is free because partition filenames are content-addressed. |
| 25 | Whether arbitrary SQL should be available to signals. | Only inside `Snapshot.sql`, on a DuckDB connection built with `enable_external_access=false` holding pre-filtered tables. | DuckDB refuses to re-enable access at runtime, so no query — however written — can reach the lake files and read around the point-in-time filter. Verified by the leakage suite. |
| 26 | Yahoo intermittently caps history at 20 years and reports `firstTradeDate` to match, so truncation is undetectable in-payload. | Request in 8-year windows (helps sometimes), verify coverage against the trading calendar, and warn when a fetch returns less history than the lake already holds. Do **not** fail the run. | Failing would make ingest unusable, since the cap is intermittent and the append-only lake means a re-run recovers the missing span. The honest cost — a first ingest can silently be short — is documented in LIMITATIONS.md with the instruction to check `data status`. |
| 27 | Whether unit tests may touch live APIs. | No. A transport-level block fails any unmarked test that opens a socket; live calls require `@pytest.mark.network`. | A test that silently hits a live API depends on a third party's uptime and on today's market data, and starts failing for reasons unrelated to the code. |
