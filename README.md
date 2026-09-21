# asof

Who traded, and when the world could first have known.

Company insiders, large funds and members of Congress all have to disclose
their trades — days, weeks or months after making them. **asof** collects those
disclosures from their official sources, keeps both dates for every one of them,
and shows the result as a feed, a timeline and a price chart that can be wound
back to any past day to see exactly what was public then and what was not.

It is two things in one repository:

- **The app** — a read-only web interface over the disclosure data. See
  [The app](#the-app).
- **The ETL underneath it** — twenty-two fetchers over seventeen free sources
  (disclosures, prices, fundamentals, macro, crypto, options), landing in a local
  parquet lake that DuckDB queries in-process. Everything the app needs is
  fetched without an API key.

The package and its command are still called `quantlab`; they are being renamed
to `asof`. Every command below is written as it works today.

Two properties are enforced structurally rather than left to discipline:

| Concern | How it is enforced |
| --- | --- |
| Point-in-time | Every row records `as_of` (what it describes) and `known_at` (when it became knowable). The lake is append-only: a revision is a new row, so "what did we believe on date D" stays answerable |
| Data quality | Every source's known biases are structured data in the catalogue, reported by `quantlab data catalogue` and rendered into [docs/DATA_CATALOGUE.md](docs/DATA_CATALOGUE.md) |

A fetcher that cannot do its job fails loudly. Nothing here substitutes a fallback
source, skips a failed job silently, or returns an empty frame to mean "error".

## Sources

| Fetcher | Dataset | Asset class | Point-in-time | Key |
| --- | --- | --- | --- | --- |
| `yahoo.ohlcv_daily` | daily bars | equity | survivorship-biased | none |
| `yahoo.corporate_actions` | splits and dividends | equity | survivorship-biased | none |
| `stooq.ohlcv_daily` | daily bars | equity | survivorship-biased | none — currently blocked by an anti-bot challenge |
| `sec_edgar.fundamentals` | XBRL company facts, by filing date | equity | vintage | none (needs a real `User-Agent`) |
| `binance.ohlcv_bars` | spot bars | crypto | as published | none |
| `binance.funding_rate` | perpetual funding | crypto | as published | none |
| `binance.instruments` | traded-universe snapshot | crypto | as published | none |
| `fred.series_observations` | macro and rates, latest vintage | macro | restated | `QUANTLAB_FRED_API_KEY` |
| `alfred.series_observations` | macro and rates, every vintage | macro | vintage | `QUANTLAB_FRED_API_KEY` |
| `eia.series_observations` | US energy inventories | commodity | restated | `QUANTLAB_EIA_API_KEY` |
| `cftc_cot.positioning` | Commitments of Traders | futures | as published | none |
| `cboe.series_observations` | VIX-family indices | options | as published | none |
| `options_snapshot.chain_snapshot` | delayed option chains | options | as published | none |
| `sec_insider.insider_transactions` | insider trades (Forms 4 and 5) | equity | vintage | none (needs a real `User-Agent`) |
| `sec_13f.institutional_holdings` | quarterly holdings of chosen managers (13F) | equity | vintage | none (needs a real `User-Agent`) |
| `sec_ftd.fails_to_deliver` | fails-to-deliver balances; the CUSIP→ticker bridge | equity | as published | none (needs a real `User-Agent`) |
| `house_clerk.congress_trades` | trades disclosed by members of the US House | equity | vintage | none |
| `house_clerk.congress_filings` | index of every House disclosure document | reference | vintage | none |
| `senate_efd.congress_trades` | trades disclosed by US senators, read through this repository's mirror | equity | vintage | none |
| `senate_efd.congress_filings` | index of Senate transaction reports, paper ones included | reference | vintage | none |
| `ken_french.series_observations` | factor returns | factors | restated | none |
| `open_asset_pricing.anomaly_catalogue` | published anomaly catalogue | reference | restated | none |

`quantlab data fetchers` prints this list with whether each can run right now.
The catalogue also documents sources that have no fetcher yet.

## Quick start

Docker is the only prerequisite.

```bash
git clone <this repo> && cd QuantBox
cp .env.example .env     # optional: the stack runs unchanged, with zero API keys
make up                  # builds images, starts worker + scheduler + web, runs doctor
make ingest              # the app is empty until there is data to read
```

The app is then at <http://127.0.0.1:8080>.

```bash
make doctor        # what this installation can and cannot do
make catalogue     # every data source, its availability, its known biases
make fetchers      # every fetcher and whether it can run right now
make ingest        # full historical pull into the lake (works with zero API keys)
make ingest-daily  # incremental update
make status        # what the lake holds and how stale it is
make test
make down
```

`make help` lists every target.

Without Docker, in a Python 3.11–3.12 environment:

```bash
uv sync
uv run quantlab init
uv run quantlab data ingest --dry-run     # show the windows without fetching
uv run quantlab data ingest
```

## What gets ingested

[configs/ingest.yaml](configs/ingest.yaml) is the ingest plan: one job per fetcher,
with its symbols, start date and options. Symbol lists are deliberately small —
widen them once you know what you need.

```bash
quantlab data ingest                              # every job in the plan
quantlab data ingest --incremental                # resume from what the lake holds
quantlab data ingest -f binance.ohlcv_bars        # one fetcher
quantlab data ingest --dry-run                    # show the windows, fetch nothing
```

The command exits non-zero if any job failed. Jobs skipped for a missing API key
are listed but are not failures.

The `scheduler` service runs the recurring jobs in
[configs/schedule.yaml](configs/schedule.yaml): a weekday incremental pull, an
hourly crypto pull, a daily pull of insider, 13F and congressional disclosures,
and a daily options-chain snapshot. **Leave the options
snapshot on** — no free source sells historical option chains, so that dataset's
history starts the day the job first runs and cannot be backfilled.

## Querying the lake

```bash
# Inspection: sees everything, including data nobody could have known at the time.
quantlab data query "select symbol, count(*) from ohlcv_daily group by 1"

# Point-in-time: only what was knowable on 2020-03-16, in a sandbox that cannot
# reach the lake files at all.
quantlab data query --as-of 2020-03-16 -d ohlcv_daily \
  "select symbol, max(as_of) from ohlcv_daily group by 1"
```

In Python:

```python
from quantlab.data.store import Store

store = Store()
store.sql("select * from ohlcv_daily limit 5")   # everything

snapshot = store.as_of("2020-03-16")
bars = snapshot.ohlcv_daily(symbols=["SPY"])     # nothing after 2020-03-16
snapshot.ohlcv_daily(end="2020-06-01")           # raises LookAheadError
```

The lake is plain parquet under `data/lake/`, partitioned by
source / dataset / asset class / year, so anything that reads parquet can read it.

## Disclosed trades: insiders, whales and politicians

Five sources cover who is buying and selling: company insiders (SEC Forms 4 and
5), large managers such as Berkshire or Bridgewater (SEC Form 13F), members of the
US House and the US Senate (STOCK Act reports), and fails-to-deliver balances. None of them observes
a trade. Each is a disclosure filed days to months later, so `as_of` is when the
trade happened and `known_at` is when anyone outside could first have known —
always query these with `--as-of`.

```bash
quantlab data ingest -f sec_insider.insider_transactions -f sec_13f.institutional_holdings \
                     -f sec_ftd.fails_to_deliver \
                     -f house_clerk.congress_filings -f house_clerk.congress_trades \
                     -f senate_efd.congress_filings -f senate_efd.congress_trades
```

```bash
# Open-market insider trades only. Grants (A), option exercises (M) and tax
# withholding (F) are most of what is filed, and say nothing about the stock.
quantlab data query --as-of 2026-09-20 -d insider_transactions \
  "select symbol, as_of::date traded, known_at::date disclosed, owner_name, officer_title,
          transaction_code, shares, price, planned_10b5_1
   from insider_transactions
   where transaction_code in ('P', 'S') and not is_derivative order by known_at desc"

# A manager's latest KNOWABLE portfolio, with tickers. A 13F knows securities by
# CUSIP only; fails_to_deliver is the free bridge to a ticker.
quantlab data query --as-of 2026-09-20 -d institutional_holdings -d fails_to_deliver \
  "with bridge as (select cusip, mode(symbol) ticker from fails_to_deliver group by cusip)
   select h.issuer_name, b.ticker, h.shares, h.value_usd
   from institutional_holdings h left join bridge b using (cusip)
   where h.symbol = '1067983'
     and h.as_of = (select max(as_of) from institutional_holdings where symbol = '1067983')
   order by h.value_usd desc"

# What Congress disclosed, and how late. `chamber` is 'house' or 'senate'.
quantlab data query --as-of 2026-09-20 -d congress_trades \
  "select chamber, member, symbol, transaction_type, amount_text, as_of::date traded,
          known_at::date disclosed, date_diff('day', as_of, known_at) days_late
   from congress_trades where symbol <> 'NO_TICKER' order by known_at desc"

# House transaction reports that could NOT be read (scanned paper filings).
# These members' trades are absent from congress_trades, not zero. Bounded to the
# period congress_trades covers: the index is cheap and the plan ingests it from
# 2015, the PDFs only from 2025, and a report nobody tried to read is not a scan.
quantlab data query --as-of 2026-09-20 -d congress_filings -d congress_trades \
  "select f.last_name, f.symbol district, f.as_of::date filed, f.url
   from congress_filings f
   where f.filing_type = 'P'
     and f.known_at >= (select min(known_at) from congress_trades)
     and f.doc_id not in (select doc_id from congress_trades)"
```

Managers are chosen in [configs/ingest.yaml](configs/ingest.yaml) by SEC CIK,
written without leading zeros (Berkshire Hathaway is `1067983`).

**The Senate comes through a mirror.** Its disclosure site answers only connections
from inside the United States. A daily GitHub Actions job
([senate-mirror.yml](.github/workflows/senate-mirror.yml)) reads it from a US-hosted
runner and commits what it saw to the `senate-mirror` branch of this repository,
and the Senate fetchers read that branch by default. On a US machine, set
`options: {direct: true}` on the two Senate jobs to read the Senate itself. A fork
must point `mirror_url` at its own branch and run the workflow once by hand.

Read the "Disclosed
trades" section of [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before drawing
conclusions from any of this.

## The app

**asof** is a read-only web app over the disclosure data: insiders, funds and
members of the House. It is built around the one thing that data is about -- the
gap between the day somebody traded and the day anyone else could know.

- **Feed.** Every disclosure, ordered by the day it became public. Pick one and
  its record opens beside the list: what happened, the chain from the trade to
  the legal deadline to the filing, and what else that person and that company
  have disclosed. Grants, option exercises and pre-scheduled sales are hidden
  until asked for, because they are most of what is filed and none of it is a
  view on the stock. The arrow keys walk the list.
- **As of.** One control, top right, moves the whole app to the end of any past
  day. Nothing disclosed later is shown, anywhere. The date lives in the URL, so
  a view of the past can be bookmarked and sent to a colleague.
- **Timeline.** The same feed as a chart: each disclosure a line from the trade
  to the day it became public. Travel back in time and a curtain is drawn at the
  as-of date, with a second block above it -- trades that had *already happened*
  and that nobody outside could see yet.
- **Ticker pages.** The price, with every disclosure drawn as a chord from the
  trade date to the disclosure date, so the slope is what the price did before
  anyone could act. Below it: every disclosure naming the ticker, the tracked
  funds that hold it (joined through the CUSIP bridge, with how stale that is),
  and the settlement fails.
- **Data health.** What the lake holds, how fresh it is, and which House reports
  were scans that could not be read.
- **Landing page**, at `/welcome`. One screen, no scrolling, and no mock-ups: the
  three panels run on the real lake. Drag the as-of line across a timeline of
  trades and watch the count of those not yet public change; read a price chart
  with its disclosure chords; adjust the weights of a portfolio estimated from a
  member's disclosed trades. Its buttons deliberately do nothing yet — there is
  no sign-up flow behind them.

```bash
make up                         # with Docker: http://127.0.0.1:8080
make ui && make serve           # without: needs Node 22 and the Python environment
make ui-dev                     # interface with hot reload on :5173, against `make serve`
```

The API is documented at `/api/docs`. It only reads: no route writes to the lake,
starts an ingest or touches the network.

> **There is no login yet.** Anyone who can reach the port can read everything.
> The defaults keep that to this machine -- `quantlab serve` binds to loopback, the
> compose file publishes to `127.0.0.1` only and mounts the lake read-only -- and
> `quantlab serve --host` says so out loud when asked for anything wider. Put it
> behind something that authenticates before exposing it. Team features that
> need to know who you are (notes, alerts, saved views) are deliberately absent
> rather than faked.

### Not built yet

These are where the product is going. None of them exists today, and the app
does not pretend otherwise.

- **Chart information.** Richer price charts on the ticker page.
- **Cloning a portfolio.** Rebuild what a member of Congress appears to hold from
  their disclosed trades, then adjust it and keep it as your own. Only the
  preview on the landing page exists, and it is an estimate: the House discloses
  value ranges, not amounts, and never a starting position.
- **Login and teams.** Accounts, shared notes, alerts and saved views.
- **Pages for a person and for a fund.** Today only a ticker has its own page.

## Running with no API keys

The stack starts and runs with none. Keyless sources cover Yahoo prices, SEC EDGAR
point-in-time fundamentals, the Ken French and Open Source Asset Pricing libraries,
CFTC positioning, CBOE volatility and Binance.

A free FRED key unlocks macro and rates, including the ALFRED vintage archive —
the only way to read a macro series as it was known at the time. `quantlab doctor`
names every unavailable source and the environment variable that would enable it.

Before ingesting SEC EDGAR, set `QUANTLAB_HTTP_USER_AGENT` to something carrying a
real contact address. The SEC requires it, and `doctor` warns while it is a
placeholder.

## Adding a source

1. Describe it in [src/quantlab/data/catalogue.py](src/quantlab/data/catalogue.py):
   URL, licence, rate limit, point-in-time quality, and every known caveat.
2. Add a module under [src/quantlab/data/sources/](src/quantlab/data/sources/) with
   a `Source` subclass decorated with `@register`, whose
   `fetch(symbols, start, end)` returns a frame conforming to a schema in
   [schemas.py](src/quantlab/data/schemas.py). Set `known_at` honestly, raise on
   failure, never fall back to another source.
3. List the module in `SOURCE_MODULES` in
   [sources/\_\_init\_\_.py](src/quantlab/data/sources/__init__.py).
4. Record a real payload under `tests/fixtures/` and test the parser against it —
   the unit suite blocks all network access.
5. Add a job to `configs/ingest.yaml`, then run
   `python scripts/gen_data_catalogue.py` to refresh the catalogue document.

## Developing

```bash
make check         # everything CI runs for Python: lint, type-check, tests
make ui-check      # type-check and test the interface
make format        # auto-format (writes to the working tree)
make lock          # rewrite uv.lock after editing pyproject.toml
```

Those run in Docker. In a local environment, run the tests as
`python -m pytest`, not bare `pytest`: the test modules import shared helpers
from `tests.conftest`, which needs the repository root on the import path. The
unit suite blocks all network access; the Docker integration tests are excluded
unless asked for.

## Layout

```
ui/                 the interface: React + TypeScript, no UI kit, built by Vite
  src/pages/        feed, ticker, data health, landing
  src/components/   the shell, as-of control, lag bar, record, timeline, price chart
  src/styles.css    every style in the app, hand-written
src/quantlab/
  api/              the web app: read-only API over the lake, serves ui/dist
    app.py          routes, security headers, one cached lens per as-of date
    events.py       one shape for three kinds of disclosure
    queries.py      what each page asks of the lake, through one as-of "lens"
    models.py       the shapes the API returns
  cli.py            the `quantlab` command
  scheduler.py      cron-style runner for recurring ingests
  runtime.py        heartbeats, so a healthcheck can tell a stuck process from a live one
  health.py         `quantlab doctor`
  config.py         settings, read from QUANTLAB_* environment variables
  paths.py          where the lake and state live, in a container or on a laptop
  logging.py        structured logs; a degraded ingest is an explicit event
  data/
    sources/        one module per external API (sec_filings.py is shared EDGAR plumbing)
    ingest.py       ingest plans and incremental windows
    http.py         rate-limited, retrying HTTP client with a hard offline mode
    schemas.py      canonical dataset schemas
    store.py        the parquet lake and DuckDB queries
    pit.py          point-in-time snapshots
    calendars.py    trading calendars
    catalogue.py    source metadata and caveats
configs/
  ingest.yaml       what to fetch
  schedule.yaml     when to fetch it
docker/             the base, worker and web images
scripts/            gen_data_catalogue.py, which writes docs/DATA_CATALOGUE.md
tests/
  unit/             offline, against recorded payloads in tests/fixtures/
  integration/      against the running Docker stack
```

## Documentation

- [docs/DATA_CATALOGUE.md](docs/DATA_CATALOGUE.md) — every source and every caveat (generated)
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — what this data cannot tell you
- [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) — the choices made under ambiguity
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — why it is built this way
