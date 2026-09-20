<!-- GENERATED FILE -- do not edit by hand.
     Source of truth: src/quantlab/data/catalogue.py
     Regenerate:      python scripts/gen_data_catalogue.py -->

# Data catalogue

Every external source this platform can read, what it provides, and how it lies to
you. Free data is never clean; the purpose of this document is to make the ways in
which it is dirty impossible to overlook.

The same information is structured data in `src/quantlab/data/catalogue.py`, is
reported by `quantlab data catalogue`, is served at `/api/sources`, and is rendered
into the data-quality warnings panel of every tearsheet.

## How to read the point-in-time column

| Value | Meaning |
| --- | --- |
| `vintage` | The source can answer "what was known on date D". Safe for signals. |
| `as_published` | Values are never revised, so today's series equals the historical one. Safe for signals. |
| `restated` | The source serves *current* values for historical dates. **Using this in a signal is look-ahead bias**, and the point-in-time layer blocks it without an explicit override. |
| `survivorship_biased` | Only entities that still exist are retrievable. Dead tickers have vanished. |

## The three biases that matter most here

1. **Survivorship bias in equities.** It cannot be fully solved without paid CRSP.
   What the platform does instead: builds universes from historical index
   constituent and SEC filer lists rather than from tickers that exist today,
   retains every delisted ticker once observed, prefers the academic factor
   datasets where the bias would dominate, and flags residual bias on every equity
   tearsheet.
2. **Restated macro data.** FRED serves the latest vintage of every series. Any
   macro signal must read ALFRED vintages instead.
3. **No free historical options data.** The options snapshot collector accumulates
   history from the day it first runs. No options backtest before that date is
   possible; a result claiming otherwise is fabricated.


## Sources at a glance

| Source | Asset classes | Point-in-time | Key | Caveats |
| --- | --- | --- | --- | --- |
| [`alfred`](#alfred) | macro, rates | `vintage` | `fred_api_key` | 2 |
| [`alpha_vantage`](#alpha-vantage) | equity, fx, crypto | `restated` | `alpha_vantage_api_key` | 2 |
| [`binance`](#binance) | crypto | `as_published` | none | 4 |
| [`bybit`](#bybit) | crypto | `as_published` | none | 1 |
| [`cboe`](#cboe) | options | `as_published` | none | 2 |
| [`cftc_cot`](#cftc-cot) | futures, commodity | `as_published` | none | 3 |
| [`coinbase`](#coinbase) | crypto | `as_published` | none | 1 |
| [`coingecko`](#coingecko) | crypto | `restated` | none | 2 |
| [`ecb`](#ecb) | rates, fx, macro | `as_published` | none | 1 |
| [`eia`](#eia) | commodity | `restated` | `eia_api_key` | 2 |
| [`exchange_settlements`](#exchange-settlements) | futures, commodity | `as_published` | none | 3 |
| [`fmp`](#fmp) | equity | `restated` | `fmp_api_key` | 1 |
| [`frankfurter`](#frankfurter) | fx | `as_published` | none | 3 |
| [`fred`](#fred) | macro, rates, credit | `restated` | `fred_api_key` | 3 |
| [`jkp_factors`](#jkp-factors) | factors, equity | `restated` | none | 2 |
| [`ken_french`](#ken-french) | factors, equity | `restated` | none | 2 |
| [`kraken`](#kraken) | crypto | `as_published` | none | 2 |
| [`nasdaq_data_link`](#nasdaq-data-link) | equity, commodity, macro | `restated` | `nasdaq_data_link_api_key` | 1 |
| [`okx`](#okx) | crypto | `as_published` | none | 1 |
| [`open_asset_pricing`](#open-asset-pricing) | factors, equity | `restated` | none | 2 |
| [`open_bond_asset_pricing`](#open-bond-asset-pricing) | credit, factors | `restated` | none | 2 |
| [`options_snapshot`](#options-snapshot) | options | `as_published` | none | 3 |
| [`sec_edgar`](#sec-edgar) | equity, reference | `vintage` | none | 5 |
| [`stooq`](#stooq) | equity, futures, fx | `survivorship_biased` | none | 3 |
| [`tiingo`](#tiingo) | equity, crypto | `restated` | `tiingo_api_key` | 1 |
| [`treasury_fiscaldata`](#treasury-fiscaldata) | rates, macro | `as_published` | none | 2 |
| [`usda_nass`](#usda-nass) | commodity | `restated` | `usda_nass_api_key` | 2 |
| [`yahoo_futures`](#yahoo-futures) | futures, commodity | `as_published` | none | 3 |
| [`yfinance`](#yfinance) | equity, futures, fx, options | `survivorship_biased` | none | 4 |


## macro

### alfred

**ALFRED (FRED vintage archive)** — <https://fred.stlouisfed.org/docs/api/fred/series_observations.html>

| Field | Value |
| --- | --- |
| Datasets | `vintage_observations`, `vintage_dates` |
| Asset classes | macro, rates |
| Point-in-time | `vintage` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | 120 requests/minute |
| Ingest throttle | 2.0 req/s |
| Licence | Free; attribution requested |
| API key | `fred_api_key` |

**Caveats**

- Vintage coverage begins at different dates per series; some series have no archive at all, in which case no macro signal may use them.
- Vintage pulls are one request per (series, vintage) pair and are therefore slow. Ingest incrementally, not as a full refresh.

> THE source for any macro signal. Spec 3.1 marks this critical.

### fred

**FRED (Federal Reserve Bank of St. Louis)** — <https://fred.stlouisfed.org/docs/api/fred/>

| Field | Value |
| --- | --- |
| Datasets | `series_observations`, `series_metadata`, `release_calendar` |
| Asset classes | macro, rates, credit |
| Point-in-time | `restated` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | 120 requests/minute |
| Ingest throttle | 2.0 req/s |
| Licence | Free for non-commercial and commercial use; attribution requested |
| API key | `fred_api_key` |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- FRED serves the LATEST vintage of every series. For any revisable macro series this is look-ahead bias: 2008 GDP as printed today is not what was known in 2008. Use the `alfred` source for macro signals.
- Non-revisable series (VIX, daily Treasury yields, ICE BofA OAS) are safe from FRED because they are never restated -- but confirm per series.
- Release timing is not encoded in the observation date: a monthly CPI dated 2024-01-01 was published mid-February. Signals must apply the publication lag, which the PIT layer enforces via the release calendar.


## rates

### ecb

**ECB Data Portal API** — <https://data.ecb.europa.eu/help/api/overview>

| Field | Value |
| --- | --- |
| Datasets | `yield_curve`, `policy_rates`, `fx_reference_rates` |
| Asset classes | rates, fx, macro |
| Point-in-time | `as_published` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | unpublished; keep under ~2 req/s |
| Ingest throttle | 2.0 req/s |
| Licence | Free reuse with attribution |
| API key | not required |

**Caveats**

- Some ECB statistical series ARE revised; revisions are not exposed as vintages. Treat macro aggregates from here as restated.

### treasury_fiscaldata

**US Treasury FiscalData API** — <https://fiscaldata.treasury.gov/api-documentation/>

| Field | Value |
| --- | --- |
| Datasets | `daily_treasury_yield_curve`, `auction_results`, `debt_outstanding` |
| Asset classes | rates, macro |
| Point-in-time | `as_published` |
| Update frequency | daily (T+1) |
| Reliability | stable |
| Rate limit | unpublished; keep under ~5 req/s |
| Ingest throttle | 4.0 req/s |
| Licence | US Government public domain |
| API key | not required |

**Caveats**

- Yield curve rates are constant-maturity par yields, not zero rates. Bootstrapping is the caller's responsibility.
- Published with a one-business-day lag; the PIT layer must apply it.


## equity

### alpha_vantage

**Alpha Vantage (free tier)** — <https://www.alphavantage.co/documentation/>

| Field | Value |
| --- | --- |
| Datasets | `ohlcv_daily_adjusted`, `fundamentals` |
| Asset classes | equity, fx, crypto |
| Point-in-time | `restated` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | ~25 requests/day on the free tier |
| Ingest throttle | 0.2 req/s |
| Licence | Free tier for personal use |
| API key | `alpha_vantage_api_key` |

> **Cross-check only.** Free-tier limits make this unusable for primary ingest; it is used to validate other sources.

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- The daily cap makes this unusable for bulk ingest. Cross-validation of a handful of symbols only (spec 3.2).
- Fundamentals are restated, with no filed date -- banned from the signal path.

### fmp

**Financial Modeling Prep (free tier)** — <https://site.financialmodelingprep.com/developer/docs>

| Field | Value |
| --- | --- |
| Datasets | `ohlcv_daily`, `fundamentals`, `delisted_companies` |
| Asset classes | equity |
| Point-in-time | `restated` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | ~250 requests/day on the free tier |
| Ingest throttle | 0.3 req/s |
| Licence | Free tier, non-commercial |
| API key | `fmp_api_key` |

> **Cross-check only.** Free-tier limits make this unusable for primary ingest; it is used to validate other sources.

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Free-tier fundamentals are restated with no as-filed date: unusable for point-in-time work. The delisted-companies endpoint is nonetheless useful for partially repairing survivorship bias in the universe.

### nasdaq_data_link

**Nasdaq Data Link (free tables only)** — <https://data.nasdaq.com/>

| Field | Value |
| --- | --- |
| Datasets | `free_tables` |
| Asset classes | equity, commodity, macro |
| Point-in-time | `restated` |
| Update frequency | varies by table |
| Reliability | stable |
| Rate limit | 50 calls/day anonymous, 300/10s with a free key |
| Ingest throttle | 1.0 req/s |
| Licence | Per-table; several formerly free tables are now paid |
| API key | `nasdaq_data_link_api_key` |

> **Cross-check only.** Free-tier limits make this unusable for primary ingest; it is used to validate other sources.

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- The free catalogue shrinks over time; tables that worked last year may now be paid. Ingest must fail loudly on a 403 rather than skipping the table.

### sec_edgar

**SEC EDGAR structured data APIs (data.sec.gov)** — <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>

| Field | Value |
| --- | --- |
| Datasets | `companyfacts`, `companyconcept`, `submissions`, `frames`, `full_text_search` |
| Asset classes | equity, reference |
| Point-in-time | `vintage` |
| Update frequency | continuous (filings), bulk refresh nightly |
| Reliability | stable |
| Rate limit | 10 requests/second, descriptive User-Agent with contact REQUIRED |
| Ingest throttle | 8.0 req/s |
| Licence | US Government public domain |
| API key | not required |

**Caveats**

- Requests without a descriptive User-Agent carrying contact details are blocked. Set QUANTLAB_HTTP_USER_AGENT.
- XBRL TAGGING IS INCONSISTENT: pre-2012 filings and small-cap filers use non-standard or custom tags, and the same economic concept appears under several us-gaap elements. Normalisation logic is mandatory and is itself a source of error -- its coverage must be reported per fundamental.
- `filed` is the point-in-time anchor, NOT `end`. Restatements appear as new facts for an old period; the PIT layer must select by filed date and keep the superseded value visible for pre-restatement dates.
- Covers SEC filers only: no foreign private issuers filing on some forms, no pre-EDGAR history (electronic filing is broadly complete from ~1996).
- Amended filings (10-K/A) can arrive years later. Ingest must be able to add a fact with an old period without rewriting history.

> The crown jewel of free fundamentals (spec 3.2).

### stooq

**Stooq CSV endpoints** — <https://stooq.com/db/h/>

| Field | Value |
| --- | --- |
| Datasets | `ohlcv_daily` |
| Asset classes | equity, futures, fx |
| Point-in-time | `survivorship_biased` |
| Update frequency | daily |
| Reliability | scraped |
| Rate limit | undocumented; throttle hard (<=1 req/s) to avoid IP blocks |
| Ingest throttle | 1.0 req/s |
| Licence | Personal use; bulk download terms unclear -- review before redistributing |
| API key | not required |

**Caveats**

- ADJUSTMENT METHODOLOGY IS UNDOCUMENTED. Whether dividends are reinvested, and on what date, is unstated. Cross-check against yfinance adjusted closes before trusting total returns.
- Delisted tickers are not reliably retained -- survivorship bias.
- No volume for some non-US venues, which breaks the ADV input to the impact model and therefore the capacity estimate.

### tiingo

**Tiingo (free tier)** — <https://www.tiingo.com/documentation/general/overview>

| Field | Value |
| --- | --- |
| Datasets | `ohlcv_daily_adjusted`, `news` |
| Asset classes | equity, crypto |
| Point-in-time | `restated` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | ~50 symbols/hour, 1000 requests/day on the free tier |
| Ingest throttle | 0.3 req/s |
| Licence | Free tier, non-commercial; registration required |
| API key | `tiingo_api_key` |

> **Cross-check only.** Free-tier limits make this unusable for primary ingest; it is used to validate other sources.

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Free tier limits preclude universe-wide ingest; cross-check only.

### yfinance

**Yahoo Finance (via yfinance)** — <https://github.com/ranaroussi/yfinance>

| Field | Value |
| --- | --- |
| Datasets | `ohlcv_daily`, `ohlcv_intraday`, `splits_dividends`, `options_chain` |
| Asset classes | equity, futures, fx, options |
| Point-in-time | `survivorship_biased` |
| Update frequency | daily / intraday |
| Reliability | unofficial |
| Rate limit | unofficial; aggressive use gets rate-limited or blocked |
| Ingest throttle | 1.0 req/s |
| Licence | Yahoo ToS prohibit redistribution; personal research use only |
| API key | not required |

**Caveats**

- UNOFFICIAL SCRAPER. Yahoo changes its endpoints without notice and the library breaks periodically. Never the ground truth for a production number.
- SEVERELY SURVIVORSHIP-BIASED: delisted tickers disappear entirely. A universe built from what Yahoo returns today is a universe of winners.
- Adjusted closes are silently restated when Yahoo reprocesses corporate actions, so a backtest re-run months later can produce different numbers. This is why ingested prices are snapshotted, not re-fetched.
- Intraday history is capped (roughly 60 days at 1m), so intraday equity research cannot be backfilled.

> Treat as convenience and cross-check, not ground truth (spec 3.2).


## futures

### cftc_cot

**CFTC Commitments of Traders** — <https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm>

| Field | Value |
| --- | --- |
| Datasets | `legacy`, `disaggregated`, `tff` |
| Asset classes | futures, commodity |
| Point-in-time | `as_published` |
| Update frequency | weekly (Tuesday snapshot, released Friday 15:30 ET) |
| Reliability | stable |
| Rate limit | bulk annual zip files; a handful of requests |
| Ingest throttle | 2.0 req/s |
| Licence | US Government public domain |
| API key | not required |

**Caveats**

- THREE-DAY PUBLICATION LAG on a Tuesday snapshot. Using the Tuesday value before Friday's release is look-ahead bias, and it is the single most common error in published COT research. The PIT layer applies the lag.
- Reclassifications between trader categories create level shifts that look like signal; positioning must be normalised within a regime, not across one.
- Weekly frequency caps the achievable Sharpe of any COT signal.

### exchange_settlements

**Exchange daily settlement files (CME / ICE public sections)** — <https://www.cmegroup.com/market-data/>

| Field | Value |
| --- | --- |
| Datasets | `daily_settlements`, `open_interest` |
| Asset classes | futures, commodity |
| Point-in-time | `as_published` |
| Update frequency | daily, after settlement |
| Reliability | scraped |
| Rate limit | self-imposed: <=0.5 req/s, business hours, cache everything |
| Ingest throttle | 0.5 req/s |
| Licence | CHECK TERMS OF SERVICE BEFORE USE. Scrape politely and identify yourself. |
| API key | not required |

**Caveats**

- The ONLY free route to a multi-contract futures curve, which carry and basis-momentum signals require (spec 3.4). Availability and file format change without notice and history is typically short -- often only recent files are posted, so the curve archive starts when WE start collecting.
- Settlement prices are not tradable prices: they are exchange-determined and can differ materially from the last trade in illiquid deferred contracts.
- Terms of service govern automated access and may forbid it. This source is disabled by default and must be enabled deliberately.

### yahoo_futures

**Yahoo continuous futures (=F tickers)** — <https://finance.yahoo.com/commodities>

| Field | Value |
| --- | --- |
| Datasets | `continuous_front_month` |
| Asset classes | futures, commodity |
| Point-in-time | `as_published` |
| Update frequency | daily |
| Reliability | unofficial |
| Rate limit | unofficial; <=1 req/s |
| Ingest throttle | 1.0 req/s |
| Licence | Yahoo ToS; personal research use only |
| API key | not required |

**Caveats**

- ROLL METHODOLOGY IS OPAQUE. The series is stitched by an unstated rule with unstated (probably no) back-adjustment, so returns across a roll date are not tradable returns. Adequate for trend research after roll-date filtering; NOT usable for carry, which needs the curve (spec 3.4).
- Gaps and stale prints around holidays and low-liquidity sessions.
- Only the front month: no second contract, hence no term structure.


## commodity

### eia

**EIA API v2** — <https://www.eia.gov/opendata/documentation.php>

| Field | Value |
| --- | --- |
| Datasets | `petroleum_stocks`, `natural_gas_storage`, `production`, `refining` |
| Asset classes | commodity |
| Point-in-time | `restated` |
| Update frequency | weekly / monthly |
| Reliability | stable |
| Rate limit | unpublished; keep under ~5 req/s |
| Ingest throttle | 4.0 req/s |
| Licence | US Government public domain |
| API key | `eia_api_key` |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- EIA REVISES weekly inventories and monthly production, and the API serves the revised value with no vintage archive. A backtest that uses the current value on the original release date has look-ahead. Mitigation: snapshot each release as ingested and use OUR OWN ingested_at as the vintage -- which only works from the day we start collecting. Pre-collection history is restated and is flagged as such.
- Release timestamps (Wed 10:30 ET crude, Thu 10:30 ET gas) matter for any signal faster than weekly.

### usda_nass

**USDA NASS QuickStats / WASDE** — <https://quickstats.nass.usda.gov/api>

| Field | Value |
| --- | --- |
| Datasets | `production`, `stocks`, `exports`, `wasde` |
| Asset classes | commodity |
| Point-in-time | `restated` |
| Update frequency | monthly / quarterly |
| Reliability | stable |
| Rate limit | unpublished; large queries are rejected -- page by year |
| Ingest throttle | 2.0 req/s |
| Licence | US Government public domain |
| API key | `usda_nass_api_key` |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Heavily revised, no vintage archive. Same ingested_at mitigation as EIA.
- WASDE is a PDF/XLS release; parsing is brittle and must be validated per release rather than assumed stable.


## fx

### frankfurter

**Frankfurter (ECB reference rates)** — <https://frankfurter.dev/>

| Field | Value |
| --- | --- |
| Datasets | `fx_spot_daily` |
| Asset classes | fx |
| Point-in-time | `as_published` |
| Update frequency | daily, ~16:00 CET |
| Reliability | stable |
| Rate limit | unpublished; be polite |
| Ingest throttle | 2.0 req/s |
| Licence | Free, open source; no key |
| API key | not required |

**Caveats**

- A SINGLE DAILY FIX (the ECB 14:15 CET reference rate), not a tradable quote. There is no bid/ask and no intraday path, so execution realism is unavailable: FX research is restricted to daily-or-slower signals and FX spread costs must be assumed, not measured (spec 3.5).
- Rates are quoted against EUR; crosses are computed and inherit two fixes' worth of non-synchronicity.
- No weekend or TARGET-holiday observations.


## credit

### open_bond_asset_pricing

**Open Source Bond Asset Pricing (error-corrected TRACE)** — <https://openbondassetpricing.com/>

| Field | Value |
| --- | --- |
| Datasets | `bond_returns`, `bond_factors` |
| Asset classes | credit, factors |
| Point-in-time | `restated` |
| Update frequency | periodic |
| Reliability | stable |
| Rate limit | static files |
| Ingest throttle | 1.0 req/s |
| Licence | Free for research with citation |
| API key | not required |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Essential: uncorrected TRACE-based bond returns contain large data errors that inflate published bond factor performance. A substantial fraction of published corporate bond factors do not survive the correction, and any credit signal built here must surface that (spec 4, Tier 2).
- Corporate bonds trade thinly; reported returns often reflect stale or matrix prices, so realised transaction costs dwarf equity-style assumptions.


## crypto

### binance

**Binance public REST** — <https://developers.binance.com/docs/binance-spot-api-docs>

| Field | Value |
| --- | --- |
| Datasets | `spot_klines`, `perp_klines`, `funding_rate`, `open_interest`, `mark_index_price` |
| Asset classes | crypto |
| Point-in-time | `as_published` |
| Update frequency | realtime |
| Reliability | stable |
| Rate limit | weight-based (~1200 weight/minute per IP); 429 then IP ban on abuse |
| Ingest throttle | 8.0 req/s |
| Licence | Free public endpoints, no key; commercial use terms apply |
| API key | not required |

**Caveats**

- SURVIVORSHIP AND LISTING BIAS: delisted symbols stop being served, and a universe of 'symbols trading today' is a universe of survivors. Retain every symbol once observed and build the universe from historical exchangeInfo.
- Funding history begins at each contract's listing date, and the funding formula/cap has changed over time -- a funding-carry backtest spanning a regime change is comparing different instruments.
- Klines are exchange-local: cross-venue basis research must align timestamps and account for differing settlement conventions.
- Geographic restrictions apply to some endpoints from some jurisdictions, and a 451 must fail loudly, not silently return empty.

> Full funding history, open interest and order books make crypto the place to build and validate the microstructure machinery (spec 3.6).

### bybit

**Bybit public API** — <https://bybit-exchange.github.io/docs/v5/intro>

| Field | Value |
| --- | --- |
| Datasets | `klines`, `funding_rate`, `open_interest` |
| Asset classes | crypto |
| Point-in-time | `as_published` |
| Update frequency | realtime |
| Reliability | stable |
| Rate limit | per-endpoint IP limits; back off on 403/429 |
| Ingest throttle | 5.0 req/s |
| Licence | Free public endpoints |
| API key | not required |

**Caveats**

- Funding interval differs by contract; do not assume 8h across venues.

### coinbase

**Coinbase Exchange public API** — <https://docs.cdp.coinbase.com/exchange/docs/welcome>

| Field | Value |
| --- | --- |
| Datasets | `candles`, `trades`, `book` |
| Asset classes | crypto |
| Point-in-time | `as_published` |
| Update frequency | realtime |
| Reliability | stable |
| Rate limit | ~10 req/s public |
| Ingest throttle | 5.0 req/s |
| Licence | Free public endpoints |
| API key | not required |

**Caveats**

- Candle requests are capped at 300 buckets; paginate.

### coingecko

**CoinGecko (free tier)** — <https://docs.coingecko.com/reference/introduction>

| Field | Value |
| --- | --- |
| Datasets | `market_cap`, `volume`, `listing_metadata` |
| Asset classes | crypto |
| Point-in-time | `restated` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | ~5-15 requests/minute on the keyless tier |
| Ingest throttle | 0.2 req/s |
| Licence | Free demo tier; attribution required |
| API key | not required |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Market-cap history uses TODAY's circulating supply for past dates in some endpoints, which silently restates the cross-section. Snapshot daily and use our own ingested_at as the vintage for any size-sorted crypto signal.
- Coin lists are survivorship-biased: dead coins are removed.

### kraken

**Kraken public API** — <https://docs.kraken.com/api/>

| Field | Value |
| --- | --- |
| Datasets | `ohlc`, `trades`, `spread` |
| Asset classes | crypto |
| Point-in-time | `as_published` |
| Update frequency | realtime |
| Reliability | stable |
| Rate limit | counter-based; ~1 req/s sustained for public endpoints |
| Ingest throttle | 1.0 req/s |
| Licence | Free public endpoints |
| API key | not required |

**Caveats**

- OHLC history is truncated to the last ~720 candles per interval: long history is only available by accumulating our own snapshots.
- Kraken asset codes are non-standard (XXBT, ZUSD) and need normalisation.

### okx

**OKX public API** — <https://www.okx.com/docs-v5/en/>

| Field | Value |
| --- | --- |
| Datasets | `klines`, `funding_rate`, `open_interest` |
| Asset classes | crypto |
| Point-in-time | `as_published` |
| Update frequency | realtime |
| Reliability | stable |
| Rate limit | per-endpoint; typically 20 req/2s |
| Ingest throttle | 5.0 req/s |
| Licence | Free public endpoints |
| API key | not required |

**Caveats**

- Historical kline depth is limited per request; paginate carefully.


## options

### cboe

**CBOE free historical data** — <https://www.cboe.com/tradable_products/vix/vix_historical_data/>

| Field | Value |
| --- | --- |
| Datasets | `vix_history`, `vix_term_structure`, `index_settlements` |
| Asset classes | options |
| Point-in-time | `as_published` |
| Update frequency | daily |
| Reliability | stable |
| Rate limit | static CSVs; cache |
| Ingest throttle | 1.0 req/s |
| Licence | Free for personal use; redistribution restricted |
| API key | not required |

**Caveats**

- VIX methodology changed in 2003 (and the pre-2003 VXO is a different index): a series spanning the change is not one instrument.
- Index level only -- no option-level greeks, so any delta-hedged strategy must be approximated rather than simulated.

### options_snapshot

**Self-collected options chain snapshots (via yfinance)** — <https://github.com/ranaroussi/yfinance>

| Field | Value |
| --- | --- |
| Datasets | `chain_snapshot` |
| Asset classes | options |
| Point-in-time | `as_published` |
| Update frequency | daily snapshot, accumulating forward |
| Reliability | unofficial |
| Rate limit | <=1 req/s |
| Ingest throttle | 1.0 req/s |
| Licence | Yahoo ToS; personal research use only |
| API key | not required |

**Caveats**

- HISTORY STARTS THE DAY WE START COLLECTING. There is no free historical options chain data, so NO options backtest before the collector's first run is possible (spec 3.7). Any result claiming otherwise is fabricated.
- A single daily snapshot has no intraday path, and quotes may be stale or crossed in illiquid strikes. Filter on non-zero volume/open interest and treat the mid as indicative only.
- Implied volatilities reported by Yahoo use an unstated model and dividend assumption; recompute from mid prices rather than trusting the field.


## factors

### jkp_factors

**Global Factor Data (Jensen, Kelly & Pedersen)** — <https://jkpfactors.com/>

| Field | Value |
| --- | --- |
| Datasets | `factor_returns`, `cluster_classification` |
| Asset classes | factors, equity |
| Point-in-time | `restated` |
| Update frequency | periodic |
| Reliability | stable |
| Rate limit | static files behind a login |
| Ingest throttle | 1.0 req/s |
| Licence | Free with registration; academic citation required |
| API key | not required |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Requires registration, so automated ingest may need a manually placed file. Ingest must fail loudly with instructions rather than silently skipping.
- Country coverage is very uneven early in the sample; small markets have few names and the factor spreads there are not tradable.

### ken_french

**Kenneth R. French Data Library** — <https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

| Field | Value |
| --- | --- |
| Datasets | `ff3`, `ff5`, `momentum`, `industry_portfolios`, `international` |
| Asset classes | factors, equity |
| Point-in-time | `restated` |
| Update frequency | monthly |
| Reliability | stable |
| Rate limit | static files; cache aggressively |
| Ingest throttle | 1.0 req/s |
| Licence | Free for research use; attribution expected |
| API key | not required |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Survivorship-bias-free and the accepted benchmark, but the whole history is rebuilt on each release, so the file is restated: it answers 'what do we now believe HML returned in 1965', not 'what was known then'. Fine as a benchmark and as a risk-model factor; not a point-in-time signal input.
- Returns are gross of transaction costs and assume free shorting.

> Acceptance test for our own factor construction: a hand-built momentum factor must correlate >0.9 with UMD or the construction has a bug (spec 3.3).

### open_asset_pricing

**Open Source Asset Pricing (Chen & Zimmermann)** — <https://www.openassetpricing.com/>

| Field | Value |
| --- | --- |
| Datasets | `predictor_portfolios`, `signal_documentation` |
| Asset classes | factors, equity |
| Point-in-time | `restated` |
| Update frequency | annual releases |
| Reliability | stable |
| Rate limit | static files |
| Ingest throttle | 1.0 req/s |
| Licence | Free for research with citation |
| API key | not required |

> **Restated data.** Blocked from the signal path by the point-in-time layer unless explicitly overridden.

**Caveats**

- Portfolio returns are reconstructions of published anomalies built on CRSP/Compustat, standardised by the authors -- they are not what the original papers reported, and differences are informative rather than errors.
- Gross of costs, and several predictors are microcap-dominated: a headline spread can be untradable at any size. Always pair with the capacity model.


## Sources deliberately not used

| Source | Why not |
| --- | --- |
| Bloomberg, Refinitiv, FactSet | Paid. Excluded by the project's hard constraints. |
| CRSP, Compustat | Paid. Their absence is the direct cause of the residual survivorship bias documented above. |
| Paid TRACE feeds | Paid. The error-corrected academic bond dataset is used instead. |
| Any broker execution API | The platform is research and paper trading only; there is no live order routing by design. |
