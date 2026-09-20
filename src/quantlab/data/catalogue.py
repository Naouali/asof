"""The data catalogue: every source, what it gives us, and how it lies to us.

This registry is the single source of truth for three things:

1. **Availability.** ``quantlab doctor`` reads it to report which sources work
   right now, given the API keys actually present. The stack must run with none
   (spec section 10), so keyless sources are marked as such.
2. **Caveats.** Free data is never clean, and this does not pretend otherwise.
   Every known bias is recorded here as structured data, reported by
   ``quantlab data catalogue`` and rendered into ``docs/DATA_CATALOGUE.md``. A
   caveat that only exists in a docstring does not reach the person consuming
   the data, so it goes here.
3. **Point-in-time honesty.** ``pit_quality`` states whether a source can answer
   "what was known on date D". Sources marked :data:`PitQuality.RESTATED` are
   forbidden from the signal path without an explicit override, because they
   serve today's restated values for historical dates.

Nothing in this module performs I/O. It is pure metadata, importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

__all__ = [
    "SOURCES",
    "AssetClass",
    "PitQuality",
    "SourceSpec",
    "get_source",
    "sources_for",
]


class AssetClass(str, Enum):
    MACRO = "macro"
    RATES = "rates"
    EQUITY = "equity"
    FUTURES = "futures"
    COMMODITY = "commodity"
    FX = "fx"
    CREDIT = "credit"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FACTORS = "factors"
    REFERENCE = "reference"


class PitQuality(str, Enum):
    """How well a source supports point-in-time reconstruction."""

    VINTAGE = "vintage"
    """Source publishes genuine vintages: we can ask what was known on date D.
    ALFRED macro vintages and SEC XBRL `filed` dates are the two examples."""

    AS_PUBLISHED = "as_published"
    """Values are not revised after publication, so today's series equals the
    historical one. True of market prices and of crypto funding history."""

    RESTATED = "restated"
    """Source serves current values for historical dates. Using these in a signal
    is look-ahead bias. Blocked from the signal path by the PIT layer."""

    SURVIVORSHIP_BIASED = "survivorship_biased"
    """Only entities that still exist are retrievable. Dead tickers vanish."""


Reliability = Literal["stable", "unofficial", "scraped"]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Metadata for one external data source."""

    key: str
    name: str
    url: str
    asset_classes: tuple[AssetClass, ...]
    datasets: tuple[str, ...]
    pit_quality: PitQuality
    update_frequency: str
    licence: str
    reliability: Reliability
    caveats: tuple[str, ...]
    #: Name of the :class:`~quantlab.config.Settings` field holding the API key,
    #: or ``None`` when the source needs no credentials.
    key_setting: str | None = None
    #: Human-readable rate limit, enforced by the ingest scheduler.
    rate_limit: str = "unspecified -- be polite"
    #: Requests per second the ingester will not exceed.
    max_requests_per_second: float = 2.0
    #: Free-tier sources we use only for cross-validation, never as primary ingest.
    cross_check_only: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def requires_key(self) -> bool:
        return self.key_setting is not None

    @property
    def point_in_time_safe(self) -> bool:
        """Whether history read from this source is what was known at the time.

        False for a restated source: it serves today's values for past dates, so
        anything computed on its history quietly knows the future.
        """
        return self.pit_quality is not PitQuality.RESTATED


# ======================================================================================
# 3.1 Macro, rates and reference data
# ======================================================================================
_MACRO: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="fred",
        name="FRED (Federal Reserve Bank of St. Louis)",
        url="https://fred.stlouisfed.org/docs/api/fred/",
        asset_classes=(AssetClass.MACRO, AssetClass.RATES, AssetClass.CREDIT),
        datasets=("series_observations", "series_metadata", "release_calendar"),
        # FRED itself serves the *current* vintage of a revised series. Anything
        # revisable (GDP, payrolls, CPI) must be pulled from ALFRED instead.
        pit_quality=PitQuality.RESTATED,
        update_frequency="daily",
        licence="Free for non-commercial and commercial use; attribution requested",
        reliability="stable",
        key_setting="fred_api_key",
        rate_limit="120 requests/minute",
        max_requests_per_second=2.0,
        caveats=(
            "FRED serves the LATEST vintage of every series. For any revisable macro "
            "series this is look-ahead bias: 2008 GDP as printed today is not what was "
            "known in 2008. Use the `alfred` source for macro signals.",
            "Non-revisable series (VIX, daily Treasury yields, ICE BofA OAS) are safe "
            "from FRED because they are never restated -- but confirm per series.",
            "Release timing is not encoded in the observation date: a monthly CPI dated "
            "2024-01-01 was published mid-February. Signals must apply the publication "
            "lag, which the PIT layer enforces via the release calendar.",
        ),
    ),
    SourceSpec(
        key="alfred",
        name="ALFRED (FRED vintage archive)",
        url="https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
        asset_classes=(AssetClass.MACRO, AssetClass.RATES),
        datasets=("vintage_observations", "vintage_dates"),
        pit_quality=PitQuality.VINTAGE,
        update_frequency="daily",
        licence="Free; attribution requested",
        reliability="stable",
        key_setting="fred_api_key",
        rate_limit="120 requests/minute",
        max_requests_per_second=2.0,
        caveats=(
            "Vintage coverage begins at different dates per series; some series have no "
            "archive at all, in which case no macro signal may use them.",
            "Vintage pulls are one request per (series, vintage) pair and are therefore "
            "slow. Ingest incrementally, not as a full refresh.",
        ),
        notes=("THE source for any macro signal. Spec 3.1 marks this critical.",),
    ),
    SourceSpec(
        key="treasury_fiscaldata",
        name="US Treasury FiscalData API",
        url="https://fiscaldata.treasury.gov/api-documentation/",
        asset_classes=(AssetClass.RATES, AssetClass.MACRO),
        datasets=("daily_treasury_yield_curve", "auction_results", "debt_outstanding"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily (T+1)",
        licence="US Government public domain",
        reliability="stable",
        key_setting=None,
        rate_limit="unpublished; keep under ~5 req/s",
        max_requests_per_second=4.0,
        caveats=(
            "Yield curve rates are constant-maturity par yields, not zero rates. "
            "Bootstrapping is the caller's responsibility.",
            "Published with a one-business-day lag; the PIT layer must apply it.",
        ),
    ),
    SourceSpec(
        key="ecb",
        name="ECB Data Portal API",
        url="https://data.ecb.europa.eu/help/api/overview",
        asset_classes=(AssetClass.RATES, AssetClass.FX, AssetClass.MACRO),
        datasets=("yield_curve", "policy_rates", "fx_reference_rates"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily",
        licence="Free reuse with attribution",
        reliability="stable",
        key_setting=None,
        rate_limit="unpublished; keep under ~2 req/s",
        max_requests_per_second=2.0,
        caveats=(
            "Some ECB statistical series ARE revised; revisions are not exposed as "
            "vintages. Treat macro aggregates from here as restated.",
        ),
    ),
    SourceSpec(
        key="frankfurter",
        name="Frankfurter (ECB reference rates)",
        url="https://frankfurter.dev/",
        asset_classes=(AssetClass.FX,),
        datasets=("fx_spot_daily",),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily, ~16:00 CET",
        licence="Free, open source; no key",
        reliability="stable",
        key_setting=None,
        rate_limit="unpublished; be polite",
        max_requests_per_second=2.0,
        caveats=(
            "A SINGLE DAILY FIX (the ECB 14:15 CET reference rate), not a tradable "
            "quote. There is no bid/ask and no intraday path, so execution realism is "
            "unavailable: FX research is restricted to daily-or-slower signals and FX "
            "spread costs must be assumed, not measured (spec 3.5).",
            "Rates are quoted against EUR; crosses are computed and inherit two fixes' "
            "worth of non-synchronicity.",
            "No weekend or TARGET-holiday observations.",
        ),
    ),
)

# ======================================================================================
# 3.2 Equities
# ======================================================================================
_EQUITY: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="stooq",
        name="Stooq CSV endpoints",
        url="https://stooq.com/db/h/",
        asset_classes=(AssetClass.EQUITY, AssetClass.FUTURES, AssetClass.FX),
        datasets=("ohlcv_daily",),
        pit_quality=PitQuality.SURVIVORSHIP_BIASED,
        update_frequency="daily",
        licence="Personal use; bulk download terms unclear -- review before redistributing",
        reliability="scraped",
        key_setting=None,
        rate_limit="undocumented; throttle hard (<=1 req/s) to avoid IP blocks",
        max_requests_per_second=1.0,
        caveats=(
            "BLOCKED AS OF 2026-09-20: the CSV endpoint serves a JavaScript "
            "proof-of-work anti-bot interstitial rather than data. QuantLab does not "
            "solve anti-bot challenges, so this source fails loudly and the "
            "independent cross-check on Yahoo's adjustments is unavailable.",
            "ADJUSTMENT METHODOLOGY IS UNDOCUMENTED. Whether dividends are reinvested, "
            "and on what date, is unstated. Cross-check against Yahoo adjusted closes "
            "before trusting total returns.",
            "Delisted tickers are not reliably retained -- survivorship bias.",
            "No volume for some non-US venues, which breaks the ADV input to the "
            "impact model and therefore the capacity estimate.",
        ),
    ),
    SourceSpec(
        key="yahoo",
        name="Yahoo Finance (public chart API)",
        url="https://query1.finance.yahoo.com/v8/finance/chart/",
        asset_classes=(
            AssetClass.EQUITY,
            AssetClass.FUTURES,
            AssetClass.FX,
            AssetClass.OPTIONS,
        ),
        datasets=("ohlcv_daily", "ohlcv_intraday", "splits_dividends", "options_chain"),
        pit_quality=PitQuality.SURVIVORSHIP_BIASED,
        update_frequency="daily / intraday",
        licence="Yahoo ToS prohibit redistribution; personal research use only",
        reliability="unofficial",
        key_setting=None,
        rate_limit="unpublished; aggressive use gets rate-limited or blocked",
        max_requests_per_second=1.0,
        caveats=(
            "UNOFFICIAL API. Yahoo publishes no contract for this endpoint and changes "
            "it without notice. Never the ground truth for a production number.",
            "SEVERELY SURVIVORSHIP-BIASED: delisted tickers disappear entirely. A "
            "universe built from what Yahoo returns today is a universe of winners.",
            "PRICES ARE RETROACTIVELY SPLIT-ADJUSTED, and not only by splits that had "
            "already happened. A May 2014 Apple close is served today divided by 28: "
            "the 7:1 split that June AND the 4:1 split six years later. Returns are "
            "unaffected, but every price-LEVEL signal -- penny-stock filters, nominal "
            "price momentum, round-number effects -- is wrong, and nothing raises.",
            "adj_close is Yahoo's own dividend adjustment, computed by an undocumented "
            "method and recomputed whenever Yahoo reprocesses a corporate action. Its "
            "known_at is therefore a fiction. Build total returns from close plus the "
            "corporate_actions dataset, whose dividends carry their own ex-dates.",
            "Intraday history is capped (roughly 60 days at 1m), so intraday equity "
            "research cannot be backfilled.",
        ),
        notes=("Treat as convenience and cross-check, not ground truth (spec 3.2).",),
    ),
    SourceSpec(
        key="sec_edgar",
        name="SEC EDGAR structured data APIs (data.sec.gov)",
        url="https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        asset_classes=(AssetClass.EQUITY, AssetClass.REFERENCE),
        datasets=("companyfacts", "companyconcept", "submissions", "frames", "full_text_search"),
        # Every XBRL fact carries the accession's `filed` date -- genuine point-in-time
        # fundamentals, which is what makes this the most valuable free equity source.
        pit_quality=PitQuality.VINTAGE,
        update_frequency="continuous (filings), bulk refresh nightly",
        licence="US Government public domain",
        reliability="stable",
        key_setting=None,
        rate_limit="10 requests/second, descriptive User-Agent with contact REQUIRED",
        max_requests_per_second=8.0,
        caveats=(
            "Requests without a descriptive User-Agent carrying contact details are "
            "blocked. Set QUANTLAB_HTTP_USER_AGENT.",
            "XBRL TAGGING IS INCONSISTENT: pre-2012 filings and small-cap filers use "
            "non-standard or custom tags, and the same economic concept appears under "
            "several us-gaap elements. Normalisation logic is mandatory and is itself a "
            "source of error -- its coverage must be reported per fundamental.",
            "`filed` is the point-in-time anchor, NOT `end`. Restatements appear as new "
            "facts for an old period; the PIT layer must select by filed date and keep "
            "the superseded value visible for pre-restatement dates.",
            "Covers SEC filers only: no foreign private issuers filing on some forms, "
            "no pre-EDGAR history (electronic filing is broadly complete from ~1996).",
            "Amended filings (10-K/A) can arrive years later. Ingest must be able to "
            "add a fact with an old period without rewriting history.",
        ),
        notes=("The crown jewel of free fundamentals (spec 3.2).",),
    ),
    SourceSpec(
        key="nasdaq_data_link",
        name="Nasdaq Data Link (free tables only)",
        url="https://data.nasdaq.com/",
        asset_classes=(AssetClass.EQUITY, AssetClass.COMMODITY, AssetClass.MACRO),
        datasets=("free_tables",),
        pit_quality=PitQuality.RESTATED,
        update_frequency="varies by table",
        licence="Per-table; several formerly free tables are now paid",
        reliability="stable",
        key_setting="nasdaq_data_link_api_key",
        rate_limit="50 calls/day anonymous, 300/10s with a free key",
        max_requests_per_second=1.0,
        cross_check_only=True,
        caveats=(
            "The free catalogue shrinks over time; tables that worked last year may now "
            "be paid. Ingest must fail loudly on a 403 rather than skipping the table.",
        ),
    ),
    SourceSpec(
        key="alpha_vantage",
        name="Alpha Vantage (free tier)",
        url="https://www.alphavantage.co/documentation/",
        asset_classes=(AssetClass.EQUITY, AssetClass.FX, AssetClass.CRYPTO),
        datasets=("ohlcv_daily_adjusted", "fundamentals"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="daily",
        licence="Free tier for personal use",
        reliability="stable",
        key_setting="alpha_vantage_api_key",
        rate_limit="~25 requests/day on the free tier",
        max_requests_per_second=0.2,
        cross_check_only=True,
        caveats=(
            "The daily cap makes this unusable for bulk ingest. Cross-validation of a "
            "handful of symbols only (spec 3.2).",
            "Fundamentals are restated, with no filed date -- banned from the signal path.",
        ),
    ),
    SourceSpec(
        key="tiingo",
        name="Tiingo (free tier)",
        url="https://www.tiingo.com/documentation/general/overview",
        asset_classes=(AssetClass.EQUITY, AssetClass.CRYPTO),
        datasets=("ohlcv_daily_adjusted", "news"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="daily",
        licence="Free tier, non-commercial; registration required",
        reliability="stable",
        key_setting="tiingo_api_key",
        rate_limit="~50 symbols/hour, 1000 requests/day on the free tier",
        max_requests_per_second=0.3,
        cross_check_only=True,
        caveats=("Free tier limits preclude universe-wide ingest; cross-check only.",),
    ),
    SourceSpec(
        key="fmp",
        name="Financial Modeling Prep (free tier)",
        url="https://site.financialmodelingprep.com/developer/docs",
        asset_classes=(AssetClass.EQUITY,),
        datasets=("ohlcv_daily", "fundamentals", "delisted_companies"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="daily",
        licence="Free tier, non-commercial",
        reliability="stable",
        key_setting="fmp_api_key",
        rate_limit="~250 requests/day on the free tier",
        max_requests_per_second=0.3,
        cross_check_only=True,
        caveats=(
            "Free-tier fundamentals are restated with no as-filed date: unusable for "
            "point-in-time work. The delisted-companies endpoint is nonetheless useful "
            "for partially repairing survivorship bias in the universe.",
        ),
    ),
)

# ======================================================================================
# 3.3 Academic factor and anomaly datasets -- the benchmark for everything we build
# ======================================================================================
_FACTORS: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="ken_french",
        name="Kenneth R. French Data Library",
        url="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        asset_classes=(AssetClass.FACTORS, AssetClass.EQUITY),
        datasets=("ff3", "ff5", "momentum", "industry_portfolios", "international"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="monthly",
        licence="Free for research use; attribution expected",
        reliability="stable",
        key_setting=None,
        rate_limit="static files; cache aggressively",
        max_requests_per_second=1.0,
        caveats=(
            "Survivorship-bias-free and the accepted benchmark, but the whole history is "
            "rebuilt on each release, so the file is restated: it answers 'what do we "
            "now believe HML returned in 1965', not 'what was known then'. Fine as a "
            "benchmark and as a risk-model factor; not a point-in-time signal input.",
            "Returns are gross of transaction costs and assume free shorting.",
        ),
        notes=(
            "Acceptance test for our own factor construction: a hand-built momentum "
            "factor must correlate >0.9 with UMD or the construction has a bug (spec 3.3).",
        ),
    ),
    SourceSpec(
        key="open_asset_pricing",
        name="Open Source Asset Pricing (Chen & Zimmermann)",
        url="https://www.openassetpricing.com/",
        asset_classes=(AssetClass.FACTORS, AssetClass.EQUITY),
        datasets=("predictor_portfolios", "signal_documentation"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="annual releases",
        licence="Free for research with citation",
        reliability="stable",
        key_setting=None,
        rate_limit="static files",
        max_requests_per_second=1.0,
        caveats=(
            "Portfolio returns are reconstructions of published anomalies built on CRSP/"
            "Compustat, standardised by the authors -- they are not what the original "
            "papers reported, and differences are informative rather than errors.",
            "Gross of costs, and several predictors are microcap-dominated: a headline "
            "spread can be untradable at any size. Always pair with the capacity model.",
        ),
    ),
    SourceSpec(
        key="jkp_factors",
        name="Global Factor Data (Jensen, Kelly & Pedersen)",
        url="https://jkpfactors.com/",
        asset_classes=(AssetClass.FACTORS, AssetClass.EQUITY),
        datasets=("factor_returns", "cluster_classification"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="periodic",
        licence="Free with registration; academic citation required",
        reliability="stable",
        key_setting=None,
        rate_limit="static files behind a login",
        max_requests_per_second=1.0,
        caveats=(
            "Requires registration, so automated ingest may need a manually placed file. "
            "Ingest must fail loudly with instructions rather than silently skipping.",
            "Country coverage is very uneven early in the sample; small markets have few "
            "names and the factor spreads there are not tradable.",
        ),
    ),
    SourceSpec(
        key="open_bond_asset_pricing",
        name="Open Source Bond Asset Pricing (error-corrected TRACE)",
        url="https://openbondassetpricing.com/",
        asset_classes=(AssetClass.CREDIT, AssetClass.FACTORS),
        datasets=("bond_returns", "bond_factors"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="periodic",
        licence="Free for research with citation",
        reliability="stable",
        key_setting=None,
        rate_limit="static files",
        max_requests_per_second=1.0,
        caveats=(
            "Essential: uncorrected TRACE-based bond returns contain large data errors "
            "that inflate published bond factor performance. A substantial fraction of "
            "published corporate bond factors do not survive the correction, and any "
            "credit signal built here must surface that (spec 4, Tier 2).",
            "Corporate bonds trade thinly; reported returns often reflect stale or "
            "matrix prices, so realised transaction costs dwarf equity-style assumptions.",
        ),
    ),
)

# ======================================================================================
# 3.4 Futures and commodities
# ======================================================================================
_FUTURES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="yahoo_futures",
        name="Yahoo continuous futures (=F tickers)",
        url="https://finance.yahoo.com/commodities",
        asset_classes=(AssetClass.FUTURES, AssetClass.COMMODITY),
        datasets=("continuous_front_month",),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily",
        licence="Yahoo ToS; personal research use only",
        reliability="unofficial",
        key_setting=None,
        rate_limit="unofficial; <=1 req/s",
        max_requests_per_second=1.0,
        caveats=(
            "ROLL METHODOLOGY IS OPAQUE. The series is stitched by an unstated rule with "
            "unstated (probably no) back-adjustment, so returns across a roll date are "
            "not tradable returns. Adequate for trend research after roll-date filtering; "
            "NOT usable for carry, which needs the curve (spec 3.4).",
            "Gaps and stale prints around holidays and low-liquidity sessions.",
            "Only the front month: no second contract, hence no term structure.",
        ),
    ),
    SourceSpec(
        key="cftc_cot",
        name="CFTC Commitments of Traders",
        url="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        asset_classes=(AssetClass.FUTURES, AssetClass.COMMODITY),
        datasets=("legacy", "disaggregated", "tff"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="weekly (Tuesday snapshot, released Friday 15:30 ET)",
        licence="US Government public domain",
        reliability="stable",
        key_setting=None,
        rate_limit="bulk annual zip files; a handful of requests",
        max_requests_per_second=2.0,
        caveats=(
            "THREE-DAY PUBLICATION LAG on a Tuesday snapshot. Using the Tuesday value "
            "before Friday's release is look-ahead bias, and it is the single most "
            "common error in published COT research. The PIT layer applies the lag.",
            "Reclassifications between trader categories create level shifts that look "
            "like signal; positioning must be normalised within a regime, not across one.",
            "Weekly frequency caps the achievable Sharpe of any COT signal.",
        ),
    ),
    SourceSpec(
        key="eia",
        name="EIA API v2",
        url="https://www.eia.gov/opendata/documentation.php",
        asset_classes=(AssetClass.COMMODITY,),
        datasets=("petroleum_stocks", "natural_gas_storage", "production", "refining"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="weekly / monthly",
        licence="US Government public domain",
        reliability="stable",
        key_setting="eia_api_key",
        rate_limit="unpublished; keep under ~5 req/s",
        max_requests_per_second=4.0,
        caveats=(
            "EIA REVISES weekly inventories and monthly production, and the API serves "
            "the revised value with no vintage archive. A backtest that uses the current "
            "value on the original release date has look-ahead. Mitigation: snapshot each "
            "release as ingested and use OUR OWN ingested_at as the vintage -- which only "
            "works from the day we start collecting. Pre-collection history is restated "
            "and is flagged as such.",
            "Release timestamps (Wed 10:30 ET crude, Thu 10:30 ET gas) matter for any "
            "signal faster than weekly.",
        ),
    ),
    SourceSpec(
        key="usda_nass",
        name="USDA NASS QuickStats / WASDE",
        url="https://quickstats.nass.usda.gov/api",
        asset_classes=(AssetClass.COMMODITY,),
        datasets=("production", "stocks", "exports", "wasde"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="monthly / quarterly",
        licence="US Government public domain",
        reliability="stable",
        key_setting="usda_nass_api_key",
        rate_limit="unpublished; large queries are rejected -- page by year",
        max_requests_per_second=2.0,
        caveats=(
            "Heavily revised, no vintage archive. Same ingested_at mitigation as EIA.",
            "WASDE is a PDF/XLS release; parsing is brittle and must be validated per "
            "release rather than assumed stable.",
        ),
    ),
    SourceSpec(
        key="exchange_settlements",
        name="Exchange daily settlement files (CME / ICE public sections)",
        url="https://www.cmegroup.com/market-data/",
        asset_classes=(AssetClass.FUTURES, AssetClass.COMMODITY),
        datasets=("daily_settlements", "open_interest"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily, after settlement",
        licence="CHECK TERMS OF SERVICE BEFORE USE. Scrape politely and identify yourself.",
        reliability="scraped",
        key_setting=None,
        rate_limit="self-imposed: <=0.5 req/s, business hours, cache everything",
        max_requests_per_second=0.5,
        caveats=(
            "The ONLY free route to a multi-contract futures curve, which carry and "
            "basis-momentum signals require (spec 3.4). Availability and file format "
            "change without notice and history is typically short -- often only recent "
            "files are posted, so the curve archive starts when WE start collecting.",
            "Settlement prices are not tradable prices: they are exchange-determined and "
            "can differ materially from the last trade in illiquid deferred contracts.",
            "Terms of service govern automated access and may forbid it. This source is "
            "disabled by default and must be enabled deliberately.",
        ),
    ),
)

# ======================================================================================
# 3.6 Crypto -- the best free data of any asset class
# ======================================================================================
_CRYPTO: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="binance",
        name="Binance public REST",
        url="https://developers.binance.com/docs/binance-spot-api-docs",
        asset_classes=(AssetClass.CRYPTO,),
        datasets=(
            "spot_klines",
            "perp_klines",
            "funding_rate",
            "open_interest",
            "mark_index_price",
        ),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="realtime",
        licence="Free public endpoints, no key; commercial use terms apply",
        reliability="stable",
        key_setting=None,
        rate_limit="weight-based (~1200 weight/minute per IP); 429 then IP ban on abuse",
        max_requests_per_second=8.0,
        caveats=(
            "SURVIVORSHIP AND LISTING BIAS: delisted symbols stop being served, and a "
            "universe of 'symbols trading today' is a universe of survivors. Retain every "
            "symbol once observed and build the universe from historical exchangeInfo.",
            "Funding history begins at each contract's listing date, and the funding "
            "formula/cap has changed over time -- a funding-carry backtest spanning a "
            "regime change is comparing different instruments.",
            "Klines are exchange-local: cross-venue basis research must align timestamps "
            "and account for differing settlement conventions.",
            "Geographic restrictions apply to some endpoints from some jurisdictions, "
            "and a 451 must fail loudly, not silently return empty.",
        ),
        notes=(
            "Full funding history, open interest and order books make crypto the place to "
            "build and validate the microstructure machinery (spec 3.6).",
        ),
    ),
    SourceSpec(
        key="bybit",
        name="Bybit public API",
        url="https://bybit-exchange.github.io/docs/v5/intro",
        asset_classes=(AssetClass.CRYPTO,),
        datasets=("klines", "funding_rate", "open_interest"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="realtime",
        licence="Free public endpoints",
        reliability="stable",
        key_setting=None,
        rate_limit="per-endpoint IP limits; back off on 403/429",
        max_requests_per_second=5.0,
        caveats=("Funding interval differs by contract; do not assume 8h across venues.",),
    ),
    SourceSpec(
        key="okx",
        name="OKX public API",
        url="https://www.okx.com/docs-v5/en/",
        asset_classes=(AssetClass.CRYPTO,),
        datasets=("klines", "funding_rate", "open_interest"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="realtime",
        licence="Free public endpoints",
        reliability="stable",
        key_setting=None,
        rate_limit="per-endpoint; typically 20 req/2s",
        max_requests_per_second=5.0,
        caveats=("Historical kline depth is limited per request; paginate carefully.",),
    ),
    SourceSpec(
        key="kraken",
        name="Kraken public API",
        url="https://docs.kraken.com/api/",
        asset_classes=(AssetClass.CRYPTO,),
        datasets=("ohlc", "trades", "spread"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="realtime",
        licence="Free public endpoints",
        reliability="stable",
        key_setting=None,
        rate_limit="counter-based; ~1 req/s sustained for public endpoints",
        max_requests_per_second=1.0,
        caveats=(
            "OHLC history is truncated to the last ~720 candles per interval: long "
            "history is only available by accumulating our own snapshots.",
            "Kraken asset codes are non-standard (XXBT, ZUSD) and need normalisation.",
        ),
    ),
    SourceSpec(
        key="coinbase",
        name="Coinbase Exchange public API",
        url="https://docs.cdp.coinbase.com/exchange/docs/welcome",
        asset_classes=(AssetClass.CRYPTO,),
        datasets=("candles", "trades", "book"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="realtime",
        licence="Free public endpoints",
        reliability="stable",
        key_setting=None,
        rate_limit="~10 req/s public",
        max_requests_per_second=5.0,
        caveats=("Candle requests are capped at 300 buckets; paginate.",),
    ),
    SourceSpec(
        key="coingecko",
        name="CoinGecko (free tier)",
        url="https://docs.coingecko.com/reference/introduction",
        asset_classes=(AssetClass.CRYPTO,),
        datasets=("market_cap", "volume", "listing_metadata"),
        pit_quality=PitQuality.RESTATED,
        update_frequency="daily",
        licence="Free demo tier; attribution required",
        reliability="stable",
        key_setting=None,
        rate_limit="~5-15 requests/minute on the keyless tier",
        max_requests_per_second=0.2,
        caveats=(
            "Market-cap history uses TODAY's circulating supply for past dates in some "
            "endpoints, which silently restates the cross-section. Snapshot daily and "
            "use our own ingested_at as the vintage for any size-sorted crypto signal.",
            "Coin lists are survivorship-biased: dead coins are removed.",
        ),
    ),
)

# ======================================================================================
# 3.7 Options and volatility -- the weakest link
# ======================================================================================
_OPTIONS: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="cboe",
        name="CBOE free historical data",
        url="https://www.cboe.com/tradable_products/vix/vix_historical_data/",
        asset_classes=(AssetClass.OPTIONS,),
        datasets=("vix_history", "vix_term_structure", "index_settlements"),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily",
        licence="Free for personal use; redistribution restricted",
        reliability="stable",
        key_setting=None,
        rate_limit="static CSVs; cache",
        max_requests_per_second=1.0,
        caveats=(
            "VIX methodology changed in 2003 (and the pre-2003 VXO is a different index): "
            "a series spanning the change is not one instrument.",
            "Index level only -- no option-level greeks, so any delta-hedged strategy must "
            "be approximated rather than simulated.",
        ),
    ),
    SourceSpec(
        key="options_snapshot",
        name="Self-collected options chain snapshots (Yahoo chart API)",
        url="https://query1.finance.yahoo.com/v8/finance/chart/",
        asset_classes=(AssetClass.OPTIONS,),
        datasets=("chain_snapshot",),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="daily snapshot, accumulating forward",
        licence="Yahoo ToS; personal research use only",
        reliability="unofficial",
        key_setting=None,
        rate_limit="<=1 req/s",
        max_requests_per_second=1.0,
        caveats=(
            "HISTORY STARTS THE DAY WE START COLLECTING. There is no free historical "
            "options chain data, so NO options backtest before the collector's first run "
            "is possible (spec 3.7). Any result claiming otherwise is fabricated.",
            "A single daily snapshot has no intraday path, and quotes may be stale or "
            "crossed in illiquid strikes. Filter on non-zero volume/open interest and "
            "treat the mid as indicative only.",
            "Implied volatilities reported by Yahoo use an unstated model and dividend "
            "assumption; recompute from mid prices rather than trusting the field.",
        ),
    ),
)

# ======================================================================================
# 3.8 Disclosed trades -- insiders, institutions and politicians
# ======================================================================================
# What these have in common is that nobody here observes a trade. Each is a legal
# DISCLOSURE, filed days to months after the fact, and the whole value of the data
# is in keeping the trade date and the disclosure date apart.
_DISCLOSURES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="sec_insider",
        name="SEC EDGAR ownership filings (Forms 4 and 5)",
        url="https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        asset_classes=(AssetClass.EQUITY,),
        datasets=("insider_transactions",),
        # The submissions API carries the instant EDGAR accepted each filing, to the
        # second, so nothing here has to be derived.
        pit_quality=PitQuality.VINTAGE,
        update_frequency="continuous; a Form 4 is due two business days after the trade",
        licence="US Government public domain",
        reliability="stable",
        key_setting=None,
        rate_limit="10 requests/second, descriptive User-Agent with contact REQUIRED",
        max_requests_per_second=8.0,
        caveats=(
            "MOST INSIDER TRANSACTIONS ARE NOT A VIEW ON THE STOCK. Grants (A), option "
            "exercises (M), shares withheld for tax (F) and gifts (G) dominate the "
            "filings of a large company. Only open-market purchases (P) and sales (S) "
            "are discretionary, and a sale under a 10b5-1 plan was scheduled months "
            "earlier. Filter on `transaction_code` before reading anything into it.",
            "One request per filing. A large issuer has hundreds of Form 4s a year, so "
            "a wide universe over a long window is tens of thousands of requests at "
            "the SEC's pace. Keep the symbol list deliberate.",
            "Amendments do not replace what they amend. A 4/A is a new filing with a "
            "new accession number, and the rows it corrects stay in the lake beside "
            "it; nothing links the two except the owner, the issuer and the dates.",
            "Coverage follows the SEC's current ticker map, so a delisted or acquired "
            "issuer cannot be requested by ticker even though its filings still exist. "
            "This is survivorship bias in what can be ASKED FOR, not in what is stored.",
            "Filings before mid-2003 are not XML and are skipped. Prices are sometimes "
            "given only in a footnote, and are then null rather than guessed.",
        ),
    ),
    SourceSpec(
        key="sec_13f",
        name="SEC EDGAR institutional holdings (Form 13F)",
        url="https://www.sec.gov/divisions/investment/13ffaq",
        asset_classes=(AssetClass.EQUITY,),
        datasets=("institutional_holdings",),
        pit_quality=PitQuality.VINTAGE,
        update_frequency="quarterly, due 45 days after quarter end",
        licence="US Government public domain",
        reliability="stable",
        key_setting=None,
        rate_limit="10 requests/second, descriptive User-Agent with contact REQUIRED",
        max_requests_per_second=8.0,
        caveats=(
            "STALE BY CONSTRUCTION. Positions are counted at quarter end and disclosed "
            "up to 45 days later, so on the day a filing becomes knowable it describes "
            "a portfolio six weeks old, and by the next one it is nineteen weeks old. "
            "A fast-trading manager's 13F says almost nothing about what it holds now.",
            "LONG POSITIONS ONLY. Short positions, cash, most derivatives, and anything "
            "not on the SEC's list of 13(f) securities are absent. A long-short fund "
            "looks like a long-only one, and a put is reported by the value of the "
            "UNDERLYING shares, not by what the option cost.",
            "Securities are identified by CUSIP and by nothing else. There is no ticker "
            "on a 13F and no free CUSIP master; `fails_to_deliver` is the only free "
            "bridge, and it covers only securities that have had a settlement fail.",
            "THE UNIT OF `value` CHANGED. Filings made before 3 January 2023 report "
            "thousands of dollars and later ones report dollars, in the same field, "
            "with nothing in the payload saying which. It is normalised here by filing "
            "date; any other copy of this data must be checked for the same thing.",
            "A restating amendment supersedes matching positions but cannot delete one: "
            "a holding dropped by a 13F-HR/A stays visible from the original filing. "
            "Managers may also omit positions under confidential treatment and disclose "
            "them up to a year later, so an early snapshot of a quarter is incomplete.",
            "Filings before mid-2013 carry the holdings as free text, not XML, and are "
            "skipped rather than parsed by guesswork.",
        ),
    ),
    SourceSpec(
        key="sec_ftd",
        name="SEC fails-to-deliver data",
        url="https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data",
        asset_classes=(AssetClass.EQUITY, AssetClass.REFERENCE),
        datasets=("fails_to_deliver",),
        pit_quality=PitQuality.AS_PUBLISHED,
        update_frequency="twice monthly, two to six weeks after the settlement dates covered",
        licence="US Government public domain",
        reliability="stable",
        key_setting=None,
        rate_limit="10 requests/second, descriptive User-Agent with contact REQUIRED",
        max_requests_per_second=4.0,
        caveats=(
            "`quantity` is a BALANCE, not a flow: the total fails outstanding in that "
            "security on that day, including ones carried over. Summing it across days "
            "counts the same undelivered shares again for every day they stay open.",
            "A fail is not evidence of naked short selling. The SEC says so on the page "
            "the data comes from: fails arise from long sales, processing delays and "
            "market-maker activity too, and the file cannot tell them apart.",
            "A security appears only on days it has a balance, so absence means no "
            "fails, not missing data -- and it means the CUSIP-to-ticker bridge built "
            "from this file never sees a security that always settles cleanly.",
            "`known_at` is the file's Last-Modified time. If the SEC ever re-uploads an "
            "old file, that history becomes knowable later than it really was, which "
            "errs in the safe direction but does err.",
        ),
    ),
    SourceSpec(
        key="house_clerk",
        name="US House of Representatives financial disclosures (Office of the Clerk)",
        url="https://disclosures-clerk.house.gov/FinancialDisclosure",
        asset_classes=(AssetClass.EQUITY, AssetClass.REFERENCE),
        datasets=("congress_filings", "congress_trades"),
        pit_quality=PitQuality.VINTAGE,
        update_frequency="continuous; a report is due within 45 days of the trade",
        licence="Public record. 5 U.S.C. 13107 prohibits use for commercial "
        "solicitation or to establish a credit rating",
        reliability="scraped",
        key_setting=None,
        rate_limit="none published -- self-limited",
        max_requests_per_second=1.0,
        caveats=(
            "SCANNED PAPER REPORTS CANNOT BE READ. Roughly one report in eight is a "
            "scan with no text layer, and some of the most active traders in the House "
            "file that way. Their trades are ABSENT from `congress_trades`, not zero. "
            "`congress_filings` lists every report, so the gap can be measured.",
            "THE HOUSE ONLY. The Senate's disclosure site refuses automated clients and "
            "sits behind a click-through agreement; it is catalogued as `senate_efd` "
            "and deliberately not fetched. Half of Congress is therefore missing.",
            "Sizes are brackets, not amounts: $1,001-$15,000 up to over $50,000,000. "
            "The top of a bracket can be fifteen times the bottom, so any aggregate "
            "dollar figure built from this is an order-of-magnitude estimate.",
            "The ticker is whatever the member typed. It is missing for bonds, funds "
            "and private holdings, occasionally wrong, and never validated here.",
            "Transactions are parsed out of a PDF's text layer by pattern. A report "
            "whose parsed row count disagrees with the dates found in it is logged, "
            "but a layout change could still lose rows quietly within one report.",
            "The Clerk publishes a filing DATE with no time and no publication "
            "timestamp, so `known_at` is taken as the end of that day in Washington.",
        ),
    ),
    SourceSpec(
        key="senate_efd",
        name="US Senate electronic financial disclosures (eFD)",
        url="https://efdsearch.senate.gov/search/",
        asset_classes=(AssetClass.EQUITY, AssetClass.REFERENCE),
        datasets=("congress_trades",),
        pit_quality=PitQuality.VINTAGE,
        update_frequency="continuous",
        licence="Public record, behind a click-through agreement restricting use",
        reliability="scraped",
        key_setting=None,
        rate_limit="automated clients are refused (HTTP 403 as of 2026-09-20)",
        max_requests_per_second=0.2,
        caveats=(
            "NOT FETCHED, by decision. The site answers automated clients with 403 and "
            "requires accepting a use agreement before any search. Getting past either "
            "is a choice for a person to make about terms they have read, not a "
            "default for a scheduler to make on their behalf.",
            "The community mirrors that once republished this data are gone or frozen: "
            "the Senate and House Stock Watcher buckets return 403, and the public "
            "GitHub copy of the Senate data ends in December 2020.",
        ),
        notes=("Catalogued so the gap in congressional coverage is explicit.",),
    ),
)

SOURCES: dict[str, SourceSpec] = {
    spec.key: spec
    for spec in (
        *_MACRO,
        *_EQUITY,
        *_FACTORS,
        *_FUTURES,
        *_CRYPTO,
        *_OPTIONS,
        *_DISCLOSURES,
    )
}


def get_source(key: str) -> SourceSpec:
    """Look up a source by key, raising a helpful error when it is unknown."""
    try:
        return SOURCES[key]
    except KeyError:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"unknown data source {key!r}; known sources: {known}") from None


def sources_for(asset_class: AssetClass) -> tuple[SourceSpec, ...]:
    """All registered sources covering an asset class."""
    return tuple(s for s in SOURCES.values() if asset_class in s.asset_classes)
