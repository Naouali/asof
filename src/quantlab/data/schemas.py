"""Canonical dataset schemas.

Every row in the lake carries three timestamps, and the distinction between them
is the entire point of this module:

``as_of``
    The instant the observation *refers to*. For a daily bar it is the session
    close, not the session date -- a bar is not information until it has closed.

``known_at``
    The instant the observation became **knowable to us**. This is what
    point-in-time queries filter on. For a vintage source (ALFRED, SEC EDGAR) it is
    the publication or filing timestamp, so history reconstructs correctly even
    though we downloaded it today. For everything else it is our download time,
    which is the best we can do and is why daily snapshotting starts early.

``ingested_at``
    When our process wrote the row. Audit only; never used for filtering.

``known_at`` is kept apart from ``ingested_at`` deliberately: collapsing "when the world learned it" into "when we learned it" makes
ALFRED vintages and EDGAR filed dates unusable, and those are the only genuine
point-in-time data available for free. See docs/ASSUMPTIONS.md.

All timestamps are timezone-aware UTC microseconds. A naive timestamp anywhere in
the lake is a bug, because a daily bar's session close is only unambiguous in UTC.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError

__all__ = [
    "COMMON_COLUMNS",
    "DATASETS",
    "TIME_COLUMNS",
    "UTC_DATETIME",
    "DatasetSchema",
    "SchemaError",
    "empty_frame",
    "get_schema",
]

#: Every timestamp column in the lake has exactly this dtype.
UTC_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")

#: Slack allowed between our clock and a venue's before a future timestamp is
#: treated as an error rather than as clock skew.
FUTURE_TOLERANCE = dt.timedelta(minutes=1)


class SchemaError(ValueError):
    """A frame does not conform to its declared dataset schema."""


#: Columns every dataset carries, in this order.
COMMON_COLUMNS: dict[str, pl.DataType] = {
    "source": pl.Utf8(),
    "dataset": pl.Utf8(),
    "symbol": pl.Utf8(),
    "as_of": UTC_DATETIME,
    "known_at": UTC_DATETIME,
    "ingested_at": UTC_DATETIME,
}

TIME_COLUMNS = ("as_of", "known_at", "ingested_at")


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """The shape of one canonical dataset."""

    name: str
    description: str
    #: Columns beyond :data:`COMMON_COLUMNS`, in order.
    columns: dict[str, pl.DataType]
    #: Columns that, together with ``as_of``, uniquely identify an observation.
    #: Duplicates on this key are resolved on read by taking the latest ``known_at``,
    #: which is how a restatement supersedes the value it replaces without erasing it.
    key: tuple[str, ...] = ("symbol",)
    #: Columns that may never be null. A null here means the source gave us
    #: something we did not understand, which must fail rather than propagate.
    required: tuple[str, ...] = ()

    @property
    def polars_schema(self) -> dict[str, pl.DataType]:
        return {**COMMON_COLUMNS, **self.columns}

    @property
    def dedup_key(self) -> tuple[str, ...]:
        return (*self.key, "as_of")

    def validate(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` coerced to this schema, or raise :class:`SchemaError`.

        Validation is strict on purpose. A source that quietly returns a column of
        strings where floats are expected produces a reader that silently skips
        those rows, and a skipped row is indistinguishable from a missing one.
        """
        expected = self.polars_schema

        missing = [name for name in expected if name not in frame.columns]
        if missing:
            raise SchemaError(f"{self.name}: missing columns {missing}")

        unexpected = [name for name in frame.columns if name not in expected]
        if unexpected:
            raise SchemaError(
                f"{self.name}: unexpected columns {unexpected}. Add them to the "
                "dataset schema rather than smuggling them through."
            )

        out = frame.select(list(expected))

        for name, dtype in expected.items():
            actual = out.schema[name]
            if actual == dtype:
                continue
            if name in TIME_COLUMNS:
                raise SchemaError(
                    f"{self.name}.{name}: expected {dtype} but got {actual}. Every "
                    "timestamp in the lake must be timezone-aware UTC; a naive "
                    "timestamp makes point-in-time comparison ambiguous."
                )
            try:
                out = out.with_columns(pl.col(name).cast(dtype, strict=True))
            except (InvalidOperationError, ComputeError) as exc:
                raise SchemaError(
                    f"{self.name}.{name}: cannot cast {actual} to {dtype}: {exc}"
                ) from exc

        for name in (*COMMON_COLUMNS, *self.required):
            if out.height and out[name].null_count():
                raise SchemaError(
                    f"{self.name}.{name}: {out[name].null_count()} null values in a "
                    "column that may not be null"
                )

        if out.height:
            horizon = dt.datetime.now(dt.UTC) + FUTURE_TOLERANCE
            ahead = out.filter(pl.col("known_at") > horizon)
            if ahead.height:
                # Venues serve the current, incomplete bar with its close time in
                # the future. Storing it stamps a mid-session price as a settled
                # close, and any snapshot taken later that day then sees it.
                sample = ahead.head(3).select("symbol", "as_of", "known_at")
                raise SchemaError(
                    f"{self.name}: {ahead.height} rows are knowable in the FUTURE "
                    f"(now is {dt.datetime.now(dt.UTC).isoformat()}). This is almost "
                    f"always an unclosed bar being treated as a completed one:\n{sample}"
                )

            future = out.filter(pl.col("as_of") > pl.col("known_at"))
            if future.height:
                # An observation cannot be knowable before it exists. This catches
                # a source that has confused a publication date with a period end.
                sample = future.head(3).select("symbol", "as_of", "known_at")
                raise SchemaError(
                    f"{self.name}: {future.height} rows have as_of after known_at, "
                    f"i.e. they claim to have been knowable before they existed:\n{sample}"
                )

        return out


_OHLCV_COLUMNS: dict[str, pl.DataType] = {
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
}

DATASETS: dict[str, DatasetSchema] = {
    "ohlcv_daily": DatasetSchema(
        name="ohlcv_daily",
        description=(
            "Daily bars. `as_of` is the session CLOSE in UTC, taken from the venue's "
            "trading calendar, not the session date: a bar is not information until "
            "it has closed. `adj_close` is the source's own adjustment and inherits "
            "that source's undocumented methodology."
        ),
        columns={
            **_OHLCV_COLUMNS,
            "adj_close": pl.Float64(),
            "currency": pl.Utf8(),
            "venue": pl.Utf8(),
        },
        key=("symbol",),
        required=("close",),
    ),
    "ohlcv_bars": DatasetSchema(
        name="ohlcv_bars",
        description=(
            "Intraday or multi-day bars at an explicit interval. `as_of` is the bar "
            "close instant as reported by the venue."
        ),
        columns={
            **_OHLCV_COLUMNS,
            "interval": pl.Utf8(),
            "quote_volume": pl.Float64(),
            "trades": pl.Int64(),
        },
        key=("symbol", "interval"),
        required=("close", "interval"),
    ),
    "series_observations": DatasetSchema(
        name="series_observations",
        description=(
            "Economic and rates time series. `symbol` is the provider's series id. "
            "`known_at` is the vintage date for vintage-capable sources (ALFRED) and "
            "the download time otherwise -- the difference decides whether a macro "
            "signal built on it is honest."
        ),
        columns={
            "value": pl.Float64(),
            "units": pl.Utf8(),
            "vintage": pl.Boolean(),
        },
        key=("symbol",),
        required=("vintage",),
    ),
    "funding_rate": DatasetSchema(
        name="funding_rate",
        description=(
            "Perpetual futures funding payments. `as_of` is the funding timestamp. "
            "The funding formula and interval have changed over time on every venue, "
            "so a series spanning a change is comparing different instruments."
        ),
        columns={
            "funding_rate": pl.Float64(),
            "mark_price": pl.Float64(),
            "interval_hours": pl.Float64(),
        },
        key=("symbol",),
        required=("funding_rate",),
    ),
    "corporate_actions": DatasetSchema(
        name="corporate_actions",
        description=(
            "Dividends and splits. Missing delisting returns are a classic way to "
            "inflate short-leg performance, so absence here is not evidence of absence."
        ),
        columns={
            "action": pl.Utf8(),
            "amount": pl.Float64(),
            "split_numerator": pl.Float64(),
            "split_denominator": pl.Float64(),
        },
        key=("symbol", "action"),
        required=("action",),
    ),
    "fundamentals": DatasetSchema(
        name="fundamentals",
        description=(
            "Point-in-time company fundamentals, one row per (symbol, metric, "
            "period). `as_of` is the period END and `known_at` is the FILED date -- "
            "the distinction is the whole value of SEC EDGAR over every restated "
            "free source. A restatement arrives as a new row with a later known_at, "
            "and the superseded value stays visible to earlier snapshots, which is "
            "what makes a historical view of fundamentals honest."
        ),
        columns={
            "metric": pl.Utf8(),
            "value": pl.Float64(),
            "unit": pl.Utf8(),
            "fiscal_period": pl.Utf8(),
            "form": pl.Utf8(),
        },
        key=("symbol", "metric", "fiscal_period"),
        required=("metric",),
    ),
    "positioning": DatasetSchema(
        name="positioning",
        description=(
            "Aggregate trader positioning, one row per (symbol, category, measure). "
            "`as_of` is the SNAPSHOT date -- the Tuesday a Commitments of Traders "
            "report counts positions on -- and `known_at` is the RELEASE instant, "
            "the following Friday at 15:30 America/New_York. The gap between them "
            "is three days wide and is the single most common look-ahead error in "
            "published work on this report: the Tuesday number simply did not exist until "
            "Friday afternoon. No CFTC payload carries the release date, so it is "
            "derived, and rows whose release cannot be established honestly are "
            "refused rather than dated optimistically."
        ),
        columns={
            "report": pl.Utf8(),
            #: The market as CFTC names it. Stored because a contract market code
            #: is opaque -- "002602" is corn, and nobody looking for corn will
            #: type that.
            "name": pl.Utf8(),
            "category": pl.Utf8(),
            "measure": pl.Utf8(),
            "value": pl.Float64(),
            "contract_units": pl.Utf8(),
            "exchange": pl.Utf8(),
        },
        # `report` is part of the identity because the reports overlap: the
        # legacy and disaggregated reports both publish a total open interest for
        # the same contract and Tuesday, and `other_reportable` is a category in
        # two of the three. Without it those rows share a key and deduplication
        # silently keeps one taxonomy's number under the other's name.
        key=("symbol", "report", "category", "measure"),
        required=("report", "category", "measure"),
    ),
    "anomaly_catalogue": DatasetSchema(
        name="anomaly_catalogue",
        description=(
            "Published cross-sectional equity predictors, with the effect size and "
            "t-statistic as reported in the original paper. `as_of` is the end of "
            "the ORIGINAL SAMPLE -- the last date the published evidence covers -- "
            "and `known_at` is publication, when the finding entered the public "
            "domain. The gap between them is the out-of-sample window every "
            "replication has to be measured over, and the two are recorded "
            "separately because post-publication decay is measured from the second "
            "while in-sample fit ends at the first."
        ),
        columns={
            "name": pl.Utf8(),
            "authors": pl.Utf8(),
            "journal": pl.Utf8(),
            "category": pl.Utf8(),
            "replication": pl.Utf8(),
            "evidence": pl.Utf8(),
            "published_return": pl.Float64(),
            "published_t_stat": pl.Float64(),
            "sign": pl.Float64(),
        },
        key=("symbol",),
        required=("name", "category"),
    ),
    "chain_snapshot": DatasetSchema(
        name="chain_snapshot",
        description=(
            "One option quote, as it stood at a point in time. `as_of` is the "
            "instant the quotes describe and `known_at` is when the snapshot was "
            "taken.\n\n"
            "This dataset is unlike every other one here: it CANNOT be backfilled. "
            "No free source sells historical option chains, so the history begins "
            "on the day the collector first runs and every day it does not run is "
            "a day that can never be recovered. Treat a gap in it as permanent."
        ),
        columns={
            "expiry": pl.Date(),
            "strike": pl.Float64(),
            "right": pl.Utf8(),
            "bid": pl.Float64(),
            "ask": pl.Float64(),
            "last": pl.Float64(),
            "volume": pl.Float64(),
            "open_interest": pl.Float64(),
            "implied_vol": pl.Float64(),
            "delta": pl.Float64(),
            "gamma": pl.Float64(),
            "vega": pl.Float64(),
            "theta": pl.Float64(),
            "underlying_price": pl.Float64(),
        },
        key=("symbol", "expiry", "strike", "right"),
        required=("expiry", "strike", "right"),
    ),
    "insider_transactions": DatasetSchema(
        name="insider_transactions",
        description=(
            "Trades by a company's own officers, directors and 10% owners, one row "
            "per transaction line of an SEC Form 4 or 5. `as_of` is the TRANSACTION "
            "date and `known_at` is the instant EDGAR accepted the filing -- usually "
            "two business days later, and sometimes months. `symbol` is the issuer's "
            "ticker. `transaction_code` is the SEC's own: P and S are open-market "
            "purchases and sales; A, M, F and G are grants, option exercises, tax "
            "withholding and gifts, which are not a view on the share price. An "
            "amendment (4/A) is a new filing with its own accession and does NOT "
            "replace the rows it corrects."
        ),
        columns={
            "issuer_cik": pl.Int64(),
            "issuer_name": pl.Utf8(),
            "owner_cik": pl.Int64(),
            "owner_name": pl.Utf8(),
            "is_director": pl.Boolean(),
            "is_officer": pl.Boolean(),
            "is_ten_percent_owner": pl.Boolean(),
            "officer_title": pl.Utf8(),
            "form": pl.Utf8(),
            "accession": pl.Utf8(),
            "line": pl.Int64(),
            "security_title": pl.Utf8(),
            "is_derivative": pl.Boolean(),
            "transaction_code": pl.Utf8(),
            "acquired_disposed": pl.Utf8(),
            "shares": pl.Float64(),
            "price": pl.Float64(),
            "shares_owned_after": pl.Float64(),
            "ownership": pl.Utf8(),
            "planned_10b5_1": pl.Boolean(),
        },
        key=("symbol", "accession", "line"),
        required=("accession", "line", "owner_name", "transaction_code", "is_derivative"),
    ),
    "institutional_holdings": DatasetSchema(
        name="institutional_holdings",
        description=(
            "Quarter-end long positions of an institutional manager, from SEC Form "
            "13F. `symbol` is the MANAGER's CIK -- the entity a job asks for -- and "
            "the security is identified by `cusip`, because a 13F carries no ticker. "
            "`as_of` is the quarter end the positions are counted on and `known_at` "
            "is the instant EDGAR accepted the filing, up to 45 days later: the "
            "holdings are already six weeks stale on the day they become knowable. "
            "One row per (cusip, put_call, shares_type), summed over the lines a "
            "manager splits a position across. `value_usd` is in dollars for every "
            "era, including filings from before 2023 that reported thousands."
        ),
        columns={
            "manager_name": pl.Utf8(),
            "cusip": pl.Utf8(),
            "issuer_name": pl.Utf8(),
            "title_of_class": pl.Utf8(),
            "put_call": pl.Utf8(),
            "shares_type": pl.Utf8(),
            "shares": pl.Float64(),
            "value_usd": pl.Float64(),
            "lines": pl.Int64(),
            "form": pl.Utf8(),
            "amendment_type": pl.Utf8(),
            "accession": pl.Utf8(),
        },
        key=("symbol", "cusip", "put_call", "shares_type"),
        required=("cusip", "put_call", "shares_type", "accession"),
    ),
    "congress_filings": DatasetSchema(
        name="congress_filings",
        description=(
            "The index of financial disclosure documents filed by members of "
            "Congress and candidates: one row per document, whatever its type. "
            "`symbol` is the state and district, `as_of` is the filing date. It "
            "exists so that the gap in `congress_trades` is queryable: a periodic "
            "transaction report (`filing_type` P) listed here with no rows there is "
            "a report that could not be read -- usually a scanned paper filing -- "
            "and NOT a member who did not trade."
        ),
        columns={
            "chamber": pl.Utf8(),
            "doc_id": pl.Utf8(),
            "filing_type": pl.Utf8(),
            "filing_type_name": pl.Utf8(),
            "prefix": pl.Utf8(),
            "first_name": pl.Utf8(),
            "last_name": pl.Utf8(),
            "suffix": pl.Utf8(),
            "year": pl.Int64(),
            "url": pl.Utf8(),
        },
        key=("symbol", "doc_id"),
        required=("chamber", "doc_id", "filing_type", "last_name"),
    ),
    "congress_trades": DatasetSchema(
        name="congress_trades",
        description=(
            "Securities transactions disclosed by members of Congress under the "
            "STOCK Act, one row per transaction line of a periodic transaction "
            "report. `as_of` is the TRANSACTION date and `known_at` when the report "
            "was filed -- the end of that day for the House, which gives no time, "
            "and the minute for the Senate, which does. The law allows 45 days "
            "between them and late filings run to months. `chamber` says which. `symbol` is the ticker the member wrote, "
            "or NO_TICKER for a bond, fund or private holding. The size is a RANGE, "
            "never a figure: `amount_min` and `amount_max` are the bounds of the "
            "bracket, and `amount_max` is null for an open-ended one."
        ),
        columns={
            "chamber": pl.Utf8(),
            "doc_id": pl.Utf8(),
            "line": pl.Int64(),
            "member": pl.Utf8(),
            "state_district": pl.Utf8(),
            "owner": pl.Utf8(),
            "asset": pl.Utf8(),
            "asset_type": pl.Utf8(),
            "transaction_type": pl.Utf8(),
            "notification_date": pl.Date(),
            "amount_min": pl.Float64(),
            "amount_max": pl.Float64(),
            "amount_text": pl.Utf8(),
            "filing_status": pl.Utf8(),
            "url": pl.Utf8(),
        },
        key=("symbol", "doc_id", "line"),
        required=("chamber", "doc_id", "line", "member", "transaction_type", "amount_min"),
    ),
    "fails_to_deliver": DatasetSchema(
        name="fails_to_deliver",
        description=(
            "Shares that failed to settle, per security per day, as reported by "
            "NSCC to the SEC. `as_of` is the SETTLEMENT date and `known_at` is when "
            "the SEC posted the file, two to six weeks later. `quantity` is the "
            "cumulative balance outstanding that day, NOT that day's new fails, so "
            "summing it across days counts the same shares repeatedly. Each row "
            "pairs a CUSIP with a ticker, which makes this the only free bridge "
            "from `institutional_holdings` to a price series."
        ),
        columns={
            "cusip": pl.Utf8(),
            "quantity": pl.Float64(),
            "description": pl.Utf8(),
            "price": pl.Float64(),
        },
        key=("symbol", "cusip"),
        required=("cusip", "quantity"),
    ),
    "instruments": DatasetSchema(
        name="instruments",
        description=(
            "Instrument reference data, appended every time the universe is observed. "
            "This is the survivorship-bias defence: once a symbol is seen it is kept "
            "forever, so a universe can be reconstructed as it was, not as it survives."
        ),
        columns={
            "venue": pl.Utf8(),
            "status": pl.Utf8(),
            "name": pl.Utf8(),
            "currency": pl.Utf8(),
            "base_asset": pl.Utf8(),
            "quote_asset": pl.Utf8(),
        },
        key=("symbol", "venue"),
        required=("venue", "status"),
    ),
}


def get_schema(dataset: str) -> DatasetSchema:
    try:
        return DATASETS[dataset]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {dataset!r}; known datasets: {known}") from None


def empty_frame(dataset: str) -> pl.DataFrame:
    """An empty frame with the dataset's exact schema.

    Sources return this when a symbol genuinely has no observations in a window --
    which is different from a failed request, and must never be produced by one.
    """
    return pl.DataFrame(schema=get_schema(dataset).polars_schema)
