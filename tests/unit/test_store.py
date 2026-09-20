"""The lake: partitioning, append-only semantics, and duplicate resolution."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.conftest import bar

from quantlab.data.store import Store

JAN_3 = dt.date(2024, 1, 3)


def test_partition_path_is_hive_style(store: Store) -> None:
    store.write(pl.DataFrame([bar("AAPL", JAN_3, 100.0)]), asset_class="equity")
    files = list(store.lake.rglob("*.parquet"))
    assert len(files) == 1
    parts = files[0].parts
    assert "source=yahoo" in parts
    assert "dataset=ohlcv_daily" in parts
    assert "asset_class=equity" in parts
    assert "year=2024" in parts


def test_rows_are_split_across_year_partitions(store: Store) -> None:
    rows = [bar("AAPL", JAN_3, 100.0), bar("AAPL", dt.date(2023, 1, 3), 90.0)]
    result = store.write(pl.DataFrame(rows), asset_class="equity")
    assert result.partitions == 2
    years = {p.parent.name for p in result.files}
    assert years == {"year=2023", "year=2024"}


def test_writing_identical_data_twice_is_a_no_op(store: Store) -> None:
    """Content-addressed filenames make re-ingest idempotent, which is what lets
    the incremental overlap window be generous without bloating the lake."""
    frame = pl.DataFrame([bar("AAPL", JAN_3, 100.0)])
    store.write(frame, asset_class="equity")
    store.write(frame, asset_class="equity")
    assert len(list(store.lake.rglob("*.parquet"))) == 1
    assert store.read("ohlcv_daily").height == 1


def test_writing_an_empty_frame_creates_no_partition(store: Store) -> None:
    """An empty result is legitimate -- a symbol may have no bars in a window --
    but it must not leave a partition that later reads as 'we have this data'."""
    from quantlab.data.schemas import empty_frame

    result = store.write(empty_frame("ohlcv_daily"), asset_class="equity")
    assert result.rows == 0
    assert not list(store.lake.rglob("*.parquet"))


def test_write_refuses_mixed_sources(store: Store) -> None:
    rows = [bar("AAPL", JAN_3, 100.0), bar("AAPL", JAN_3, 100.0, source="stooq")]
    with pytest.raises(ValueError, match="one source and one dataset"):
        store.write(pl.DataFrame(rows), asset_class="equity")


def test_latest_known_value_wins_on_read(populated_store: Store) -> None:
    frame = populated_store.read("ohlcv_daily", symbols=["AAPL"])
    restated = frame.filter(pl.col("as_of") == dt.datetime(2024, 1, 3, 21, tzinfo=dt.UTC))
    assert restated.height == 1, "duplicates must collapse to one row per key"
    assert restated["close"].item() == 999.0


def test_superseded_values_stay_on_disk(populated_store: Store) -> None:
    """Append-only is what makes 'what did we believe then' answerable. If the
    restatement overwrote the original, no snapshot could ever recover it."""
    raw = populated_store.scan("ohlcv_daily", symbols=["AAPL"], dedup=False).collect()
    jan3 = raw.filter(pl.col("as_of") == dt.datetime(2024, 1, 3, 21, tzinfo=dt.UTC))
    assert jan3.height == 2
    assert set(jan3["close"].to_list()) == {100.0, 999.0}


def test_known_before_filter_precedes_duplicate_resolution(populated_store: Store) -> None:
    """Order matters: filtering after deduplication would let a future restatement
    win the key and then be removed, leaving no value at all."""
    frame = populated_store.read("ohlcv_daily", symbols=["AAPL"], known_before=dt.date(2024, 3, 1))
    jan3 = frame.filter(pl.col("as_of") == dt.datetime(2024, 1, 3, 21, tzinfo=dt.UTC))
    assert jan3.height == 1
    assert jan3["close"].item() == 100.0


def test_scan_filters_symbols_and_window(populated_store: Store) -> None:
    frame = populated_store.read(
        "ohlcv_daily",
        symbols=["MSFT"],
        start=dt.date(2024, 1, 4),
        end=dt.date(2024, 1, 5),
    )
    assert frame["symbol"].unique().to_list() == ["MSFT"]
    assert frame.height == 2


def test_scan_of_an_empty_dataset_returns_the_right_schema(store: Store) -> None:
    from quantlab.data.schemas import get_schema

    frame = store.read("funding_rate")
    assert frame.height == 0
    assert dict(frame.schema) == get_schema("funding_rate").polars_schema


def test_results_are_sorted_deterministically(populated_store: Store) -> None:
    """Spec section 1: a given commit and snapshot must produce identical results.
    Unordered reads make downstream floating-point reductions vary run to run."""
    first = populated_store.read("ohlcv_daily")
    second = populated_store.read("ohlcv_daily")
    assert first.equals(second)
    assert first["symbol"].to_list() == sorted(first["symbol"].to_list())


def test_symbols_listing(populated_store: Store) -> None:
    assert populated_store.symbols("ohlcv_daily") == ["AAPL", "MSFT"]


def test_stats_report_span_and_staleness(populated_store: Store) -> None:
    stats = populated_store.stats()
    assert len(stats) == 1
    stat = stats[0]
    assert stat.source == "yahoo"
    assert stat.symbols == 2
    assert stat.rows == 7
    assert stat.first_as_of == dt.datetime(2024, 1, 3, 21, tzinfo=dt.UTC)
    assert stat.staleness is not None and stat.staleness.total_seconds() > 0


def test_duckdb_view_is_registered_per_dataset(populated_store: Store) -> None:
    out = populated_store.sql("SELECT count(*) AS n FROM ohlcv_daily")
    assert out.row(0, named=True)["n"] == 7


def test_bare_date_means_end_of_day(store: Store) -> None:
    """Documented in docs/ASSUMPTIONS.md: a bare date is the END of that day, so
    `as_of(T)` includes T's close, matching the rebalance convention."""
    snapshot = store.as_of(dt.date(2024, 1, 4))
    assert snapshot.as_of.hour == 23
    assert snapshot.as_of.date() == dt.date(2024, 1, 4)


def test_single_day_window_is_not_silently_empty(populated_store: Store) -> None:
    """`start` widens to the beginning of its day and `end` to the end of its day.

    Before this was fixed, both widened to end-of-day, so asking for a single
    session returned nothing at all -- an empty result indistinguishable from
    "this symbol did not trade".
    """
    frame = populated_store.read(
        "ohlcv_daily", symbols=["MSFT"], start=dt.date(2024, 1, 4), end=dt.date(2024, 1, 4)
    )
    assert frame.height == 1
    assert frame["as_of"].item() == dt.datetime(2024, 1, 4, 21, tzinfo=dt.UTC)


def test_reingesting_the_same_observations_later_is_still_a_no_op(store: Store) -> None:
    """The content hash covers the observations, not `ingested_at`.

    Found by re-running a live ingest: the same 5,030 bars appeared twice in the
    raw lake because the download timestamp differed. Reads deduplicate, so the
    numbers were right, but a year of nightly overlap windows would leave hundreds
    of redundant partitions for every read to collapse.
    """
    first = pl.DataFrame([bar("AAPL", JAN_3, 100.0)])
    later = first.with_columns(
        ingested_at=pl.lit(dt.datetime(2026, 1, 1, tzinfo=dt.UTC)).cast(first.schema["ingested_at"])
    )
    store.write(first, asset_class="equity")
    store.write(later, asset_class="equity")

    assert len(list(store.lake.rglob("*.parquet"))) == 1
    assert store.scan("ohlcv_daily", dedup=False).collect().height == 1


def test_a_restatement_still_gets_its_own_partition(store: Store) -> None:
    """The exclusion must not swallow real corrections: a changed value is a
    different observation and must be kept alongside the one it supersedes."""
    original = pl.DataFrame([bar("AAPL", JAN_3, 100.0)])
    restated = pl.DataFrame(
        [bar("AAPL", JAN_3, 999.0, known_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC))]
    )
    store.write(original, asset_class="equity")
    store.write(restated, asset_class="equity")

    assert store.scan("ohlcv_daily", dedup=False).collect().height == 2
    assert store.read("ohlcv_daily")["close"].item() == 999.0
