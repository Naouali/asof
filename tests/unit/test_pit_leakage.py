"""The leakage suite: deliberately try to see the future, and assert it fails.

Spec section 3.9 asks for exactly this. Every test here is an attack on the
point-in-time contract, written the way the bug would actually arrive -- as an
innocent-looking argument, a convenient SQL query, or a filter that silently
does not hold.

A passing suite does not prove the contract is unbreakable. It proves that the
specific ways a person is likely to break it by accident all raise.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import polars as pl
import pytest
from tests.conftest import bar

from quantlab.data.pit import LookAheadError
from quantlab.data.store import Store

JAN_4 = dt.date(2024, 1, 4)
JAN_5_CLOSE = dt.datetime(2024, 1, 5, 21, tzinfo=dt.UTC)


# ----------------------------------------------------------------------------------
# The basic contract
# ----------------------------------------------------------------------------------
def test_snapshot_hides_bars_that_had_not_closed(populated_store: Store) -> None:
    frame = populated_store.as_of(JAN_4).ohlcv_daily(symbols=["AAPL"])
    assert frame["as_of"].max() < JAN_5_CLOSE
    assert frame.height == 2, "the 5 Jan bar had not closed on 4 Jan"


def test_snapshot_serves_the_value_that_was_on_the_record(populated_store: Store) -> None:
    """The restatement of the 3 Jan close was filed on 1 June.

    A snapshot in March must return the original 100.0, not the revised 999.0.
    Getting this backwards is the single most common way a fundamentals-driven
    backtest becomes fiction.
    """
    march = populated_store.as_of(dt.date(2024, 3, 1)).ohlcv_daily(symbols=["AAPL"])
    assert (
        march.filter(pl.col("as_of") == dt.datetime(2024, 1, 3, 21, tzinfo=dt.UTC))["close"].item()
        == 100.0
    )

    july = populated_store.as_of(dt.date(2024, 7, 1)).ohlcv_daily(symbols=["AAPL"])
    assert (
        july.filter(pl.col("as_of") == dt.datetime(2024, 1, 3, 21, tzinfo=dt.UTC))["close"].item()
        == 999.0
    )


def test_earlier_snapshot_is_a_subset_of_a_later_one(populated_store: Store) -> None:
    """Knowledge only accumulates. A snapshot can never know *less* about a past
    observation than an earlier snapshot did."""
    early = populated_store.as_of(dt.date(2024, 1, 4)).ohlcv_daily()
    late = populated_store.as_of(dt.date(2024, 12, 31)).ohlcv_daily()
    assert set(early["as_of"].to_list()) <= set(late["as_of"].to_list())
    assert early.height <= late.height


# ----------------------------------------------------------------------------------
# Attack 1: ask for it directly
# ----------------------------------------------------------------------------------
def test_requesting_an_end_beyond_the_snapshot_raises(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(LookAheadError, match="was knowable"):
        snapshot.ohlcv_daily(end=dt.date(2024, 6, 1))


def test_requesting_a_start_beyond_the_snapshot_raises(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(LookAheadError, match="after the snapshot instant"):
        snapshot.ohlcv_daily(start=dt.date(2024, 6, 1))


def test_future_window_is_never_silently_truncated(populated_store: Store) -> None:
    """Silently clamping `end` to the snapshot would be friendlier and worse: the
    caller would believe they had tested a window they never received."""
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(LookAheadError):
        snapshot.frame("ohlcv_daily", end=dt.datetime(2030, 1, 1, tzinfo=dt.UTC))


@pytest.mark.parametrize(
    "accessor",
    ["ohlcv_daily", "bars", "series", "funding", "corporate_actions", "instruments"],
)
def test_every_typed_accessor_enforces_the_window(populated_store: Store, accessor: str) -> None:
    """A new accessor that forgets the guard is the obvious future regression."""
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(LookAheadError):
        getattr(snapshot, accessor)(end=dt.date(2030, 1, 1))


# ----------------------------------------------------------------------------------
# Attack 2: go around the accessors with SQL
# ----------------------------------------------------------------------------------
def test_sandboxed_sql_cannot_read_the_lake_from_disk(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    lake = populated_store.lake
    with pytest.raises(duckdb.Error):
        snapshot.sql(
            f"SELECT * FROM read_parquet('{lake}/**/*.parquet')",
            datasets=["ohlcv_daily"],
        )


def test_sandboxed_sql_cannot_glob_the_filesystem(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(duckdb.Error):
        snapshot.sql(f"SELECT * FROM glob('{populated_store.lake}/*')", datasets=["ohlcv_daily"])


def test_sandboxed_sql_cannot_install_an_extension(populated_store: Store) -> None:
    """httpfs would let a query fetch the lake over HTTP and sidestep the filter."""
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(duckdb.Error):
        snapshot.sql("INSTALL httpfs", datasets=["ohlcv_daily"])


def test_sandboxed_sql_cannot_re_enable_external_access(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(duckdb.Error):
        snapshot.sql("SET enable_external_access=true", datasets=["ohlcv_daily"])


def test_sandboxed_sql_sees_only_filtered_rows(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    result = snapshot.sql(
        "SELECT count(*) AS n, max(as_of) AS newest, max(close) AS mx "
        "FROM ohlcv_daily WHERE symbol = 'AAPL'",
        datasets=["ohlcv_daily"],
    )
    row = result.row(0, named=True)
    assert row["newest"] < JAN_5_CLOSE
    assert row["mx"] == 101.0, "the 999.0 restatement was not knowable on 4 Jan"


def test_sandboxed_sql_requires_naming_its_datasets(populated_store: Store) -> None:
    """The sandbox holds nothing by default, so a query cannot reach a dataset the
    caller did not think about."""
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(ValueError, match="name the datasets"):
        snapshot.sql("SELECT 1", datasets=[])
    with pytest.raises(duckdb.Error):
        snapshot.sql("SELECT * FROM funding_rate", datasets=["ohlcv_daily"])


# ----------------------------------------------------------------------------------
# Attack 3: corrupt the data so the filter silently stops working
# ----------------------------------------------------------------------------------
def test_post_condition_catches_a_broken_filter(
    populated_store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a future bug that drops the known_at filter.

    The defence-in-depth check must turn that into a crash rather than into a
    flattering backtest. Without this check, such a regression would be invisible:
    every number would still look plausible.
    """
    original = Store.scan

    def unfiltered(self: Store, dataset: str, **kwargs: object) -> pl.LazyFrame:
        kwargs.pop("known_before", None)  # the bug
        return original(self, dataset, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Store, "scan", unfiltered)

    with pytest.raises(LookAheadError, match="bug in QuantLab"):
        populated_store.as_of(JAN_4).ohlcv_daily()


def test_schema_rejects_data_knowable_before_it_existed(store: Store) -> None:
    """A row whose known_at precedes its as_of claims to have been knowable before
    it happened. That is a source confusing a period end with a publication date,
    and it must never reach the lake."""
    from quantlab.data.schemas import SchemaError

    row = bar("AAPL", dt.date(2024, 1, 3), 100.0)
    row["known_at"] = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    with pytest.raises(SchemaError, match="knowable before they existed"):
        store.write(pl.DataFrame([row]), asset_class="equity")


def test_naive_timestamps_are_rejected(store: Store) -> None:
    """A naive timestamp makes point-in-time comparison ambiguous by hours, which
    is exactly enough to leak a US close into an Asian open."""
    from quantlab.data.schemas import SchemaError

    row = bar("AAPL", dt.date(2024, 1, 3), 100.0)
    row["as_of"] = dt.datetime(2024, 1, 3, 21)
    with pytest.raises(SchemaError, match="timezone-aware UTC"):
        store.write(pl.DataFrame([row]), asset_class="equity")


def test_snapshot_rejects_a_naive_as_of(populated_store: Store) -> None:
    with pytest.raises(ValueError, match="naive datetime"):
        populated_store.as_of(dt.datetime(2024, 1, 4))


# ----------------------------------------------------------------------------------
# Attack 4: the store itself is not point-in-time, and must not pretend to be
# ----------------------------------------------------------------------------------
def test_store_sql_is_not_point_in_time_and_that_is_documented(
    populated_store: Store,
) -> None:
    """`Store.sql` deliberately sees everything. This test pins that it is the
    snapshot, not the store, that carries the guarantee -- so nobody later
    'fixes' the store into a false sense of safety."""
    everything = populated_store.sql("SELECT max(close) AS mx FROM ohlcv_daily")
    assert everything.row(0, named=True)["mx"] == 999.0

    assert "not point-in-time" in (Store.sql.__doc__ or "").lower()


def test_snapshots_are_immutable(populated_store: Store) -> None:
    snapshot = populated_store.as_of(JAN_4)
    with pytest.raises(AttributeError):
        snapshot.as_of = dt.datetime(2030, 1, 1, tzinfo=dt.UTC)  # type: ignore[misc]

    later = snapshot.with_as_of(dt.date(2024, 7, 1))
    assert snapshot.as_of < later.as_of
