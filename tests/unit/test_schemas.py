"""Schema validation is strict on purpose.

A source that quietly returns strings where floats are expected produces a
backtest that silently skips those rows, and a skipped row is indistinguishable
from a missing one.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.conftest import bar

from quantlab.data.schemas import (
    DATASETS,
    TIME_COLUMNS,
    UTC_DATETIME,
    SchemaError,
    empty_frame,
    get_schema,
)


def _frame(**overrides: object) -> pl.DataFrame:
    row = bar("AAPL", dt.date(2024, 1, 3), 100.0)
    row.update(overrides)
    return pl.DataFrame([row])


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_dataset_carries_the_three_timestamps(name: str) -> None:
    schema = get_schema(name).polars_schema
    for column in TIME_COLUMNS:
        assert schema[column] == UTC_DATETIME, f"{name}.{column} is not UTC microseconds"


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_dataset_documents_itself(name: str) -> None:
    assert len(DATASETS[name].description) > 80, f"{name} has no substantive description"


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_dedup_key_always_includes_as_of(name: str) -> None:
    """Without as_of in the key, two observations of the same symbol on different
    days would collapse into one."""
    assert "as_of" in DATASETS[name].dedup_key


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_empty_frame_matches_the_schema(name: str) -> None:
    frame = empty_frame(name)
    assert frame.height == 0
    assert dict(frame.schema) == get_schema(name).polars_schema


def test_valid_frame_passes_through_with_column_order_normalised() -> None:
    schema = get_schema("ohlcv_daily")
    shuffled = _frame().select(reversed(_frame().columns))
    out = schema.validate(shuffled)
    assert out.columns == list(schema.polars_schema)


def test_missing_column_is_named() -> None:
    with pytest.raises(SchemaError, match=r"missing columns \['close'\]"):
        get_schema("ohlcv_daily").validate(_frame().drop("close"))


def test_unexpected_column_is_rejected_rather_than_dropped() -> None:
    """Silently dropping an extra column hides a source that started returning
    something new -- which is usually the first sign its format changed."""
    with pytest.raises(SchemaError, match="unexpected columns"):
        get_schema("ohlcv_daily").validate(_frame().with_columns(surprise=pl.lit(1)))


def test_castable_type_is_coerced() -> None:
    frame = _frame().with_columns(pl.col("volume").cast(pl.Int64()))
    out = get_schema("ohlcv_daily").validate(frame)
    assert out.schema["volume"] == pl.Float64()


def test_uncastable_type_raises() -> None:
    frame = _frame().with_columns(close=pl.lit("not a number"))
    with pytest.raises(SchemaError, match="cannot cast"):
        get_schema("ohlcv_daily").validate(frame)


def test_naive_timestamp_raises_rather_than_being_localised() -> None:
    frame = _frame().with_columns(pl.col("as_of").dt.replace_time_zone(None))
    with pytest.raises(SchemaError, match="timezone-aware UTC"):
        get_schema("ohlcv_daily").validate(frame)


def test_non_utc_timestamp_raises() -> None:
    """A New York-localised timestamp compares fine but partitions by the wrong
    year around new year, which is the kind of off-by-one nobody finds."""
    frame = _frame().with_columns(pl.col("as_of").dt.convert_time_zone("America/New_York"))
    with pytest.raises(SchemaError, match="timezone-aware UTC"):
        get_schema("ohlcv_daily").validate(frame)


def test_null_in_a_required_column_raises() -> None:
    frame = _frame().with_columns(close=pl.lit(None, dtype=pl.Float64()))
    with pytest.raises(SchemaError, match="may not be null"):
        get_schema("ohlcv_daily").validate(frame)


def test_null_in_an_optional_column_is_fine() -> None:
    frame = _frame().with_columns(adj_close=pl.lit(None, dtype=pl.Float64()))
    assert get_schema("ohlcv_daily").validate(frame).height == 1


def test_known_at_before_as_of_raises() -> None:
    frame = _frame(known_at=dt.datetime(2020, 1, 1, tzinfo=dt.UTC))
    with pytest.raises(SchemaError, match="knowable before they existed"):
        get_schema("ohlcv_daily").validate(frame)


def test_unknown_dataset_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known datasets"):
        get_schema("ohlcv_hourly_but_typoed")
