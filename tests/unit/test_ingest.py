"""Ingest orchestration: plan parsing, incremental resume, and failure handling."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest
from tests.conftest import bar

from quantlab.config import Settings
from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError, SourceUnavailableError
from quantlab.data.ingest import (
    IngestJob,
    IngestPlan,
    load_plan,
    recent_runs,
    run_plan,
)
from quantlab.data.sources.base import SOURCE_REGISTRY, Source
from quantlab.data.store import Store


class _FakeSource(Source):
    """A fetcher that records its window and returns one bar per requested day."""

    name: ClassVar[str] = "yahoo.fake"
    spec: ClassVar = get_source("yahoo")
    dataset: ClassVar[str] = "ohlcv_daily"
    asset_class: ClassVar = AssetClass.EQUITY
    default_symbols: ClassVar[tuple[str, ...]] = ("AAPL",)

    calls: ClassVar[list[tuple[dt.datetime, dt.datetime]]] = []
    fail_with: ClassVar[Exception | None] = None

    def fetch(self, symbols, start, end):
        type(self).calls.append((start, end))
        if type(self).fail_with is not None:
            raise type(self).fail_with
        sessions = [dt.date(2024, 1, 3), dt.date(2024, 1, 4), dt.date(2024, 1, 5)]
        rows = [
            bar(symbol, day, 100.0 + index)
            for symbol in symbols
            for index, day in enumerate(sessions)
            if start <= dt.datetime(day.year, day.month, day.day, 21, tzinfo=dt.UTC) <= end
        ]
        return pl.DataFrame(rows, schema=self.schema.polars_schema) if rows else self.empty()


@pytest.fixture(autouse=True)
def fake_source() -> None:
    SOURCE_REGISTRY["yahoo.fake"] = _FakeSource
    _FakeSource.calls = []
    _FakeSource.fail_with = None
    yield
    SOURCE_REGISTRY.pop("yahoo.fake", None)


def plan_with(**kwargs: object) -> IngestPlan:
    return IngestPlan(
        jobs=(IngestJob(fetcher="yahoo.fake", **kwargs),),  # type: ignore[arg-type]
        default_start=dt.date(2024, 1, 1),
    )


# ------------------------------------------------------------------ plan parsing --
def test_repo_plan_parses(repo_root: Path) -> None:
    plan = load_plan(repo_root / "configs" / "ingest.yaml")
    assert plan.jobs
    assert {"binance.ohlcv_bars", "yahoo.ohlcv_daily", "alfred.series_observations"} <= {
        job.fetcher for job in plan.jobs
    }


def test_repo_plan_only_references_real_fetchers(repo_root: Path) -> None:
    """A typo would otherwise surface at 05:30 as a job nobody is watching."""
    for job in load_plan(repo_root / "configs" / "ingest.yaml").jobs:
        assert job.fetcher in SOURCE_REGISTRY


def test_stooq_is_disabled_in_the_shipped_plan(repo_root: Path) -> None:
    """Stooq serves an anti-bot challenge rather than CSV; an enabled job that
    fails every night trains the operator to ignore ingest failures."""
    plan = load_plan(repo_root / "configs" / "ingest.yaml")
    stooq = next(job for job in plan.jobs if job.fetcher == "stooq.ohlcv_daily")
    assert not stooq.enabled


def test_unknown_fetcher_fails_at_load(tmp_path: Path) -> None:
    path = tmp_path / "ingest.yaml"
    path.write_text("jobs:\n  - fetcher: bloomberg.everything\n")
    with pytest.raises(KeyError, match="known fetchers"):
        load_plan(path)


def test_bad_date_fails_at_load(tmp_path: Path) -> None:
    path = tmp_path / "ingest.yaml"
    path.write_text("jobs:\n  - fetcher: yahoo.ohlcv_daily\n    start: last tuesday\n")
    with pytest.raises(ValueError, match="not an ISO date"):
        load_plan(path)


def test_select_narrows_the_plan(repo_root: Path) -> None:
    plan = load_plan(repo_root / "configs" / "ingest.yaml")
    narrowed = plan.select(["binance.funding_rate"])
    assert [job.fetcher for job in narrowed.jobs] == ["binance.funding_rate"]
    with pytest.raises(KeyError, match="no job in the plan"):
        plan.select(["nope.nope"])


# ------------------------------------------------------------------------ running --
def test_run_writes_to_the_lake(store: Store, settings: Settings) -> None:
    result = run_plan(plan_with(), store=store, settings=settings)
    assert result.ok
    assert result.rows == 3
    assert store.read("ohlcv_daily").height == 3


def test_default_symbols_are_used_when_none_are_given(store: Store, settings: Settings) -> None:
    run_plan(plan_with(), store=store, settings=settings)
    assert store.symbols("ohlcv_daily") == ["AAPL"]


def test_dry_run_touches_nothing(store: Store, settings: Settings) -> None:
    result = run_plan(plan_with(), store=store, settings=settings, dry_run=True)
    assert not _FakeSource.calls
    assert result.skipped
    assert not list(store.lake.rglob("*.parquet"))
    assert recent_runs(settings.layout.state) == []


def test_incremental_resumes_with_an_overlap(store: Store, settings: Settings) -> None:
    """The overlap re-fetches recent days so late corrections are seen. It is free
    because the store is content-addressed."""
    run_plan(plan_with(), store=store, settings=settings)
    _FakeSource.calls = []

    run_plan(plan_with(overlap_days=1), store=store, settings=settings, incremental=True)
    start, _end = _FakeSource.calls[0]
    newest = dt.datetime(2024, 1, 5, 21, tzinfo=dt.UTC)
    assert start == newest - dt.timedelta(days=1)


def test_incremental_never_starts_before_the_configured_start(
    store: Store, settings: Settings
) -> None:
    """A generous overlap must not silently widen the window past the job's own
    start date -- that would re-request years of history on every nightly run."""
    run_plan(plan_with(), store=store, settings=settings)
    _FakeSource.calls = []

    run_plan(plan_with(overlap_days=365), store=store, settings=settings, incremental=True)
    start, _end = _FakeSource.calls[0]
    assert start == dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def test_full_run_ignores_what_is_already_stored(store: Store, settings: Settings) -> None:
    run_plan(plan_with(), store=store, settings=settings)
    _FakeSource.calls = []
    run_plan(plan_with(), store=store, settings=settings, incremental=False)
    start, _end = _FakeSource.calls[0]
    assert start == dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def test_reingesting_identical_data_does_not_duplicate(store: Store, settings: Settings) -> None:
    run_plan(plan_with(), store=store, settings=settings)
    run_plan(plan_with(), store=store, settings=settings)
    assert store.read("ohlcv_daily").height == 3
    assert len(list(store.lake.rglob("*.parquet"))) == 1


# ----------------------------------------------------------------------- failures --
def test_a_failing_job_is_reported_and_the_run_exits_not_ok(
    store: Store, settings: Settings
) -> None:
    _FakeSource.fail_with = SourceError("yahoo", "upstream is down")
    result = run_plan(plan_with(), store=store, settings=settings)
    assert not result.ok
    assert result.failures[0].error is not None
    assert "upstream is down" in result.failures[0].error


def test_nothing_is_substituted_for_a_failed_source(store: Store, settings: Settings) -> None:
    """Spec section 13. A failed job must leave no data behind, so that a later
    reader cannot mistake a partial pull for a complete one."""
    _FakeSource.fail_with = SourceError("yahoo", "boom")
    run_plan(plan_with(), store=store, settings=settings)
    assert store.read("ohlcv_daily").height == 0


def test_a_missing_api_key_is_skipped_not_failed(store: Store, settings: Settings) -> None:
    """The platform is required to run with zero keys, so an unavailable source is
    reported, never fatal."""
    _FakeSource.fail_with = SourceUnavailableError("yahoo", "no key configured")
    result = run_plan(plan_with(), store=store, settings=settings)
    assert result.ok
    assert result.skipped[0].skipped_reason is not None


def test_later_jobs_still_run_after_an_earlier_failure(store: Store, settings: Settings) -> None:
    class _Good(_FakeSource):
        name: ClassVar[str] = "yahoo.good"
        calls: ClassVar[list[tuple[dt.datetime, dt.datetime]]] = []
        fail_with: ClassVar[Exception | None] = None

    SOURCE_REGISTRY["yahoo.good"] = _Good
    _FakeSource.fail_with = SourceError("yahoo", "boom")
    try:
        plan = IngestPlan(
            jobs=(IngestJob(fetcher="yahoo.fake"), IngestJob(fetcher="yahoo.good")),
            default_start=dt.date(2024, 1, 1),
        )
        result = run_plan(plan, store=store, settings=settings)
    finally:
        SOURCE_REGISTRY.pop("yahoo.good", None)

    assert not result.ok
    assert result.rows == 3, "the healthy job's data was kept"


def test_disabled_jobs_are_not_run(store: Store, settings: Settings) -> None:
    run_plan(plan_with(enabled=False), store=store, settings=settings)
    assert not _FakeSource.calls


# ------------------------------------------------------------------ run registry --
def test_runs_are_recorded_for_audit(store: Store, settings: Settings) -> None:
    run_plan(plan_with(), store=store, settings=settings)
    run_plan(plan_with(), store=store, settings=settings, incremental=True)
    runs = recent_runs(settings.layout.state)
    assert len(runs) == 2
    assert runs[0]["incremental"] is False
    assert runs[1]["incremental"] is True
    assert runs[0]["jobs"][0]["fetcher"] == "yahoo.fake"


def test_coverage_regression_is_warned_about(
    store: Store, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Yahoo intermittently serves 20 years instead of the full history, and the
    response gives no sign of it -- firstTradeDate is moved to match. The only
    reliable detector is comparing against what the lake already holds.
    """
    import logging

    early = [bar("AAPL", dt.date(2020, 1, 2), 50.0)]
    store.write(pl.DataFrame(early), asset_class="equity")

    with caplog.at_level(logging.WARNING):
        run_plan(plan_with(), store=store, settings=settings)

    assert "ingest.coverage_regression" in caplog.text
    assert "AAPL" in caplog.text


def test_coverage_regression_does_not_delete_the_older_history(
    store: Store, settings: Settings
) -> None:
    """Append-only is what limits the damage: a truncated re-fetch adds nothing and
    removes nothing, so the 2020 bar survives."""
    store.write(pl.DataFrame([bar("AAPL", dt.date(2020, 1, 2), 50.0)]), asset_class="equity")
    run_plan(plan_with(), store=store, settings=settings)

    held = store.read("ohlcv_daily", symbols=["AAPL"])
    assert held["as_of"].min() == dt.datetime(2020, 1, 2, 21, tzinfo=dt.UTC)


def test_no_regression_warning_on_a_first_ingest(
    store: Store, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """There is nothing to compare against, which is exactly why a first ingest can
    be silently short. Documented in LIMITATIONS.md rather than papered over."""
    import logging

    with caplog.at_level(logging.WARNING):
        run_plan(plan_with(), store=store, settings=settings)
    assert "ingest.coverage_regression" not in caplog.text
