"""Ingest orchestration.

A plan is a list of jobs; a job is one fetcher, a symbol list and a date window.
Jobs run in order, each writing to the lake as it completes, so a failure halfway
through a long pull keeps everything already fetched.

Design decisions worth stating:

**Incremental mode has no cursor file.** The resume point is derived from the lake
itself -- the newest ``as_of`` already stored for that (source, dataset) -- minus
an overlap window. A cursor file is state that can disagree with the data it
describes; the data cannot disagree with itself.

**The overlap window is deliberate.** Re-fetching the last few days catches late
corrections and bars that were provisional when first seen. Re-ingesting identical
rows is free: the store is content-addressed, so an unchanged partition is written
once. A genuine correction arrives as a new row with a later ``known_at``, and the
old value stays on disk where a point-in-time query can still find it.

**Failure is loud and total.** A failing job is recorded and the run exits
non-zero. Nothing is substituted, nothing is skipped silently (spec section 13).
"""

from __future__ import annotations

import datetime as dt
import json
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from polars.exceptions import ComputeError

from quantlab.config import Settings, get_settings
from quantlab.data.http import SourceError, SourceUnavailableError
from quantlab.data.schemas import get_schema
from quantlab.data.sources import load_all_sources
from quantlab.data.sources.base import Source, get_fetcher
from quantlab.data.store import Store, utcnow
from quantlab.logging import get_logger

__all__ = [
    "IngestJob",
    "IngestPlan",
    "IngestResult",
    "JobResult",
    "iter_fetchers",
    "load_plan",
    "recent_runs",
    "run_plan",
]

log = get_logger("quantlab.data.ingest")

DEFAULT_OVERLAP_DAYS = 7


@dataclass(frozen=True, slots=True)
class IngestJob:
    """One fetcher, one symbol list, one window."""

    fetcher: str
    symbols: tuple[str, ...] = ()
    start: dt.date | None = None
    end: dt.date | None = None
    options: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    overlap_days: int = DEFAULT_OVERLAP_DAYS
    description: str = ""

    def build(self, settings: Settings) -> Source:
        return get_fetcher(self.fetcher)(settings=settings, **self.options)


@dataclass(frozen=True, slots=True)
class IngestPlan:
    jobs: tuple[IngestJob, ...]
    default_start: dt.date

    def enabled(self) -> tuple[IngestJob, ...]:
        return tuple(job for job in self.jobs if job.enabled)

    def select(self, fetchers: Sequence[str] | None) -> IngestPlan:
        if not fetchers:
            return self
        wanted = set(fetchers)
        unknown = wanted - {job.fetcher for job in self.jobs}
        if unknown:
            raise KeyError(
                f"no job in the plan for {sorted(unknown)}; plan covers "
                f"{sorted({j.fetcher for j in self.jobs})}"
            )
        return IngestPlan(
            jobs=tuple(job for job in self.jobs if job.fetcher in wanted),
            default_start=self.default_start,
        )


@dataclass(frozen=True, slots=True)
class JobResult:
    fetcher: str
    symbols: int
    rows: int
    start: dt.datetime
    end: dt.datetime
    seconds: float
    ok: bool
    error: str | None = None
    skipped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetcher": self.fetcher,
            "symbols": self.symbols,
            "rows": self.rows,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "seconds": round(self.seconds, 2),
            "ok": self.ok,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True, slots=True)
class IngestResult:
    started_at: dt.datetime
    finished_at: dt.datetime
    incremental: bool
    jobs: tuple[JobResult, ...]

    @property
    def rows(self) -> int:
        return sum(job.rows for job in self.jobs)

    @property
    def failures(self) -> tuple[JobResult, ...]:
        return tuple(job for job in self.jobs if not job.ok)

    @property
    def skipped(self) -> tuple[JobResult, ...]:
        return tuple(job for job in self.jobs if job.skipped_reason)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "incremental": self.incremental,
            "rows": self.rows,
            "ok": self.ok,
            "jobs": [job.as_dict() for job in self.jobs],
        }


def load_plan(path: Path) -> IngestPlan:
    """Parse an ingest plan, failing loudly on anything malformed."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    default_start = _as_date(defaults.get("start", "2015-01-01"), f"{path}: defaults.start")

    entries = raw.get("jobs", [])
    if not isinstance(entries, list):
        raise ValueError(f"{path}: `jobs` must be a list, got {type(entries).__name__}")

    load_all_sources()
    jobs: list[IngestJob] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: job #{index} must be a mapping")
        if "fetcher" not in entry:
            raise ValueError(f"{path}: job #{index} has no `fetcher`")
        fetcher = str(entry["fetcher"])
        get_fetcher(fetcher)  # validate the name now, not at 05:30

        jobs.append(
            IngestJob(
                fetcher=fetcher,
                symbols=tuple(str(s) for s in entry.get("symbols", ())),
                start=_as_date(entry["start"], f"{path}:{fetcher}.start")
                if "start" in entry
                else None,
                end=_as_date(entry["end"], f"{path}:{fetcher}.end") if "end" in entry else None,
                options=dict(entry.get("options") or {}),
                enabled=bool(entry.get("enabled", True)),
                overlap_days=int(entry.get("overlap_days", DEFAULT_OVERLAP_DAYS)),
                description=str(entry.get("description", "")),
            )
        )
    return IngestPlan(jobs=tuple(jobs), default_start=default_start)


def _as_date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{where}: {value!r} is not an ISO date (YYYY-MM-DD)") from exc


def _coverage_regressions(
    store: Store, source: str, dataset: str, frame: pl.DataFrame
) -> list[str]:
    """Symbols whose newly-fetched history starts later than what we already hold.

    This is the only reliable detector of Yahoo's intermittent 20-year lookback
    cap. The payload itself cannot be trusted: when Yahoo truncates it also reports
    ``firstTradeDate`` as the start of the truncated range, so the response is
    internally consistent and every in-payload check passes.

    The lake is append-only, so a truncated re-fetch does not destroy the older
    bars -- but on a FIRST ingest there is nothing to compare against and the loss
    is silent. Hence the warning, and the advice in docs/LIMITATIONS.md to re-run
    ingest until `data status` shows the span you expect.
    """
    if frame.height == 0:
        return []

    # Grouped by the schema's identity columns, not by symbol alone. One symbol
    # can carry several independent series in the same dataset -- a CFTC contract
    # appears under the legacy, disaggregated and TFF reports, which begin in
    # 1992, 2006 and 2006 -- and comparing one against another reports a
    # regression on every ingest. A detector that cries wolf is a detector people
    # learn to ignore, and this one is the only defence against Yahoo's silent
    # truncation.
    keys = [c for c in get_schema(dataset).key if c in frame.columns]
    existing = (
        store.scan(dataset, source=source, dedup=False)
        .group_by(keys)
        .agg(pl.col("as_of").min().alias("held_from"))
        .collect()
    )
    if existing.height == 0:
        return []

    incoming = frame.group_by(keys).agg(pl.col("as_of").min().alias("fetched_from"))
    merged = existing.join(incoming, on=keys, how="inner")
    short = merged.filter(pl.col("fetched_from") > pl.col("held_from") + pl.duration(days=10))
    return [
        f"{' / '.join(str(row[k]) for k in keys)}: held from "
        f"{row['held_from'].date()} but this fetch started at "
        f"{row['fetched_from'].date()}"
        for row in short.iter_rows(named=True)
    ]


def _resume_point(
    store: Store, source: str, dataset: str, symbols: Sequence[str]
) -> dt.datetime | None:
    """Newest ``as_of`` already stored for this (source, dataset)."""
    try:
        frame = store.scan(dataset, source=source, symbols=symbols or None, dedup=False)
        newest = frame.select(pl.col("as_of").max()).collect().item()
    except (FileNotFoundError, ComputeError):  # pragma: no cover - empty lake
        return None
    return newest if isinstance(newest, dt.datetime) else None


def run_plan(
    plan: IngestPlan,
    *,
    store: Store | None = None,
    settings: Settings | None = None,
    incremental: bool = False,
    end: dt.date | None = None,
    dry_run: bool = False,
) -> IngestResult:
    """Execute every enabled job in ``plan``."""
    settings = settings or get_settings()
    store = store or Store(settings.layout)
    settings.layout.ensure()
    load_all_sources()

    started_at = utcnow()
    end_at = dt.datetime.combine(end or started_at.date(), dt.time.max, tzinfo=dt.UTC)
    results: list[JobResult] = []

    for job in plan.enabled():
        fetcher_class = get_fetcher(job.fetcher)
        dataset = fetcher_class.dataset
        source_key = fetcher_class.spec.key
        symbols = list(job.symbols) or list(fetcher_class.default_symbols)

        start_at = dt.datetime.combine(job.start or plan.default_start, dt.time.min, tzinfo=dt.UTC)
        if incremental:
            resume = _resume_point(store, source_key, dataset, symbols)
            if resume is not None:
                # Step back over the overlap window so late corrections are seen.
                start_at = max(start_at, resume - dt.timedelta(days=job.overlap_days))

        job_end = dt.datetime.combine(job.end, dt.time.max, tzinfo=dt.UTC) if job.end else end_at
        began = utcnow()

        if dry_run:
            results.append(
                JobResult(
                    fetcher=job.fetcher,
                    symbols=len(symbols),
                    rows=0,
                    start=start_at,
                    end=job_end,
                    seconds=0.0,
                    ok=True,
                    skipped_reason="dry run",
                )
            )
            continue

        if start_at > job_end:
            results.append(
                JobResult(
                    fetcher=job.fetcher,
                    symbols=len(symbols),
                    rows=0,
                    start=start_at,
                    end=job_end,
                    seconds=0.0,
                    ok=True,
                    skipped_reason="already up to date",
                )
            )
            continue

        source = job.build(settings)
        available, reason = source.availability()
        if not available:
            # A missing API key is not a failure of the run: the platform is
            # required to work with zero keys. It is reported, never hidden.
            source.close()
            log.warning("ingest.unavailable", fetcher=job.fetcher, reason=reason)
            results.append(
                JobResult(
                    fetcher=job.fetcher,
                    symbols=len(symbols),
                    rows=0,
                    start=start_at,
                    end=job_end,
                    seconds=0.0,
                    ok=True,
                    skipped_reason=reason,
                )
            )
            continue

        log.info(
            "ingest.job.start",
            fetcher=job.fetcher,
            dataset=dataset,
            symbols=len(symbols),
            start=start_at.isoformat(),
            end=job_end.isoformat(),
        )
        try:
            with source:
                frame = source.fetch(symbols, start_at, job_end)
                get_schema(dataset).validate(frame)
                if not incremental:
                    for regression in _coverage_regressions(store, source_key, dataset, frame):
                        log.warning(
                            "ingest.coverage_regression",
                            fetcher=job.fetcher,
                            detail=regression,
                            action="existing rows are retained; re-run to try for the full span",
                        )
                store.write(frame, asset_class=source.asset_class)
            results.append(
                JobResult(
                    fetcher=job.fetcher,
                    symbols=len(symbols),
                    rows=frame.height,
                    start=start_at,
                    end=job_end,
                    seconds=(utcnow() - began).total_seconds(),
                    ok=True,
                )
            )
            log.info("ingest.job.finish", fetcher=job.fetcher, rows=frame.height)
        except SourceUnavailableError as exc:
            log.warning("ingest.unavailable", fetcher=job.fetcher, error=str(exc))
            results.append(
                JobResult(
                    fetcher=job.fetcher,
                    symbols=len(symbols),
                    rows=0,
                    start=start_at,
                    end=job_end,
                    seconds=(utcnow() - began).total_seconds(),
                    ok=True,
                    skipped_reason=str(exc),
                )
            )
        except (SourceError, ValueError, KeyError) as exc:
            log.error(
                "ingest.job.failed",
                fetcher=job.fetcher,
                error=str(exc),
                traceback=traceback.format_exc(limit=3),
            )
            results.append(
                JobResult(
                    fetcher=job.fetcher,
                    symbols=len(symbols),
                    rows=0,
                    start=start_at,
                    end=job_end,
                    seconds=(utcnow() - began).total_seconds(),
                    ok=False,
                    error=str(exc),
                )
            )

    result = IngestResult(
        started_at=started_at,
        finished_at=utcnow(),
        incremental=incremental,
        jobs=tuple(results),
    )
    if not dry_run:
        _record(settings.layout.state / "ingest_runs.jsonl", result)
    return result


def _record(path: Path, result: IngestResult) -> None:
    """Append the run to the registry, for audit and the ingest-health panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.as_dict()) + "\n")


def recent_runs(state_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Read back the most recent ingest runs."""
    path = state_dir / "ingest_runs.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # pragma: no cover - truncated write
            continue
    return out


def iter_fetchers() -> Iterable[tuple[str, type[Source]]]:
    """Every registered fetcher, loading the source modules first."""
    from quantlab.data.sources.base import SOURCE_REGISTRY

    load_all_sources()
    return sorted(SOURCE_REGISTRY.items())
