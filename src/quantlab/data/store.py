"""The parquet lake and its DuckDB query layer.

Layout, Hive-style, as the spec requires::

    lake/source=<source>/dataset=<dataset>/asset_class=<class>/year=<YYYY>/part-*.parquet

Three properties are load-bearing:

**Append-only.** Ingest never rewrites a file. A restatement arrives as a new row
with a later ``known_at``; the superseded value stays on disk, which is what makes
"what did we believe on date D" answerable at all. Reads resolve duplicates by
taking the latest ``known_at`` at or before the query's as-of instant.

**Path pruning.** Filters are pushed into the glob, so a query for one dataset in
one year touches one directory rather than scanning the lake.

**No server.** DuckDB runs in-process over the parquet files. Postgres exists in
this system only for paper-trading state and the run registry.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from quantlab.config import get_settings
from quantlab.data.catalogue import AssetClass
from quantlab.data.schemas import DATASETS, DatasetSchema, get_schema
from quantlab.logging import get_logger
from quantlab.paths import Layout

if TYPE_CHECKING:  # pragma: no cover
    import duckdb

    from quantlab.data.pit import Snapshot

__all__ = ["DatasetStat", "Store", "WriteResult", "utcnow"]

log = get_logger("quantlab.data.store")


def utcnow() -> dt.datetime:
    """Current UTC instant, timezone-aware. The only clock this package reads."""
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True, slots=True)
class WriteResult:
    dataset: str
    source: str
    rows: int
    files: tuple[Path, ...]

    @property
    def partitions(self) -> int:
        return len(self.files)


@dataclass(frozen=True, slots=True)
class DatasetStat:
    source: str
    dataset: str
    asset_class: str
    rows: int
    symbols: int
    files: int
    bytes: int
    first_as_of: dt.datetime | None
    last_as_of: dt.datetime | None
    last_known_at: dt.datetime | None

    @property
    def staleness(self) -> dt.timedelta | None:
        """How long since the newest observation became knowable."""
        if self.last_known_at is None:
            return None
        return utcnow() - self.last_known_at


def _as_utc(
    moment: dt.datetime | dt.date | str, *, boundary: Literal["start", "end"] = "end"
) -> dt.datetime:
    """Normalise a moment to a timezone-aware UTC datetime.

    A bare ``date`` is widened to whichever end of that day the caller means:
    ``start`` gives 00:00:00, ``end`` gives 23:59:59.999999. Both are needed. If a
    bare date always meant end-of-day, then ``scan(start=T, end=T)`` would exclude
    every bar that closed during T and return nothing -- a silently empty result,
    which is precisely the failure this platform exists to prevent.

    The snapshot instant uses the ``end`` boundary, so ``as_of(T)`` includes the
    close of T. That matches the rebalance convention in spec section 6.4 -- signal
    computed on the close of T, traded at T+1 -- and is recorded in
    docs/ASSUMPTIONS.md. Pass a datetime when you need intraday precision.
    """
    if isinstance(moment, str):
        # "2024-01-04" is a date and must be widened to a day boundary. Passing it
        # to datetime.fromisoformat yields a NAIVE midnight, which then trips the
        # timezone guard below with a message about a timezone the caller never
        # wrote -- confusing, and it makes `--as-of 2024-01-04` fail outright.
        moment = (
            dt.datetime.fromisoformat(moment)
            if ("T" in moment or " " in moment)
            else dt.date.fromisoformat(moment)
        )
    if isinstance(moment, dt.datetime):
        if moment.tzinfo is None:
            raise ValueError(
                f"naive datetime {moment!r}: every instant in QuantLab is explicit "
                "UTC, because a session close is only unambiguous with a timezone"
            )
        return moment.astimezone(dt.UTC)
    edge = dt.time.min if boundary == "start" else dt.time.max
    return dt.datetime.combine(moment, edge, tzinfo=dt.UTC)


class Store:
    """Read and write the parquet lake."""

    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout or get_settings().layout

    # ------------------------------------------------------------------ paths --
    @property
    def lake(self) -> Path:
        return self.layout.lake

    def partition_dir(self, source: str, dataset: str, asset_class: str, year: int) -> Path:
        return (
            self.lake
            / f"source={source}"
            / f"dataset={dataset}"
            / f"asset_class={asset_class}"
            / f"year={year}"
        )

    def _globs(
        self,
        dataset: str,
        *,
        source: str | None = None,
        asset_class: str | None = None,
        years: Iterable[int] | None = None,
    ) -> list[str]:
        source_part = f"source={source}" if source else "source=*"
        class_part = f"asset_class={asset_class}" if asset_class else "asset_class=*"
        year_parts = [f"year={y}" for y in years] if years is not None else ["year=*"]
        return [
            str(self.lake / source_part / f"dataset={dataset}" / class_part / yp / "*.parquet")
            for yp in year_parts
        ]

    def _existing_files(self, globs: Sequence[str]) -> list[Path]:
        found: list[Path] = []
        for pattern in globs:
            relative = Path(pattern).relative_to(self.lake)
            found.extend(sorted(self.lake.glob(str(relative))))
        return found

    # ------------------------------------------------------------------ write --
    def write(
        self,
        frame: pl.DataFrame,
        *,
        asset_class: AssetClass | str,
        validate: bool = True,
    ) -> WriteResult:
        """Append a validated frame to the lake, partitioned by year of ``as_of``.

        ``source`` and ``dataset`` are read from the frame's own columns, so a
        frame cannot be filed under a source that did not produce it.
        """
        if frame.height == 0:
            # Writing nothing is legitimate -- a symbol may genuinely have no
            # observations in a window -- but it must not create empty partitions
            # that later read as "we have this data".
            return WriteResult(dataset="", source="", rows=0, files=())

        for column in ("source", "dataset"):
            if column not in frame.columns:
                raise ValueError(f"frame has no `{column}` column; cannot place it in the lake")

        sources = frame["source"].unique().to_list()
        datasets = frame["dataset"].unique().to_list()
        if len(sources) != 1 or len(datasets) != 1:
            raise ValueError(
                f"write() takes one source and one dataset per call, got "
                f"sources={sources} datasets={datasets}"
            )
        source, dataset = str(sources[0]), str(datasets[0])
        schema = get_schema(dataset)
        if validate:
            frame = schema.validate(frame)

        klass = asset_class.value if isinstance(asset_class, AssetClass) else str(asset_class)
        written: list[Path] = []

        for (year,), part in frame.with_columns(pl.col("as_of").dt.year().alias("_year")).group_by(
            "_year", maintain_order=True
        ):
            part = part.drop("_year")
            directory = self.partition_dir(source, dataset, klass, int(year))  # type: ignore[arg-type,unused-ignore]
            directory.mkdir(parents=True, exist_ok=True)

            # Filename is content-addressed over the OBSERVATIONS, deliberately
            # excluding `ingested_at`. Without that exclusion every nightly run
            # would write a fresh partition for the same overlap window -- the
            # rows are identical except for the download timestamp -- and a year
            # of incremental ingest would leave 365 redundant files that reads
            # then have to deduplicate. A genuine restatement still changes a
            # value, so it still gets its own partition.
            digest = hashlib.sha256(
                part.drop("ingested_at").sort(schema.dedup_key).write_ipc(None).getbuffer()
            ).hexdigest()[:16]
            path = directory / f"part-{digest}.parquet"
            if path.exists():
                log.debug("store.write.duplicate", path=str(path), rows=part.height)
                written.append(path)
                continue

            part.write_parquet(path, compression="zstd", statistics=True)
            written.append(path)

        log.info(
            "store.write",
            source=source,
            dataset=dataset,
            asset_class=klass,
            rows=frame.height,
            files=len(written),
        )
        return WriteResult(dataset=dataset, source=source, rows=frame.height, files=tuple(written))

    # ------------------------------------------------------------------- read --
    def scan(
        self,
        dataset: str,
        *,
        source: str | None = None,
        asset_class: str | None = None,
        symbols: Sequence[str] | None = None,
        start: dt.datetime | dt.date | str | None = None,
        end: dt.datetime | dt.date | str | None = None,
        known_before: dt.datetime | dt.date | str | None = None,
        dedup: bool = True,
    ) -> pl.LazyFrame:
        """Lazily scan one dataset.

        ``known_before`` is the point-in-time filter: only rows that were knowable
        at or before that instant are returned. It is applied *before* duplicate
        resolution, so a restatement filed after the as-of date cannot displace the
        value that was actually on the record then.
        """
        schema: DatasetSchema = get_schema(dataset)
        start_at = _as_utc(start, boundary="start") if start is not None else None
        end_at = _as_utc(end, boundary="end") if end is not None else None
        known_at = _as_utc(known_before, boundary="end") if known_before is not None else None

        years = None
        if start_at is not None and end_at is not None:
            years = range(start_at.year, end_at.year + 1)

        files = self._existing_files(
            self._globs(dataset, source=source, asset_class=asset_class, years=years)
        )
        if not files:
            return pl.LazyFrame(schema=schema.polars_schema)

        frame = pl.scan_parquet([str(p) for p in files])

        if known_at is not None:
            frame = frame.filter(pl.col("known_at") <= known_at)
        if symbols is not None:
            frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
        if start_at is not None:
            frame = frame.filter(pl.col("as_of") >= start_at)
        if end_at is not None:
            frame = frame.filter(pl.col("as_of") <= end_at)

        if dedup:
            # Latest known value wins. Sorting by ingested_at breaks ties between two
            # rows that became knowable at the same instant, which happens whenever a
            # source reports to date rather than timestamp precision.
            frame = frame.sort(["known_at", "ingested_at"]).unique(
                subset=list(schema.dedup_key), keep="last", maintain_order=True
            )

        return frame.sort(["symbol", "as_of"])

    def read(self, dataset: str, **kwargs: Any) -> pl.DataFrame:
        """Eager :meth:`scan`."""
        return self.scan(dataset, **kwargs).collect()

    def symbols(self, dataset: str, *, source: str | None = None) -> list[str]:
        frame = self.scan(dataset, source=source, dedup=False)
        return sorted(frame.select("symbol").unique().collect()["symbol"].to_list())

    # --------------------------------------------------------------- duckdb ----
    def connect(self) -> duckdb.DuckDBPyConnection:
        """An in-memory DuckDB connection with every populated dataset as a view.

        This connection can read the whole lake, including rows that were not
        knowable at any particular date. It is for data inspection and ingest
        health checks, **not** for signals -- use :meth:`as_of` for those.
        """
        import duckdb

        connection = duckdb.connect(database=":memory:", read_only=False)
        connection.execute("SET TimeZone='UTC'")
        for dataset in DATASETS:
            files = self._existing_files(self._globs(dataset))
            if not files:
                continue
            paths = ", ".join(f"'{p}'" for p in files)
            # Both interpolations are ours: `dataset` is a key of the DATASETS
            # registry and the paths come from globbing our own lake. No external
            # input reaches this string.
            statement = f"CREATE OR REPLACE VIEW {dataset} AS SELECT * FROM read_parquet([{paths}])"  # noqa: S608
            connection.execute(statement)
        return connection

    def sql(self, query: str) -> pl.DataFrame:
        """Run SQL against the whole lake. Not point-in-time; not for signals."""
        with self.connect() as connection:
            return connection.execute(query).pl()

    # -------------------------------------------------------------- snapshot ---
    def as_of(self, moment: dt.datetime | dt.date | str) -> Snapshot:
        """Return a point-in-time view of everything knowable at ``moment``.

        This is the only interface signals may use. See :mod:`quantlab.data.pit`.
        """
        from quantlab.data.pit import Snapshot

        return Snapshot(self, _as_utc(moment, boundary="end"))

    # ----------------------------------------------------------------- stats ---
    def stats(self) -> list[DatasetStat]:
        """Per-partition-group inventory, used by `quantlab data status`."""
        out: list[DatasetStat] = []
        for dataset in DATASETS:
            files = self._existing_files(self._globs(dataset))
            if not files:
                continue
            groups: dict[tuple[str, str], list[Path]] = {}
            for path in files:
                parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
                groups.setdefault((parts["source"], parts["asset_class"]), []).append(path)

            for (source, asset_class), paths in sorted(groups.items()):
                frame = (
                    pl.scan_parquet([str(p) for p in paths])
                    .select(
                        pl.len().alias("rows"),
                        pl.col("symbol").n_unique().alias("symbols"),
                        pl.col("as_of").min().alias("first_as_of"),
                        pl.col("as_of").max().alias("last_as_of"),
                        pl.col("known_at").max().alias("last_known_at"),
                    )
                    .collect()
                )
                row = frame.row(0, named=True)
                out.append(
                    DatasetStat(
                        source=source,
                        dataset=dataset,
                        asset_class=asset_class,
                        rows=int(row["rows"]),
                        symbols=int(row["symbols"]),
                        files=len(paths),
                        bytes=sum(p.stat().st_size for p in paths),
                        first_as_of=row["first_as_of"],
                        last_as_of=row["last_as_of"],
                        last_known_at=row["last_known_at"],
                    )
                )
        return out
