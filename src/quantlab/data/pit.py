"""The point-in-time contract.

    snapshot = store.as_of(date)   # a view of ALL data knowable at `date`

Anything that asks "what was known on date D" reads through this object and
through nothing else: the app's as-of control is a thin layer over it. The goal is
not to make reading the future *discouraged* -- it is to make it structurally
unavailable, so that a tired analyst at 1am cannot do it by accident.

Four independent mechanisms enforce that:

1. **Filtering.** Every query filters ``known_at <= as_of`` and ``as_of <= as_of``.
   A restatement filed after the snapshot date cannot displace the value that was
   actually on the record then.
2. **Argument guards.** Asking for data beyond the snapshot instant raises
   :class:`LookAheadError` rather than silently returning less than requested.
   A silent truncation trains you to trust a window you did not get.
3. **Post-condition checks.** Every frame leaving this module is re-inspected, and
   a single row with ``known_at`` after the snapshot instant raises. This is
   redundant with (1) on purpose: it converts a future filtering bug from a
   quietly wrong answer into a crash.
4. **A sandboxed SQL surface.** :meth:`Snapshot.sql` runs against a DuckDB
   connection built with ``enable_external_access=false``, holding only
   already-filtered tables. The lake files are unreachable from it, DuckDB refuses
   to re-enable access at runtime, and so arbitrary SQL cannot reach around the
   filter.

There is deliberately no escape hatch. If you need unfiltered data -- for ingest
health, for plotting the full history -- use :class:`~quantlab.data.store.Store`
directly and know that what you are holding is not point-in-time.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from quantlab.data.schemas import DATASETS, get_schema
from quantlab.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from quantlab.data.store import Store

__all__ = ["LookAheadError", "Snapshot"]

log = get_logger("quantlab.data.pit")


class LookAheadError(RuntimeError):
    """An attempt was made to see data that was not knowable at the as-of instant.

    This is never a warning. A view of the past that has seen the future is not
    slightly wrong; it describes a day that never existed.
    """


class Snapshot:
    """An immutable view of the lake as it was knowable at one instant."""

    __slots__ = ("_as_of", "_store")

    def __init__(self, store: Store, as_of: dt.datetime) -> None:
        if as_of.tzinfo is None:
            raise ValueError("snapshot instant must be timezone-aware UTC")
        self._store = store
        self._as_of = as_of.astimezone(dt.UTC)

    @property
    def as_of(self) -> dt.datetime:
        """The instant this snapshot is taken at. Read-only.

        Mutability here would be a hole in every other guarantee in this module:
        code could pass the point-in-time checks, then move the instant forward and
        re-read. `with_as_of` returns a new snapshot instead.
        """
        return self._as_of

    # ------------------------------------------------------------------ core --
    def frame(
        self,
        dataset: str,
        *,
        symbols: Sequence[str] | None = None,
        start: dt.datetime | dt.date | str | None = None,
        end: dt.datetime | dt.date | str | None = None,
        source: str | None = None,
    ) -> pl.DataFrame:
        """Point-in-time read of one dataset.

        ``end`` defaults to the snapshot instant. Passing an ``end`` beyond it, or
        a ``start`` beyond it, raises :class:`LookAheadError`.
        """
        from quantlab.data.store import _as_utc

        get_schema(dataset)  # fail fast on an unknown dataset name

        if end is not None:
            end_at = _as_utc(end, boundary="end")
            if end_at > self.as_of:
                raise LookAheadError(
                    f"requested data up to {end_at.isoformat()} from a snapshot taken "
                    f"at {self.as_of.isoformat()}. Nothing after the snapshot instant "
                    "was knowable. Move the snapshot forward if that is what you mean; "
                    "this is not silently truncated because a window you did not get "
                    "is a window you should not trust."
                )
        else:
            end_at = self.as_of

        if start is not None:
            start_at = _as_utc(start, boundary="start")
            if start_at > self.as_of:
                raise LookAheadError(
                    f"requested data starting {start_at.isoformat()}, which is after "
                    f"the snapshot instant {self.as_of.isoformat()}"
                )
        else:
            start_at = None

        frame = self._store.scan(
            dataset,
            source=source,
            symbols=symbols,
            start=start_at,
            end=end_at,
            known_before=self.as_of,
            dedup=True,
        ).collect()

        return self._enforce(dataset, frame)

    def _enforce(self, dataset: str, frame: pl.DataFrame) -> pl.DataFrame:
        """Post-condition: nothing leaving this object may postdate the snapshot.

        Redundant with the query filter, and kept anyway. The filter is code that
        can acquire a bug; this check turns that bug into a crash instead of into
        a number somebody believes.
        """
        if frame.height == 0:
            return frame
        for column in ("known_at", "as_of"):
            worst = frame[column].max()
            if not isinstance(worst, dt.datetime):
                # A non-datetime here means the schema was bypassed on write.
                raise LookAheadError(
                    f"{dataset}.{column} is {type(worst).__name__}, not a datetime; the "
                    "schema was bypassed and point-in-time safety cannot be verified"
                )
            if worst > self.as_of:
                offenders = frame.filter(pl.col(column) > self.as_of)
                raise LookAheadError(
                    f"point-in-time violation in {dataset}: {offenders.height} rows "
                    f"have {column} after the snapshot instant {self.as_of.isoformat()} "
                    f"(worst {worst.isoformat()}). The snapshot filter did not hold; "
                    "this is a bug in QuantLab, not in your query."
                )
        return frame

    # ------------------------------------------------------- typed accessors --
    def ohlcv_daily(self, **kwargs: Any) -> pl.DataFrame:
        """Daily bars. ``as_of`` is the session close, so an unclosed bar is absent."""
        return self.frame("ohlcv_daily", **kwargs)

    def bars(self, **kwargs: Any) -> pl.DataFrame:
        return self.frame("ohlcv_bars", **kwargs)

    def series(self, **kwargs: Any) -> pl.DataFrame:
        """Macro and rates series, at the vintage that was current at the as-of date."""
        return self.frame("series_observations", **kwargs)

    def funding(self, **kwargs: Any) -> pl.DataFrame:
        return self.frame("funding_rate", **kwargs)

    def corporate_actions(self, **kwargs: Any) -> pl.DataFrame:
        return self.frame("corporate_actions", **kwargs)

    def instruments(self, **kwargs: Any) -> pl.DataFrame:
        """Instrument reference data as observed at or before the as-of date.

        This is how a universe is reconstructed without survivorship bias: a
        symbol delisted in 2019 is present in a 2018 snapshot because we recorded
        it then, and absent from a 2024 one.
        """
        return self.frame("instruments", **kwargs)

    def insider_transactions(self, **kwargs: Any) -> pl.DataFrame:
        """Insider trades whose REPORT had been accepted by the as-of date. A trade
        made last week and not yet reported is absent, as it was for everyone."""
        return self.frame("insider_transactions", **kwargs)

    def institutional_holdings(self, **kwargs: Any) -> pl.DataFrame:
        """13F positions from filings accepted by the as-of date, so for six weeks
        after a quarter ends this still shows the quarter before it."""
        return self.frame("institutional_holdings", **kwargs)

    def congress_trades(self, **kwargs: Any) -> pl.DataFrame:
        """Congressional trades whose report had been filed by the as-of date."""
        return self.frame("congress_trades", **kwargs)

    # ------------------------------------------------------------------- sql --
    def sql(
        self,
        query: str,
        *,
        datasets: Sequence[str],
        symbols: Sequence[str] | None = None,
        start: dt.datetime | dt.date | str | None = None,
    ) -> pl.DataFrame:
        """Run SQL against point-in-time data in a sandbox that cannot reach the lake.

        The named datasets are materialised, already filtered, as tables in a
        DuckDB connection created with ``enable_external_access=false``. That
        connection cannot open a file, install an extension, or re-enable access,
        so no query -- however it is written -- can see past the snapshot instant.

        Materialising is the price of that guarantee. Filter with ``symbols`` and
        ``start`` rather than pulling the whole lake into memory.
        """
        import duckdb

        unknown = [name for name in datasets if name not in DATASETS]
        if unknown:
            raise KeyError(f"unknown datasets {unknown}; known: {sorted(DATASETS)}")
        if not datasets:
            raise ValueError(
                "name the datasets your query reads; the sandbox holds nothing by default"
            )

        connection = duckdb.connect(":memory:", config={"enable_external_access": False})
        try:
            connection.execute("SET TimeZone='UTC'")
            for name in datasets:
                frame = self.frame(name, symbols=symbols, start=start)
                connection.register(name, frame.to_arrow())
            return connection.execute(query).pl()
        finally:
            connection.close()

    # ----------------------------------------------------------------- misc ---
    def with_as_of(self, moment: dt.datetime | dt.date | str) -> Snapshot:
        """A new snapshot at a different instant. Snapshots are never mutated."""
        return self._store.as_of(moment)

    def __repr__(self) -> str:
        return f"Snapshot(as_of={self.as_of.isoformat()})"
