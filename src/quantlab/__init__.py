"""QuantLab -- multi-asset market data ETL built on free sources.

Connects to external sources, fetches their data and lands it in a parquet lake
queried in-process by DuckDB. Every row carries when it became knowable, so the
lake can be read point-in-time; every source's known biases are recorded in the
catalogue rather than left to be rediscovered.

Layout:

    cli, scheduler          -- run and schedule ingests
    data.sources            -- one fetcher per external source
    data.ingest             -- ingest plans, incremental windows
    data.store, data.pit    -- the lake and its point-in-time view
    data.catalogue, schemas -- what each source is, and what its rows must look like
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
