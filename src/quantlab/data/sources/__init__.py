"""One module per external API.

Every source module exposes :class:`~quantlab.data.sources.base.Source` subclasses
with a uniform ``fetch(symbols, start, end) -> polars.DataFrame`` interface, and
registers them under ``"<source>.<dataset>"``.

Modules are imported explicitly by :func:`load_all_sources` rather than scanned,
so that adding a file is a deliberate act and an import error surfaces as an
import error rather than as a mysteriously absent fetcher.
"""

from __future__ import annotations

import importlib

__all__ = ["SOURCE_MODULES", "load_all_sources"]

SOURCE_MODULES: tuple[str, ...] = (
    "quantlab.data.sources.binance",
    "quantlab.data.sources.eia",
    "quantlab.data.sources.options",
    "quantlab.data.sources.open_asset_pricing",
    "quantlab.data.sources.cboe",
    "quantlab.data.sources.edgar",
    "quantlab.data.sources.cftc",
    "quantlab.data.sources.fred",
    "quantlab.data.sources.house_clerk",
    "quantlab.data.sources.senate_efd",
    "quantlab.data.sources.ken_french",
    "quantlab.data.sources.sec_13f",
    "quantlab.data.sources.sec_ftd",
    "quantlab.data.sources.sec_insider",
    "quantlab.data.sources.stooq",
    "quantlab.data.sources.yahoo",
)


def load_all_sources() -> None:
    """Import every source module, populating the fetcher registry. Idempotent."""
    for module in SOURCE_MODULES:
        importlib.import_module(module)
