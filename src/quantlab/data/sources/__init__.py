"""One module per external API.

Every source module exposes the uniform interface declared in `sources.base`:

    fetch(symbols, start, end) -> polars.DataFrame

plus a module-level ``SPEC`` naming its entry in :mod:`quantlab.data.catalogue`.
Source modules land in Milestone 2 (FRED, Stooq/Yahoo, Binance) and Milestone 9
(the rest).
"""

from __future__ import annotations

__all__: list[str] = []
