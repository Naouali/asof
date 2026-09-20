"""Investable universe construction and filters.

Milestone 2/9. The universe is built from *historical* index constituent lists and
SEC filer lists, never from "tickers that exist today" (spec section 13). Once a
symbol is observed it is retained forever, including after delisting.
"""

from __future__ import annotations

__all__: list[str] = []
