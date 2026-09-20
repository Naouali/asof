"""The instrument view: everything the lake holds about one asset.

The rest of the dashboard is strategy-centric -- it answers "does this idea
work". This answers the question that comes first in practice: *what do I
actually have on this ticker, and what does its history look like?*

**Names, not just codes.** A CFTC contract market code is opaque: nobody looking
for corn types 002602. Names are taken from the data where the source supplies
one, and the *latest* name is used because sources rename markets -- the same
CFTC code has been "WHEAT", "WHEAT-SRW" and, for Treasuries, both "2 YEAR U.S.
TREASURY NOTES" and "UST 2Y NOTE".

**Statistics here are buy-and-hold, and labelled as such.** The Sharpe of holding
an instrument is not a strategy result and must never be read as one; it is
context for whatever is traded on top. It carries no deflation because there is
nothing to deflate -- no search produced it.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from quantlab.config import get_settings
from quantlab.logging import get_logger

__all__ = ["instrument_detail", "instrument_index"]

log = get_logger("quantlab.dashboard.instruments")

#: Datasets scanned for the index, with the column that carries a display name.
SCANNED: dict[str, str | None] = {
    "ohlcv_daily": None,
    "ohlcv_bars": None,
    "fundamentals": None,
    "positioning": "name",
    "series_observations": None,
    "funding_rate": None,
    "chain_snapshot": None,
    "anomaly_catalogue": "name",
}

#: Points kept in a price history sent to a browser.
CHART_POINTS = 900


def _store() -> Any:
    from quantlab.data.store import Store

    return Store(get_settings().layout)


def instrument_index() -> dict[str, Any]:
    """Every symbol in the lake, with where it appears and what it is called.

    Built from the lake rather than from a hand-maintained list, so an
    instrument cannot be present in the data and absent from the catalogue --
    which is the failure that makes a data browser untrustworthy.
    """
    store = _store()
    found: dict[str, dict[str, Any]] = {}

    for dataset, name_column in SCANNED.items():
        try:
            frame = store.scan(dataset, dedup=False).select(
                ["symbol", "as_of", *([name_column] if name_column else [])]
            )
            summary = (
                frame.group_by("symbol")
                .agg(
                    pl.len().alias("rows"),
                    pl.col("as_of").min().alias("first"),
                    pl.col("as_of").max().alias("last"),
                    *(
                        [
                            pl.col(name_column).last().alias("name"),
                            # Every name the source has used. Sources rename
                            # markets -- the CFTC's two-year note has been "2
                            # YEAR U.S. TREASURY NOTES" and is now "UST 2Y NOTE"
                            # -- and someone searching "treasury" should still
                            # find it under either.
                            pl.col(name_column).unique().alias("aliases"),
                        ]
                        if name_column
                        else []
                    ),
                )
                .collect()
            )
        except Exception as exc:
            log.debug("instruments.dataset_skipped", dataset=dataset, error=str(exc)[:120])
            continue

        for row in summary.iter_rows(named=True):
            symbol = str(row["symbol"])
            entry = found.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "name": None,
                    "aliases": set(),
                    "datasets": [],
                    "rows": 0,
                    "first": None,
                    "last": None,
                },
            )
            entry["datasets"].append(dataset)
            entry["rows"] += int(row["rows"])
            entry["name"] = entry["name"] or row.get("name")
            entry["aliases"].update(a for a in (row.get("aliases") or []) if a)
            for key, value in (("first", row["first"]), ("last", row["last"])):
                current = entry[key]
                if value is None:
                    continue
                if current is None or (value < current if key == "first" else value > current):
                    entry[key] = value

    instruments = []
    for entry in found.values():
        instruments.append(
            {
                "symbol": entry["symbol"],
                "name": entry["name"],
                # Searched but not displayed: a table showing four historical
                # names per row is a table nobody scans.
                "aliases": sorted(entry["aliases"] - {entry["name"]}),
                "datasets": sorted(entry["datasets"]),
                "rows": entry["rows"],
                "first": str(entry["first"])[:10] if entry["first"] else None,
                "last": str(entry["last"])[:10] if entry["last"] else None,
                "priced": bool({"ohlcv_daily", "ohlcv_bars"} & set(entry["datasets"])),
            }
        )
    instruments.sort(key=lambda i: (not i["priced"], i["symbol"]))
    return {"instruments": instruments, "count": len(instruments)}


def instrument_detail(symbol: str, as_of: str | None = None) -> dict[str, Any]:
    """Price history, buy-and-hold statistics, and every other row held."""
    store = _store()
    moment = as_of or dt.datetime.now(tz=dt.UTC).date().isoformat()
    snapshot = store.as_of(moment)
    upper = symbol.upper()

    prices, source_dataset = _price_history(snapshot, symbol, upper)
    out: dict[str, Any] = {
        "ok": True,
        "symbol": symbol,
        "as_of": moment,
        "price_dataset": source_dataset,
        "history": [],
        "stats": None,
        "datasets": _dataset_rows(snapshot, symbol, upper),
    }
    if prices is not None and prices.height >= 2:
        out["history"], out["stats"] = _chart_and_stats(prices)

    out["fundamentals"] = _fundamentals(snapshot, upper)
    out["positioning"] = _positioning(snapshot, symbol)
    if not out["history"] and not out["datasets"]:
        return {"ok": False, "error": f"nothing in the lake for {symbol!r}"}
    return out


# ---------------------------------------------------------------------- parts --
def _price_history(
    snapshot: Any, symbol: str, upper: str
) -> tuple[pl.DataFrame | None, str | None]:
    for dataset, column in (("ohlcv_daily", "adj_close"), ("ohlcv_bars", "close")):
        try:
            frame = snapshot.frame(dataset, symbols=[symbol, upper])
        except (KeyError, ValueError):
            continue
        if frame.height == 0:
            continue
        price = column if column in frame.columns else "close"
        return frame.sort("as_of").select("as_of", pl.col(price).alias("close")), dataset
    return None, None


def _chart_and_stats(frame: pl.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    closes = frame["close"].to_numpy().astype(float)
    dates = frame["as_of"].to_list()
    returns = np.diff(closes) / closes[:-1]
    returns = returns[np.isfinite(returns)]

    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    cagr = (closes[-1] / closes[0]) ** (1 / years) - 1 if closes[0] > 0 else 0.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if returns.size > 1 else 0.0
    peak = np.maximum.accumulate(closes)
    drawdown = closes / peak - 1.0

    step = max(1, frame.height // CHART_POINTS)
    history = [
        {"t": str(dates[i])[:10], "close": float(closes[i]), "drawdown": float(drawdown[i])}
        for i in range(0, frame.height, step)
    ]
    stats = {
        "bars": frame.height,
        "first": str(dates[0])[:10],
        "last": str(dates[-1])[:10],
        "years": years,
        "last_close": float(closes[-1]),
        "cagr": float(cagr),
        "volatility": volatility,
        "max_drawdown": float(drawdown.min()),
        # Buy and hold, and labelled so in the UI. Not a strategy result, and it
        # carries no deflation because no search produced it.
        "buy_hold_sharpe": float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
        if returns.size > 1 and returns.std(ddof=1) > 0
        else 0.0,
    }
    return history, stats


def _dataset_rows(snapshot: Any, symbol: str, upper: str) -> list[dict[str, Any]]:
    rows = []
    for dataset in SCANNED:
        try:
            frame = snapshot.frame(dataset, symbols=[symbol, upper])
        except (KeyError, ValueError):
            continue
        if frame.height == 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "rows": frame.height,
                "first": str(frame["as_of"].min())[:10],
                "last": str(frame["as_of"].max())[:10],
            }
        )
    return rows


def _fundamentals(snapshot: Any, upper: str) -> list[dict[str, Any]]:
    """Latest annual figure per metric, on a consistent basis."""
    try:
        frame = snapshot.frame("fundamentals", symbols=[upper])
    except (KeyError, ValueError):
        return []
    if frame.height == 0:
        return []

    from quantlab.signals.equity.profitability import DEFAULT_MAX_STALENESS, select_annual

    picked = select_annual(frame, as_of=snapshot.as_of, max_staleness=DEFAULT_MAX_STALENESS)
    return [
        {"metric": str(row["metric"]), "value": float(row["value"])}
        for row in picked.sort("metric").iter_rows(named=True)
    ]


def _positioning(snapshot: Any, symbol: str) -> list[dict[str, Any]]:
    try:
        frame = snapshot.frame("positioning", symbols=[symbol])
    except (KeyError, ValueError):
        return []
    if frame.height == 0:
        return []
    latest = frame["as_of"].max()
    recent = frame.filter(pl.col("as_of") == latest)
    return [
        {
            "report": str(row["report"]),
            "category": str(row["category"]),
            "measure": str(row["measure"]),
            "value": float(row["value"]),
            "as_of": str(latest)[:10],
        }
        for row in recent.sort("report", "category", "measure").iter_rows(named=True)
    ]
