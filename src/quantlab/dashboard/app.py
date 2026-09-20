"""Reporting UI.

Milestone 1 ships the parts that must exist for the stack to be operable: a
liveness endpoint for the Docker healthcheck, the environment report from
:mod:`quantlab.health`, the data catalogue with its caveats, and worker/scheduler
heartbeat status. Paper-trading P&L, strategy comparison and the signal decay
monitor arrive in Milestone 11.

Everything is served from local assets. The dashboard must work with no internet
access (spec section 1), so there are no CDN references anywhere.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from quantlab import __version__
from quantlab.config import get_settings
from quantlab.data.catalogue import SOURCES
from quantlab.health import Status, run_checks, source_availability
from quantlab.runtime import read_heartbeat

__all__ = ["app", "create_app"]

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _heartbeats() -> list[dict[str, Any]]:
    state_dir = get_settings().layout.state
    if not state_dir.exists():
        return []
    out = []
    for path in sorted(state_dir.glob("*.heartbeat.json")):
        status = read_heartbeat(path)
        out.append(
            {
                "name": status.name,
                "healthy": status.healthy,
                "age_seconds": status.age_seconds,
                "detail": status.detail,
            }
        )
    return out


def _lake() -> list[dict[str, Any]]:
    """What the lake holds, and how stale each dataset is."""
    from quantlab.data.store import Store

    out: list[dict[str, Any]] = []
    for stat in Store(get_settings().layout).stats():
        age = stat.staleness
        out.append(
            {
                "source": stat.source,
                "dataset": stat.dataset,
                "asset_class": stat.asset_class,
                "rows": stat.rows,
                "symbols": stat.symbols,
                "megabytes": round(stat.bytes / 1e6, 1),
                "first_as_of": stat.first_as_of.isoformat() if stat.first_as_of else None,
                "last_as_of": stat.last_as_of.isoformat() if stat.last_as_of else None,
                "age_days": round(age.total_seconds() / 86400, 1) if age else None,
            }
        )
    return out


def _symbol_list(value: object) -> list[str] | None:
    """Symbols from a comma or whitespace separated box, or None for the lot."""
    if not value:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = ",".join(str(item) for item in value)
    else:
        return None
    symbols = [part.strip().upper() for part in text.replace("\n", ",").split(",")]
    kept = [s for s in symbols if s]
    return kept or None


def _parse_parameters(raw: str) -> dict[str, list[object]] | None:
    """``name=v1,v2`` lines into a grid, or None if any line is malformed.

    Returning None rather than skipping a bad line is deliberate: a typo that
    silently drops an axis changes the size of the search, and the size of the
    search is the number everything else is judged against.
    """
    axes: dict[str, list[object]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            return None
        name, _, values = line.partition("=")
        parsed = [_coerce_value(v.strip()) for v in values.split(",") if v.strip()]
        if not name.strip() or not parsed:
            return None
        axes[name.strip()] = parsed
    return axes or None


def _coerce_value(text: str) -> object:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _paper_books() -> list[dict[str, Any]]:
    """Every paper book's latest cycle.

    Read-only, and deliberately shows the staleness of each book. A dashboard
    that displays a position without saying when it was last updated invites the
    reader to assume it is current, and a paper loop that silently stopped
    running looks exactly like one holding steady.
    """
    from quantlab.paper.state import PaperState

    directory = get_settings().layout.state / "paper"
    if not directory.exists():
        return []

    books: list[dict[str, Any]] = []
    now = dt.datetime.now(tz=dt.UTC)
    for path in sorted(directory.glob("*.jsonl")):
        state = PaperState(path, path.stem)
        latest = state.latest()
        if latest is None:
            continue
        as_of = dt.datetime.fromisoformat(latest["as_of"])
        books.append(
            {
                "strategy": latest["strategy"],
                "as_of": latest["as_of"],
                "stale_days": round((now - as_of).total_seconds() / 86400.0, 1),
                "cycles": len(state.history()),
                "equity": latest["equity"],
                "gross_exposure": latest["gross_exposure"],
                "net_exposure": latest["net_exposure"],
                "positions": len(latest["positions"]),
                "periods_per_year": round(state.periods_per_year()),
            }
        )
    return books


def _last_ingest() -> dict[str, Any] | None:
    from quantlab.data.ingest import recent_runs

    runs = recent_runs(get_settings().layout.state, limit=1)
    return runs[-1] if runs else None


def create_app() -> FastAPI:
    app = FastAPI(
        title="QuantLab",
        version=__version__,
        description="Multi-asset quantitative research platform.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.get("/health", include_in_schema=False)
    def health() -> JSONResponse:
        """Container liveness. Deliberately cheap: no disk scans, no network."""
        return JSONResponse({"status": "ok", "version": __version__})

    @app.get("/api/health")
    def api_health() -> JSONResponse:
        checks = run_checks()
        worst = max((c.status for c in checks), key=lambda s: ["ok", "warn", "fail"].index(s))
        return JSONResponse(
            {
                "status": worst.value,
                "version": __version__,
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "detail": c.detail,
                        "hint": c.hint,
                    }
                    for c in checks
                ],
                "heartbeats": _heartbeats(),
            }
        )

    @app.get("/api/lake")
    def api_lake() -> JSONResponse:
        return JSONResponse({"datasets": _lake(), "last_ingest": _last_ingest()})

    @app.get("/api/sources")
    def api_sources() -> JSONResponse:
        return JSONResponse(
            {"sources": [a.as_dict() for a in source_availability()], "count": len(SOURCES)}
        )

    @app.get("/api/instruments")
    def api_instruments() -> JSONResponse:
        from quantlab.dashboard.instruments import instrument_index

        return JSONResponse(instrument_index())

    @app.get("/api/instruments/{symbol}")
    def api_instrument(symbol: str) -> JSONResponse:
        from quantlab.dashboard.instruments import instrument_detail

        return JSONResponse(instrument_detail(symbol))

    @app.post("/api/portfolio")
    def api_portfolio(body: dict[str, Any]) -> JSONResponse:
        from quantlab.dashboard.actions import today
        from quantlab.dashboard.portfolio_view import build_portfolio

        symbols = _symbol_list(body.get("symbols")) or []
        return JSONResponse(
            build_portfolio(
                symbols,
                as_of=str(body.get("as_of") or today()),
                method=str(body.get("method") or "risk-parity"),
                lookback=int(body.get("lookback") or 504),
            )
        )

    @app.post("/api/portfolio/history")
    def api_portfolio_history(body: dict[str, Any]) -> JSONResponse:
        from quantlab.dashboard.actions import today
        from quantlab.dashboard.portfolio_view import portfolio_history

        return JSONResponse(
            portfolio_history(
                list(body.get("symbols") or []),
                [float(w) for w in (body.get("weights") or [])],
                str(body.get("as_of") or today()),
            )
        )

    @app.get("/api/strategies")
    def api_strategies() -> JSONResponse:
        from quantlab.dashboard.actions import saved_strategies

        return JSONResponse(saved_strategies())

    @app.get("/api/strategies/{name}")
    def api_strategy(name: str) -> JSONResponse:
        from quantlab.dashboard.actions import saved_strategy

        return JSONResponse(saved_strategy(name))

    @app.get("/api/lake/summary")
    def api_lake_summary() -> JSONResponse:
        from quantlab.dashboard.actions import lake_summary

        return JSONResponse(lake_summary())

    @app.get("/api/signals")
    def api_signals() -> JSONResponse:
        from quantlab.dashboard.actions import available_signals

        return JSONResponse({"signals": available_signals()})

    @app.get("/api/trials")
    def api_trials() -> JSONResponse:
        from quantlab.dashboard.actions import trial_summary

        return JSONResponse(trial_summary())

    @app.post("/api/run/signal")
    def api_run_signal(body: dict[str, Any]) -> JSONResponse:
        from quantlab.dashboard.actions import run_signal_scores, today

        return JSONResponse(
            run_signal_scores(
                str(body.get("signal", "")),
                str(body.get("as_of") or today()),
                _symbol_list(body.get("symbols")),
            )
        )

    @app.post("/api/run/sweep")
    def api_run_sweep(body: dict[str, Any]) -> JSONResponse:
        from quantlab.dashboard.actions import run_sweep, today

        raw = str(body.get("parameters") or "").strip()
        parameters = _parse_parameters(raw) if raw else None
        if raw and parameters is None:
            return JSONResponse(
                {"ok": False, "error": "parameters must look like name=v1,v2 (one per line)"}
            )
        return JSONResponse(
            run_sweep(
                (str(body.get("signal")) or None) if body.get("signal") else None,
                str(body.get("as_of") or today()),
                _symbol_list(body.get("symbols")),
                parameters,
                int(body.get("lookback") or 252),
            )
        )

    @app.post("/api/run/backtest")
    def api_run_backtest(body: dict[str, Any]) -> JSONResponse:
        from quantlab.dashboard.actions import run_backtest

        return JSONResponse(run_backtest(str(body.get("config", ""))))

    @app.get("/api/paper")
    def api_paper() -> JSONResponse:
        return JSONResponse({"books": _paper_books()})

    @app.get("/research", response_class=HTMLResponse, include_in_schema=False)
    def research(request: Request) -> HTMLResponse:
        from quantlab.dashboard.actions import today

        return _TEMPLATES.TemplateResponse(
            request=request,
            name="research.html",
            context={"version": __version__, "today": today()},
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        checks = run_checks()
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "version": __version__,
                "checks": checks,
                "paper": _paper_books(),
                "sources": source_availability(),
                "heartbeats": _heartbeats(),
                "lake": _lake(),
                "last_ingest": _last_ingest(),
                "status_ok": Status.OK,
                "status_warn": Status.WARN,
            },
        )

    return app


app = create_app()
