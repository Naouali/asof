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

    @app.get("/api/sources")
    def api_sources() -> JSONResponse:
        return JSONResponse(
            {"sources": [a.as_dict() for a in source_availability()], "count": len(SOURCES)}
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
                "sources": source_availability(),
                "heartbeats": _heartbeats(),
                "status_ok": Status.OK,
                "status_warn": Status.WARN,
            },
        )

    return app


app = create_app()
