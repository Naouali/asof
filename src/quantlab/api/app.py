"""The web app: a read-only API over the lake, which also serves the built UI.

**Read-only, entirely.** No route writes to the lake, starts an ingest or touches
the network. Data arrives by ``quantlab data ingest``; this process only reads what
is there, which is what makes it safe to leave running.

**There is no login yet.** Everything here is readable by anyone who can reach the
port, so the defaults keep that to this machine: ``quantlab serve`` binds to
loopback and the compose file publishes to 127.0.0.1 only. Identity is the next
thing to build, and nothing below pretends to have it -- ``/api/health`` reports
``authentication: none`` so that a deployment cannot mistake this for protected.

**"As of" is resolved in exactly one place**, :func:`_instant`. A date means the
END of that day in Washington, because that is when the last thing filed on it
became public: EDGAR accepts until 22:00 Eastern, and a House report is knowable
once its filing day is over.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from quantlab import __version__
from quantlab.api import analytics, portfolios, queries
from quantlab.api import rules as app_rules
from quantlab.api.events import EASTERN, eastern_date
from quantlab.api.models import (
    ContractsResponse,
    DataHealth,
    FeedResponse,
    Health,
    PortfolioMembers,
    PortfolioResponse,
    PriceMovesResponse,
    Rules,
    SearchHit,
    TickerResponse,
    TradedResponse,
)
from quantlab.api.queries import Lens
from quantlab.config import Settings, get_settings
from quantlab.data.store import Store, utcnow
from quantlab.logging import get_logger

__all__ = ["create_app"]

log = get_logger("quantlab.api")

#: How long a built lens is reused. Long enough that paging around is instant,
#: short enough that a finished ingest shows up without a restart.
LENS_TTL_SECONDS = 300.0
MAX_LENSES = 12

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # Everything the interface needs is served from here: no CDN, no webfont host.
    # It keeps working offline, and it tells a browser to refuse anything injected.
    "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; frame-ancestors 'none'",
}

AsOf = Annotated[
    dt.date | None,
    Query(description="See the lake as it was knowable at the end of this day. Omit for now."),
]


class _Lenses:
    """A small cache of lenses, keyed by the day they look at."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._built: dict[str, tuple[float, Lens]] = {}

    def get(self, as_of: dt.date | None) -> tuple[Lens, Lens | None]:
        """The lens for ``as_of``, and the live lens beside it when that is in the past."""
        live = self._lens("live", utcnow)
        if as_of is None or as_of >= eastern_date(live.as_of):
            return live, None
        return self._lens(as_of.isoformat(), lambda: _instant(as_of)), live

    def _lens(self, key: str, instant: Callable[[], dt.datetime]) -> Lens:
        with self._lock:
            cached = self._built.get(key)
            if cached is not None and time.monotonic() - cached[0] < LENS_TTL_SECONDS:
                return cached[1]
            lens = Lens(self._store, instant())
            self._built[key] = (time.monotonic(), lens)
            while len(self._built) > MAX_LENSES:
                del self._built[min(self._built, key=lambda k: self._built[k][0])]
            return lens


def _instant(day: dt.date) -> dt.datetime:
    """The last moment of ``day`` in Washington, as UTC."""
    return dt.datetime.combine(day, dt.time.max, tzinfo=EASTERN).astimezone(dt.UTC)


def create_app(
    settings: Settings | None = None, *, ui_dir: Path | None = None, warm: bool = False
) -> FastAPI:
    settings = settings or get_settings()
    store = Store(settings.layout)
    lenses = _Lenses(store)
    if warm:
        # Building the live lens reads every disclosure in the lake: seconds, once.
        # Done now, in the background, so that it is not the first visitor who waits.
        threading.Thread(target=lambda: lenses.get(None)[0].events, daemon=True).start()
    ui = ui_dir if ui_dir is not None else settings.layout.repo_root / "ui" / "dist"

    app = FastAPI(
        title="QuantLab",
        version=__version__,
        description="Read-only API over the disclosure lake. There is no authentication yet.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )

    @app.middleware("http")
    async def _headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        if not request.url.path.startswith("/api/docs"):
            for name, value in _SECURITY_HEADERS.items():
                response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health", response_model=Health)
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "today": eastern_date(utcnow()),
            "authentication": "none",
        }

    @app.get("/api/rules", response_model=Rules)
    def rules() -> dict[str, Any]:
        return app_rules.rules(settings)

    @app.get("/api/feed", response_model=FeedResponse)
    def feed(
        as_of: AsOf = None,
        days: Annotated[int, Query(ge=1, le=400)] = 7,
        actor: Annotated[str | None, Query(max_length=200)] = None,
    ) -> dict[str, Any]:
        lens, live = lenses.get(as_of)
        return queries.feed(lens, days=days, actor=actor, live=live)

    @app.get("/api/tickers/{ticker}", response_model=TickerResponse)
    def ticker(ticker: str, as_of: AsOf = None) -> dict[str, Any]:
        lens, _ = lenses.get(as_of)
        return queries.ticker_page(lens, ticker[:12])

    @app.get("/api/search", response_model=list[SearchHit])
    def search(
        q: Annotated[str, Query(max_length=80)] = "", as_of: AsOf = None
    ) -> list[dict[str, Any]]:
        lens, _ = lenses.get(as_of)
        return queries.search(lens, q)

    @app.get("/api/analytics/traded", response_model=TradedResponse)
    def analytics_traded(
        as_of: AsOf = None,
        days: Annotated[int, Query(ge=1, le=400)] = 90,
        kind: Annotated[Literal["insider", "congress", "fund"] | None, Query()] = None,
    ) -> dict[str, Any]:
        lens, _ = lenses.get(as_of)
        return analytics.traded(lens, days=days, kind=kind)

    @app.get("/api/analytics/price-moves", response_model=PriceMovesResponse)
    def analytics_price_moves(
        as_of: AsOf = None, days: Annotated[int, Query(ge=1, le=400)] = 365
    ) -> dict[str, Any]:
        lens, _ = lenses.get(as_of)
        return analytics.price_moves(lens, days=days)

    @app.get("/api/analytics/contracts", response_model=ContractsResponse)
    def analytics_contracts(as_of: AsOf = None) -> dict[str, Any]:
        lens, live = lenses.get(as_of)
        return analytics.contracts(lens, live=live)

    @app.get("/api/portfolios", response_model=PortfolioMembers)
    def portfolio_members(as_of: AsOf = None) -> dict[str, Any]:
        lens, _ = lenses.get(as_of)
        return portfolios.members(lens)

    @app.get("/api/portfolio", response_model=PortfolioResponse)
    def portfolio(
        actor: Annotated[str, Query(min_length=1, max_length=200)], as_of: AsOf = None
    ) -> Any:
        lens, _ = lenses.get(as_of)
        found = portfolios.portfolio(lens, actor)
        if found is None:
            return JSONResponse(
                {"detail": f"no disclosed trades by {actor!r} as of that date"}, status_code=404
            )
        return found

    @app.get("/api/data", response_model=DataHealth)
    def data() -> dict[str, Any]:
        lens, _ = lenses.get(None)
        return queries.data_health(store, lens)

    @app.get("/api/{rest:path}", include_in_schema=False)
    def unknown_api(rest: str) -> JSONResponse:
        return JSONResponse({"detail": f"no such endpoint: /api/{rest}"}, status_code=404)

    _mount_ui(app, ui)
    return app


def _mount_ui(app: FastAPI, ui: Path) -> None:
    index = ui / "index.html"
    if not index.exists():
        log.warning(
            "api.ui_missing", looked_in=str(ui), fix="cd ui && npm install && npm run build"
        )

        @app.get("/{rest:path}", include_in_schema=False)
        def not_built(rest: str) -> JSONResponse:
            return JSONResponse(
                {
                    "detail": "The interface has not been built. Run `make ui`, or "
                    "`cd ui && npm install && npm run build`. The API is at /api/docs."
                },
                status_code=503,
            )

        return

    assets = ui / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{rest:path}", include_in_schema=False)
    def spa(rest: str) -> FileResponse:
        # Any path that is not an asset is a page of the single-page app, so a
        # reload on /t/AAPL lands on the app rather than on a 404.
        candidate = (ui / rest).resolve()
        if rest and candidate.is_file() and ui.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
