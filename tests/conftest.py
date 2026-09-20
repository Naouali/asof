"""Shared fixtures.

Every test runs against an isolated data root and a settings object built from a
controlled environment. Two things make that necessary:

* :func:`quantlab.config.get_settings` is ``lru_cache``d, so a test that mutates the
  environment would otherwise leak into every later test.
* ``Settings`` reads a ``.env`` file when one is present. A developer's real ``.env``
  must never change what the test suite asserts.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from quantlab.config import Settings, get_settings

if TYPE_CHECKING:
    from quantlab.data.store import Store

_QUANTLAB_ENV_PREFIX = "QUANTLAB_"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test a clean environment and its own data root."""
    for name in list(os.environ):
        if name.startswith(_QUANTLAB_ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANTLAB_ENV", "ci")
    monkeypatch.setenv("QUANTLAB_DATA_ROOT", str(tmp_path / "data"))
    # Neutralise any .env sitting in the working directory.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(isolated_env: None) -> Settings:
    return get_settings()


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Fail any test that tries to reach the real network.

    The unit suite must run offline, in a locked-down CI container. More
    importantly, a test that silently hits a live API is a test whose result
    depends on a third party's uptime and on today's market data: it will start
    failing for reasons that have nothing to do with the code.

    The block is installed on ``httpx.HTTPTransport``, which is the seam where
    real sockets are opened. FastAPI's ``TestClient`` drives the app through
    ``ASGITransport`` and is unaffected, so in-process HTTP tests still work.

    Tests that genuinely need a live API declare ``@pytest.mark.network``.
    """
    if request.node.get_closest_marker("network"):
        return

    import httpx

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "this test attempted a real network call. Use a recorded fixture, or "
            "mark it @pytest.mark.network if it must reach a live API."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _blocked)


# ----------------------------------------------------------------------------------
# Data layer fixtures
# ----------------------------------------------------------------------------------
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load_json_fixture(name: str) -> object:
    import json

    return json.loads(load_fixture(name))


@pytest.fixture
def store(settings: Settings) -> Store:
    from quantlab.data.store import Store

    settings.layout.ensure()
    return Store(settings.layout)


def bar(
    symbol: str,
    session: dt.date,
    close: float,
    *,
    known_at: dt.datetime | None = None,
    source: str = "yahoo",
) -> dict[str, object]:
    """One canonical daily bar, closing at 21:00 UTC."""
    import datetime as dt

    as_of = dt.datetime(session.year, session.month, session.day, 21, tzinfo=dt.UTC)
    knowable = known_at or as_of
    return {
        "source": source,
        "dataset": "ohlcv_daily",
        "symbol": symbol,
        "as_of": as_of,
        "known_at": knowable,
        "ingested_at": knowable,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1_000_000.0,
        "adj_close": close,
        "currency": "USD",
        "venue": "XNAS",
    }


@pytest.fixture
def populated_store(store: Store) -> Store:
    """A small lake with a known shape, including a restatement.

    Three sessions of one symbol, plus a fourth row that restates the first
    session's close months later. Point-in-time queries before the restatement
    must still see the original value -- that is the property every test in
    test_pit_leakage.py is checking.
    """
    import datetime as dt

    import polars as pl

    sessions = [dt.date(2024, 1, 3), dt.date(2024, 1, 4), dt.date(2024, 1, 5)]
    rows = [bar("AAPL", day, 100.0 + index) for index, day in enumerate(sessions)]
    rows += [bar("MSFT", day, 300.0 + index) for index, day in enumerate(sessions)]
    # The restatement: the 3 Jan close, revised on 1 June.
    rows.append(bar("AAPL", sessions[0], 999.0, known_at=dt.datetime(2024, 6, 1, tzinfo=dt.UTC)))
    store.write(pl.DataFrame(rows), asset_class="equity")
    return store
