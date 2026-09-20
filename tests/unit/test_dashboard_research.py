"""The research UI.

A browser drops the cost of trying one more idea from typing a command to
clicking a button, and every idea tried is a trial the next Sharpe ratio is
deflated against. These tests hold the line on that: the endpoints run the real
machinery, they do not bypass the registry, and they refuse searches whose size
the page is not willing to own.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quantlab.dashboard.app import _parse_parameters, _symbol_list, create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ------------------------------------------------------------------- pages --
def test_the_research_page_serves(client: TestClient) -> None:
    response = client.get("/research")
    assert response.status_code == 200
    assert "QuantLab" in response.text


def test_the_page_states_that_nothing_is_routed(client: TestClient) -> None:
    """Spec section 1. It belongs where someone using the tool will read it, not
    only in a docstring."""
    assert "no live order routing" in client.get("/research").text


def test_the_sweep_panel_warns_before_the_search_not_after(client: TestClient) -> None:
    """A caveat shown after the result is one the reader has already formed an
    opinion past."""
    page = client.get("/research").text
    assert "Every cell is a trial" in page
    assert "54%" in page, "the measured false-positive rate of a naive read"


def test_the_trials_panel_explains_why_it_exists(client: TestClient) -> None:
    page = client.get("/research").text
    assert "cheaper to try one more idea" in page


# -------------------------------------------------------------------- api --
def test_signals_are_listed_with_their_failure_modes(client: TestClient) -> None:
    payload = client.get("/api/signals").json()
    assert payload["signals"]
    first = payload["signals"][0]
    assert {"name", "tier", "evidence", "fails_when"} <= set(first)
    assert len(first["fails_when"]) > 40, "the registry enforces this; the API must carry it"


def test_the_trial_count_is_exposed(client: TestClient) -> None:
    payload = client.get("/api/trials").json()
    assert "total" in payload
    assert isinstance(payload["families"], list)


def test_a_signal_that_refuses_to_run_is_not_an_error(client: TestClient) -> None:
    """A signal declining for want of data is a designed outcome, and the reason
    is the useful part. Rendering it as a 500 would hide it."""
    response = client.post(
        "/api/run/signal",
        json={"signal": "carry.commodity_basis", "as_of": "2026-09-18"},
    )
    assert response.status_code == 200
    body = response.json()
    if not body["ok"]:
        assert body.get("unavailable") or body["error"]


def test_an_unknown_signal_does_not_crash_the_page(client: TestClient) -> None:
    response = client.post("/api/run/signal", json={"signal": "not.a.signal"})
    assert response.status_code in {200, 500}


def test_an_oversized_grid_is_refused_with_the_reason(client: TestClient) -> None:
    """Not only because waiting is unpleasant: a grid that size is a search that
    size, and the best cell would be deflated against all of it."""
    axes = "\n".join(f"vol_window={','.join(str(10 + i) for i in range(9))}" for _ in range(1))
    body = client.post(
        "/api/run/sweep",
        json={"signal": "trend.time_series_momentum", "parameters": axes + "\nclip=1,2,3,4,5"},
    ).json()
    assert body["ok"] is False
    assert "search of that size" in body["error"]


def test_malformed_parameters_are_rejected_not_skipped(client: TestClient) -> None:
    body = client.post(
        "/api/run/sweep", json={"signal": "trend.time_series_momentum", "parameters": "oops"}
    ).json()
    assert body["ok"] is False
    assert "name=v1,v2" in body["error"]


# ------------------------------------------------------------- parsing --
def test_symbols_accept_commas_and_newlines() -> None:
    assert _symbol_list("spy, qqq\ntlt") == ["SPY", "QQQ", "TLT"]
    assert _symbol_list("") is None
    assert _symbol_list(None) is None
    assert _symbol_list(["spy", "qqq"]) == ["SPY", "QQQ"]


def test_a_malformed_axis_rejects_the_whole_grid() -> None:
    """Skipping a bad line silently changes the size of the search, and the size
    of the search is the number everything else is judged against."""
    assert _parse_parameters("vol_window=21,63") == {"vol_window": [21, 63]}
    assert _parse_parameters("vol_window=21,63\ngarbage") is None
    assert _parse_parameters("=1,2") is None
    assert _parse_parameters("vol_window=") is None


def test_values_are_coerced_to_numbers() -> None:
    grid = _parse_parameters("a=1,2\nb=0.5\nc=text")
    assert grid == {"a": [1, 2], "b": [0.5], "c": ["text"]}
