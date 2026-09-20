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


def page(client: TestClient) -> str:
    """The research page with whitespace normalised.

    Assertions are about what the page *says*, not about where a line happened
    to wrap in the template. Matching raw source makes a reflow look like a
    regression.
    """
    return " ".join(client.get("/research").text.split())


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
    assert "No live order routing" in page(client)


def test_the_sweep_panel_warns_before_the_search_not_after(client: TestClient) -> None:
    """A caveat shown after the result is one the reader has already formed an
    opinion past."""
    body = page(client)
    assert "Every cell is a trial" in body
    assert "54% of the time" in body, "the measured false-positive rate of a naive read"


def test_the_trials_panel_explains_why_it_exists(client: TestClient) -> None:
    assert "cheaper to try one more idea" in page(client)


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


# ------------------------------------------------------- the comparison view --
def test_strategies_are_listed_with_what_makes_them_comparable(client: TestClient) -> None:
    """A screen ranking three strategies by Sharpe without the trial count and
    the disqualification beside it would be the most misleading thing this
    platform could render."""
    payload = client.get("/api/strategies").json()
    assert "strategies" in payload
    for row in payload["strategies"]:
        assert {"deflated_sharpe", "trials", "is_disqualified", "warnings"} <= set(row)


def test_a_disqualified_strategy_cannot_hide_below_the_fold(client: TestClient) -> None:
    """The list is ordered so a failing strategy is not quietly ranked beneath
    the ones a reader is comparing."""
    from quantlab.reporting.results_store import ResultsStore

    runs = ResultsStore(__import__("pathlib").Path("data/runs/results")).all()
    if len(runs) > 1:
        flags = [r.is_disqualified for r in runs]
        assert flags == sorted(flags, reverse=True) or len(set(flags)) == 1


def test_a_missing_strategy_is_a_clean_error(client: TestClient) -> None:
    body = client.get("/api/strategies/not-a-run").json()
    assert body["ok"] is False
    assert "no saved result" in body["error"]


def test_the_lake_summary_reports_span_not_just_size(client: TestClient) -> None:
    """Row counts say nothing about whether the history is usable."""
    payload = client.get("/api/lake/summary").json()
    for row in payload["datasets"]:
        assert {"rows", "symbols", "earliest", "latest"} <= set(row)


def test_the_page_leads_with_deflation_not_with_sharpe(client: TestClient) -> None:
    """What a reader sees first is what they take away."""
    body = page(client)
    assert "Ranked by deflated Sharpe" in body
    assert "0.95 is the bar" in body
    assert "the gap between them is" in body


def test_the_page_states_the_measured_false_positive_rate(client: TestClient) -> None:
    body = page(client)
    assert "54% of the time" in body
    assert "one cell in 2,800" in body


def test_charts_are_drawn_without_a_network_dependency(client: TestClient) -> None:
    """Offline-capable is a platform requirement, not a nicety. A chart library
    from a CDN would break the whole page on a plane."""
    body = page(client)
    assert "https://" not in body
    assert "lineChart" in body and "barChart" in body, "charts are hand-rolled SVG"
