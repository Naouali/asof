"""The instrument explorer and the portfolio view.

Two questions the rest of the dashboard does not answer: *what do I actually
hold data on for this ticker*, and *how do these go together*. The second is the
more reliable one -- correlation is estimable from a few hundred observations in
a way expected return is not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from quantlab.dashboard.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ------------------------------------------------------------------- index --
def test_the_index_is_built_from_the_lake(client: TestClient) -> None:
    """Not from a maintained list. An instrument present in the data and absent
    from the catalogue is the failure that makes a data browser untrustworthy."""
    payload = client.get("/api/instruments").json()
    assert payload["count"] == len(payload["instruments"])
    for row in payload["instruments"]:
        assert row["datasets"], f"{row['symbol']} listed with no dataset"


def test_an_instrument_reports_every_dataset_it_appears_in(client: TestClient) -> None:
    payload = client.get("/api/instruments").json()
    multi = [i for i in payload["instruments"] if len(i["datasets"]) > 1]
    if multi:
        assert multi[0]["datasets"] == sorted(multi[0]["datasets"]), "listed in a stable order"


def test_priced_instruments_sort_first(client: TestClient) -> None:
    """A ticker you can chart is more useful than one you cannot, and a browser
    opens on the first page."""
    rows = client.get("/api/instruments").json()["instruments"]
    priced = [i["priced"] for i in rows]
    assert priced == sorted(priced, reverse=True)


def test_opaque_codes_carry_a_name(client: TestClient) -> None:
    """Nobody looking for corn types 002602."""
    rows = client.get("/api/instruments").json()["instruments"]
    coded = [i for i in rows if i["symbol"].isdigit() or i["symbol"][:1].isdigit()]
    if coded:
        assert any(i["name"] for i in coded), "CFTC codes must resolve to market names"


def test_former_names_are_searchable(client: TestClient) -> None:
    """Sources rename markets. The CFTC's two-year note is now 'UST 2Y NOTE' and
    was '2 YEAR U.S. TREASURY NOTES'; someone typing 'treasury' means that one,
    and indexing only the latest name loses them."""
    rows = client.get("/api/instruments").json()["instruments"]
    searchable = [
        i
        for i in rows
        if "treasury" in (i["name"] or "").lower()
        or any("treasury" in a.lower() for a in i.get("aliases", []))
    ]
    if any(i["symbol"] in {"042601", "043602"} for i in rows):
        assert searchable, "a renamed contract is unreachable by its former name"


# ------------------------------------------------------------------ detail --
def test_a_missing_instrument_is_a_clean_error(client: TestClient) -> None:
    body = client.get("/api/instruments/NOTATICKER").json()
    assert body["ok"] is False
    assert "nothing in the lake" in body["error"]


def test_buy_and_hold_statistics_are_labelled_as_such(client: TestClient) -> None:
    """The Sharpe of holding an instrument is not a strategy result and carries
    no deflation, because no search produced it."""
    rows = client.get("/api/instruments").json()["instruments"]
    priced = next((i for i in rows if i["priced"]), None)
    if priced is None:
        pytest.skip("no priced instrument in this lake")

    body = client.get(f"/api/instruments/{priced['symbol']}").json()
    assert body["ok"]
    if body["stats"]:
        assert "buy_hold_sharpe" in body["stats"]
        assert {"cagr", "volatility", "max_drawdown", "bars"} <= set(body["stats"])
    assert "not a strategy result" in client.get("/research").text


# --------------------------------------------------------------- portfolio --
def test_a_portfolio_needs_at_least_two_instruments(client: TestClient) -> None:
    body = client.post("/api/portfolio", json={"symbols": "SPY"}).json()
    assert body["ok"] is False
    assert "at least two" in body["error"]


def test_an_unknown_method_is_refused(client: TestClient) -> None:
    body = client.post("/api/portfolio", json={"symbols": "SPY,QQQ", "method": "kelly"}).json()
    assert body["ok"] is False
    assert "unknown method" in body["error"]


def test_risk_share_is_returned_beside_every_weight(client: TestClient) -> None:
    """The point of the view. An equal-weighted book of correlated assets puts
    most of its risk in one place while looking diversified on the weights."""
    body = client.post(
        "/api/portfolio", json={"symbols": "SPY,QQQ,TLT,GLD", "as_of": "2026-09-18"}
    ).json()
    if not body["ok"]:
        pytest.skip(body["error"])
    for position in body["positions"]:
        assert {"weight", "risk_share", "volatility"} <= set(position)
    assert sum(p["risk_share"] for p in body["positions"]) == pytest.approx(1.0, abs=1e-6)


def test_risk_parity_equalises_risk_and_equal_weight_does_not(client: TestClient) -> None:
    """The comparison the page exists to make."""
    args = {"symbols": "SPY,QQQ,TLT,GLD,USO", "as_of": "2026-09-18"}
    parity = client.post("/api/portfolio", json={**args, "method": "risk-parity"}).json()
    equal = client.post("/api/portfolio", json={**args, "method": "equal"}).json()
    if not (parity["ok"] and equal["ok"]):
        pytest.skip("lake lacks the history")

    spread = lambda b: max(p["risk_share"] for p in b["positions"]) - min(  # noqa: E731
        p["risk_share"] for p in b["positions"]
    )
    assert spread(parity) < 0.01, "risk parity equalises risk contribution"
    assert spread(equal) > spread(parity), "equal weight does not"


def test_the_correlation_matrix_is_square_and_unit_diagonal(client: TestClient) -> None:
    body = client.post(
        "/api/portfolio", json={"symbols": "SPY,QQQ,TLT", "as_of": "2026-09-18"}
    ).json()
    if not body["ok"]:
        pytest.skip(body["error"])
    matrix = body["correlation"]["matrix"]
    assert len(matrix) == len(body["correlation"]["symbols"])
    for i, row in enumerate(matrix):
        assert len(row) == len(matrix)
        assert row[i] == pytest.approx(1.0, abs=1e-9)


def test_independent_bets_are_reported_not_just_instrument_count(client: TestClient) -> None:
    """Six correlated ETFs are not six bets, and the difference is the whole
    reason to look at a correlation matrix."""
    body = client.post(
        "/api/portfolio", json={"symbols": "SPY,QQQ,IWM,DIA", "as_of": "2026-09-18"}
    ).json()
    if not body["ok"]:
        pytest.skip(body["error"])
    stats = body["portfolio"]
    assert stats["independent_bets"] < stats["instruments"]


def test_no_expected_returns_are_estimated(client: TestClient) -> None:
    """Sample means are noisy enough that optimising on them reliably produces a
    worse portfolio than equal weighting."""
    body = client.post(
        "/api/portfolio",
        json={"symbols": "SPY,QQQ,TLT,GLD", "method": "min-variance", "as_of": "2026-09-18"},
    ).json()
    if not body["ok"]:
        pytest.skip(body["error"])
    assert "No expected returns are estimated" in body["note"]


def test_the_history_view_says_it_is_not_a_backtest(client: TestClient) -> None:
    """A costless daily-rebalanced weight vector applied to past prices answers
    'how did these move together', not 'what would this have returned'."""
    body = client.post(
        "/api/portfolio/history",
        json={"symbols": ["SPY", "TLT"], "weights": [0.5, 0.5], "as_of": "2026-09-18"},
    ).json()
    if not body["ok"]:
        pytest.skip(body["error"])
    assert "not a backtest" in body["note"]


# --------------------------------------------- the portfolio maths, exercised --
def _lake(store: object, *, sessions: int = 600) -> None:
    """Four instruments with deliberately unequal volatility.

    The point of the portfolio view is that equal weights do not mean equal
    risk, and that only shows up when the instruments differ.
    """
    import datetime as dt

    import numpy as np
    import polars as pl
    from tests.conftest import bar

    rng = np.random.default_rng(4)
    days = [dt.date(2024, 1, 2) + dt.timedelta(days=i) for i in range(sessions)]
    common = rng.normal(0.0, 0.006, sessions)
    rows = []
    for symbol, vol in (("CALM", 0.004), ("MID", 0.011), ("WILD", 0.030), ("OTHER", 0.012)):
        price = 100.0
        for step, day in enumerate(days):
            price *= 1.0 + 0.4 * common[step] + rng.normal(0.0, vol)
            row = bar(symbol, day, price)
            row["adj_close"] = price
            row["volume"] = 4_000_000.0
            rows.append(row)
    store.write(pl.DataFrame(rows), asset_class="equity")  # type: ignore[attr-defined]


def test_equal_weights_do_not_mean_equal_risk(store: object) -> None:
    """The single insight the page exists to show. A quiet instrument and a wild
    one held at the same weight are not the same bet, and the weights column
    cannot tell you that."""
    from quantlab.dashboard.portfolio_view import build_portfolio

    _lake(store)
    equal = build_portfolio(["CALM", "MID", "WILD", "OTHER"], as_of="2025-08-01", method="equal")
    assert equal["ok"], equal.get("error")

    shares = {p["symbol"]: p["risk_share"] for p in equal["positions"]}
    assert shares["WILD"] > 3 * shares["CALM"], "the loud instrument dominates the risk"
    assert all(abs(p["weight"] - 0.25) < 1e-9 for p in equal["positions"]), "weights are equal"


def test_risk_parity_equalises_what_equal_weight_does_not(store: object) -> None:
    from quantlab.dashboard.portfolio_view import build_portfolio

    _lake(store)
    parity = build_portfolio(
        ["CALM", "MID", "WILD", "OTHER"], as_of="2025-08-01", method="risk-parity"
    )
    assert parity["ok"], parity.get("error")

    shares = [p["risk_share"] for p in parity["positions"]]
    assert max(shares) - min(shares) < 0.01, "every position carries the same risk"
    weights = {p["symbol"]: p["weight"] for p in parity["positions"]}
    assert weights["CALM"] > weights["WILD"], "the quiet instrument gets the larger weight"


def test_correlated_instruments_are_fewer_bets_than_they_look(store: object) -> None:
    from quantlab.dashboard.portfolio_view import build_portfolio

    _lake(store)
    book = build_portfolio(["CALM", "MID", "WILD", "OTHER"], as_of="2025-08-01")
    assert book["ok"], book.get("error")

    stats = book["portfolio"]
    assert stats["instruments"] == 4
    assert stats["independent_bets"] < 4
    assert stats["average_correlation"] > 0


def test_an_instrument_with_gaps_is_dropped_loudly(store: object) -> None:
    """Padding a short history understates both the volatility and the
    correlation of whichever instrument has least data."""
    from quantlab.dashboard.portfolio_view import build_portfolio

    _lake(store)
    book = build_portfolio(["CALM", "MID", "WILD", "OTHER", "ABSENT"], as_of="2025-08-01")
    assert book["ok"]
    assert "ABSENT" not in [p["symbol"] for p in book["positions"]]
