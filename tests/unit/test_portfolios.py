"""A member's portfolio, compiled from disclosed trades.

Everything here is an estimate of something nobody is required to disclose, so
each test pins one way the estimate could quietly claim more than it knows: a
starting position it never saw, a range read as an amount, an amendment counted
twice, a return nobody could have earned.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_analytics import _prices
from tests.unit.test_api import at, congress_line, frame

from quantlab.api import create_app, portfolios
from quantlab.api.queries import Lens
from quantlab.config import Settings
from quantlab.data.store import Store

NOW = at(2026, 9, 18, 23)
VILLAREAL = "congress:tx21:tom villareal"


def trade(doc: str, line: int, **overrides: Any) -> dict[str, Any]:
    return congress_line(doc_id=doc, line=line, **overrides)


@pytest.fixture
def lake(store: Store) -> Store:
    rows = [
        # OSPR: bought twice, sold part. $100,001-$250,000 each purchase.
        trade("d1", 1),
        trade("d1", 2, as_of=at(2026, 7, 9)),
        trade(
            "d2",
            1,
            transaction_type="sale_partial",
            as_of=at(2026, 8, 3),
            known_at=at(2026, 9, 17, 3, 59),
            amount_min=15_001.0,
            amount_max=50_000.0,
            amount_text="$15,001 - $50,000",
        ),
        # KITE: sold without ever being seen bought.
        trade(
            "d2",
            2,
            symbol="KITE",
            asset="Kite Logistics (KITE)",
            transaction_type="sale",
            amount_min=1_001.0,
            amount_max=15_000.0,
            amount_text="$1,001 - $15,000",
        ),
        # HERN: bought and then sold for more than was bought.
        trade("d3", 1, symbol="HERN", asset="Heron Foods (HERN)", amount_min=1_001.0,
              amount_max=15_000.0, amount_text="$1,001 - $15,000"),
        trade("d3", 2, symbol="HERN", asset="Heron Foods (HERN)", transaction_type="sale",
              as_of=at(2026, 8, 1), amount_min=15_001.0, amount_max=50_000.0,
              amount_text="$15,001 - $50,000"),
        # Left out: an option, an exchange, a bond with no ticker.
        trade("d4", 1, asset_type="OP"),
        trade("d4", 2, transaction_type="exchange"),
        trade("d4", 3, symbol="NO_TICKER", asset="Parkland, PA School District Bond"),
        # Someone else entirely.
        trade("d9", 1, member="Hon. Ada Reyes", state_district="NM01", owner="SP"),
    ]  # fmt: skip
    store.write(frame("congress_trades", "house_clerk", rows), asset_class="equity")
    return store


def _portfolio(lake: Store, moment: dt.datetime = NOW) -> dict[str, Any]:
    found = portfolios.portfolio(Lens(lake, moment), VILLAREAL)
    assert found is not None
    return found


def test_a_position_is_the_midpoints_netted_with_its_honest_range_beside_it(lake: Store) -> None:
    (holding,) = _portfolio(lake)["holdings"]

    assert holding["ticker"] == "OSPR"
    assert (holding["purchases"], holding["sales"]) == (2, 1)
    # Two purchases at the midpoint of $100,001-$250,000, less a sale at $15,001-$50,000.
    assert holding["mid_usd"] == pytest.approx(2 * 175_000.5 - 32_500.5)
    # Least: purchases at their floors, the sale at its ceiling. Most: the reverse.
    assert holding["low_usd"] == pytest.approx(2 * 100_001 - 50_000)
    assert holding["high_usd"] == pytest.approx(2 * 250_000 - 15_001)
    assert holding["weight_pct"] == 100.0
    assert holding["first_bought"] == dt.date(2026, 7, 2)


def test_a_sale_never_seen_bought_is_a_holding_from_before_not_a_short(lake: Store) -> None:
    found = _portfolio(lake)

    assert [row["ticker"] for row in found["held_before"]] == ["KITE"]
    assert found["held_before"][0]["sold_high_usd"] == 15_000.0
    assert all(holding["mid_usd"] > 0 for holding in found["holdings"])


def test_a_position_sold_for_more_than_was_bought_is_closed(lake: Store) -> None:
    assert [row["ticker"] for row in _portfolio(lake)["closed"]] == ["HERN"]


def test_options_exchanges_and_unnamed_assets_are_counted_and_left_out(lake: Store) -> None:
    found = _portfolio(lake)
    assert found["left_out"] == {"options": 1, "exchanges": 1, "no_ticker": 1}
    assert found["trades"] == 9


def test_an_open_ended_range_counts_at_its_floor(store: Store) -> None:
    row = trade("d1", 1, amount_min=50_000_001.0, amount_max=None, amount_text="Over $50,000,000")
    store.write(frame("congress_trades", "house_clerk", [row]), asset_class="equity")
    (holding,) = _portfolio(store)["holdings"]
    assert holding["low_usd"] == holding["mid_usd"] == holding["high_usd"] == 50_000_001.0


def test_an_amendments_copy_of_a_line_replaces_the_original(store: Store) -> None:
    rows = [
        trade("orig", 1),
        trade("orig", 2),  # the same bracket twice in ONE report is two trades
        trade("amend", 1, filing_status="amended", known_at=at(2026, 9, 18, 3, 59)),
        trade("amend", 2, filing_status="amended", known_at=at(2026, 9, 18, 3, 59)),
    ]
    store.write(frame("congress_trades", "house_clerk", rows), asset_class="equity")
    (holding,) = _portfolio(store)["holdings"]
    assert holding["purchases"] == 2


def test_a_deleted_line_is_not_a_trade(store: Store) -> None:
    rows = [trade("d1", 1), trade("d1", 2, filing_status="deleted")]
    store.write(frame("congress_trades", "house_clerk", rows), asset_class="equity")
    assert _portfolio(store)["holdings"][0]["purchases"] == 1


def test_the_portfolio_on_a_past_date_holds_only_what_was_public_then(lake: Store) -> None:
    """The purchases of 2 and 9 July surfaced on 17 September."""
    assert portfolios.portfolio(Lens(lake, at(2026, 9, 1, 23)), VILLAREAL) is None
    assert len(_portfolio(lake)["holdings"]) == 1


def test_the_record_begins_with_the_first_report_not_the_first_trade(lake: Store) -> None:
    assert _portfolio(lake)["since"] == dt.date(2026, 9, 16)  # 03:59 UTC is the evening before


def test_a_follower_gets_the_return_from_the_day_it_became_public(lake: Store) -> None:
    lake.write(
        _prices("OSPR", {at(2026, 7, 2): 30.0, at(2026, 7, 9): 30.0, at(2026, 9, 16): 45.0,
                         at(2026, 9, 18): 54.0}),
        asset_class="equity",
    )  # fmt: skip
    found = _portfolio(lake)
    (holding,) = found["holdings"]

    assert holding["return_since_bought_pct"] == 80.0  # 30 to 54
    assert holding["return_since_public_pct"] == 20.0  # 45 to 54: what was left to copy
    assert found["priced_share"] == 1.0
    assert found["returns"] == {"since_bought_pct": 80.0, "since_public_pct": 20.0}


def test_no_overall_return_is_offered_when_prices_cover_too_little(lake: Store) -> None:
    found = _portfolio(lake)
    assert found["priced_share"] == 0.0 and found["returns"] is None
    assert found["holdings"][0]["return_since_public_pct"] is None


def test_members_are_listed_once_with_what_they_traded(lake: Store) -> None:
    listed = portfolios.members(Lens(lake, NOW))["members"]

    assert [member["actor"] for member in listed] == ["Rep. Tom Villareal", "Rep. Ada Reyes"]
    assert listed[0]["actor_id"] == VILLAREAL
    assert (listed[0]["trades"], listed[0]["tickers"]) == (9, 3)


def test_the_api_serves_both_and_says_so_when_there_is_nobody(
    lake: Store, settings: Settings, tmp_path: Path
) -> None:
    client = TestClient(create_app(settings, ui_dir=tmp_path / "no-ui"))

    listed = client.get("/api/portfolios", params={"as_of": "2026-09-18"})
    found = client.get("/api/portfolio", params={"as_of": "2026-09-18", "actor": VILLAREAL})
    nobody = client.get("/api/portfolio", params={"actor": "congress:zz00:no one"})

    assert listed.status_code == found.status_code == 200
    assert len(listed.json()["members"]) == 2
    assert found.json()["holdings"][0]["ticker"] == "OSPR"
    assert nobody.status_code == 404
    assert client.get("/api/rules").json()["portfolios"]["min_priced_share_pct"] == round(
        portfolios.MIN_PRICED_SHARE * 100
    )


# ------------------------------------------------------------------ performance --
def _priced(
    store: Store, rows: list[dict[str, Any]], closes: dict[dt.datetime, float]
) -> dict[str, Any]:
    store.write(frame("congress_trades", "house_clerk", rows), asset_class="equity")
    store.write(_prices("OSPR", closes), asset_class="equity")
    found = portfolios.portfolio(Lens(store, NOW), VILLAREAL)
    assert found is not None
    return found


def _on(found: dict[str, Any], day: dt.date) -> dict[str, Any]:
    return next(point for point in found["performance"]["points"] if point["date"] == day)


def test_an_open_trade_is_marked_from_its_entry_price_to_each_day(store: Store) -> None:
    found = _priced(
        store,
        [trade("d1", 1, known_at=at(2026, 7, 20, 3, 59))],  # bought 2 Jul, public 19 Jul
        {
            at(2026, 7, 2, 21): 100.0,
            at(2026, 7, 10, 21): 110.0,
            at(2026, 7, 19, 21): 120.0,
            at(2026, 9, 17, 21): 150.0,
        },
    )
    assert _on(found, dt.date(2026, 7, 2))["member_pct"] == 0.0
    assert _on(found, dt.date(2026, 7, 10))["member_pct"] == 10.0
    assert found["performance"]["member_pct"] == 50.0  # 100 to 150, still held


def test_a_closed_trade_earns_its_exit_over_its_entry_and_then_stands_still(store: Store) -> None:
    rows = [
        trade("d1", 1, known_at=at(2026, 7, 20, 3, 59)),
        trade(
            "d2", 1, transaction_type="sale", as_of=at(2026, 8, 3), known_at=at(2026, 8, 20, 3, 59)
        ),
    ]
    found = _priced(
        store,
        rows,
        {
            at(2026, 7, 2, 21): 100.0,
            at(2026, 8, 3, 21): 130.0,
            at(2026, 8, 10, 21): 90.0,
            at(2026, 9, 17, 21): 300.0,
        },
    )
    assert _on(found, dt.date(2026, 8, 3))["member_pct"] == 30.0
    # Sold at 130: what the price did afterwards is nobody's gain or loss.
    assert _on(found, dt.date(2026, 8, 10))["member_pct"] == 30.0
    assert found["performance"]["member_pct"] == 30.0
    assert found["closed"][0]["return_pct"] == 30.0


def test_a_partial_sale_closes_part_and_leaves_the_rest_marked_to_market(store: Store) -> None:
    rows = [
        trade("d1", 1, amount_min=100_000.0, amount_max=100_000.0),
        trade("d2", 1, transaction_type="sale_partial", as_of=at(2026, 8, 3),
              amount_min=60_000.0, amount_max=60_000.0),
    ]  # fmt: skip
    found = _priced(
        store,
        rows,
        {at(2026, 7, 2, 21): 100.0, at(2026, 8, 3, 21): 120.0, at(2026, 9, 17, 21): 60.0},
    )
    # 1,000 shares at 100. 500 sold at 120 (+10,000); 500 left at 60 (-20,000).
    assert found["performance"]["member_pct"] == -10.0


def test_the_return_is_everything_gained_over_everything_put_in(store: Store) -> None:
    rows = [
        trade("d1", 1, amount_min=100_000.0, amount_max=100_000.0),
        trade("d1", 2, as_of=at(2026, 8, 3), amount_min=100_000.0, amount_max=100_000.0),
    ]
    found = _priced(
        store,
        rows,
        {at(2026, 7, 2, 21): 100.0, at(2026, 8, 3, 21): 200.0, at(2026, 9, 17, 21): 200.0},
    )
    # The first lot doubled (+100,000); the second, bought at 200, did nothing.
    assert _on(found, dt.date(2026, 8, 3))["member_pct"] == 50.0
    assert found["performance"]["purchases"] == 2 and found["performance"]["tickers"] == 1


def test_a_follower_enters_and_exits_on_the_days_things_became_public(store: Store) -> None:
    rows = [
        trade("d1", 1, known_at=at(2026, 7, 20, 3, 59)),  # public 19 Jul
        trade(
            "d2", 1, transaction_type="sale", as_of=at(2026, 8, 3), known_at=at(2026, 8, 20, 3, 59)
        ),
    ]
    closes = {at(2026, 7, 2, 21): 100.0, at(2026, 7, 10, 21): 100.0, at(2026, 7, 19, 21): 125.0, at(2026, 8, 3, 21): 130.0,
              at(2026, 8, 19, 21): 100.0, at(2026, 9, 17, 21): 100.0}  # fmt: skip
    found = _priced(store, rows, closes)

    assert found["performance"]["member_pct"] == 30.0  # 100 to 130
    assert found["performance"]["follower_pct"] == -20.0  # in at 125, out at 100
    assert _on(found, dt.date(2026, 7, 10)) == {
        "date": dt.date(2026, 7, 10),
        "member_pct": 0.0,
        "follower_pct": None,  # nothing was public yet: there was nothing to copy
    }


def test_a_sale_of_shares_never_seen_bought_takes_no_part(store: Store) -> None:
    rows = [trade("d1", 1, transaction_type="sale", as_of=at(2026, 8, 3))]
    found = _priced(store, rows, {at(2026, 8, 3, 21): 100.0, at(2026, 9, 17, 21): 50.0})
    assert found["performance"] is None


def test_without_prices_there_is_no_graph(lake: Store) -> None:
    assert _portfolio(lake)["performance"] is None
