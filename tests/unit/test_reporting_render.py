"""Rendering, and the ways a chart can lie.

The renderers share one source of content, so the tests check that they agree,
that the HTML is self-contained and escaped, and -- the part that matters -- that
downsampling a long equity curve cannot hide the worst drawdown.
"""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import pytest
from tests.unit.test_reporting_tearsheet import FakeCapacity, FakeEvidence, FakeResult, sheet

from quantlab.reporting.render import equity_curve_svg, to_html, to_markdown, to_text


def curve(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "as_of": [
                dt.datetime(2020, 1, 1, tzinfo=dt.UTC) + dt.timedelta(days=i)
                for i in range(len(values))
            ],
            "equity": values,
            "drawdown": np.zeros(len(values)),
        }
    )


# ------------------------------------------------------------------------ text --
def test_the_text_report_leads_with_the_verdict() -> None:
    body = to_text(sheet(evidence=FakeEvidence(deflated=0.2, trials=50)))
    head = body[: body.index("PERFORMANCE")]
    assert "NOT EVIDENCE" in head


def test_disqualifying_panels_are_marked_in_the_margin() -> None:
    body = to_text(sheet(evidence=FakeEvidence(deflated=0.2, trials=50)))
    assert "!! PERFORMANCE" in body
    assert "   ACCOUNTING" in body


def test_every_panel_appears_in_every_format() -> None:
    built = sheet()
    text, markdown, html = to_text(built), to_markdown(built), to_html(built)
    for panel in built.panels:
        assert panel.title.upper() in text
        assert f"## {panel.title}" in markdown
        assert f"<h2>{panel.title}</h2>" in html


def test_the_formats_agree_on_the_verdict() -> None:
    built = sheet(capacity=FakeCapacity(break_even_aum=None))
    fragment = "no AUM at which"
    assert all(fragment in render(built) for render in (to_text, to_markdown, to_html))


# -------------------------------------------------------------------- markdown --
def test_markdown_flags_severity_in_the_heading() -> None:
    body = to_markdown(sheet(evidence=FakeEvidence(deflated=0.2, trials=50)))
    assert "## Performance — **disqualifying**" in body
    assert "## Accounting\n" in body


# ------------------------------------------------------------------------ html --
def test_the_page_is_self_contained() -> None:
    """It has to open offline. A tearsheet that needs a CDN is a tearsheet that
    stops working exactly when you are on a plane reading it."""
    body = to_html(sheet())
    for forbidden in ("<script", "http://", "https://", "@import"):
        assert forbidden not in body


def test_the_html_is_structurally_complete() -> None:
    """Not parsed as XML -- an HTML5 doctype is not a valid XML prolog. The SVG
    inside it is XML and is parsed in its own test."""
    body = to_html(sheet())
    assert body.startswith("<!doctype html>")
    assert body.rstrip().endswith("</body></html>")
    for tag in ("section", "table", "h2"):
        assert body.count(f"<{tag}") == body.count(f"</{tag}>")


def test_a_hostile_strategy_name_is_escaped() -> None:
    built = sheet(name="<script>alert(1)</script>")
    body = to_html(built)
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_the_verdict_class_reflects_the_outcome() -> None:
    assert "verdict ok" in to_html(sheet())
    assert "verdict bad" in to_html(sheet(evidence=FakeEvidence(deflated=0.1, trials=20)))


# ------------------------------------------------------------------------- svg --
def test_downsampling_cannot_hide_the_worst_drawdown() -> None:
    """The single most flattering thing a downsample can do is drop the trough.

    A crash lasting one bar in four thousand would vanish under plain decimation.
    The drawdown series takes the minimum of each bucket precisely so it cannot.
    """
    values = list(np.full(4000, 100.0))
    values[1999] = 40.0  # a one-bar, 60% crash
    for i in range(2000, 4000):
        values[i] = 100.0

    svg = equity_curve_svg(curve(values), max_points=50)
    assert "-60.0%" in svg


def test_a_long_curve_is_downsampled() -> None:
    long_svg = equity_curve_svg(curve(list(np.linspace(100, 300, 5000))), max_points=400)
    short_svg = equity_curve_svg(curve(list(np.linspace(100, 300, 5000))), max_points=5000)
    assert len(long_svg) < len(short_svg) / 5


def test_a_short_curve_is_not_downsampled() -> None:
    svg = equity_curve_svg(curve([100.0, 110.0, 105.0, 120.0]), max_points=600)
    assert svg.count(",") >= 8  # four equity points and four drawdown points


def test_the_svg_is_valid_xml() -> None:
    ET.fromstring(  # noqa: S314 - our own output
        equity_curve_svg(curve(list(np.linspace(100, 250, 900))))
    )


def test_a_degenerate_curve_says_so_rather_than_drawing_nonsense() -> None:
    assert "no equity curve" in equity_curve_svg(curve([100.0]))
    assert "non-positive" in equity_curve_svg(curve([100.0, 0.0, 50.0]))
    assert "no equity curve" in equity_curve_svg(pl.DataFrame({"as_of": [], "other": []}))


def test_a_flat_curve_does_not_divide_by_zero() -> None:
    svg = equity_curve_svg(curve([100.0] * 50))
    assert "<polyline" in svg
    assert "nan" not in svg.lower()


def test_the_chart_is_on_a_log_scale() -> None:
    """A linear chart makes every strategy look like it did all its work at the
    end. On a log scale a constant growth rate is a straight line, so the test is
    that an exponential curve renders as one."""
    values = list(100.0 * np.exp(np.linspace(0, 2, 200)))
    svg = equity_curve_svg(curve(values), max_points=200)
    points = svg[svg.index("points='") + 8 :]
    ys = [float(p.split(",")[1]) for p in points[: points.index("'")].split()]
    steps = np.diff(ys)
    # Coordinates are rounded to a tenth of a pixel, so equal steps of ~0.75px
    # alternate between 0.7 and 0.8. That quantisation sets the floor here; a
    # linear-scale render of this curve varies by orders of magnitude, not 15%.
    assert np.std(steps) / abs(np.mean(steps)) < 0.15


def test_the_curve_defaults_to_the_result_when_none_is_given() -> None:
    built = sheet()
    assert to_html(built) == to_html(built, curve=FakeResult().equity_curve)


@pytest.mark.parametrize("render", [to_text, to_markdown, to_html])
def test_source_caveats_survive_into_every_format(render: object) -> None:
    built = sheet(caveats=("Yahoo: delisted tickers disappear entirely",))
    assert "delisted tickers" in render(built)  # type: ignore[operator]
