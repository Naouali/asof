"""Tearsheets and dashboard data.

A tearsheet without a capacity estimate, a deflated Sharpe and its trial count,
and a data-quality warnings panel is not a finished tearsheet (spec sections 9
and 13). :meth:`Tearsheet.build` enforces the first two by refusing to assemble
a result that lacks them, and the third is collected from the source catalogue
rather than written by hand.

Panels are ordered by severity, so a strategy that fails deflation says so above
its equity curve rather than beneath it.
"""

from __future__ import annotations

from quantlab.reporting.caveats import caveats_for, sources_behind
from quantlab.reporting.panels import Panel, Row, Severity
from quantlab.reporting.render import equity_curve_svg, to_html, to_markdown, to_text
from quantlab.reporting.tearsheet import Tearsheet, TearsheetError

__all__ = [
    "Panel",
    "Row",
    "Severity",
    "Tearsheet",
    "TearsheetError",
    "caveats_for",
    "equity_curve_svg",
    "sources_behind",
    "to_html",
    "to_markdown",
    "to_text",
]
