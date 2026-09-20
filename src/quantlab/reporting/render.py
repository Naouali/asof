"""Rendering a tearsheet to text, Markdown and HTML.

All three read the same panels, so they cannot disagree about what the strategy
did. Each opens with the verdict and the disqualifying warnings; the numbers come
afterwards. That ordering is the point of the module -- a reader who stops after
the first paragraph should already know whether to believe the rest.

No charting library is used. An equity curve rendered as an inline SVG has no
dependency, no fonts to embed and no rendering surprises, and a tearsheet that
opens in any browser offline is worth more than one that needs a toolchain.
"""

from __future__ import annotations

import html
import itertools
from typing import TYPE_CHECKING

import numpy as np

from quantlab.reporting.panels import Severity

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl

    from quantlab.reporting.tearsheet import Tearsheet

__all__ = ["equity_curve_svg", "to_html", "to_markdown", "to_text"]

_MARKER = {
    Severity.DISQUALIFYING: "!!",
    Severity.WARNING: " !",
    Severity.INFORMATIONAL: "  ",
}


def to_text(sheet: Tearsheet) -> str:
    """A terminal tearsheet, plain enough to paste into an email."""
    width = 78
    lines = [
        "=" * width,
        f"{sheet.name}  --  generated {sheet.generated_at:%Y-%m-%d %H:%M UTC}",
        "=" * width,
        "",
        sheet.verdict(),
        "",
    ]
    for panel in sheet.panels:
        marker = _MARKER[panel.severity]
        lines.append(f"{marker} {panel.title.upper()}")
        lines.append("-" * width)
        for row in panel.rows:
            note = f"   ({row.note})" if row.note else ""
            lines.append(f"     {row.label:<30} {row.value:>22}{note}")
        if panel.footer:
            lines.append(f"     {panel.footer}")
        for warning in panel.warnings:
            lines.append("")
            lines.extend(f"     {line}" for line in _wrap(warning, width - 5))
        lines.append("")
    return "\n".join(lines)


def to_markdown(sheet: Tearsheet) -> str:
    """Markdown, for a pull request or a research log."""
    lines = [
        f"# {sheet.name}",
        "",
        f"*Generated {sheet.generated_at:%Y-%m-%d %H:%M UTC}*",
        "",
        f"> {sheet.verdict()}".replace("\n", "\n> "),
        "",
    ]
    for panel in sheet.panels:
        flag = {
            Severity.DISQUALIFYING: " — **disqualifying**",
            Severity.WARNING: " — warning",
            Severity.INFORMATIONAL: "",
        }[panel.severity]
        lines += [f"## {panel.title}{flag}", "", "| | | |", "| --- | --- | --- |"]
        lines += [f"| {row.label} | {row.value} | {row.note} |" for row in panel.rows]
        lines.append("")
        if panel.footer:
            lines += [panel.footer, ""]
        for warning in panel.warnings:
            lines += [f"**⚠ {warning}**", ""]
    return "\n".join(lines)


def to_html(sheet: Tearsheet, *, curve: pl.DataFrame | None = None) -> str:
    """A self-contained HTML page: no scripts, no fonts, no network.

    It opens offline, which is a requirement of the platform rather than a nicety.
    """
    curve = curve if curve is not None else sheet.result.equity_curve
    body = [
        f"<h1>{html.escape(sheet.name)}</h1>",
        f"<p class='meta'>Generated {sheet.generated_at:%Y-%m-%d %H:%M UTC}</p>",
        f"<div class='verdict {'bad' if sheet.is_disqualified else 'ok'}'>"
        f"{html.escape(sheet.verdict()).replace(chr(10), '<br>')}</div>",
        equity_curve_svg(curve),
    ]
    for panel in sheet.panels:
        css = panel.severity.name.lower()
        body.append(f"<section class='{css}'><h2>{html.escape(panel.title)}</h2><table>")
        for row in panel.rows:
            body.append(
                f"<tr><th>{html.escape(row.label)}</th>"
                f"<td class='num'>{html.escape(row.value)}</td>"
                f"<td class='note'>{html.escape(row.note)}</td></tr>"
            )
        body.append("</table>")
        if panel.footer:
            body.append(f"<p class='footer'>{html.escape(panel.footer)}</p>")
        for warning in panel.warnings:
            body.append(f"<p class='warn'>{html.escape(warning)}</p>")
        body.append("</section>")

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(sheet.name)} tearsheet</title><style>{_CSS}</style>"
        f"</head><body>{''.join(body)}</body></html>"
    )


def equity_curve_svg(
    curve: pl.DataFrame, *, width: int = 720, height: int = 220, max_points: int = 600
) -> str:
    """The equity curve as an inline SVG, on a log scale.

    Log scale because a linear equity chart makes every strategy look like it did
    all its work at the end: the same 20% gain is a larger rise the later it
    happens. The drawdown is drawn beneath, because the pair is what a reader
    needs and a curve without one invites reading the peak as the outcome.

    Long series are downsampled to keep the page small -- a 720-pixel chart cannot
    resolve four thousand points, and emitting them costs 96 KB. The downsampling
    is deliberately asymmetric: equity takes the last value in each bucket, but
    drawdown takes the **minimum**, so the worst drawdown in the sample survives
    into the picture. Decimating both the same way would let a chart quietly drop
    the trough, which is the single most flattering thing a downsample can do.
    """
    if curve.height < 2 or "equity" not in curve.columns:
        return "<p class='note'>no equity curve</p>"

    equity = curve["equity"].to_numpy().astype(float)
    if not np.isfinite(equity).all() or equity.min() <= 0:
        return "<p class='note'>equity curve contains non-positive values</p>"

    # Drawdown is computed on the full series, before any downsampling.
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    worst = float(drawdown.min())

    equity, drawdown = _downsample(equity, drawdown, max_points)

    pad, plot_h = 8, height * 0.68
    log_equity = np.log(equity)
    span = float(log_equity.max() - log_equity.min()) or 1.0
    x = pad + (width - 2 * pad) * np.linspace(0.0, 1.0, len(equity))
    y = pad + plot_h * (1.0 - (log_equity - log_equity.min()) / span)
    line = " ".join(f"{xi:.1f},{yi:.1f}" for xi, yi in zip(x, y, strict=True))

    dd_top = pad + plot_h + 12
    dd_h = height - dd_top - pad
    dd_y = dd_top + dd_h * (drawdown / (worst or -1.0))
    dd_area = (
        f"{x[0]:.1f},{dd_top:.1f} "
        + " ".join(f"{xi:.1f},{yi:.1f}" for xi, yi in zip(x, dd_y, strict=True))
        + f" {x[-1]:.1f},{dd_top:.1f}"
    )

    return (
        f"<svg class='curve' viewBox='0 0 {width} {height}' width='100%' "
        f"role='img' aria-label='equity curve, log scale, with drawdown'>"
        f"<polyline points='{line}' fill='none' stroke='#1a3d6d' stroke-width='1.6'/>"
        f"<polygon points='{dd_area}' fill='#c0392b' fill-opacity='0.25'/>"
        f"<text x='{pad}' y='{height - 2}' class='axis'>"
        f"log equity (top), drawdown to {worst:.1%} (bottom)</text>"
        "</svg>"
    )


def _downsample(
    equity: np.ndarray, drawdown: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bucket a long series, keeping the last equity and the worst drawdown."""
    if len(equity) <= max_points:
        return equity, drawdown
    edges = np.linspace(0, len(equity), max_points + 1).astype(int)
    buckets = [(lo, hi) for lo, hi in itertools.pairwise(edges) if hi > lo]
    return (
        np.array([equity[hi - 1] for _, hi in buckets]),
        np.array([drawdown[lo:hi].min() for lo, hi in buckets]),
    )


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


_CSS = """
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
 max-width:820px;margin:2rem auto;padding:0 1rem;color:#1c1c1c}
h1{margin-bottom:0}h2{font-size:1.05rem;margin:0 0 .4rem}
.meta{color:#666;margin-top:.2rem}
.verdict{padding:.8rem 1rem;border-radius:4px;margin:1rem 0;font-weight:600}
.verdict.ok{background:#eef6ee;border-left:4px solid #2e7d32}
.verdict.bad{background:#fdecea;border-left:4px solid #c0392b}
section{margin:1.4rem 0;padding-left:.8rem;border-left:3px solid #ddd}
section.warning{border-left-color:#e8a33d}
section.disqualifying{border-left-color:#c0392b}
table{border-collapse:collapse;width:100%}
th{text-align:left;font-weight:500;padding:.18rem 0;width:38%}
td.num{text-align:right;font-variant-numeric:tabular-nums;width:22%;padding-right:1rem}
td.note{color:#666;font-size:.86em}
.warn{background:#fdf3e3;border-left:3px solid #e8a33d;padding:.5rem .7rem;margin:.5rem 0}
.footer{color:#444;font-size:.9em}
.curve{margin:1rem 0;background:#fafafa;border:1px solid #eee;border-radius:3px}
.axis{font-size:10px;fill:#888}
"""
