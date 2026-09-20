#!/usr/bin/env python
"""Render docs/DATA_CATALOGUE.md from the source registry.

Documentation that is maintained by hand drifts from the code, and a data caveat
that has drifted is worse than no caveat: it is a false assurance. The catalogue
document is therefore generated, and `tests/unit/test_docs.py` fails if the
committed file does not match what this script produces.

    python scripts/gen_data_catalogue.py           # write the file
    python scripts/gen_data_catalogue.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantlab.data.catalogue import SOURCES, AssetClass, PitQuality  # noqa: E402

OUTPUT = REPO / "docs" / "DATA_CATALOGUE.md"

HEADER = """<!-- GENERATED FILE -- do not edit by hand.
     Source of truth: src/quantlab/data/catalogue.py
     Regenerate:      python scripts/gen_data_catalogue.py -->

# Data catalogue

Every external source this ETL can read, what it provides, and how it lies to
you. Free data is never clean; the purpose of this document is to make the ways in
which it is dirty impossible to overlook.

The same information is structured data in `src/quantlab/data/catalogue.py` and is
reported by `quantlab data catalogue`.

## How to read the point-in-time column

| Value | Meaning |
| --- | --- |
| `vintage` | The source can answer "what was known on date D". Safe for signals. |
| `as_published` | Values are never revised, so today's series equals the historical one. Safe for signals. |
| `restated` | The source serves *current* values for historical dates. **Using this in a signal is look-ahead bias**, and the point-in-time layer blocks it without an explicit override. |
| `survivorship_biased` | Only entities that still exist are retrievable. Dead tickers have vanished. |

## The three biases that matter most here

1. **Survivorship bias in equities.** It cannot be fully solved without paid CRSP.
   What the lake does instead: retains every delisted ticker once observed, and
   marks each affected source `survivorship_biased` so a consumer cannot miss it.
2. **Restated macro data.** FRED serves the latest vintage of every series. Any
   macro signal must read ALFRED vintages instead.
3. **No free historical options data.** The options snapshot collector accumulates
   history from the day it first runs. No options backtest before that date is
   possible; a result claiming otherwise is fabricated.

"""

FOOTER = """
## Sources deliberately not used

| Source | Why not |
| --- | --- |
| Bloomberg, Refinitiv, FactSet | Paid. Excluded by the project's hard constraints. |
| CRSP, Compustat | Paid. Their absence is the direct cause of the residual survivorship bias documented above. |
| Paid TRACE feeds | Paid. The error-corrected academic bond dataset is used instead. |
| Any broker execution API | This is a data pipeline; it has no order routing by design. |
"""


def _table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def render() -> str:
    parts = [HEADER]

    parts.append("## Sources at a glance\n")
    parts.append(_table_row(["Source", "Asset classes", "Point-in-time", "Key", "Caveats"]))
    parts.append(_table_row(["---"] * 5))
    for spec in sorted(SOURCES.values(), key=lambda s: s.key):
        key_note = f"`{spec.key_setting}`" if spec.key_setting else "none"
        parts.append(
            _table_row(
                [
                    f"[`{spec.key}`](#{spec.key.replace('_', '-')})",
                    ", ".join(a.value for a in spec.asset_classes),
                    f"`{spec.pit_quality.value}`",
                    key_note,
                    str(len(spec.caveats)),
                ]
            )
        )
    parts.append("")

    for asset_class in AssetClass:
        specs = sorted(
            (s for s in SOURCES.values() if s.asset_classes[0] is asset_class),
            key=lambda s: s.key,
        )
        if not specs:
            continue
        parts.append(f"\n## {asset_class.value}\n")
        for spec in specs:
            parts.append(f"### {spec.key}\n")
            parts.append(f"**{spec.name}** — <{spec.url}>\n")
            parts.append(_table_row(["Field", "Value"]))
            parts.append(_table_row(["---", "---"]))
            parts.append(_table_row(["Datasets", ", ".join(f"`{d}`" for d in spec.datasets)]))
            parts.append(
                _table_row(["Asset classes", ", ".join(a.value for a in spec.asset_classes)])
            )
            parts.append(_table_row(["Point-in-time", f"`{spec.pit_quality.value}`"]))
            parts.append(_table_row(["Update frequency", spec.update_frequency]))
            parts.append(_table_row(["Reliability", spec.reliability]))
            parts.append(_table_row(["Rate limit", spec.rate_limit]))
            parts.append(_table_row(["Ingest throttle", f"{spec.max_requests_per_second} req/s"]))
            parts.append(_table_row(["Licence", spec.licence]))
            parts.append(
                _table_row(
                    ["API key", f"`{spec.key_setting}`" if spec.key_setting else "not required"]
                )
            )
            parts.append("")
            if spec.cross_check_only:
                parts.append(
                    "> **Cross-check only.** Free-tier limits make this unusable for "
                    "primary ingest; it is used to validate other sources.\n"
                )
            if spec.pit_quality is PitQuality.RESTATED:
                parts.append(
                    "> **Restated data.** Blocked from the signal path by the "
                    "point-in-time layer unless explicitly overridden.\n"
                )
            parts.append("**Caveats**\n")
            for caveat in spec.caveats:
                parts.append(f"- {caveat}")
            parts.append("")
            for note in spec.notes:
                parts.append(f"> {note}\n")

    parts.append(FOOTER)
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print(f"{OUTPUT.relative_to(REPO)} is stale; run python scripts/gen_data_catalogue.py")
            return 1
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
