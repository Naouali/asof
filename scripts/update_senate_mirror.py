"""Bring the Senate mirror up to date.

The Senate's disclosure site answers only connections from inside the United
States, so this runs where that is true -- the ``senate-mirror`` workflow, on a
US-hosted runner -- and writes what it read as JSON for everyone else to ingest.
See ``quantlab.data.sources.senate_efd`` for what the files hold and why.

    python scripts/update_senate_mirror.py mirror/senate --start 2025-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from quantlab.data.http import HttpClient, SourceError
from quantlab.data.sources.senate_efd import SenateSession, SenateTrades, update_mirror


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="where the ptr-YEAR.json files live")
    parser.add_argument(
        "--start",
        type=dt.date.fromisoformat,
        default=dt.date(2025, 1, 1),
        help="earliest filing date to mirror (default: 2025-01-01)",
    )
    args = parser.parse_args()

    try:
        with HttpClient(SenateTrades.spec) as client:
            added = update_mirror(
                SenateSession(client, "senate_efd.mirror"), args.directory, args.start
            )
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{added} new report(s) mirrored into {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
