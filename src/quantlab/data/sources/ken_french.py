"""Kenneth R. French Data Library.

The accepted benchmark for equity factor research: survivorship-bias-free,
decades of history, free, and built on CRSP/Compustat data nobody else gets for
nothing. Spec section 3.3 is direct about the use: *"If your hand-built momentum
factor doesn't correlate >0.9 with Ken French's UMD, your construction has a bug."*

Pulled forward from Milestone 9 because Milestone 6's acceptance criterion is that
benchmark, and a benchmark you cannot run is not one.

**On the restatement.** The whole history is rebuilt on each release, so these
files answer *"what do we now believe UMD returned in 1965"* rather than *"what was
known then"*. The catalogue marks the source ``restated`` for that reason. What is
restated is the **construction** -- the underlying returns are realised portfolio
returns, not forecasts -- so as a benchmark and as a risk-model factor these are
sound. As a point-in-time signal input they are not, and ``vintage`` is recorded as
false on every row so that distinction survives into the lake.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.data.store import utcnow
from quantlab.logging import get_logger

__all__ = ["KEN_FRENCH_FILES", "KenFrenchFactors", "parse_ken_french_csv"]

log = get_logger("quantlab.data.sources.ken_french")

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: Zip file to the factors it contains, with our canonical symbol names.
KEN_FRENCH_FILES: dict[str, dict[str, str]] = {
    "F-F_Research_Data_Factors_daily_CSV.zip": {
        "Mkt-RF": "KF_MKT_RF",
        "SMB": "KF_SMB",
        "HML": "KF_HML",
        "RF": "KF_RF",
    },
    "F-F_Momentum_Factor_daily_CSV.zip": {"Mom": "KF_MOM"},
}

#: Files are refreshed monthly, so a daily observation is not published until the
#: following release. Charged conservatively; these are a benchmark, not a signal,
#: but a lag that is too short is the kind of error that silently becomes one.
PUBLICATION_LAG_DAYS = 60


def parse_ken_french_csv(text: str, columns: dict[str, str]) -> list[dict[str, Any]]:
    """Parse one Ken French CSV into ``(symbol, date, value)`` records.

    The files carry several paragraphs of prose, then a header row beginning with
    a comma, then ``YYYYMMDD,value...`` rows, then a copyright footer -- and
    sometimes a second table of annual data after a blank line. Parsing is done by
    recognising the shape of a data row rather than by counting header lines,
    because the number of prose lines differs per file and changes between
    releases.
    """
    records: list[dict[str, Any]] = []
    header: list[str] | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # A blank line ends the daily table; anything after it is a second,
            # lower-frequency table that would silently corrupt the series.
            if header is not None and records:
                break
            continue

        parts = [part.strip() for part in line.split(",")]
        if header is None:
            if parts[0] == "" and len(parts) > 1:
                header = parts
            continue

        stamp = parts[0]
        if not (len(stamp) == 8 and stamp.isdigit()):
            if records:
                break  # the copyright footer, or an annual table
            continue

        try:
            date = dt.datetime.strptime(stamp, "%Y%m%d").replace(tzinfo=dt.UTC)
        except ValueError:  # pragma: no cover - guarded by the isdigit check
            continue

        for index, name in enumerate(header[1:], start=1):
            if name not in columns or index >= len(parts):
                continue
            try:
                value = float(parts[index])
            except ValueError:
                continue
            # Ken French publishes returns in percent.
            if value <= -99.0:
                continue  # the library's missing-value marker
            records.append({"symbol": columns[name], "date": date, "value": value / 100.0})

    if header is None:
        raise SourceError("ken_french", "no header row found; the file format has changed")
    return records


@register
class KenFrenchFactors(Source):
    """Daily factor returns from the Ken French library."""

    name: ClassVar[str] = "ken_french.series_observations"
    spec: ClassVar = get_source("ken_french")
    dataset: ClassVar[str] = "series_observations"
    asset_class: ClassVar = AssetClass.FACTORS
    default_symbols: ClassVar[tuple[str, ...]] = (
        "KF_MKT_RF",
        "KF_SMB",
        "KF_HML",
        "KF_RF",
        "KF_MOM",
    )

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        wanted = {s.upper() for s in symbols} if symbols else None
        lag = dt.timedelta(days=PUBLICATION_LAG_DAYS)
        observed = utcnow()
        rows: list[dict[str, Any]] = []

        for filename, columns in KEN_FRENCH_FILES.items():
            if wanted is not None and not (set(columns.values()) & wanted):
                continue

            payload = self.client.request(BASE_URL + filename).content
            try:
                archive = zipfile.ZipFile(io.BytesIO(payload))
            except zipfile.BadZipFile as exc:
                raise SourceError(
                    self.spec.key,
                    f"{filename} is not a zip archive; the library's layout has "
                    f"changed or the request was redirected: {exc}",
                ) from exc

            members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise SourceError(self.spec.key, f"{filename} contains no CSV")
            text = archive.read(members[0]).decode("utf-8", errors="replace")

            for record in parse_ken_french_csv(text, columns):
                if wanted is not None and record["symbol"] not in wanted:
                    continue
                date = record["date"]
                if not (start <= date <= end):
                    continue
                rows.append(
                    {
                        "symbol": record["symbol"],
                        "as_of": date,
                        # Published with a lag, and restated on every release --
                        # sound as a benchmark, not as a point-in-time input.
                        #
                        # Capped at the download instant: the lag is an upper bound
                        # on when an observation became knowable, but holding the
                        # data bounds it too. Without the cap the most recent two
                        # months arrive stamped as knowable in the future, which the
                        # schema guard rejects -- correctly, since a row nothing can
                        # ever have seen is not a row.
                        "known_at": min(date + lag, observed),
                        "value": record["value"],
                        "units": "daily return",
                        "vintage": False,
                    }
                )

            log.info("ken_french.parsed", file=filename, rows=len(rows))

        return self.finalise(rows)
