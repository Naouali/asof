"""Open Source Asset Pricing — the published anomaly catalogue.

Chen and Zimmermann's replication of the cross-sectional equity literature: every
predictor they could find in a published paper, with the effect size and
t-statistic the original authors reported, and their own assessment of whether it
replicates.

**Why this platform carries it.** Spec section 0 says the system exists to tell
you whether a signal works, and section 7 makes the trial count mandatory on every
Sharpe ratio. This file is the trial count of the entire literature. Of 212
published predictors the median reported t-statistic is 4.0 and only 2.7% fall
below 2.0 -- a distribution truncated exactly where journals stop accepting
papers. A new signal with a t-statistic of 2.5 is not unusual against that
backdrop; it is below the median of a set that is itself selected on significance.

``as_of`` is the end of the original sample and ``known_at`` is publication,
because those are different questions: in-sample fit ends at the first, and
post-publication decay is measured from the second.

**The distribution channel is fragile, and that is the source's own choice.** The
files are on Google Drive, behind IDs that change whenever the authors re-upload.
There is no versioned URL, no content hash and no API. The ID is pinned here and
the payload's shape is checked on arrival, so a re-upload fails loudly rather
than silently loading a different vintage.
"""

from __future__ import annotations

import datetime as dt
import io
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from quantlab.data.catalogue import AssetClass, get_source
from quantlab.data.http import SourceError
from quantlab.data.sources.base import Source, register
from quantlab.logging import get_logger

__all__ = ["DOCUMENTATION_FILE_ID", "OpenAssetPricingCatalogue"]

log = get_logger("quantlab.data.sources.open_asset_pricing")

#: Pinned. A re-upload changes this and the fetch fails, which is the intent:
#: silently loading a different vintage of the literature would change every
#: multiple-testing number derived from it.
DOCUMENTATION_FILE_ID = "1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2"
DRIVE_URL = "https://drive.google.com/uc?export=download&id={file_id}"

#: Columns the parser depends on. Checked on arrival rather than assumed.
REQUIRED_COLUMNS = (
    "Acronym",
    "Cat.Signal",
    "Authors",
    "Year",
    "LongDescription",
    "SampleEndYear",
)

#: ``Cat.Signal`` values. "Placebo" entries are predictors the authors could not
#: find published evidence for; they are kept, because a catalogue of what was
#: tried is worth more than a catalogue of what worked.
PREDICTOR = "Predictor"


@register
class OpenAssetPricingCatalogue(Source):
    """The signal documentation table: one row per published predictor."""

    name: ClassVar[str] = "open_asset_pricing.anomaly_catalogue"
    dataset: ClassVar[str] = "anomaly_catalogue"
    asset_class: ClassVar[AssetClass] = AssetClass.REFERENCE
    spec: ClassVar = get_source("open_asset_pricing")

    def fetch(self, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        self.require_available()
        table = self._download()

        wanted = {s.upper() for s in symbols}
        rows: list[dict[str, Any]] = []
        for record in table.iter_rows(named=True):
            acronym = str(record.get("Acronym") or "").strip()
            if not acronym or (wanted and acronym.upper() not in wanted):
                continue
            row = _row_for(record, acronym)
            if row is None:
                continue
            if not (start <= row["known_at"] <= end):
                continue
            rows.append(row)

        if not rows:
            raise SourceError(
                self.name,
                f"no catalogued predictors published between {start:%Y-%m-%d} and "
                f"{end:%Y-%m-%d}. The window filters on PUBLICATION date; the "
                "literature here runs from 1973 to 2016.",
            )
        log.info(
            "open_asset_pricing.catalogue",
            rows=len(rows),
            predictors=sum(1 for r in rows if r["category"] == PREDICTOR),
        )
        return self.finalise(rows)

    def _download(self) -> pl.DataFrame:
        payload = self.client.get_bytes(DRIVE_URL.format(file_id=DOCUMENTATION_FILE_ID))
        if payload[:15].lower().startswith(b"<!doctype html") or payload[:6] == b"<html>":
            raise SourceError(
                self.name,
                "Google Drive returned an HTML page rather than the CSV. The pinned "
                f"file id {DOCUMENTATION_FILE_ID} has probably been re-uploaded, or "
                "the file now exceeds Drive's scan limit and serves a confirmation "
                "page. The id is pinned deliberately: loading a different vintage "
                "of the literature silently would change every multiple-testing "
                "number derived from it.",
            )
        try:
            table = pl.read_csv(
                io.BytesIO(payload), infer_schema_length=2000, truncate_ragged_lines=True
            )
        except Exception as exc:
            raise SourceError(self.name, f"signal documentation did not parse: {exc}") from exc

        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise SourceError(
                self.name,
                f"signal documentation is missing columns {missing}; got "
                f"{table.columns[:12]}. The layout changed and the parser must be "
                "updated rather than guessing.",
            )
        return table


def _row_for(record: dict[str, Any], acronym: str) -> dict[str, Any] | None:
    """One catalogue row, or ``None`` when the dates cannot be established."""
    published = _year(record.get("Year"))
    if published is None:
        return None
    sample_end = _year(record.get("SampleEndYear")) or published

    return {
        "symbol": acronym,
        # The last date the published evidence covers ...
        "as_of": dt.datetime(sample_end, 12, 31, tzinfo=dt.UTC),
        # ... and the year it became public. Anything after this is out of sample
        # for the original paper, which is what makes decay measurable.
        "known_at": dt.datetime(published, 12, 31, tzinfo=dt.UTC),
        "name": str(record.get("LongDescription") or acronym)[:400],
        "authors": _text(record.get("Authors")),
        "journal": _text(record.get("Journal")),
        "category": _text(record.get("Cat.Signal")) or "unknown",
        "replication": _text(record.get("Signal Rep Quality")),
        "evidence": _text(record.get("Predictability in OP")),
        "published_return": _number(record.get("Return")),
        "published_t_stat": _number(record.get("T-Stat")),
        "sign": _number(record.get("Sign")),
    }


def _year(value: Any) -> int | None:
    try:
        year = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def _number(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text if text and text.upper() != "NA" else None
