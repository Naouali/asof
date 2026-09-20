"""The published anomaly catalogue.

This is the trial count of the cross-sectional equity literature, so the tests
are about keeping its two dates distinct -- the end of the original sample and
the date of publication -- and about failing loudly when the fragile Google Drive
distribution changes underneath it.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from tests.unit.test_sources import fixture_source, load_fixture

from quantlab.config import Settings
from quantlab.data.http import SourceError
from quantlab.data.sources.open_asset_pricing import (
    DOCUMENTATION_FILE_ID,
    OpenAssetPricingCatalogue,
)

WINDOW = (dt.datetime(1900, 1, 1, tzinfo=dt.UTC), dt.datetime(2026, 9, 20, tzinfo=dt.UTC))


def fetch(settings: Settings, symbols: list[str] | None = None) -> pl.DataFrame:
    payload = load_fixture("osap_signal_documentation.csv")
    with fixture_source(OpenAssetPricingCatalogue, payload, settings) as source:
        return source.fetch(symbols or [], *WINDOW)


def test_the_catalogue_parses(settings: Settings) -> None:
    frame = fetch(settings)

    assert frame.height > 0
    assert frame["symbol"].n_unique() == frame.height, "one row per predictor"
    assert frame["name"].null_count() == 0


def test_sample_end_and_publication_are_kept_apart(settings: Settings) -> None:
    """They answer different questions. In-sample fit ends at the sample end;
    post-publication decay is measured from publication. Collapsing them into one
    date makes the out-of-sample window disappear."""
    frame = fetch(settings)

    assert (frame["known_at"] >= frame["as_of"]).all()
    gaps = (frame["known_at"] - frame["as_of"]).dt.total_days()
    assert gaps.max() > 365, "a paper takes years to appear after its sample ends"


def test_the_published_effect_size_is_carried(settings: Settings) -> None:
    frame = fetch(settings)
    stats = frame["published_t_stat"].drop_nulls()

    assert stats.len() > 0
    assert stats.min() > 0, "published t-statistics are reported as magnitudes"


def test_placebos_are_kept_not_filtered(settings: Settings) -> None:
    """A catalogue of what was tried is worth more than a catalogue of what
    worked. Keeping only the predictors would reproduce, inside this platform,
    exactly the selection that makes the published distribution misleading."""
    frame = fetch(settings)
    assert "category" in frame.columns
    assert frame["category"].null_count() == 0


def test_a_single_predictor_can_be_requested(settings: Settings) -> None:
    frame = fetch(settings, ["Accruals"])
    assert frame["symbol"].to_list() == ["Accruals"]


def test_an_html_page_instead_of_the_csv_fails_loudly(settings: Settings) -> None:
    """Google Drive serves a confirmation page when a file id is stale or the
    payload outgrows its scanner. Parsing that as CSV would produce an empty
    catalogue, and an empty trial count reads as 'nothing has been tried'."""
    with (
        fixture_source(OpenAssetPricingCatalogue, "<!doctype html><html>...", settings) as source,
        pytest.raises(SourceError, match="re-uploaded"),
    ):
        source.fetch([], *WINDOW)


def test_a_changed_layout_fails_rather_than_guessing(settings: Settings) -> None:
    with (
        fixture_source(OpenAssetPricingCatalogue, "Foo,Bar\n1,2\n", settings) as source,
        pytest.raises(SourceError, match="missing columns"),
    ):
        source.fetch([], *WINDOW)


def test_a_window_outside_the_literature_fails_loudly(settings: Settings) -> None:
    payload = load_fixture("osap_signal_documentation.csv")
    with (
        fixture_source(OpenAssetPricingCatalogue, payload, settings) as source,
        pytest.raises(SourceError, match="PUBLICATION date"),
    ):
        source.fetch(
            [],
            dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2021, 1, 1, tzinfo=dt.UTC),
        )


def test_the_drive_file_id_is_pinned() -> None:
    """Not a constant for tidiness: an unpinned id silently loads whatever
    vintage of the literature is current, and every multiple-testing number
    derived from it would move without anything in the repository changing."""
    assert DOCUMENTATION_FILE_ID
    assert OpenAssetPricingCatalogue.dataset == "anomaly_catalogue"
