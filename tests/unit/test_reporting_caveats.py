"""Source caveats reaching the tearsheet.

The caveats are not written in the reporting layer. They live in the source
catalogue as structured data, recorded once when a source is registered, so a
limitation discovered while writing a fetcher appears on every tearsheet built
from that fetcher's data without anyone remembering to mention it.
"""

from __future__ import annotations

import pytest

from quantlab.data.catalogue import SOURCES, PitQuality
from quantlab.reporting.caveats import caveats_for, sources_behind


def test_a_sources_caveats_are_collected_and_attributed() -> None:
    collected = caveats_for(["yahoo"])
    spec = SOURCES["yahoo"]

    assert len(collected) >= len(spec.caveats)
    assert all(text.startswith(spec.name) for text in collected)
    assert any("survivorship" in text.lower() for text in collected)


def test_restated_data_earns_an_extra_warning_of_its_own() -> None:
    """A source whose history is rewritten is not point-in-time, and that is a
    different kind of problem from any individual caveat it lists."""
    restated = [key for key, spec in SOURCES.items() if spec.pit_quality is PitQuality.RESTATED]
    assert restated, "the catalogue should record at least one restated source"

    collected = caveats_for(restated[:1])
    assert any("RESTATED" in text for text in collected)
    assert any("upper bound" in text for text in collected)


def test_an_unknown_source_raises_rather_than_being_skipped() -> None:
    """A tearsheet that silently dropped the caveats of a source it could not
    identify would show a short, clean list and imply the data was clean."""
    with pytest.raises(KeyError):
        caveats_for(["yahoo", "not-a-real-source"])


def test_repeated_sources_are_collected_once() -> None:
    assert caveats_for(["yahoo", "yahoo"]) == caveats_for(["yahoo"])


def test_order_is_stable() -> None:
    first = caveats_for(["yahoo", "binance"])
    assert first == caveats_for(["yahoo", "binance"])
    assert first != caveats_for(["binance", "yahoo"])


def test_the_list_can_be_truncated() -> None:
    assert len(caveats_for(["yahoo"], limit=2)) == 2


def test_no_sources_means_no_caveats() -> None:
    assert caveats_for([]) == ()


def test_sources_behind_a_dataset_is_a_superset() -> None:
    """Used when a run did not record which source it read. Erring toward showing
    a limitation that did not apply beats hiding one that did."""
    keys = sources_behind(["ohlcv_daily"])

    assert "yahoo" in keys
    assert all(key in SOURCES for key in keys)
    assert list(keys) == sorted(keys)  # deterministic


def test_an_unknown_dataset_yields_nothing_rather_than_guessing() -> None:
    assert sources_behind(["not_a_dataset"]) == ()


def test_every_catalogue_source_can_be_rendered() -> None:
    """A source whose caveats crash the reporting layer would be discovered on
    the day someone used it, which is the wrong day."""
    for key in SOURCES:
        assert isinstance(caveats_for([key]), tuple)
