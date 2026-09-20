"""Documentation that the code can falsify.

A stale data caveat is worse than a missing one: it is a false assurance. The
catalogue document is generated, so staleness is a test failure rather than
something noticed months later.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REQUIRED_DOCS = (
    "README.md",
    "docs/DATA_CATALOGUE.md",
    "docs/ASSUMPTIONS.md",
    "docs/LIMITATIONS.md",
    "docs/ARCHITECTURE.md",
    "docs/MILESTONES.md",
)


@pytest.mark.parametrize("relative", REQUIRED_DOCS)
def test_required_docs_exist_and_are_substantive(relative: str) -> None:
    path = REPO / relative
    assert path.exists(), f"{relative} is missing"
    assert len(path.read_text(encoding="utf-8")) > 500, f"{relative} is a stub"


def test_generated_catalogue_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_data_catalogue.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_source_appears_in_the_generated_catalogue() -> None:
    from quantlab.data.catalogue import SOURCES

    text = (REPO / "docs" / "DATA_CATALOGUE.md").read_text(encoding="utf-8")
    for key in SOURCES:
        assert f"### {key}" in text, f"{key} has no section in DATA_CATALOGUE.md"


def test_limitations_names_the_biases_it_cannot_fix() -> None:
    """Spec section 12 asks for an honest LIMITATIONS.md. These are the admissions
    that make it honest rather than decorative."""
    text = (REPO / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8").lower()
    for admission in (
        "survivorship",
        "no free historical options data",
        "live sharpe",
        "no live trading",
        "term structure",
    ):
        assert admission in text, f"LIMITATIONS.md does not address: {admission}"


def test_env_example_documents_every_api_key_setting() -> None:
    """A key that Settings reads but .env.example does not mention is a source the
    user has no way of discovering they could enable."""
    from quantlab.config import Settings

    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for field in Settings.model_fields:
        if field.endswith("_api_key"):
            assert f"QUANTLAB_{field.upper()}" in text, f"{field} is undocumented"


def test_env_example_contains_no_real_secrets() -> None:
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "_API_KEY=" in line and not line.strip().startswith("#"):
            assert line.strip().endswith("="), f"committed a value: {line}"


def test_makefile_exposes_every_target_the_spec_requires() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    for target in (
        "up",
        "down",
        "ingest",
        "ingest-daily",
        "backtest",
        "paper",
        "test",
        "lint",
        "shell",
        "notebook",
        "clean-data",
    ):
        assert f"\n{target}:" in text, f"Makefile has no `{target}` target"


# ----------------------------------------------------------------------------------
# Keeping the documentation honest
# ----------------------------------------------------------------------------------
def test_limitations_lists_every_documented_component() -> None:
    """Part II gained a section per subsystem as each was built. A component
    with no entry is one whose limitations were never written down."""
    text = (REPO / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    for component in (
        "Portfolio construction and risk",
        "Tearsheets",
        "Option chains",
        "The event-driven engine",
        "Paper trading",
    ):
        assert f"### {component}" in text, component


def test_the_signal_inventory_in_limitations_matches_the_registry() -> None:
    """Section 7 claimed for several milestones that only two Tier 1 signals
    could run. EDGAR, FRED and the Milestone 9 sources changed that and the
    document did not notice. A count that drifts is worse than no count: it is a
    number a reader will trust."""
    from quantlab.signals import SIGNAL_REGISTRY, load_all_signals

    load_all_signals()
    text = (REPO / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    section = text[text.index("## 7.") : text.index("## 8.")]

    for name in SIGNAL_REGISTRY:
        assert f"`{name}`" in section, f"{name} is missing from the signal inventory"

    counted = section.count("| Runs") + section.count("| **Blocked.**")
    assert counted == len(SIGNAL_REGISTRY), (
        f"the inventory lists {counted} signals but {len(SIGNAL_REGISTRY)} are registered"
    )


def test_limitations_does_not_claim_unbuilt_execution_modelling() -> None:
    """An early draft claimed the event-driven engine modelled order types,
    partial fills, latency and rejects. It models none of them, and the document
    whose job is to prevent overselling was doing the overselling."""
    text = (REPO / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    section = text[text.index("## 5.") : text.index("## 6.")]

    assert "does **not** add order types, partial fills, latency or rejects" in section
    assert "sequencing" in section


def test_the_signal_guide_covers_the_mistakes_that_do_not_look_like_mistakes() -> None:
    """Each of these cost real time in this build, and none of them raises."""
    guide = (REPO / "docs" / "ADDING_A_SIGNAL.md").read_text(encoding="utf-8")

    assert "reversed sign" in guide
    assert "reporting basis" in guide
    assert "calendar date" in guide
    assert "SignalUnavailableError" in guide
