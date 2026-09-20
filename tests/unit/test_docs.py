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
    """These are the admissions that make LIMITATIONS.md honest rather than
    decorative."""
    text = (REPO / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8").lower()
    for admission in (
        "survivorship",
        "no free historical options data",
        "restated",
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


def test_makefile_exposes_the_operational_targets() -> None:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    for target in (
        "up",
        "down",
        "ingest",
        "ingest-daily",
        "status",
        "test",
        "lint",
        "shell",
        "clean-data",
    ):
        assert f"\n{target}:" in text, f"Makefile has no `{target}` target"
