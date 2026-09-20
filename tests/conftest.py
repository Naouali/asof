"""Shared fixtures.

Every test runs against an isolated data root and a settings object built from a
controlled environment. Two things make that necessary:

* :func:`quantlab.config.get_settings` is ``lru_cache``d, so a test that mutates the
  environment would otherwise leak into every later test.
* ``Settings`` reads a ``.env`` file when one is present. A developer's real ``.env``
  must never change what the test suite asserts.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from quantlab.config import Settings, get_settings

_QUANTLAB_ENV_PREFIX = "QUANTLAB_"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test a clean environment and its own data root."""
    for name in list(os.environ):
        if name.startswith(_QUANTLAB_ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANTLAB_ENV", "ci")
    monkeypatch.setenv("QUANTLAB_DATA_ROOT", str(tmp_path / "data"))
    # Neutralise any .env sitting in the working directory.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(isolated_env: None) -> Settings:
    return get_settings()


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
