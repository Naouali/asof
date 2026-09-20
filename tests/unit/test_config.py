from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantlab.config import Settings, get_settings


def test_stack_runs_with_zero_api_keys(settings: Settings) -> None:
    """Spec section 10: the stack must start and run with no API keys at all."""
    for field in Settings.model_fields:
        if field.endswith("_api_key"):
            assert settings.secret_for(field) is None


def test_secrets_are_not_printed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTLAB_FRED_API_KEY", "super-secret-value")
    get_settings.cache_clear()
    settings = get_settings()
    assert "super-secret-value" not in repr(settings)
    assert settings.secret_for("fred_api_key") == "super-secret-value"


def test_empty_key_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-empty variable (the compose default) must not read as present."""
    monkeypatch.setenv("QUANTLAB_FRED_API_KEY", "")
    get_settings.cache_clear()
    assert get_settings().secret_for("fred_api_key") is None


def test_settings_are_frozen(settings: Settings) -> None:
    with pytest.raises(ValidationError):
        settings.http_max_retries = 1  # type: ignore[misc]


def test_default_user_agent_is_obviously_unconfigured(settings: Settings) -> None:
    """SEC EDGAR blocks anonymous requests; the default must trip `doctor`, not the SEC."""
    assert "unconfigured" in settings.http_user_agent


def test_layout_ensure_is_idempotent(settings: Settings) -> None:
    layout = settings.layout
    layout.ensure()
    layout.ensure()
    for directory in layout.all_data_dirs():
        assert directory.is_dir()
