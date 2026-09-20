from __future__ import annotations

import pytest

from quantlab.config import Settings, get_settings
from quantlab.data.catalogue import SOURCES
from quantlab.health import (
    Status,
    _check_user_agent,
    _user_agent_problem,
    run_checks,
    source_availability,
)


def test_checks_never_raise_and_never_fail_on_a_bare_install(settings: Settings) -> None:
    """Spec section 10: zero keys must degrade gracefully, not error."""
    settings.layout.ensure()
    checks = run_checks(settings)
    assert checks
    assert not [c for c in checks if c.status is Status.FAIL]


def test_keyless_sources_are_always_available(settings: Settings) -> None:
    availability = {a.spec.key: a for a in source_availability(settings)}
    assert len(availability) == len(SOURCES)
    for item in availability.values():
        if not item.spec.requires_key:
            assert item.available


def test_tier_one_signal_sources_work_without_any_key(settings: Settings) -> None:
    """The claim in the README and .env.example, asserted rather than trusted:
    trend following, crypto carry and equity profitability need no credentials."""
    available = {a.spec.key for a in source_availability(settings) if a.available}
    assert {"yahoo_futures", "binance", "sec_edgar", "ken_french", "stooq"} <= available


def test_missing_key_names_the_environment_variable(settings: Settings) -> None:
    fred = next(a for a in source_availability(settings) if a.spec.key == "fred")
    assert not fred.available
    assert "QUANTLAB_FRED_API_KEY" in fred.reason


def test_present_key_flips_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTLAB_FRED_API_KEY", "abc123")
    get_settings.cache_clear()
    fred = next(a for a in source_availability() if a.spec.key == "fred")
    assert fred.available


def test_unwritable_data_root_is_a_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.layout.ensure()
    monkeypatch.setattr("quantlab.health.os.access", lambda *_a, **_k: False)
    statuses = {c.name: c for c in run_checks(settings)}
    assert statuses["data_root"].status is Status.FAIL


def test_availability_dict_carries_caveats_to_the_ui(settings: Settings) -> None:
    for item in source_availability(settings):
        assert item.as_dict()["caveats"] == list(item.spec.caveats)


# ----------------------------------------------------------------------------------
# The SEC contact address
# ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "user_agent",
    [
        "QuantLab/0.1 (unconfigured -- set QUANTLAB_HTTP_USER_AGENT)",
        "QuantLab/0.1 (your.name@example.com)",  # .env.example, copied unchanged
        "QuantLab/0.1 (youremail@gmail.com)",
        "QuantLab/0.1 (test@company.com)",
        "QuantLab/0.1 (noreply@company.com)",
        "QuantLab/0.1 (changeme@company.com)",
        "QuantLab/0.1",  # no address at all
        "QuantLab/0.1 (someone@localhost)",  # no domain dot
    ],
)
def test_a_placeholder_contact_address_is_not_configured(user_agent: str) -> None:
    """Copying `.env.example` unchanged is the documented setup path, and it
    leaves a placeholder behind. Passing that as configured is how a fabricated
    contact address reaches the SEC looking like a real one."""
    assert _user_agent_problem(user_agent) is not None


@pytest.mark.parametrize(
    "user_agent",
    [
        "QuantLab/0.1 (ada@lovelace.org)",
        "QuantLab/0.1 (research@some-fund.co.uk)",
        "quantlab/0.1 (a.b+c@sub.domain.io)",
    ],
)
def test_a_real_contact_address_passes(user_agent: str) -> None:
    assert _user_agent_problem(user_agent) is None


def test_the_check_explains_why_a_placeholder_is_worse_than_nothing() -> None:
    settings = Settings(http_user_agent="QuantLab/0.1 (your.name@example.com)")
    check = _check_user_agent(settings)

    assert check.status is Status.WARN
    assert "placeholder" in check.detail
    assert check.hint is not None
    assert "fabricated identity" in check.hint
