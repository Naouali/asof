"""The signal contract, and the metadata every signal is required to declare."""

from __future__ import annotations

import pytest

from quantlab.backtest.conventions import RebalanceFrequency
from quantlab.data.catalogue import AssetClass
from quantlab.signals import SIGNAL_REGISTRY, get_signal
from quantlab.signals.base import (
    EvidenceGrade,
    SignalOutput,
    SignalSpec,
)


def spec(**overrides: object) -> SignalSpec:
    base: dict[str, object] = {
        "name": "test.signal",
        "asset_class": AssetClass.EQUITY,
        "tier": 1,
        "output": SignalOutput.CROSS_SECTIONAL_SCORE,
        "required_datasets": ("ohlcv_daily",),
        "rebalance": RebalanceFrequency.MONTHLY,
        "expected_turnover_annual": 1.0,
        "evidence": EvidenceGrade.STRONG,
        "reference": "Someone (2020)",
        "known_failure_modes": "x" * 60,
    }
    base.update(overrides)
    return SignalSpec(**base)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------
# Every signal must declare how it fails
# ----------------------------------------------------------------------------------
def test_a_signal_must_declare_substantive_failure_modes() -> None:
    """Not decoration. A library accumulates entries faster than anyone remembers
    the caveats, and the tearsheet quoting a Sharpe should quote these beside it."""
    with pytest.raises(ValueError, match="do not understand it well enough"):
        spec(known_failure_modes="it sometimes loses money")


def test_a_signal_must_declare_the_data_it_reads() -> None:
    with pytest.raises(ValueError, match="declare the data it reads"):
        spec(required_datasets=())


def test_tier_must_be_one_two_or_three() -> None:
    with pytest.raises(ValueError, match="tier must be"):
        spec(tier=4)


@pytest.mark.parametrize("name", sorted(SIGNAL_REGISTRY))
def test_every_registered_signal_declares_its_failure_modes(name: str) -> None:
    signal_spec = SIGNAL_REGISTRY[name].spec
    assert len(signal_spec.known_failure_modes) > 100, (
        f"{name}: the failure modes are too terse to be useful to a reader deciding "
        "whether to trade it"
    )


@pytest.mark.parametrize("name", sorted(SIGNAL_REGISTRY))
def test_every_registered_signal_cites_its_source(name: str) -> None:
    assert len(SIGNAL_REGISTRY[name].spec.reference) > 20


@pytest.mark.parametrize("name", sorted(SIGNAL_REGISTRY))
def test_every_registered_signal_declares_expected_turnover(name: str) -> None:
    """The single most useful number for guessing whether a signal survives costs,
    before any of it is built."""
    assert SIGNAL_REGISTRY[name].spec.expected_turnover_annual > 0


@pytest.mark.parametrize("name", sorted(SIGNAL_REGISTRY))
def test_every_registered_signal_reads_a_real_dataset(name: str) -> None:
    from quantlab.data.schemas import DATASETS

    for dataset in SIGNAL_REGISTRY[name].spec.required_datasets:
        assert dataset in DATASETS, f"{name} declares unknown dataset {dataset!r}"


# ----------------------------------------------------------------------------------
# Tiering reflects evidence, not interest
# ----------------------------------------------------------------------------------
def test_tier_one_signals_claim_strong_evidence() -> None:
    for name, cls in SIGNAL_REGISTRY.items():
        if cls.spec.tier == 1:
            assert cls.spec.evidence is EvidenceGrade.STRONG, (
                f"{name} is Tier 1 but its evidence is {cls.spec.evidence.value}; "
                "tiering is by evidence quality, not by interest"
            )


def test_tier_three_signals_are_marked_decayed_and_say_so() -> None:
    """Spec section 4: the platform must be argumentative. A Tier 3 signal surfaces
    the prior evidence that it is decayed, rather than being implemented and left
    to speak for itself."""
    tier_three = [c for c in SIGNAL_REGISTRY.values() if c.spec.tier == 3]
    assert tier_three, "the library should carry at least one decayed effect to re-test"
    for cls in tier_three:
        assert cls.spec.evidence is EvidenceGrade.DECAYED
        assert "STOPPED WORKING" in cls.spec.known_failure_modes.upper() or (
            "decay" in cls.spec.known_failure_modes.lower()
        )


def test_the_tier_one_signals_the_spec_names_are_all_present() -> None:
    """Spec section 4, Tier 1: trend, cross-asset carry, equity profitability."""
    names = set(SIGNAL_REGISTRY)
    assert "trend.time_series_momentum" in names
    assert "equity.profitability" in names
    carry = {n for n in names if n.startswith("carry.")}
    assert {"carry.rates", "carry.commodity_basis", "carry.crypto_perp_funding"} <= carry


# ----------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------
def test_unknown_signal_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known signals"):
        get_signal("trend.something_invented")


def test_signals_are_addressable_by_name() -> None:
    assert get_signal("trend.time_series_momentum").spec.tier == 1


def test_spec_describes_itself() -> None:
    text = get_signal("trend.time_series_momentum").spec.describe()
    assert "fails when" in text
    assert "turnover" in text
