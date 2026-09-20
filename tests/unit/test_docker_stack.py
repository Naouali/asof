"""The compose stack is configuration the operator depends on, so it is tested.

Every assertion here corresponds to a requirement in spec section 10 that would
otherwise only be discovered on a fresh machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SERVICES = ("db", "worker", "scheduler", "dashboard", "research")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    # Compose's `${VAR:-default}` substitution is not YAML, but it is only ever used
    # inside scalars, so the document still parses.
    return yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))


def test_expected_services_exist(compose: dict[str, Any]) -> None:
    assert set(compose["services"]) == set(SERVICES)


def test_there_is_no_market_data_database_service(compose: dict[str, Any]) -> None:
    """Spec 3.8: bulk market data lives in the parquet lake queried by embedded
    DuckDB. Postgres is for paper-trading state and the run registry only."""
    images = " ".join(str(s.get("image", "")) for s in compose["services"].values())
    for engine in ("clickhouse", "questdb", "influxdb", "mongo"):
        assert engine not in images


def test_every_service_declares_resource_limits(compose: dict[str, Any]) -> None:
    """Spec section 10: it must not eat a laptop."""
    for name, service in compose["services"].items():
        limits = service.get("deploy", {}).get("resources", {}).get("limits", {})
        assert limits.get("cpus"), f"{name} declares no cpu limit"
        assert limits.get("memory"), f"{name} declares no memory limit"


def test_published_ports_bind_to_loopback_only(compose: dict[str, Any]) -> None:
    """Neither the dashboard nor Jupyter authenticates; exposing them on the local
    network would be a real vulnerability, not a theoretical one."""
    for name, service in compose["services"].items():
        for mapping in service.get("ports", []):
            assert str(mapping).startswith("127.0.0.1:"), (
                f"{name} publishes {mapping} on all interfaces"
            )


def test_database_is_not_published(compose: dict[str, Any]) -> None:
    assert "ports" not in compose["services"]["db"]


def test_data_volume_is_named_and_survives_rebuilds(compose: dict[str, Any]) -> None:
    assert compose["volumes"]["quantlab-data"]["name"] == "quantlab-data"
    for name in ("worker", "scheduler", "dashboard", "research"):
        mounts = compose["services"][name]["volumes"]
        assert any(str(m).startswith("quantlab-data:/data") for m in mounts), name


def test_services_wait_for_a_healthy_database(compose: dict[str, Any]) -> None:
    for name in ("worker", "scheduler", "dashboard", "research"):
        depends = compose["services"][name]["depends_on"]
        assert depends["db"]["condition"] == "service_healthy", name


def test_every_api_key_defaults_to_empty(compose: dict[str, Any]) -> None:
    """Spec section 10: the stack must start with zero API keys. A `${VAR}` without
    a default would make compose warn and, worse, suggest the key is required."""
    env = compose["services"]["worker"]["environment"]
    keys = [k for k in env if k.endswith("_API_KEY")]
    assert keys
    for key in keys:
        assert env[key].endswith(":-}"), f"{key} has no empty default: {env[key]!r}"


def test_db_has_a_healthcheck(compose: dict[str, Any]) -> None:
    assert compose["services"]["db"]["healthcheck"]["test"]


@pytest.mark.parametrize("name", ["worker", "dashboard", "research"])
def test_service_images_declare_a_healthcheck(name: str) -> None:
    text = (REPO / "docker" / f"Dockerfile.{name}").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text


@pytest.mark.parametrize("name", ["base", "research"])
def test_images_do_not_end_as_root(name: str) -> None:
    text = (REPO / "docker" / f"Dockerfile.{name}").read_text(encoding="utf-8")
    users = [line.split()[1] for line in text.splitlines() if line.startswith("USER ")]
    assert users and users[-1] != "root", (
        f"Dockerfile.{name} ends as {users[-1] if users else 'root'}"
    )


def test_base_image_pins_determinism_env() -> None:
    text = (REPO / "docker" / "Dockerfile.base").read_text(encoding="utf-8")
    assert "PYTHONHASHSEED=0" in text


def test_dependency_install_is_frozen() -> None:
    """`uv sync --frozen` refuses to update the lockfile, so an image can never be
    built from a resolution that is not the committed one (spec section 1)."""
    for name in ("base", "research"):
        text = (REPO / "docker" / f"Dockerfile.{name}").read_text(encoding="utf-8")
        for line in text.splitlines():
            if "uv sync" in line:
                assert "--frozen" in line, f"Dockerfile.{name}: unfrozen sync: {line.strip()}"


def test_lockfile_is_committed_and_covers_every_dependency() -> None:
    lock = REPO / "uv.lock"
    assert lock.exists(), "uv.lock must be committed: it is the reproducibility guarantee"
    text = lock.read_text(encoding="utf-8")
    for package in ("polars", "duckdb", "pyarrow", "fastapi", "pytest", "ruff"):
        assert f'name = "{package}"' in text, f"{package} missing from uv.lock"


def test_override_example_exists_and_real_override_is_ignored() -> None:
    assert (REPO / "docker-compose.override.yml.example").exists()
    assert "docker-compose.override.yml" in (REPO / ".gitignore").read_text(encoding="utf-8")
