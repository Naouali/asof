"""End-to-end test of the compose stack.

Marked `integration` and skipped unless a Docker daemon is reachable, so the unit
suite stays runnable anywhere. CI runs this in its own job (see .github/workflows).

This is the automated half of the acceptance criterion: on a clean machine
with only Docker installed, `make up` must produce a healthy stack with no manual
intervention. The other half -- a genuinely fresh VM -- has to be done by hand.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return (
        subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False).returncode
        == 0
    )


requires_docker = pytest.mark.skipif(not _docker_available(), reason="no reachable Docker daemon")


def _run(*argv: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False
    )


@requires_docker
def test_compose_file_is_valid() -> None:
    result = _run("docker", "compose", "config", "--quiet", timeout=120)
    assert result.returncode == 0, result.stderr


@requires_docker
@pytest.mark.timeout(3600)
def test_stack_comes_up_healthy_with_no_api_keys() -> None:
    """The acceptance criterion: `make up` on a bare checkout, zero keys."""
    try:
        build = _run("make", "up")
        assert build.returncode == 0, build.stdout + build.stderr

        state = json.loads(_run("docker", "compose", "ps", "--format", "json").stdout or "[]")
        services = state if isinstance(state, list) else [state]
        for service in services:
            health = service.get("Health", "")
            assert health in {"healthy", ""}, f"{service.get('Service')} is {health}"

        # `doctor` must succeed with no keys configured.
        doctor = _run("docker", "compose", "exec", "-T", "worker", "quantlab", "doctor")
        assert doctor.returncode == 0, doctor.stdout + doctor.stderr

        # The ingest plan must resolve with no keys configured. Dry run: nothing
        # is fetched.
        ingest = _run(
            "docker", "compose", "exec", "-T", "worker", "quantlab", "data", "ingest", "--dry-run"
        )
        assert ingest.returncode == 0, ingest.stdout + ingest.stderr
    finally:
        _run("docker", "compose", "down", "--remove-orphans", timeout=300)


@requires_docker
@pytest.mark.timeout(3600)
def test_base_image_builds_for_both_architectures() -> None:
    """Must build and run on linux/amd64 and linux/arm64."""
    result = _run(
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64,linux/arm64",
        "-f",
        "docker/Dockerfile.base",
        "--target",
        "runtime",
        ".",
    )
    assert result.returncode == 0, result.stdout + result.stderr
