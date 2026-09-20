from __future__ import annotations

from fastapi.testclient import TestClient

from quantlab.config import Settings
from quantlab.dashboard.app import create_app


def test_health_is_cheap_and_ok() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_health_reports_checks(settings: Settings) -> None:
    settings.layout.ensure()
    with TestClient(create_app()) as client:
        payload = client.get("/api/health").json()
    assert payload["status"] in {"ok", "warn"}
    assert {c["name"] for c in payload["checks"]} >= {"python", "data_root", "sources"}


def test_api_sources_exposes_every_caveat() -> None:
    with TestClient(create_app()) as client:
        payload = client.get("/api/sources").json()
    assert payload["count"] == len(payload["sources"])
    for source in payload["sources"]:
        assert source["caveats"], f"{source['key']} exposes no caveats to the UI"


def test_index_renders_and_carries_the_data_quality_warning() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "QuantLab" in response.text
    assert "caveats" in response.text.lower()


def test_dashboard_references_no_external_assets() -> None:
    """Spec section 1: the platform must be fully usable with no internet access.
    A CDN reference would make the UI silently degrade offline."""
    with TestClient(create_app()) as client:
        body = client.get("/").text
    for scheme in ("http://", "https://"):
        assert scheme not in body, "dashboard must not reference external assets"
