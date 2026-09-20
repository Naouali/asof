"""System self-diagnosis -- the engine behind `quantlab doctor` and `/health`.

Spec section 10 requires the stack to start with zero API keys and to "degrade
gracefully with a clear message for each unavailable source". That message is
produced here, once, and rendered by both the CLI and the dashboard so the two can
never disagree.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from quantlab import __version__
from quantlab.config import Settings, get_settings
from quantlab.data.catalogue import SOURCES, SourceSpec

__all__ = ["Check", "SourceAvailability", "Status", "run_checks", "source_availability"]


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class SourceAvailability:
    spec: SourceSpec
    available: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "name": self.spec.name,
            "available": self.available,
            "reason": self.reason,
            "requires_key": self.spec.requires_key,
            "key_setting": self.spec.key_setting,
            "asset_classes": [a.value for a in self.spec.asset_classes],
            "pit_quality": self.spec.pit_quality.value,
            "cross_check_only": self.spec.cross_check_only,
            "caveats": list(self.spec.caveats),
        }


def source_availability(settings: Settings | None = None) -> list[SourceAvailability]:
    """Report which catalogued sources are usable with the credentials present.

    A missing key is a WARN, never an error: keyless sources cover macro rates, US
    equities, academic factors, futures positioning and all of crypto, which is
    enough to run the whole Tier 1 signal set.
    """
    settings = settings or get_settings()
    out: list[SourceAvailability] = []
    for spec in SOURCES.values():
        if spec.key_setting is None:
            out.append(SourceAvailability(spec, True, "keyless"))
            continue
        env_name = f"QUANTLAB_{spec.key_setting.upper()}"
        if settings.secret_for(spec.key_setting):
            out.append(SourceAvailability(spec, True, f"{env_name} is set"))
        else:
            out.append(
                SourceAvailability(spec, False, f"{env_name} is not set -- source unavailable")
            )
    return sorted(out, key=lambda s: (not s.available, s.spec.key))


def _check_python() -> Check:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        return Check(
            "python",
            Status.FAIL,
            f"Python {major}.{minor} is too old",
            "QuantLab requires Python 3.11+ (spec section 1).",
        )
    return Check("python", Status.OK, f"Python {major}.{minor}")


def _check_data_root(settings: Settings) -> Check:
    layout = settings.layout
    root = layout.data_root
    if not root.exists():
        return Check(
            "data_root",
            Status.WARN,
            f"{root} does not exist yet",
            "Run `quantlab init` (or `make up`, which runs it) to create the lake.",
        )
    if not os.access(root, os.W_OK):
        return Check(
            "data_root",
            Status.FAIL,
            f"{root} is not writable by uid {os.getuid()}",
            "Check the named-volume ownership; containers run as a non-root user.",
        )
    usage = shutil.disk_usage(root)
    free_gb = usage.free / 1e9
    status = Status.OK if free_gb >= 5 else Status.WARN
    missing = [d.name for d in layout.all_data_dirs() if not d.exists()]
    detail = f"{root} writable, {free_gb:.1f} GB free"
    if missing:
        detail += f"; missing subdirectories: {', '.join(missing)}"
        status = Status.WARN if status is Status.OK else status
    return Check("data_root", status, detail)


def _check_user_agent(settings: Settings) -> Check:
    ua = settings.http_user_agent
    if "unconfigured" in ua:
        return Check(
            "http_user_agent",
            Status.WARN,
            "User-Agent is not configured",
            "SEC EDGAR rejects requests without a descriptive User-Agent carrying "
            "contact details. Set QUANTLAB_HTTP_USER_AGENT before ingesting fundamentals.",
        )
    return Check("http_user_agent", Status.OK, ua)


def _check_sources(settings: Settings) -> Check:
    availability = source_availability(settings)
    unavailable = [a for a in availability if not a.available]
    total = len(availability)
    if not unavailable:
        return Check("sources", Status.OK, f"all {total} catalogued sources available")
    names = ", ".join(a.spec.key for a in unavailable)
    return Check(
        "sources",
        Status.WARN,
        f"{total - len(unavailable)}/{total} sources available; missing keys for: {names}",
        "The stack runs without these. Keyless sources cover macro, US equities, "
        "academic factors and all of crypto -- enough for the Tier 1 signals.",
    )


def _check_offline(settings: Settings) -> Check:
    if settings.offline:
        return Check(
            "offline",
            Status.OK,
            "offline mode: network calls are refused, research reads the lake only",
        )
    return Check("offline", Status.OK, "online: ingestion permitted")


def _check_determinism(settings: Settings) -> Check:
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != "0":
        return Check(
            "determinism",
            Status.WARN,
            f"PYTHONHASHSEED={hash_seed!r}",
            "Set PYTHONHASHSEED=0 for byte-identical results across runs "
            "(the Docker images already do).",
        )
    return Check("determinism", Status.OK, f"PYTHONHASHSEED=0, random_seed={settings.random_seed}")


def run_checks(settings: Settings | None = None) -> list[Check]:
    """Run every environment check. Never raises; failures are reported as data."""
    settings = settings or get_settings()
    return [
        Check("version", Status.OK, f"quantlab {__version__} (env={settings.env})"),
        _check_python(),
        _check_data_root(settings),
        _check_determinism(settings),
        _check_offline(settings),
        _check_user_agent(settings),
        _check_sources(settings),
    ]
