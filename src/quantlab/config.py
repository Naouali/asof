"""Runtime configuration.

Loaded from environment variables (and `.env` outside containers), with a
`QUANTLAB_` prefix. The stack is required to start and run with **zero API keys**:
every key is optional here, and the sources that need one are
reported as unavailable by `quantlab doctor` rather than failing at import time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quantlab.paths import Layout

__all__ = ["Settings", "get_layout", "get_settings"]

LogFormat = Literal["console", "json"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def _default_repo_root() -> Path:
    # src/quantlab/config.py -> src/quantlab -> src -> repo root
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process-wide settings. Construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="QUANTLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -- environment ------------------------------------------------------------
    env: Literal["local", "docker", "ci"] = "local"
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "console"

    # -- paths ------------------------------------------------------------------
    data_root: Path = Field(default=Path("./data"))
    repo_root: Path = Field(default_factory=_default_repo_root)

    # -- offline mode -----------------------------------------------------------
    # Consumers of the lake must be able to run with no internet access after the
    # initial ingest. When true, any source attempting a network call raises instead of hanging.
    offline: bool = False

    # -- ingest politeness ------------------------------------------------------
    # SEC requires a descriptive User-Agent with contact details; requests without
    # one are rejected. This default is deliberately obviously-wrong so
    # that an unconfigured install is caught by `doctor` rather than by the SEC.
    http_user_agent: str = "QuantLab/0.1 (unconfigured -- set QUANTLAB_HTTP_USER_AGENT)"
    http_timeout_seconds: float = 30.0
    http_max_retries: int = 5

    # -- API keys (every one optional; see quantlab.data.catalogue) --------------
    fred_api_key: SecretStr | None = None
    eia_api_key: SecretStr | None = None
    usda_nass_api_key: SecretStr | None = None
    nasdaq_data_link_api_key: SecretStr | None = None
    alpha_vantage_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    fmp_api_key: SecretStr | None = None

    @field_validator("data_root", "repo_root")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @property
    def layout(self) -> Layout:
        return Layout(data_root=self.data_root, repo_root=self.repo_root)

    def secret_for(self, name: str) -> str | None:
        """Return the configured value of an API key setting, or None if unset.

        Used by the source catalogue to report availability without each source
        module reaching into settings by hand.
        """
        value = getattr(self, name, None)
        if value is None:
            return None
        if isinstance(value, SecretStr):
            secret = value.get_secret_value()
            return secret or None
        return str(value) or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def get_layout() -> Layout:
    """Resolved filesystem layout for the current settings."""
    return get_settings().layout
