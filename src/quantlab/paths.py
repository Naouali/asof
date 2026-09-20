"""Filesystem layout.

Paths are resolved once, from configuration, so that the same code runs unchanged
inside a container (where the lake is a named volume at /data) and on a developer
laptop (where it is ./data).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["Layout"]


class Layout:
    """Resolved directory layout for one QuantLab installation."""

    def __init__(self, data_root: Path, repo_root: Path) -> None:
        self.data_root = data_root
        self.repo_root = repo_root

    # -- the data lake ----------------------------------------------------------
    @property
    def lake(self) -> Path:
        """Parquet lake root. Partitioned source/dataset/asset_class/year."""
        return self.data_root / "lake"

    @property
    def state(self) -> Path:
        """Process state: heartbeats, ingest cursors, lock files."""
        return self.data_root / "state"

    # -- the repo ---------------------------------------------------------------
    @property
    def configs(self) -> Path:
        return self.repo_root / "configs"

    def all_data_dirs(self) -> tuple[Path, ...]:
        return (self.lake, self.state)

    def ensure(self) -> None:
        """Create every data directory. Safe to call repeatedly."""
        for directory in self.all_data_dirs():
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Layout(data_root={self.data_root!s}, repo_root={self.repo_root!s})"
