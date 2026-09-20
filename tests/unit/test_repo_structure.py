"""Structural invariants of the repository itself.

Rules that are cheap to enforce mechanically and expensive to notice by eye.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "quantlab"


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_every_package_directory_has_an_init() -> None:
    missing = [
        str(d.relative_to(SRC))
        for d in SRC.rglob("*")
        if d.is_dir()
        and d.name not in {"__pycache__", "templates"}
        and not (d / "__init__.py").exists()
    ]
    assert not missing, f"packages without __init__.py: {missing}"


def test_pandas_is_absent() -> None:
    """No pandas indexed joins where a time-misalignment bug could hide. Ruff
    enforces this on new code; this test enforces it on all code, and fails even
    if someone adds a `# noqa`."""
    offenders = [
        str(path.relative_to(SRC))
        for path in _python_files()
        if any(m == "pandas" or m.startswith("pandas.") for m in _imported_modules(path))
    ]
    assert not offenders, f"pandas imported: {offenders}"


def test_no_live_order_routing(repo_root: Path) -> None:
    """This is a data pipeline. A coarse guard, but it fails loudly the first time
    a broker SDK is added."""
    banned = {"ib_insync", "ibapi", "alpaca_trade_api", "alpaca", "ccxt", "oandapyV20", "tda"}
    for path in _python_files():
        found = banned & {m.split(".")[0] for m in _imported_modules(path)}
        assert not found, f"{path.relative_to(repo_root)} imports broker SDK(s) {found}"


def test_secrets_are_not_committed(repo_root: Path) -> None:
    assert not (repo_root / ".env").exists() or ".env" in (repo_root / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert (repo_root / ".env.example").exists()
