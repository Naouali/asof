"""Structural invariants of the repository itself.

These are the rules from spec sections 2 and 13 that are cheap to enforce
mechanically and expensive to notice by eye once the codebase is large.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "quantlab"

# Layer order, lowest first. A module may import from its own layer or below.
LAYERS: tuple[tuple[str, ...], ...] = (
    ("data", "universe"),
    ("costs", "signals"),
    ("portfolio", "risk"),
    ("backtest", "validation"),
    ("reporting", "dashboard"),
)


def _layer_of(package: str) -> int | None:
    for index, names in enumerate(LAYERS):
        if package in names:
            return index
    return None


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


@pytest.mark.parametrize("package", ["signals", "backtest", "costs", "portfolio"])
def test_pandas_is_absent_from_the_signal_path(package: str) -> None:
    """Spec section 13: no pandas indexed joins where a time-misalignment bug could
    hide. Ruff enforces this on new code; this test enforces it on all code, and
    fails even if someone adds a `# noqa`."""
    offenders = [
        str(path.relative_to(SRC))
        for path in (SRC / package).rglob("*.py")
        if any(m == "pandas" or m.startswith("pandas.") for m in _imported_modules(path))
    ]
    assert not offenders, f"pandas imported in the signal path: {offenders}"


def test_layers_depend_downward_only() -> None:
    """Spec section 2: layers only depend downward."""
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC)
        if not relative.parts or len(relative.parts) == 1:
            continue
        own_layer = _layer_of(relative.parts[0])
        if own_layer is None:
            continue
        for module in _imported_modules(path):
            if not module.startswith("quantlab."):
                continue
            parts = module.split(".")
            if len(parts) < 2:
                continue
            other_layer = _layer_of(parts[1])
            if other_layer is not None and other_layer > own_layer:
                violations.append(f"{relative} imports {module} (upward dependency)")
    assert not violations, "\n".join(violations)


def test_no_live_order_routing(repo_root: Path) -> None:
    """Spec section 1 and 13: v1 is research and paper trading only. This is a
    coarse guard, but it fails loudly the first time a broker SDK is added."""
    banned = {"ib_insync", "ibapi", "alpaca_trade_api", "alpaca", "ccxt", "oandapyV20", "tda"}
    for path in _python_files():
        found = banned & {m.split(".")[0] for m in _imported_modules(path)}
        assert not found, f"{path.relative_to(repo_root)} imports broker SDK(s) {found}"


def test_secrets_are_not_committed(repo_root: Path) -> None:
    assert not (repo_root / ".env").exists() or ".env" in (repo_root / ".gitignore").read_text(
        encoding="utf-8"
    )
    assert (repo_root / ".env.example").exists()
