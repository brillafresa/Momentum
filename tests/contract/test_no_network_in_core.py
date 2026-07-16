"""Contract: core package must not import network / UI stacks."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN = frozenset(
    {
        "yfinance",
        "finvizfinance",
        "requests",
        "streamlit",
        "urllib",
        "httpx",
        "aiohttp",
    }
)


def _iter_python_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*.py"):
        if path.name == "__init__.py" and path.read_text(encoding="utf-8").strip() == "":
            continue
        yield path


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_package_has_no_forbidden_network_imports() -> None:
    """Forbid live-API imports under core/ at the AST level.

    ``core/__init__.py`` must stay free of network stacks and must not
    re-export ``analysis_utils`` (which imports yfinance). Scoring remains
    in ``analysis_utils`` until modules are migrated into ``core/``.
    """
    core_root = Path(__file__).resolve().parents[2] / "core"
    offenders = []
    for path in _iter_python_files(core_root):
        imported = _imported_modules(path)
        bad = imported & FORBIDDEN
        if bad:
            offenders.append(f"{path.relative_to(core_root.parent)}: {sorted(bad)}")
    assert offenders == [], "core/ must not import network stacks:\n" + "\n".join(offenders)
