"""
Contract: Finviz US performance prefilter must never be stricter than local.

Purpose
-------
The Finviz prefilter only shrinks the batch early. On Perf Quarter / Perf Half
it must not reject names that the local post-filter would keep.

Invariant (per axis): ``finviz_exclusive_floor <= local_exclusive_floor``.

Usage
-----
    python -m pytest tests/contract/test_prefilter_not_stricter_than_local.py -q
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from universe_utils import (
    FINVIZ_PERF_HALF_LABEL,
    FINVIZ_PERF_QUARTER_LABEL,
    LOCAL_PERF_HALF_GT,
    LOCAL_PERF_QUARTER_GT,
    assert_prefilter_not_stricter_than_local,
    finviz_perf_exclusive_floor,
    update_universe_file,
    us_finviz_performance_filters,
)


def test_prefilter_floors_not_above_local() -> None:
    """Policy constants: Finviz exclusive floors must be <= local floors."""
    assert_prefilter_not_stricter_than_local()
    assert finviz_perf_exclusive_floor(FINVIZ_PERF_QUARTER_LABEL) <= LOCAL_PERF_QUARTER_GT
    assert finviz_perf_exclusive_floor(FINVIZ_PERF_HALF_LABEL) <= LOCAL_PERF_HALF_GT


def test_legacy_plus_percent_labels_are_stricter_than_up() -> None:
    """Sanity: +10%/+20% map to higher floors than Up (documents the model)."""
    assert finviz_perf_exclusive_floor("Quarter +10%") > finviz_perf_exclusive_floor("Quarter Up")
    assert finviz_perf_exclusive_floor("Half +20%") > finviz_perf_exclusive_floor("Half Up")


def test_unknown_finviz_label_fails_loudly() -> None:
    with pytest.raises(KeyError, match="Unknown Finviz Performance label"):
        finviz_perf_exclusive_floor("Quarter +99%")


def test_assert_helper_detects_stricter_prefilter(monkeypatch: pytest.MonkeyPatch) -> None:
    """If someone points Finviz at Half +20% while local stays >0, harness fails."""
    import universe_utils as uu

    monkeypatch.setattr(uu, "FINVIZ_PERF_HALF_LABEL", "Half +20%")
    with pytest.raises(AssertionError, match="Half prefilter"):
        uu.assert_prefilter_not_stricter_than_local()


def test_update_universe_file_uses_policy_ssot() -> None:
    """Source of ``update_universe_file`` must call the shared filter helper.

    Guards against re-introducing hardcoded ``Quarter +10%`` / ``Half +20%``
    literals that bypass the harness constants.
    """
    src = inspect.getsource(update_universe_file)
    assert "us_finviz_performance_filters" in src
    assert "finviz_screener_view_resilient" in src
    assert "LOCAL_PERF_QUARTER_GT" in src
    assert "LOCAL_PERF_HALF_GT" in src
    # Hardcoded legacy stricter labels must not reappear in the function body.
    assert "Quarter +10%" not in src
    assert "Half +20%" not in src


def test_us_finviz_performance_filters_match_module_labels() -> None:
    filters = us_finviz_performance_filters()
    assert filters["Performance"] == FINVIZ_PERF_QUARTER_LABEL
    assert filters["Performance 2"] == FINVIZ_PERF_HALF_LABEL


def test_universe_utils_module_has_no_stricter_perf_literals_outside_map() -> None:
    """AST scan: Performance string literals in update path stay within the map.

    Allows the known label map and policy constants; fails if a +N% label is
    assigned as an active Finviz filter outside ``_FINVIZ_PERF_EXCLUSIVE_FLOOR``.
    """
    path = Path(__file__).resolve().parents[2] / "universe_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_active = {"Quarter +10%", "Half +20%", "Half +10%", "Quarter +5%"}
    # Collect string constants that appear as dict values near Performance keys
    # by simply forbidding those literals anywhere except inside the floor map
    # assignment name ``_FINVIZ_PERF_EXCLUSIVE_FLOOR``.
    floor_map_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_FINVIZ_PERF_EXCLUSIVE_FLOOR":
                    floor_map_node = node
    assert floor_map_node is not None

    def _strings_under(node: ast.AST) -> set[str]:
        out: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                out.add(child.value)
        return out

    map_strings = _strings_under(floor_map_node)
    all_strings = _strings_under(tree)
    active = (all_strings - map_strings) & forbidden_active
    assert not active, (
        f"Stricter Finviz Perf labels used outside floor map: {sorted(active)}"
    )
