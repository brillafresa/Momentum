# -*- coding: utf-8 -*-
"""Unit tests for DetailViewAtom identity ↔ series binding.

Purpose
-------
Fail-closed contract: detail-view ticker label and chart series must stay
bound in one atom so index/session drift cannot show the wrong chart.

Run
---
    python -m pytest tests/unit/test_detail_view_atom.py -q
"""

from __future__ import annotations

import pandas as pd

from adapters.ui_data_bundle import (
    DetailViewAtom,
    build_detail_view_atom,
    reconcile_detail_selection,
)


def _prices():
    idx = pd.bdate_range("2024-01-01", periods=10)
    return pd.DataFrame(
        {
            "AAA": range(10, 20),
            "BBB": range(20, 30),
        },
        index=idx,
    )


def _mom():
    return pd.DataFrame(
        {
            "FMS": [1.0, 2.0],
            "R_1M": [0.1, -0.1],
            "AboveEMA50": [1.0, 0.0],
            "R_3M": [0.2, 0.0],
            "ΔFMS_1D": [0.1, 0.0],
            "ΔFMS_5D": [0.2, -0.1],
        },
        index=["AAA", "BBB"],
    )


def test_build_atom_binds_symbol_series_and_name():
    prices = _prices()
    atom = build_detail_view_atom(
        "AAA",
        prices,
        name_map={"AAA": "Alpha Co"},
        mom=_mom(),
        period_key="3M",
    )
    assert atom is not None
    assert atom.symbol == "AAA"
    assert atom.display_name == "Alpha Co"
    assert atom.series_krw.name == "AAA"
    assert atom.series_krw.equals(prices["AAA"])
    assert atom.is_consistent()
    assert atom.fms_row is not None
    assert atom.fms_row.name == "AAA"
    assert atom.period_key == "3M"


def test_build_atom_missing_symbol_returns_none():
    assert build_detail_view_atom("ZZZ", _prices()) is None


def test_build_atom_never_substitutes_other_column():
    prices = _prices()
    atom = build_detail_view_atom("ZZZ", prices)
    assert atom is None
    # Existing columns unchanged / not returned under wrong identity
    other = build_detail_view_atom("BBB", prices)
    assert other is not None
    assert other.symbol == "BBB"
    assert other.series_krw.name == "BBB"
    assert list(other.series_krw.values) == list(prices["BBB"].values)


def test_inconsistent_atom_detected():
    prices = _prices()
    series = prices["AAA"].copy()
    series.name = "WRONG"
    bad = DetailViewAtom(
        symbol="AAA",
        display_name="AAA",
        series_krw=series,
        period_key="6M",
        fingerprint=("x",),
        fms_row=None,
    )
    assert not bad.is_consistent()


def test_fms_row_mismatch_fails_consistency():
    prices = _prices()
    series = prices["AAA"].copy()
    series.name = "AAA"
    row = _mom().loc["BBB"].copy()
    row.name = "BBB"
    bad = DetailViewAtom(
        symbol="AAA",
        display_name="AAA",
        series_krw=series,
        period_key="6M",
        fingerprint=("x",),
        fms_row=row,
    )
    assert not bad.is_consistent()


def test_reconcile_keeps_valid_symbol():
    sym, idx = reconcile_detail_selection(["A", "B", "C"], "B", "A")
    assert sym == "B"
    assert idx == 1


def test_reconcile_falls_back_when_removed():
    sym, idx = reconcile_detail_selection(["A", "C"], "B", "A")
    assert sym == "A"
    assert idx == 0


def test_reconcile_empty_options():
    sym, idx = reconcile_detail_selection([], "B", "DEFAULT")
    assert sym == "DEFAULT"
    assert idx == 0
