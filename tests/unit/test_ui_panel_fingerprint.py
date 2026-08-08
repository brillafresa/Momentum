# -*- coding: utf-8 -*-
"""Unit tests for watchlist panel fingerprints (UI session memo keys).

Purpose
-------
Guard the session-memo fingerprint used to skip FMS recomputation when the
watchlist KRW/OHLC panel has not changed (detail-view navigation).

Run
---
    python -m pytest tests/unit/test_ui_panel_fingerprint.py -q
"""

from __future__ import annotations

import pandas as pd

from adapters.ui_data_bundle import (
    fingerprint_ohlc_panel,
    fingerprint_price_panel,
    watchlist_bundle_key,
)


def _panel(cols, end="2024-06-10", periods=5):
    idx = pd.bdate_range(end=end, periods=periods)
    data = {c: range(1, periods + 1) for c in cols}
    return pd.DataFrame(data, index=idx)


def test_fingerprint_identical_for_same_panel():
    df = _panel(["A", "B"])
    assert fingerprint_price_panel(df) == fingerprint_price_panel(df.copy())


def test_fingerprint_changes_when_last_valid_moves():
    df = _panel(["A"], end="2024-06-10")
    fp1 = fingerprint_price_panel(df)
    df2 = _panel(["A"], end="2024-06-11")
    fp2 = fingerprint_price_panel(df2)
    assert fp1 != fp2


def test_fingerprint_changes_when_column_added():
    fp1 = fingerprint_price_panel(_panel(["A"]))
    fp2 = fingerprint_price_panel(_panel(["A", "B"]))
    assert fp1 != fp2


def test_fingerprint_empty_safe():
    assert fingerprint_price_panel(pd.DataFrame()) == ("empty",)
    assert fingerprint_ohlc_panel(None) == ("ohlc_empty",)
    assert fingerprint_ohlc_panel(pd.DataFrame()) == ("ohlc_empty",)


def test_watchlist_bundle_key_includes_mode_and_period():
    df = _panel(["AAA"])
    k1 = watchlist_bundle_key("FREE", ["AAA"], "1y", df, None)
    k2 = watchlist_bundle_key("IRP", ["AAA"], "1y", df, None)
    k3 = watchlist_bundle_key("FREE", ["AAA"], "2y", df, None)
    assert k1 != k2
    assert k1 != k3
    assert k1 == watchlist_bundle_key("FREE", ["AAA"], "1y", df.copy(), None)
