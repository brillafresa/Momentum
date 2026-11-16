"""
Unit tests for detail-view MED3 overlay helper (Validation Harness).

Purpose
-------
Lock the UI-only 3-day close median used by the detail Rebased chart.
Production FMS and tradeability SMA5 must remain untouched by this helper.

Covered behaviors
-----------------
- ``rolling_close_median(window=3)`` matches ``Series.rolling(3).median()``
- First ``window - 1`` bars are NaN (``min_periods=window``)
- Odd-window median is the center value of the window
- Helper lives in ``adapters.ui_data_bundle`` (no Streamlit / network)

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_detail_chart_med3.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adapters.ui_data_bundle import rolling_close_median


def test_med3_matches_rolling_median() -> None:
    """MED3 must be the 3-bar rolling median of the input series."""
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    s = pd.Series([10.0, 30.0, 20.0, 40.0, 5.0, 15.0], index=idx, name="AAA")
    got = rolling_close_median(s, window=3)
    expected = s.rolling(window=3, min_periods=3).median()
    pd.testing.assert_series_equal(got, expected)


def test_med3_leading_bars_are_nan() -> None:
    """With min_periods=window, the first two MED3 values are NaN."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)
    got = rolling_close_median(s, window=3)
    assert np.isnan(got.iloc[0])
    assert np.isnan(got.iloc[1])
    assert got.iloc[2] == pytest.approx(2.0)  # median(1,2,3)
    assert got.iloc[3] == pytest.approx(3.0)  # median(2,3,4)


def test_med3_is_not_sma5_mean() -> None:
    """Regression guard: MED3 must not silently become a 5-day mean."""
    s = pd.Series([1.0, 100.0, 2.0, 3.0, 4.0], dtype=float)
    med3 = rolling_close_median(s, window=3)
    sma5 = s.rolling(window=5, min_periods=5).mean()
    assert med3.iloc[2] == pytest.approx(2.0)  # median(1,100,2)
    assert np.isnan(sma5.iloc[2])
    assert sma5.iloc[4] == pytest.approx(s.mean())
