"""
Unit tests for core price indicators (Validation Harness).

Purpose
-------
Lock ``core.indicators`` (``ema``, ``returns_pct``, ``r_squared_3m``) as the
pure, offline source of truth — no network I/O.

Covered behaviors
-----------------
- ``ema``: exponential smoother matches ``ewm(adjust=False)``
- ``returns_pct``: n-period pct change on last row; short series → empty floats
- ``r_squared_3m``: trend panel ranks by smoothness; short series → NaN;
  non-positive price glitches do not emit ``log`` RuntimeWarnings
- ``analysis_utils`` re-export shim stays identity-equal to ``core.indicators``

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_indicators.py -q
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from core.indicators import ema, r_squared_3m, returns_pct


def test_ema_matches_ewm_adjust_false() -> None:
    """EMA must be the standard ewm(span, adjust=False) mean."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    expected = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(ema(s, 3), expected)


def test_returns_pct_last_n_period_change() -> None:
    """Last-row n-period return: (last / lag_n) - 1."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0, 140.0]}, index=idx)
    # n=2: 140/120 - 1 = 1/6
    got = returns_pct(df, 2)
    assert got["A"] == pytest.approx(140.0 / 120.0 - 1.0)


def test_returns_pct_short_series_returns_empty_floats() -> None:
    """When rows <= n, return a float Series indexed by columns (all NaN-like empty)."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=idx)
    got = returns_pct(df, 5)
    assert list(got.index) == ["A", "B"]
    assert got.dtype == float
    assert got.isna().all()


def test_r_squared_3m_trend_outranks_flat(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Smooth uptrend should score higher R² than a flat series on the same panel."""
    r2 = r_squared_3m(synthetic_prices_krw)
    assert r2["TREND_UP"] > r2["FLAT"]
    assert r2["TREND_UP"] > r2["CRASHY"]
    assert 0.0 <= r2["TREND_UP"] <= 1.0


def test_r_squared_3m_short_history_is_nan() -> None:
    """Fewer than 63 bars → NaN (insufficient window)."""
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    df = pd.DataFrame({"SHORT": np.linspace(100.0, 120.0, 20)}, index=idx)
    r2 = r_squared_3m(df)
    assert pd.isna(r2["SHORT"])


def test_r_squared_3m_non_positive_glitch_emits_no_log_warnings(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Zero/negative Adj Close ticks must not spam np.log RuntimeWarnings."""
    glitch = synthetic_prices_krw.copy()
    col = glitch.columns[0]
    glitch.iloc[-30, glitch.columns.get_loc(col)] = -5.0
    glitch.iloc[-40, glitch.columns.get_loc(col)] = 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = r_squared_3m(glitch)

    log_warnings = [w for w in caught if "in log" in str(w.message)]
    assert log_warnings == [], [str(w.message) for w in log_warnings]
    assert out.name == "R2_3M"
    assert out.notna().any()


def test_analysis_utils_reexports_core_indicators() -> None:
    """Transitional facade must expose the same callables as core."""
    import analysis_utils as au
    import core.indicators as ci

    assert au.ema is ci.ema
    assert au.returns_pct is ci.returns_pct
    assert au.r_squared_3m is ci.r_squared_3m
