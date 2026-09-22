"""
Unit tests for ``core.indicators.naive_kelly`` (Validation Harness).

Purpose
-------
Lock the diagnostic ``NAIVE_KELLY_20D`` metric used by the momentum table /
optional sidebar sort: ``mean(last N daily returns) / var(last N, ddof=1)`` with
risk-free rate treated as 0 and no cross-asset covariance.

This is **not** an FMS production input. Default UI / ``momentum_now_and_delta``
sort remains FMS-descending; Kelly is available as an alternate rank key.

This is **not** an FMS production input. FMS remains ``alive_pullback`` over
SEG_* / residual features. Kelly is attached in ``momentum_now_and_delta``
after scoring for interpretability and ranking.

Covered behaviors
-----------------
- Closed-form mean/var contract on a synthetic return path
- Zero variance (flat prices) → NaN (no div-by-zero)
- Short history (< window returns) → NaN
- Column-native as-of: trailing NaN after last valid must not extend the window

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_naive_kelly.py -q
    python -m pytest tests/unit/test_naive_kelly.py tests/unit/test_fms_scoring.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.indicators import naive_kelly


def _prices_from_returns(rets: list[float], start: float = 100.0) -> pd.Series:
    """Rebuild a price path from simple daily returns."""
    prices = [start]
    for r in rets:
        prices.append(prices[-1] * (1.0 + r))
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=idx, dtype=float)


def test_naive_kelly_matches_mean_over_var() -> None:
    """NAIVE_KELLY_20D = mean(last 20 daily rets) / var(last 20, ddof=1)."""
    rets = [0.01 * ((-1) ** i) + 0.002 for i in range(25)]
    prices = _prices_from_returns(rets)
    df = pd.DataFrame({"A": prices})
    got = naive_kelly(df, window=20)

    tail = pd.Series(rets[-20:], dtype=float)
    expected = float(tail.mean() / tail.var(ddof=1))
    assert got["A"] == pytest.approx(expected)


def test_naive_kelly_zero_variance_is_nan() -> None:
    """Flat prices → zero variance → NaN (no division by zero)."""
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    df = pd.DataFrame({"FLAT": np.full(30, 100.0)}, index=idx)
    got = naive_kelly(df, window=20)
    assert pd.isna(got["FLAT"])


def test_naive_kelly_short_history_is_nan() -> None:
    """Fewer than ``window`` daily returns → NaN."""
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({"SHORT": np.linspace(100.0, 110.0, 10)}, index=idx)
    got = naive_kelly(df, window=20)
    assert pd.isna(got["SHORT"])


def test_naive_kelly_uses_column_native_asof() -> None:
    """Trailing NaN after last valid must not extend the Kelly window."""
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    live = np.linspace(100.0, 140.0, 30)
    dead = np.full(10, np.nan)
    df = pd.DataFrame({"A": np.concatenate([live, dead])}, index=idx)

    got = naive_kelly(df, window=20)
    truncated = pd.DataFrame({"A": live}, index=idx[:30])
    expected = naive_kelly(truncated, window=20)
    assert got["A"] == pytest.approx(expected["A"])
