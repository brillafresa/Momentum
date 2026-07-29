"""Tests for offline short-horizon candidate feature semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibration.short_horizon_features import compute_short_horizon_candidates


def _frame(**series: np.ndarray) -> pd.DataFrame:
    """Build a deterministic business-day price frame."""
    length = len(next(iter(series.values())))
    return pd.DataFrame(series, index=pd.bdate_range("2026-01-01", periods=length))


def test_steady_uptrend_has_clean_positive_short_trend() -> None:
    prices = _frame(STEADY=100.0 * np.exp(0.01 * np.arange(80)))

    row = compute_short_horizon_candidates(prices).loc["STEADY"]

    assert row["LOG_SLOPE_15D"] == pytest.approx(0.01)
    assert row["TREND_R2_15D"] == pytest.approx(1.0)
    assert row["TREND_EFFICIENCY_15D"] == pytest.approx(1.0)
    assert row["DOWNSIDE_RMS_20D"] == pytest.approx(0.0)
    assert row["UNDER_EMA20_STREAK"] == 0


def test_recent_three_day_recovery_beats_prior_decline() -> None:
    log_returns = np.r_[np.full(69, 0.002), np.full(7, -0.01), np.full(3, 0.02)]
    prices = _frame(RECOVERY=100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)]))

    row = compute_short_horizon_candidates(prices).loc["RECOVERY"]

    assert row["R_3D_LOG"] > 0
    assert row["RECOVERY_3D_VS_PRIOR7"] > 0
    assert row["EMA20_ACCEL_3D_VS_10D"] > 0


def test_unrecovered_break_records_ema_streak_and_downside_risk() -> None:
    smooth = 100.0 * np.exp(0.003 * np.arange(80))
    broken = smooth.copy()
    broken[-6:] *= np.exp(np.linspace(0.0, -0.30, 6))
    prices = _frame(SMOOTH=smooth, BROKEN=broken)

    features = compute_short_horizon_candidates(prices)

    assert features.loc["BROKEN", "UNDER_EMA20_STREAK"] >= 3
    assert features.loc["BROKEN", "PRICE_VS_EMA20"] < 0
    assert (
        features.loc["BROKEN", "DOWNSIDE_RMS_20D"]
        > features.loc["SMOOTH", "DOWNSIDE_RMS_20D"]
    )


def test_post_spike_stall_exceeds_continuing_trend() -> None:
    continuing = 100.0 * np.exp(0.004 * np.arange(80))
    stalled = np.full(80, 100.0)
    stalled[30:33] = [110.0, 130.0, 150.0]
    stalled[33:43] = 150.0
    stalled[43:53] = np.linspace(145.0, 100.0, 10)
    prices = _frame(CONTINUING=continuing, STALLED=stalled)

    features = compute_short_horizon_candidates(prices)

    assert features.loc["STALLED", "POST_SPIKE_STALL_3M"] > 0
    assert features.loc["CONTINUING", "POST_SPIKE_STALL_3M"] == pytest.approx(0.0)


def test_stale_age_penalizes_spike_then_long_flat() -> None:
    continuing = 100.0 * np.exp(0.004 * np.arange(80))
    stale = 100.0 * np.exp(0.001 * np.arange(80))
    stale[59] = 130.0
    stale[60:77] = 105.0
    stale[77:] = 106.0
    prices = _frame(CONTINUING=continuing, STALE=stale)

    features = compute_short_horizon_candidates(prices)

    assert features.loc["STALE", "STALE_AGE"] > features.loc["CONTINUING", "STALE_AGE"]
    assert features.loc["CONTINUING", "STALE_AGE"] == pytest.approx(0.0, abs=1e-6)
