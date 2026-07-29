"""Unit tests for core.fms_features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.fms_features import build_symbol_feature_frame, candidate_feature_columns


def _make_uptrend(n: int = 80, daily: float = 0.003) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 100.0 * np.exp(np.cumsum(np.full(n, daily)))
    return pd.Series(prices, index=idx)


def test_steady_uptrend_has_positive_short_horizon_features() -> None:
    prices = pd.DataFrame({"AAA": _make_uptrend()})
    feats = build_symbol_feature_frame(prices)
    row = feats.loc["AAA"]
    assert row["LOG_SLOPE_15D"] > 0
    assert row["TREND_EFFICIENCY_15D"] > 0.5
    assert row["TREND_R2_15D"] > 0.8


def test_spike_then_flat_has_stale_age() -> None:
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    values = np.full(n, 100.0)
    values[40:43] = [110.0, 112.0, 111.0]
    # Drift up slightly after the spike so max_gain in the lookback is visible.
    values[43:] = np.linspace(111.0, 112.5, n - 43)
    prices = pd.DataFrame({"BBB": values}, index=idx)
    feats = build_symbol_feature_frame(prices)
    assert feats.loc["BBB", "STALE_AGE"] >= 0


def test_discontinuous_steps_exceed_steady_uptrend_jumpiness() -> None:
    steady = _make_uptrend()
    stepped = steady.copy()
    values = np.full(len(stepped), 100.0)
    values[20:] *= 1.12
    values[40:] *= 1.12
    values[55:] *= 1.12
    values *= np.exp(np.sin(np.arange(len(values))) * 0.015)
    stepped[:] = values
    feats = build_symbol_feature_frame(
        pd.DataFrame({"steady": steady, "stepped": stepped})
    )
    assert (
        feats.loc["stepped", "JUMP_DISCONTINUITY_3M"]
        > feats.loc["steady", "JUMP_DISCONTINUITY_3M"]
    )


def test_recent_decline_breaks_prior_21d_trend() -> None:
    healthy = _make_uptrend()
    broken = healthy.copy()
    broken.iloc[-3:] *= np.array([0.98, 0.95, 0.92])
    feats = build_symbol_feature_frame(
        pd.DataFrame({"healthy": healthy, "broken": broken})
    )
    assert (
        feats.loc["broken", "RECENT_3D_VS_21D_TREND"]
        < feats.loc["healthy", "RECENT_3D_VS_21D_TREND"]
    )


def test_candidate_columns_exclude_r4m() -> None:
    frame = pd.DataFrame(
        {
            "R_3M": [0.1],
            "R_4M": [0.2],
            "TREND_EFFICIENCY_15D": [0.5],
            "rank": [1],
        },
        index=["X"],
    )
    cols = candidate_feature_columns(frame)
    assert "R_4M" not in cols
    assert "TREND_EFFICIENCY_15D" in cols
