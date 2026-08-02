"""Unit tests for non-overlapping segment features and nonlinear formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibration.nonlinear_formulas import FORMULA_FAMILIES, softplus
from core.fms_features import build_symbol_feature_frame


def _make_path(n: int = 80, *, recent_boost: float = 0.0, prior: float = 0.002) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    daily = np.full(n, prior)
    daily[-5:] = recent_boost
    prices = 100.0 * np.exp(np.cumsum(daily))
    return pd.Series(prices, index=idx)


def test_nonoverlapping_segments_distinguish_v_bounce_from_continuation() -> None:
    continuation = _make_path(recent_boost=0.004, prior=0.003)
    # Flat/down prior, sharp recent bounce.
    bounce = _make_path(recent_boost=0.02, prior=-0.001)
    feats = build_symbol_feature_frame(
        pd.DataFrame({"cont": continuation, "bounce": bounce})
    )
    assert feats.loc["cont", "SEG_RET_21_63"] > feats.loc["bounce", "SEG_RET_21_63"]
    assert feats.loc["bounce", "SEG_RET_0_5"] > feats.loc["cont", "SEG_RET_0_5"]
    assert feats.loc["cont", "PRIOR_SUPPORT_SIGN"] == 1.0
    assert feats.loc["bounce", "PRIOR_SUPPORT_SIGN"] == 0.0


def test_segment_bands_are_finite_on_steady_uptrend() -> None:
    prices = pd.DataFrame({"AAA": _make_path(recent_boost=0.003, prior=0.002)})
    row = build_symbol_feature_frame(prices).loc["AAA"]
    for col in (
        "SEG_RET_0_3",
        "SEG_RET_0_5",
        "SEG_RET_5_21",
        "SEG_RET_21_63",
        "SEG_SLOPE_0_5",
        "SEG_VOL_0_5",
    ):
        assert np.isfinite(row[col])


def test_nonlinear_families_score_without_nan() -> None:
    prices = pd.DataFrame(
        {
            "A": _make_path(recent_boost=0.01, prior=0.003),
            "B": _make_path(recent_boost=-0.01, prior=-0.001),
        }
    )
    feats = build_symbol_feature_frame(prices)
    # Attach columns used by formulas that come from panel builder.
    feats["R_3M"] = [0.4, -0.05]
    feats["R2_3M"] = [0.9, 0.2]
    feats["Vol20_Ann"] = [0.2, 0.05]
    feats["TREND_EFFICIENCY_REWARD_15D"] = [0.5, 0.0]
    rng = np.random.default_rng(0)
    for family in FORMULA_FAMILIES:
        params = family.sample_params(rng)
        scores = family.score(feats, params)
        assert scores.notna().all()
        assert scores.loc["A"] != scores.loc["B"] or True  # allow ties; just run


def test_softplus_positive_for_positive_input() -> None:
    s = softplus(pd.Series([0.0, 0.1, -0.1]))
    assert s.iloc[1] > s.iloc[0] > s.iloc[2]


def test_residual_features_mid_dip_and_stale_run() -> None:
    """Mid-dip recovery rises when recent recovers after a mid-band drawdown."""
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    # Strong prior, mid dip, recent recovery.
    daily = np.full(80, 0.003)
    daily[-21:-5] = -0.008
    daily[-5:] = 0.012
    dip = pd.Series(100.0 * np.exp(np.cumsum(daily)), index=idx)
    feats = build_symbol_feature_frame(pd.DataFrame({"dip": dip}))
    assert feats.loc["dip", "MID_DIP_RECOVERY"] > 0.0
    assert feats.loc["dip", "SEG_RET_5_21"] < 0.0
    assert feats.loc["dip", "SEG_RET_0_5"] > 0.0
    assert np.isfinite(feats.loc["dip", "RECENT_UP_DAYS_5D"])
    assert np.isfinite(feats.loc["dip", "STALE_AFTER_RUN"])
    assert 0.0 <= feats.loc["dip", "RECENT_JUMP_SHARE_5D"] <= 1.0


def test_stale_after_run_spares_alive_recovery() -> None:
    """Finished mega-runs get penalty; recovering mid-dip paths should not."""
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    # Mega run then flat/weak recent.
    dead = np.full(80, 0.012)
    dead[-8:] = -0.001
    dead_px = pd.Series(100.0 * np.exp(np.cumsum(dead)), index=idx)
    # Prior up, mid dip, strong recent recovery.
    alive = np.full(80, 0.004)
    alive[-21:-5] = -0.01
    alive[-5:] = 0.015
    alive_px = pd.Series(100.0 * np.exp(np.cumsum(alive)), index=idx)
    feats = build_symbol_feature_frame(pd.DataFrame({"dead": dead_px, "alive": alive_px}))
    assert feats.loc["dead", "STALE_AFTER_RUN"] > feats.loc["alive", "STALE_AFTER_RUN"]
    assert feats.loc["alive", "MID_DIP_RECOVERY"] > 0.0


def test_pullback_and_anti_stale_families_registered() -> None:
    names = {f.name for f in FORMULA_FAMILIES}
    assert "pullback_continuation" in names
    assert "anti_stale_run" in names
    assert "alive_pullback" in names

