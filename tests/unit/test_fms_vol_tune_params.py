"""
Contract: vol-penalty tune helper delegates to core FMS params path.
"""

from __future__ import annotations

import pandas as pd

from core.fms import compute_fms_snapshot, score_fms_from_feature_frame
from core.indicators import returns_pct
from fms_recalib_tune_vol_penalty import fms_score_with_vol_params


def _feature_frame_from_prices(prices_krw: pd.DataFrame) -> pd.DataFrame:
    snap = compute_fms_snapshot(
        prices_krw,
        reference_prices_krw=prices_krw,
        ohlc_data=None,
        symbols=list(prices_krw.columns),
    )
    feat = snap.drop(columns=["FMS", "Filter_Status"], errors="ignore").copy()
    feat["R_4M"] = returns_pct(prices_krw, 84)
    feat = feat.rename(columns={"Vol20(ann)": "Vol20_Ann"})
    return feat


def test_vol_tune_default_like_point_matches_core_override(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    feat = _feature_frame_from_prices(synthetic_prices_krw)
    got = fms_score_with_vol_params(feat, q_pct=70.0, hard_power=1.5, hard_scale=1.0)
    expected = score_fms_from_feature_frame(
        feat,
        params={"vol_q_pct": 70.0, "vol_hard_power": 1.5, "vol_hard_scale": 1.0},
    )
    pd.testing.assert_series_equal(
        got.sort_index(),
        expected.sort_index(),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_vol_tune_custom_point_matches_core_override(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    feat = _feature_frame_from_prices(synthetic_prices_krw)
    got = fms_score_with_vol_params(feat, q_pct=55.0, hard_power=2.25, hard_scale=0.5)
    expected = score_fms_from_feature_frame(
        feat,
        params={"vol_q_pct": 55.0, "vol_hard_power": 2.25, "vol_hard_scale": 0.5},
    )
    pd.testing.assert_series_equal(
        got.sort_index(),
        expected.sort_index(),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )

