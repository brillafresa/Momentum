"""
Contract: recalib feature→score path must match production FMS.

Purpose
-------
``score_fms_from_feature_frame`` (used by recalib ``f_current``) must reproduce
``compute_fms_snapshot(..., reference_prices_krw=prices)`` on the same panel.

Usage
-----
    python -m pytest tests/unit/test_fms_recalib_parity.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.fms import compute_fms_snapshot, score_fms_from_feature_frame
from core.indicators import returns_pct
from fms_recalib_evaluate_formulas import f_current, f_proposed


def _feature_frame_from_prices(prices_krw: pd.DataFrame) -> pd.DataFrame:
    """Build a recalib-style feature table from a production snapshot."""
    snap = compute_fms_snapshot(
        prices_krw,
        reference_prices_krw=prices_krw,
        ohlc_data=None,
        symbols=list(prices_krw.columns),
    )
    feat = snap.drop(columns=["FMS", "Filter_Status"], errors="ignore").copy()
    feat["R_6M"] = returns_pct(prices_krw, 126)
    feat = feat.rename(columns={"Vol20(ann)": "Vol20_Ann"})
    return feat, snap["FMS"]


def test_score_fms_from_feature_frame_matches_production_snapshot(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Self-referenced snapshot FMS == feature-frame scorer (SSOT)."""
    feat, expected = _feature_frame_from_prices(synthetic_prices_krw)
    got = score_fms_from_feature_frame(feat)
    pd.testing.assert_series_equal(
        got.sort_index(),
        expected.sort_index(),
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_f_current_and_f_proposed_delegate_to_core(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Recalib entrypoints must not carry an independent formula fork."""
    feat, expected = _feature_frame_from_prices(synthetic_prices_krw)
    pd.testing.assert_series_equal(
        f_current(feat).sort_index(),
        expected.sort_index(),
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )
    pd.testing.assert_series_equal(
        f_proposed(feat).sort_index(),
        f_current(feat).sort_index(),
        check_names=False,
    )


def test_score_fms_from_feature_frame_requires_core_columns() -> None:
    """Missing required feature columns must fail loudly."""
    with pytest.raises(KeyError, match="missing columns"):
        score_fms_from_feature_frame(pd.DataFrame({"R_1M": [0.1]}))
