"""
Contract: parameterized FMS scoring stays SSOT with production defaults.

Purpose
-------
``score_fms_from_feature_frame(..., params=...)`` must:
- match default / production scores when ``params`` is omitted or production
- accept Mapping overrides for offline Monte-Carlo search (tune scripts)
- keep tune ``fms_score`` as a thin core delegate (no formula fork)

Usage
-----
    python -m pytest tests/unit/test_fms_params.py -q
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from core.fms import (
    FmsScoreParams,
    compute_fms_snapshot,
    production_fms_score_params,
    score_fms_from_feature_frame,
)
from core.indicators import returns_pct
from fms_recalib_tune_weights_and_transitions import baseline_params, fms_score


def _feature_frame_from_prices(prices_krw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    snap = compute_fms_snapshot(
        prices_krw,
        reference_prices_krw=prices_krw,
        ohlc_data=None,
        symbols=list(prices_krw.columns),
    )
    feat = snap.drop(columns=["FMS", "Filter_Status"], errors="ignore").copy()
    feat["R_4M"] = returns_pct(prices_krw, 84)
    feat = feat.rename(columns={"Vol20(ann)": "Vol20_Ann"})
    return feat, snap["FMS"]


def test_production_params_match_default_scoring(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Explicit production params must equal the no-params path."""
    feat, _ = _feature_frame_from_prices(synthetic_prices_krw)
    default = score_fms_from_feature_frame(feat)
    explicit = score_fms_from_feature_frame(feat, params=production_fms_score_params())
    pd.testing.assert_series_equal(
        default.sort_index(),
        explicit.sort_index(),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_production_params_match_snapshot_fms(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Parameterized feature scorer remains parity with snapshot FMS."""
    feat, expected = _feature_frame_from_prices(synthetic_prices_krw)
    got = score_fms_from_feature_frame(feat, params=production_fms_score_params())
    pd.testing.assert_series_equal(
        got.sort_index(),
        expected.sort_index(),
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_mapping_params_override_changes_score(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Dict overrides must be accepted and change at least one finite score."""
    feat, _ = _feature_frame_from_prices(synthetic_prices_krw)
    base = score_fms_from_feature_frame(feat)
    tweaked = dataclasses.replace(production_fms_score_params(), w_r3=0.05, w_r4=0.05)
    alt = score_fms_from_feature_frame(feat, params=tweaked)
    finite = base.replace(-999.0, pd.NA).dropna()
    assert not finite.empty
    assert not alt.reindex(finite.index).equals(finite)


def test_mapping_dict_coerces_to_params(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Tune-style dict must coerce via FmsScoreParams.from_mapping."""
    feat, _ = _feature_frame_from_prices(synthetic_prices_krw)
    p = production_fms_score_params()
    as_dict = dataclasses.asdict(p)
    via_dict = score_fms_from_feature_frame(feat, params=as_dict)
    via_obj = score_fms_from_feature_frame(feat, params=p)
    pd.testing.assert_series_equal(
        via_dict.sort_index(),
        via_obj.sort_index(),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_tune_fms_score_delegates_to_core(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Tune search entrypoint must not carry an independent formula body."""
    feat, _ = _feature_frame_from_prices(synthetic_prices_krw)
    p = baseline_params()
    pd.testing.assert_series_equal(
        fms_score(feat, p).sort_index(),
        score_fms_from_feature_frame(feat, params=p).sort_index(),
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_baseline_params_equal_production() -> None:
    """Tune search baseline must start from production SSOT weights."""
    prod = dataclasses.asdict(production_fms_score_params())
    base = baseline_params()
    for key, value in base.items():
        assert key in prod, f"unexpected tune key: {key}"
        assert value == pytest.approx(prod[key], rel=0, abs=1e-12), key


def test_fms_score_params_unknown_key_rejected() -> None:
    with pytest.raises(TypeError, match="unknown"):
        FmsScoreParams.from_mapping({"w_r3": 0.1, "not_a_real_param": 1.0})
