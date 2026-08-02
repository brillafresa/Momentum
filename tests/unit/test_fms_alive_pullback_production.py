"""
Production alive_pullback FMS contracts (v5.0.0 Validation Harness).

Purpose
-------
Lock the promoted nonlinear scorer: frozen params, absolute (reference-invariant)
scoring, cash-like paths not topping ranks, and parity with the calibration
family wrapper.

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_fms_alive_pullback_production.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibration.nonlinear_formulas import family_by_name
from core.fms import compute_fms_snapshot
from core.fms_features import (
    PRODUCTION_ALIVE_PULLBACK_PARAMS,
    PRODUCTION_FORMULA_ID,
    build_panel_feature_frame,
    score_alive_pullback_from_params,
    score_production_fms_features,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CASH_PANEL = FIXTURES / "cash_like_paths_prices_krw.csv"


def test_production_formula_id() -> None:
    assert PRODUCTION_FORMULA_ID == "alive_pullback_v5"


def test_calibration_family_matches_core_scorer() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    daily = np.full(80, 0.004)
    daily[-5:] = 0.01
    prices = pd.DataFrame(
        {"A": 100.0 * np.exp(np.cumsum(daily))},
        index=idx,
    )
    feats = build_panel_feature_frame(prices)
    core_score = score_alive_pullback_from_params(
        feats, PRODUCTION_ALIVE_PULLBACK_PARAMS
    )
    fam = family_by_name("alive_pullback")
    cal_score = fam.score(feats, PRODUCTION_ALIVE_PULLBACK_PARAMS)
    pd.testing.assert_series_equal(
        core_score, cal_score.rename("FMS"), check_names=False
    )


def test_production_scorer_applies_disqualification() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    prices = pd.DataFrame(
        {
            "OK": 100.0 * np.exp(np.cumsum(np.full(80, 0.003))),
            "BAD": 100.0 * np.exp(np.cumsum(np.full(80, 0.003))),
        },
        index=idx,
    )
    feats = build_panel_feature_frame(prices)
    scores = score_production_fms_features(
        feats, disqualified_symbols={"BAD"}
    )
    assert scores.loc["BAD"] == pytest.approx(-999.0)
    assert scores.loc["OK"] != pytest.approx(-999.0)


@pytest.mark.skipif(not CASH_PANEL.exists(), reason="cash fixture missing")
def test_cash_like_path_not_top_under_alive_pullback() -> None:
    prices = pd.read_csv(CASH_PANEL, index_col=0, parse_dates=True)
    snap = compute_fms_snapshot(prices, symbols=list(prices.columns))
    ranked = snap["FMS"].sort_values(ascending=False)
    assert ranked.index[0] != "CASH_LIKE"
    assert float(ranked.loc["CASH_LIKE"]) < float(ranked.iloc[0])
