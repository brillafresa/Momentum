"""
Contracts for the cash-like path quality-bonus gate (v4.6.1+) under
watchlist-relative Z-score normalization (v4.7.0).

Policy
------
Suppress **positive** quality-axis contributions only when
``low_return(R_3M) × ultra_low_vol(Vol20_Ann) × high_smooth(R2_3M)`` is high.
``R_3M`` itself and existing penalties are unchanged. Tradeability ``-999``
semantics are unchanged.

Also locks relative-Z contracts that the cash fixture panel exercises:
self-reference centering, zero-variance axes, and batch-vs-watchlist parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.fms import compute_fms_snapshot, momentum_now_and_delta
from core.fms_features import (
    CASH_R2_FULL,
    CASH_R2_NONE,
    CASH_R3M_FULL,
    CASH_R3M_NONE,
    CASH_VOL_FULL,
    CASH_VOL_NONE,
    PRODUCTION_FMS_COLUMNS,
    build_panel_feature_frame,
    cash_like_strength,
    production_axis_contributions,
    score_production_fms_features,
    smoothstep_series,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PANEL_PATH = FIXTURES / "cash_like_paths_prices_krw.csv"
GEN_PATH = ROOT / "scripts" / "fixtures" / "generate_cash_like_panel.py"


@pytest.fixture(scope="module")
def cash_like_prices() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("cash_like_gen", GEN_PATH)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        panel = mod.build_cash_like_panel()
        panel.to_csv(PANEL_PATH)
    df = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def test_smoothstep_boundaries() -> None:
    x = pd.Series([0.0, 0.01, 0.03, 0.05, 0.10])
    s = smoothstep_series(x, 0.01, 0.05)
    assert float(s.iloc[0]) == pytest.approx(0.0)
    assert float(s.iloc[1]) == pytest.approx(0.0)
    assert float(s.iloc[3]) == pytest.approx(1.0)
    assert float(s.iloc[4]) == pytest.approx(1.0)
    assert 0.0 < float(s.iloc[2]) < 1.0


def test_cash_like_strength_is_high_for_smooth_cash_path(
    cash_like_prices: pd.DataFrame,
) -> None:
    feats = build_panel_feature_frame(cash_like_prices)
    strength = cash_like_strength(feats)
    assert float(strength.loc["CASH_LIKE"]) > 0.9
    assert float(strength.loc["CASH_STAIR"]) > 0.5
    assert float(strength.loc["NOISY_LOW"]) == pytest.approx(0.0, abs=1e-9)
    assert float(strength.loc["BOND_RALLY"]) == pytest.approx(0.0, abs=1e-6)
    assert float(strength.loc["EQUITY_TREND"]) == pytest.approx(0.0, abs=1e-9)
    assert float(strength.loc["SMOOTH_STRONG"]) == pytest.approx(0.0, abs=1e-6)


def test_cash_like_path_does_not_rank_as_top_momentum(
    cash_like_prices: pd.DataFrame,
) -> None:
    """Cash-rate smoothness must not outrank real momentum paths."""
    result = momentum_now_and_delta(
        cash_like_prices,
        reference_prices_krw=cash_like_prices,
        ohlc_data=None,
        symbols=list(cash_like_prices.columns),
    )
    cash_fms = float(result.loc["CASH_LIKE", "FMS"])
    equity_fms = float(result.loc["EQUITY_TREND", "FMS"])
    smooth_strong_fms = float(result.loc["SMOOTH_STRONG", "FMS"])
    bond_fms = float(result.loc["BOND_RALLY", "FMS"])

    assert cash_fms < 0.5
    assert cash_fms < equity_fms
    assert cash_fms < smooth_strong_fms
    assert cash_fms < bond_fms


def test_reference_axis_zscores_are_centered_on_current_watchlist(
    cash_like_prices: pd.DataFrame,
) -> None:
    """Ungated axis contributions use current-watchlist mean/std, not frozen fit."""
    feats = build_panel_feature_frame(cash_like_prices)
    contrib = production_axis_contributions(
        feats,
        reference_features=feats,
        apply_cash_gate=False,
    )
    # Every non-degenerate axis is centered; weighted contributions retain mean 0.
    for col in PRODUCTION_FMS_COLUMNS:
        assert float(contrib[col].mean()) == pytest.approx(0.0, abs=1e-10)


def test_batch_targets_are_scored_against_watchlist_reference(
    cash_like_prices: pd.DataFrame,
) -> None:
    """Changing the account watchlist reference must change candidate FMS."""
    feats = build_panel_feature_frame(cash_like_prices)
    targets = feats.loc[["EQUITY_TREND", "BOND_RALLY"]]
    weak_ref = feats.loc[["CASH_LIKE", "CASH_STAIR", "NOISY_LOW"]]
    strong_ref = feats.loc[["SMOOTH_STRONG", "EQUITY_TREND", "BOND_RALLY"]]
    weak_scores = score_production_fms_features(
        targets, reference_features=weak_ref
    )
    strong_scores = score_production_fms_features(
        targets, reference_features=strong_ref
    )
    assert not np.allclose(weak_scores, strong_scores)
    assert float(weak_scores.mean()) > float(strong_scores.mean())


def test_zero_variance_reference_axis_contributes_zero() -> None:
    """A current-watchlist axis with no dispersion has no relative information."""
    row = {col: 1.0 for col in PRODUCTION_FMS_COLUMNS}
    row["Vol20_Ann"] = 0.2
    targets = pd.DataFrame([row, row], index=["A", "B"])
    contrib = production_axis_contributions(
        targets,
        reference_features=targets,
        apply_cash_gate=False,
    )
    assert np.allclose(contrib.to_numpy(), 0.0)


def test_strong_smooth_trend_keeps_quality_bonus(
    cash_like_prices: pd.DataFrame,
) -> None:
    """Sufficient return + smoothness retains positive quality credit."""
    feats = build_panel_feature_frame(cash_like_prices)
    gated = score_production_fms_features(feats, reference_features=feats)
    # Ungated baseline: force cash_strength=0 by raising R_3M above NONE.
    ungated_feats = feats.copy()
    ungated_feats.loc["SMOOTH_STRONG", "R_3M"] = max(
        float(ungated_feats.loc["SMOOTH_STRONG", "R_3M"]), CASH_R3M_NONE + 0.01
    )
    ungated = score_production_fms_features(
        ungated_feats, reference_features=ungated_feats
    )
    # Path already has high R_3M; gated score must match ungated for that row.
    assert float(gated.loc["SMOOTH_STRONG"]) == pytest.approx(
        float(ungated.loc["SMOOTH_STRONG"]), abs=1e-9
    )
    assert float(gated.loc["SMOOTH_STRONG"]) > 1.0


def test_noisy_low_return_equity_score_unchanged_vs_no_gate_identity(
    cash_like_prices: pd.DataFrame,
) -> None:
    """Low return alone is not enough; equity-like vol keeps full score."""
    feats = build_panel_feature_frame(cash_like_prices)
    assert float(cash_like_strength(feats).loc["NOISY_LOW"]) == pytest.approx(0.0)
    snap = compute_fms_snapshot(cash_like_prices, symbols=list(cash_like_prices.columns))
    assert np.isfinite(float(snap.loc["NOISY_LOW", "FMS"]))


def test_gate_is_continuous_across_r3m_boundary(
    cash_like_prices: pd.DataFrame,
) -> None:
    """Scores must not cliff at the cash R_3M edges."""
    feats = build_panel_feature_frame(cash_like_prices)
    base = feats.loc[["CASH_LIKE"]].copy()
    # Force ultra-low vol + high R² so only R_3M varies the strength.
    base["Vol20_Ann"] = CASH_VOL_FULL * 0.5
    base["R2_3M"] = CASH_R2_FULL

    scores = []
    for r3m in np.linspace(CASH_R3M_FULL - 0.005, CASH_R3M_NONE + 0.01, 25):
        row = base.copy()
        row["R_3M"] = r3m
        scores.append(float(score_production_fms_features(row).iloc[0]))
    diffs = np.abs(np.diff(scores))
    assert float(np.max(diffs)) < 0.35


def test_penalties_and_r3m_term_not_boosted_by_gate() -> None:
    """Gate must never inflate negative quality contributions."""
    # Construct a cash-like feature row with a penalty axis that is already bad.
    row = {
        col: 0.0 for col in PRODUCTION_FMS_COLUMNS
    }
    row.update(
        {
            "R2_3M": 0.999,
            "DD_RECOVERY": 1.0,
            "TREND_QUALITY_21D": 0.0,
            "JUMP_DISCONTINUITY_3M": 0.2,  # bad → negative contribution direction -1
            "UNDER_EMA20_DAYS": 0.0,
            "R_3M": 0.005,
            "STALE_AGE": 0.0,
            "UP_STREAK_5D": 0.0,
            "TREND_EFFICIENCY_REWARD_15D": 0.0,
            "RANGE_COMPRESSION_20D": 1.0,
            "Vol20_Ann": 0.001,
        }
    )
    feats = pd.DataFrame([row], index=["X"])
    strength = float(cash_like_strength(feats).iloc[0])
    assert strength > 0.9
    # Use a varied watchlist reference; singleton references intentionally
    # contribute zero because they contain no relative dispersion.
    low = {col: 0.0 for col in PRODUCTION_FMS_COLUMNS}
    high = {col: 1.0 for col in PRODUCTION_FMS_COLUMNS}
    low["Vol20_Ann"] = 0.1
    high["Vol20_Ann"] = 0.3
    reference = pd.DataFrame([low, high], index=["LOW", "HIGH"])
    gated = float(
        score_production_fms_features(
            feats, reference_features=reference, apply_cash_gate=True
        ).iloc[0]
    )
    ungated = float(
        score_production_fms_features(
            feats, reference_features=reference, apply_cash_gate=False
        ).iloc[0]
    )
    contrib = production_axis_contributions(
        feats, reference_features=reference
    ).loc["X"]
    assert float(contrib["JUMP_DISCONTINUITY_3M"]) <= 0.0
    assert gated <= ungated


def test_tradeability_minus_999_unchanged(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
) -> None:
    symbols = list(synthetic_prices_krw.columns)
    result = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=symbols,
    )
    assert float(result.loc["CRASHY", "FMS"]) == -999.0


def test_existing_golden_rank_order_preserved(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
    golden_fms_ranks: dict,
) -> None:
    result = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=list(synthetic_prices_krw.columns),
    )
    assert list(result.index) == golden_fms_ranks["symbols_desc_fms"]


def test_threshold_constants_match_plan_candidate_range() -> None:
    assert CASH_R3M_FULL == pytest.approx(0.01)
    assert CASH_R3M_NONE == pytest.approx(0.05)
    assert CASH_VOL_FULL == pytest.approx(0.005)
    assert CASH_VOL_NONE == pytest.approx(0.03)
    assert CASH_R2_NONE == pytest.approx(0.95)
    assert CASH_R2_FULL == pytest.approx(0.99)
