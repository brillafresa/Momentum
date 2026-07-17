"""
Unit tests for offline FMS scoring (Validation Harness).

Purpose
-------
Verify production scoring entrypoints
(``compute_fms_snapshot``, ``momentum_now_and_delta``) using checked-in
synthetic KRW / OHLC fixtures — no live market API.

Covered behaviors
-----------------
- Golden rank order (TREND_UP > MILD_UP > FLAT > CRASHY)
- Tradeability disqualification → FMS == -999 for CRASHY
- Missing OHLC skips -999 filter path
- yfinance.download is never called
- All-NaN column does not crash scoring
- Non-positive price glitches do not emit ``log`` RuntimeWarnings

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_fms_scoring.py -q

Fixtures: ``tests/fixtures/synthetic_*.csv``, ``golden_fms_ranks.json``.
Regenerate panels: ``python scripts/fixtures/generate_synthetic_panel.py``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict
from unittest.mock import patch

import pandas as pd
import pytest

from analysis_utils import compute_fms_snapshot, momentum_now_and_delta, r_squared_3m


def test_momentum_now_and_delta_rank_order_matches_golden(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
    golden_fms_ranks: Dict[str, Any],
) -> None:
    """Strong trend should outrank mild/flat; CRASHY is filtered to -999."""
    symbols = list(synthetic_prices_krw.columns)
    result = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=symbols,
    )

    assert list(result.index) == golden_fms_ranks["symbols_desc_fms"]
    for symbol, expected in golden_fms_ranks["disqualified"].items():
        assert result.loc[symbol, "FMS"] == pytest.approx(expected)


def test_compute_fms_snapshot_matches_momentum_fms_column(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
) -> None:
    """Public snapshot API must match the FMS column used by momentum_now_and_delta."""
    symbols = list(synthetic_prices_krw.columns)
    snap = compute_fms_snapshot(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=symbols,
    )
    full = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=synthetic_ohlc,
        symbols=symbols,
    )
    aligned = snap["FMS"].reindex(full.index)
    pd.testing.assert_series_equal(aligned, full["FMS"], check_names=False)


def test_missing_ohlc_skips_disqualification(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Without OHLC, tradeability filter does not force FMS=-999."""
    symbols = list(synthetic_prices_krw.columns)
    result = momentum_now_and_delta(
        synthetic_prices_krw,
        reference_prices_krw=synthetic_prices_krw,
        ohlc_data=None,
        symbols=symbols,
    )
    assert (result["FMS"] == -999).sum() == 0
    assert result["FMS"].notna().all()


def test_fms_scoring_does_not_call_yfinance_download(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
) -> None:
    """Injected fixtures must not trigger live yfinance downloads."""
    symbols = list(synthetic_prices_krw.columns)
    with patch("yfinance.download", side_effect=AssertionError("network forbidden")):
        momentum_now_and_delta(
            synthetic_prices_krw,
            reference_prices_krw=synthetic_prices_krw,
            ohlc_data=synthetic_ohlc,
            symbols=symbols,
        )
        compute_fms_snapshot(
            synthetic_prices_krw,
            reference_prices_krw=synthetic_prices_krw,
            ohlc_data=synthetic_ohlc,
            symbols=symbols,
        )


def test_non_positive_price_glitch_emits_no_log_warnings(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """Yahoo Adj Close glitches (negative/zero ticks) must not spam np.log warnings.

    Regression for batch-scan console noise:
    ``RuntimeWarning: invalid value encountered in log`` from
    ``r_squared_3m`` / EMA20 slope-curvature log regressions.
    """
    glitch = synthetic_prices_krw.copy()
    col = glitch.columns[0]
    glitch.iloc[-30, glitch.columns.get_loc(col)] = -5.0
    glitch.iloc[-40, glitch.columns.get_loc(col)] = 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r_squared_3m(glitch)
        result = momentum_now_and_delta(
            glitch,
            reference_prices_krw=glitch,
            ohlc_data=None,
            symbols=list(glitch.columns),
        )

    log_warnings = [w for w in caught if "in log" in str(w.message)]
    assert log_warnings == [], [str(w.message) for w in log_warnings]
    assert "FMS" in result.columns
    assert result["FMS"].notna().all()


def test_nan_column_does_not_crash_scoring(
    synthetic_prices_krw: pd.DataFrame,
) -> None:
    """All-NaN series is an edge case; scoring should still return a frame."""
    prices = synthetic_prices_krw.copy()
    prices["ALL_NAN"] = float("nan")
    result = momentum_now_and_delta(
        prices,
        reference_prices_krw=prices,
        ohlc_data=None,
        symbols=list(prices.columns),
    )
    assert "ALL_NAN" in result.index
    assert "FMS" in result.columns
