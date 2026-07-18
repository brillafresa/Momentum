"""
Unit tests for tradeability filters (Validation Harness).

Purpose
-------
Lock ``core.tradeability.calculate_tradeability_filters`` as the pure,
offline True Range / downside-risk disqualification path — no network I/O.

Covered behaviors
-----------------
- Synthetic panel: CRASHY disqualified; TREND_UP / MILD_UP / FLAT pass
- Missing OHLC columns → disqualify (``OHLC 데이터 부족``)
- History shorter than 63 bars → ``데이터 기간 부족``
- Single extreme True Range day (>30%) → fatal volatility DQ
- Four downside days (<-7% in 20d) → repeated downside DQ
- Zero high/low replaced by prior bar does not invent false extremes
- ``analysis_utils`` re-export shim stays identity-equal to ``core``

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_tradeability.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.tradeability import calculate_tradeability_filters


def _multiindex_ohlc(
    symbols: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build MultiIndex (symbol, field) OHLC from per-symbol frames."""
    pieces = {}
    for sym, frame in symbols.items():
        for field in ("High", "Low", "Close"):
            pieces[(sym, field)] = frame[field]
    return pd.DataFrame(pieces)


def _quiet_ohlc(n: int = 80, start: float = 100.0) -> pd.DataFrame:
    """Stable path: ~1% daily range, no extreme TR / downside."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(start, start * 1.05, n), index=idx)
    high = close * 1.005
    low = close * 0.995
    return pd.DataFrame({"High": high, "Low": low, "Close": close})


def test_synthetic_panel_crashy_disqualified_others_pass(
    synthetic_ohlc: pd.DataFrame,
) -> None:
    """Fixture CRASHY trips filters; trend/mild/flat stay tradeable."""
    symbols = ["TREND_UP", "MILD_UP", "FLAT", "CRASHY"]
    flags, reasons = calculate_tradeability_filters(synthetic_ohlc, symbols)

    assert flags["CRASHY"] is True
    assert "치명적 변동성" in reasons["CRASHY"] or "반복적 하방리스크" in reasons["CRASHY"]
    for sym in ("TREND_UP", "MILD_UP", "FLAT"):
        assert flags[sym] is False
        assert reasons[sym] == "정상"


def test_missing_ohlc_columns_disqualify() -> None:
    """Symbol absent from MultiIndex OHLC → OHLC 데이터 부족."""
    quiet = _quiet_ohlc()
    ohlc = _multiindex_ohlc({"KEEP": quiet})
    flags, reasons = calculate_tradeability_filters(ohlc, ["KEEP", "MISSING"])
    assert flags["KEEP"] is False
    assert flags["MISSING"] is True
    assert reasons["MISSING"] == "OHLC 데이터 부족"


def test_short_history_disqualify() -> None:
    """Fewer than 63 closes → 데이터 기간 부족."""
    short = _quiet_ohlc(n=40)
    ohlc = _multiindex_ohlc({"SHORT": short})
    flags, reasons = calculate_tradeability_filters(ohlc, ["SHORT"])
    assert flags["SHORT"] is True
    assert "데이터 기간 부족" in reasons["SHORT"]


def test_extreme_true_range_day_disqualifies() -> None:
    """One day with True Range / prev_close > 30% → fatal volatility."""
    frame = _quiet_ohlc(n=80)
    # Spike high vs prior close by >30%
    frame.iloc[-5, frame.columns.get_loc("High")] = frame["Close"].iloc[-6] * 1.35
    ohlc = _multiindex_ohlc({"SPIKE": frame})
    flags, reasons = calculate_tradeability_filters(ohlc, ["SPIKE"])
    assert flags["SPIKE"] is True
    assert "치명적 변동성" in reasons["SPIKE"]


def test_repeated_downside_disqualifies() -> None:
    """Four lows more than 7% below prior close within 20d → DQ."""
    frame = _quiet_ohlc(n=80)
    for offset in (-2, -4, -6, -8):
        prev = frame["Close"].iloc[offset - 1]
        frame.iloc[offset, frame.columns.get_loc("Low")] = prev * 0.90
    ohlc = _multiindex_ohlc({"DOWN": frame})
    flags, reasons = calculate_tradeability_filters(ohlc, ["DOWN"])
    assert flags["DOWN"] is True
    assert "반복적 하방리스크" in reasons["DOWN"]


def test_zero_high_low_replaced_by_prior_does_not_false_dq() -> None:
    """Yahoo open-print glitch (H=L=0) uses prior H/L; quiet path stays OK."""
    frame = _quiet_ohlc(n=80)
    frame.iloc[-1, frame.columns.get_loc("High")] = 0.0
    frame.iloc[-1, frame.columns.get_loc("Low")] = 0.0
    ohlc = _multiindex_ohlc({"GLITCH": frame})
    flags, reasons = calculate_tradeability_filters(ohlc, ["GLITCH"])
    assert flags["GLITCH"] is False
    assert reasons["GLITCH"] == "정상"


def test_flat_ohlc_columns_single_symbol() -> None:
    """Non-MultiIndex High/Low/Close frame is supported (single-symbol path)."""
    frame = _quiet_ohlc(n=80)
    flags, reasons = calculate_tradeability_filters(frame, ["ONLY"])
    assert flags["ONLY"] is False
    assert reasons["ONLY"] == "정상"


def test_analysis_utils_reexports_core_tradeability() -> None:
    """Transitional facade must expose the same callable as core."""
    import analysis_utils as au
    import core.tradeability as ct

    assert au.calculate_tradeability_filters is ct.calculate_tradeability_filters
