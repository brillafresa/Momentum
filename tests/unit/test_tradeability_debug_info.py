"""
Validation Harness: get_filter_debug_info outputs
match core tradeability True Range / downside-risk decisions.

Goal
----
Lock ``analysis_utils.get_filter_debug_info`` as the transitional debug
surface for the tradeability filters. When we migrate the function into
``core/tradeability.py`` later, this test suite will guard against logic drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis_utils import get_filter_debug_info
from core.tradeability import calculate_tradeability_filters


def _multiindex_ohlc(symbol_to_frame: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build MultiIndex (symbol, field) OHLC from per-symbol frames."""
    pieces: dict[tuple[str, str], pd.Series] = {}
    for sym, frame in symbol_to_frame.items():
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


def test_debug_info_empty_ohlc() -> None:
    debug = get_filter_debug_info(None, "ANY")
    assert debug["has_ohlc"] is False
    assert debug["data_points"] == 0
    assert debug["error"] == "OHLC 데이터 없음"


def test_debug_info_missing_columns_multindex() -> None:
    # Build MultiIndex OHLC but omit the requested symbol.
    quiet = _quiet_ohlc()
    ohlc = _multiindex_ohlc({"KEEP": quiet})
    debug = get_filter_debug_info(ohlc, "MISSING")
    assert debug["has_ohlc"] is False
    assert debug["error"] == "OHLC 컬럼 없음 (MultiIndex)"


def test_debug_info_short_history_matches_tradeability_reason() -> None:
    short = _quiet_ohlc(n=40)
    ohlc = _multiindex_ohlc({"SHORT": short})

    flags, reasons = calculate_tradeability_filters(ohlc, ["SHORT"])
    debug = get_filter_debug_info(ohlc, "SHORT")

    assert flags["SHORT"] is True
    assert "데이터 기간 부족" in reasons["SHORT"]
    assert debug["error"] is not None
    assert "데이터 부족" in debug["error"]


def test_debug_info_zero_high_low_replaced_by_prior() -> None:
    # When High==Low==0 (open-print glitch), the debug surface should report
    # high_low_fixed and use the prior H/L in the computed range components.
    frame = _quiet_ohlc(n=80)
    prev_high = frame["High"].iloc[-2]
    prev_low = frame["Low"].iloc[-2]

    frame.iloc[-1, frame.columns.get_loc("High")] = 0.0
    frame.iloc[-1, frame.columns.get_loc("Low")] = 0.0

    ohlc = _multiindex_ohlc({"GLITCH": frame})

    flags, reasons = calculate_tradeability_filters(ohlc, ["GLITCH"])
    assert flags["GLITCH"] is False
    assert reasons["GLITCH"] == "정상"

    debug = get_filter_debug_info(ohlc, "GLITCH")
    assert debug["error"] is None
    assert debug["has_ohlc"] is True
    assert debug["recent_data"]["high_low_fixed"] is True
    assert debug["recent_data"]["last_high"] == pytest.approx(prev_high)
    assert debug["recent_data"]["last_low"] == pytest.approx(prev_low)


def test_debug_info_extreme_and_repeated_downside_counts() -> None:
    # Construct a panel where:
    # - 1 day in the last 63 triggers fatal volatility (>30% TR/prev_close)
    # - 4 days in the last 20 trigger repeated downside (<-7% low/prev_close - 1)
    frame = _quiet_ohlc(n=80)

    # Fatal volatility: spike high on one day.
    frame.iloc[-5, frame.columns.get_loc("High")] = frame["Close"].iloc[-6] * 1.35

    # Repeated downside: set low to 90% of prev_close for 4 days.
    for offset in (-2, -4, -6, -8):
        prev = frame["Close"].iloc[offset - 1]
        frame.iloc[offset, frame.columns.get_loc("Low")] = prev * 0.90

    ohlc = _multiindex_ohlc({"DQ": frame})

    flags, reasons = calculate_tradeability_filters(ohlc, ["DQ"])
    assert flags["DQ"] is True
    assert "치명적 변동성" in reasons["DQ"]
    assert "반복적 하방리스크" in reasons["DQ"]

    debug = get_filter_debug_info(ohlc, "DQ")
    assert debug["error"] is None
    assert debug["has_ohlc"] is True
    assert debug["extreme_days_count"] == 1
    assert debug["severe_days_count"] == 4

