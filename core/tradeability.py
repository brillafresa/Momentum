# -*- coding: utf-8 -*-
"""Tradeability (True Range) disqualification filters — pure logic, no I/O.

Migrated from ``analysis_utils.calculate_tradeability_filters``.
``analysis_utils`` re-exports this callable for transitional callers.

Rules (see HARNESS_RULES / .cursorrules):
- Fatal volatility: any day in last 63 with True Range / prev_close > 30%
- Repeated downside: >= 4 days in last 20 with (low / prev_close - 1) < -7%
- Open-print glitch: if high==low==0 and prior bar valid, substitute prior H/L
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def calculate_tradeability_filters(
    ohlc_data: pd.DataFrame, symbols: List[str]
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """Return per-symbol disqualification flags and human-readable reasons.

    Args:
        ohlc_data: Raw OHLC panel. MultiIndex columns ``(symbol, field)`` with
            ``High``/``Low``/``Close``, or a flat frame with those field names
            (single-symbol path).
        symbols: Symbols to evaluate.

    Returns:
        ``(disqualification, filter_reasons)`` where ``disqualification[sym]``
        is True when the symbol fails tradeability, and ``filter_reasons[sym]``
        is ``정상`` or a semicolon-joined reason string.
    """
    disqualification: Dict[str, bool] = {}
    filter_reasons: Dict[str, str] = {}
    for symbol in symbols:
        try:
            if isinstance(ohlc_data.columns, pd.MultiIndex):
                if (
                    (symbol, "High") in ohlc_data.columns
                    and (symbol, "Low") in ohlc_data.columns
                    and (symbol, "Close") in ohlc_data.columns
                ):
                    high = ohlc_data[(symbol, "High")].dropna()
                    low = ohlc_data[(symbol, "Low")].dropna()
                    close = ohlc_data[(symbol, "Close")].dropna()
                else:
                    disqualification[symbol] = True
                    filter_reasons[symbol] = "OHLC 데이터 부족"
                    continue
            else:
                if all(c in ohlc_data.columns for c in ["High", "Low", "Close"]):
                    high = ohlc_data["High"].dropna()
                    low = ohlc_data["Low"].dropna()
                    close = ohlc_data["Close"].dropna()
                else:
                    disqualification[symbol] = True
                    filter_reasons[symbol] = "OHLC 데이터 부족"
                    continue

            if len(close) < 63:
                disqualification[symbol] = True
                filter_reasons[symbol] = "데이터 기간 부족 (63일 미만)"
                continue

            prev_close = close.shift(1)
            prev_high = high.shift(1)
            prev_low = low.shift(1)

            # 당일 고가/저가가 0인 경우 전일 데이터로 대체
            # 조건: 당일 고가=0 AND 당일 저가=0 AND 전일 종가>0 AND 전일 고가>0 AND 전일 저가>0
            invalid_high_low = (
                (high == 0)
                & (low == 0)
                & (prev_close > 0)
                & (prev_high > 0)
                & (prev_low > 0)
            )
            high_fixed = high.copy()
            low_fixed = low.copy()
            high_fixed[invalid_high_low] = prev_high[invalid_high_low]
            low_fixed[invalid_high_low] = prev_low[invalid_high_low]

            true_range = pd.concat(
                [
                    high_fixed - low_fixed,
                    (high_fixed - prev_close).abs(),
                    (low_fixed - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1, skipna=False)
            daily_true_range_vol = true_range / prev_close
            daily_downside_risk = (low_fixed / prev_close) - 1

            extreme_days = daily_true_range_vol.tail(63)
            extreme_days_filtered = extreme_days[extreme_days > 0.30]
            severe_days = daily_downside_risk.tail(20)
            severe_days_filtered = severe_days[severe_days < -0.07]

            reasons = []
            if len(extreme_days_filtered) > 0:
                reasons.append(f"치명적 변동성 ({len(extreme_days_filtered)}일 30% 초과)")
            if len(severe_days_filtered) >= 4:
                reasons.append(
                    f"반복적 하방리스크 ({len(severe_days_filtered)}일 -7% 미만)"
                )

            disqualification[symbol] = len(reasons) > 0
            filter_reasons[symbol] = "; ".join(reasons) if reasons else "정상"
        except Exception as e:
            disqualification[symbol] = True
            filter_reasons[symbol] = f"계산 오류: {e}"
    return disqualification, filter_reasons
