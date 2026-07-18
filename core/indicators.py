# -*- coding: utf-8 -*-
"""Pure price / return indicators (no network I/O).

Migrated from ``analysis_utils`` so scoring helpers can live under ``core/``
while the transitional facade re-exports these callables.

See HARNESS_RULES.md §2.1 / §2.5.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import linregress


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with ``adjust=False`` (recursive EMA)."""
    return s.ewm(span=span, adjust=False).mean()


def returns_pct(df: pd.DataFrame, n: int) -> pd.Series:
    """Last-row n-period percentage return per column.

    If the panel has ``<= n`` rows, returns an empty float Series indexed by
    ``df.columns`` (all NaN).
    """
    if df.shape[0] <= n:
        return pd.Series(index=df.columns, dtype=float)
    dff = df.ffill()
    r = dff.pct_change(periods=n, fill_method=None).iloc[-1]
    return r


def r_squared_3m(prices_krw: pd.DataFrame) -> pd.Series:
    """3개월(63거래일) 로그 수익률 기반 결정계수(R²)를 계산합니다.

    R²는 추세의 매끄러움을 평가하며, 높을수록 안정적인 우상향 추세를 의미합니다.

    Args:
        prices_krw: KRW 환산 가격 데이터 (컬럼: 종목, 인덱스: 날짜)

    Returns:
        각 종목별 R² 값 (0~1 사이, NaN 가능), name ``R2_3M``
    """
    r2_dict = {}
    for col in prices_krw.columns:
        s = prices_krw[col].dropna()
        if len(s) < 63:
            r2_dict[col] = np.nan
            continue

        # 최근 63거래일 데이터
        recent = s.tail(63)
        if len(recent) < 2:
            r2_dict[col] = np.nan
            continue

        # 로그 수익률 계산 (0/음수 글리치 가격은 NaN 처리해 log 경고 방지)
        ratio = recent / recent.iloc[0]
        log_returns = np.log(ratio.where(ratio > 0))

        # 선형 회귀: 유효 값만 사용 (글리치 지점 제외, 원래 시간축 위치 유지)
        valid = log_returns.notna().values
        if valid.sum() < 2:
            r2_dict[col] = np.nan
            continue
        x = np.flatnonzero(valid)
        y = log_returns.values[valid]

        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r2 = r_value ** 2
            r2_dict[col] = r2
        except Exception:
            r2_dict[col] = np.nan

    return pd.Series(r2_dict, name="R2_3M")


def ytd_return(df: pd.DataFrame) -> pd.Series:
    """Year-to-date return from the nearest session on/after Jan 1 to last close."""
    if df.empty:
        return pd.Series(dtype=float)
    dff = df.ffill()
    last = dff.index[-1]
    y0 = pd.Timestamp(datetime(last.year, 1, 1))
    start_idx = dff.index.get_indexer([y0], method="nearest")[0]
    return dff.iloc[-1] / dff.iloc[start_idx] - 1.0


def last_vol_annualized(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized volatility of the last ``window`` daily returns (sqrt(252))."""
    rets = df.ffill().pct_change(fill_method=None).dropna()
    if rets.empty:
        return pd.Series(index=df.columns, dtype=float)
    vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252.0)
    return vol

