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


def mask_non_positive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-positive prices with NaN (Yahoo Adj Close hygiene).

    Some tickers (e.g. certain KR ETFs) publish large **negative** Adj Close
    histories that later jump back to normal positives. Feeding those into EMA /
    return features inflates ``AboveEMA50`` and FMS into triple-digit outliers.
    Callers that score FMS must mask before indicator math.
    """
    if df is None or df.empty:
        return df
    out = df.astype(float).copy()
    return out.where(out > 0.0)


def align_bday_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a single-market frame to weekdays and forward-fill gaps.

    Intended for one market's native calendar before multi-market concat.
    Does not clip to last_valid (the frame's own span *is* the native span).
    """
    if df is None or len(df) == 0:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    return df.reindex(idx).ffill()


def harmonize_calendar(df: pd.DataFrame, coverage: float = 0.9) -> pd.DataFrame:
    """Union B-day panel with ffill that does **not** extend past native as-of.

    Multi-market concat introduces trailing NaNs when one market prints a day
    another has not. Blanket ``ffill`` would fabricate flat bars and shift
    SEG_* windows. This helper:

    1. Records each column's ``last_valid_index`` before reindex.
    2. Reindexes to a shared business-day grid and forward-fills (interior gaps).
    3. Restores NaN for dates **after** that column's last real observation.
    4. Drops columns whose non-NaN coverage is below ``coverage``.

    Market labels are unused — US/KR/HK/JPN/future markets are treated the same.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    work = df.astype(float).copy()
    last_valid = {col: work[col].last_valid_index() for col in work.columns}
    idx = pd.date_range(work.index.min(), work.index.max(), freq="B")
    out = work.reindex(idx).ffill()
    for col, lv in last_valid.items():
        if lv is None:
            out[col] = np.nan
        else:
            out.loc[out.index > lv, col] = np.nan

    valid_ratio = out.count().div(len(out))
    keep_cols = valid_ratio[valid_ratio >= coverage].index
    return out[keep_cols] if len(keep_cols) > 0 else pd.DataFrame()


def returns_pct(df: pd.DataFrame, n: int) -> pd.Series:
    """n-period percentage return at each column's last valid observation.

    Trailing NaNs (other-market-only days after native as-of) are ignored.
    Interior gaps are forward-filled only within the native span.
    If a column has ``<= n`` valid points after that, its value is NaN.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for col in df.columns:
        s = df[col].astype(float)
        lv = s.last_valid_index()
        if lv is None:
            out[col] = np.nan
            continue
        hist = s.loc[:lv].ffill().dropna()
        if len(hist) <= n:
            out[col] = np.nan
            continue
        out[col] = float(hist.iloc[-1] / hist.iloc[-(n + 1)] - 1.0)
    return pd.Series(out, dtype=float)


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
    """Year-to-date return to each column's last valid close (native as-of)."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for col in df.columns:
        s = df[col].astype(float)
        lv = s.last_valid_index()
        if lv is None:
            out[col] = np.nan
            continue
        hist = s.loc[:lv].ffill().dropna()
        if hist.empty:
            out[col] = np.nan
            continue
        y0 = pd.Timestamp(datetime(lv.year, 1, 1))
        start_idx = hist.index.get_indexer([y0], method="nearest")[0]
        base = float(hist.iloc[start_idx])
        last = float(hist.iloc[-1])
        if base == 0.0:
            out[col] = np.nan
        else:
            out[col] = last / base - 1.0
    return pd.Series(out, dtype=float)


def last_vol_annualized(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized volatility at each column's last valid observation (sqrt(252))."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for col in df.columns:
        s = df[col].astype(float)
        lv = s.last_valid_index()
        if lv is None:
            out[col] = np.nan
            continue
        hist = s.loc[:lv].ffill()
        rets = hist.pct_change(fill_method=None).dropna()
        if len(rets) < window:
            out[col] = np.nan
            continue
        out[col] = float(rets.iloc[-window:].std(ddof=1) * np.sqrt(252.0))
    return pd.Series(out, dtype=float)
