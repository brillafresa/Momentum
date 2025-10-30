# -*- coding: utf-8 -*-
"""
공통 분석 유틸리티: 데이터 다운로드/전처리, FMS 계산, 거래 적합성 필터
app.py 및 run_scan_batch.py가 이 모듈만 참조하도록 표준화합니다.
"""

from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf


def classify(sym: str) -> str:
    s = str(sym)
    if s.endswith('.KS'):
        return 'KOR'
    if s.endswith('.T'):
        return 'JPN'
    return 'USA'


def _extract_adj_close(df_chunk: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if df_chunk is None or len(df_chunk) == 0:
        return pd.DataFrame(columns=tickers, dtype=float)
    if isinstance(df_chunk.columns, pd.MultiIndex):
        if 'Adj Close' in df_chunk.columns.get_level_values(0):
            adj = df_chunk['Adj Close'].copy()
        elif 'Close' in df_chunk.columns.get_level_values(0):
            adj = df_chunk['Close'].copy()
        else:
            parts = []
            for t in tickers:
                try:
                    if ('Adj Close', t) in df_chunk.columns:
                        s = df_chunk[('Adj Close', t)].rename(t)
                    elif ('Close', t) in df_chunk.columns:
                        s = df_chunk[('Close', t)].rename(t)
                    else:
                        s = pd.Series(dtype=float, name=t)
                except Exception:
                    s = pd.Series(dtype=float, name=t)
                parts.append(s)
            adj = pd.concat(parts, axis=1)
    else:
        cols = df_chunk.columns
        if 'Adj Close' in cols:
            adj = df_chunk[['Adj Close']].copy(); adj.columns = tickers[:1]
        elif 'Close' in cols:
            adj = df_chunk[['Close']].copy(); adj.columns = tickers[:1]
        else:
            adj = df_chunk.copy()
            keep = [c for c in adj.columns if c in tickers]
            adj = adj[keep] if keep else pd.DataFrame(columns=tickers, dtype=float)
    adj = adj.loc[:, ~adj.columns.duplicated()]
    return adj


def download_prices(tickers: List[str], period_: str = '1y', interval: str = '1d', chunk: int = 25) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        try:
            raw = yf.download(part, period=period_, interval=interval, auto_adjust=False,
                              group_by='column', progress=False, threads=True)
            adj = _extract_adj_close(raw, part)
        except Exception:
            adj = pd.DataFrame()
        if adj.empty or adj.isna().all().all():
            pframes = []
            for t in part:
                try:
                    r = yf.download(t, period=period_, interval=interval, auto_adjust=False,
                                    group_by='column', progress=False, threads=False)
                    a = _extract_adj_close(r, [t])
                    pframes.append(a)
                except Exception:
                    missing.append(t)
            if pframes:
                frames.append(pd.concat(pframes, axis=1))
        else:
            frames.append(adj)
    if not frames:
        return pd.DataFrame(), missing
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    all_nan = out.columns[out.isna().all()]
    out = out.drop(columns=all_nan)
    return out, sorted(list(dict.fromkeys(list(missing) + list(all_nan))))


def download_ohlc_prices(tickers: List[str], period_: str = '1y', interval: str = '1d', chunk: int = 25) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        try:
            raw = yf.download(part, period=period_, interval=interval, auto_adjust=False,
                              group_by='column', progress=False, threads=True)
            if raw.empty:
                missing.extend(part)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                ohlc_map: Dict[str, pd.DataFrame] = {}
                for t in part:
                    if ('High', t) in raw.columns and ('Low', t) in raw.columns and ('Close', t) in raw.columns:
                        ohlc_map[t] = pd.DataFrame({'High': raw[('High', t)], 'Low': raw[('Low', t)], 'Close': raw[('Close', t)]})
                if ohlc_map:
                    frames.append(pd.concat(ohlc_map, axis=1))
            else:
                if len(part) == 1 and all(c in raw.columns for c in ['High', 'Low', 'Close']):
                    t = part[0]
                    frames.append(pd.concat({t: raw[['High', 'Low', 'Close']].copy()}, axis=1))
                else:
                    missing.extend(part)
        except Exception:
            missing.extend(part)
    if not frames:
        return pd.DataFrame(), missing
    all_ohlc = pd.concat(frames, axis=1)
    all_ohlc = all_ohlc.loc[:, ~all_ohlc.columns.duplicated()].sort_index()
    return all_ohlc, sorted(list(dict.fromkeys(missing)))


def download_fx(period_: str = '1y', interval: str = '1d') -> Tuple[pd.Series, pd.Series, pd.Series]:
    fx_krw, _ = download_prices(['KRW=X'], period_, interval)
    fx_jpy, _ = download_prices(['JPY=X'], period_, interval)
    usdkrw = fx_krw.iloc[:, 0].rename('USDKRW') if not fx_krw.empty else pd.Series(dtype=float, name='USDKRW')
    usdjpy = fx_jpy.iloc[:, 0].rename('USDJPY') if not fx_jpy.empty else pd.Series(dtype=float, name='USDJPY')
    if not usdkrw.empty and not usdjpy.empty:
        start = min(usdkrw.index.min(), usdjpy.index.min())
        end = max(usdkrw.index.max(), usdjpy.index.max())
        idx = pd.date_range(start, end, freq='B')
        usdkrw = usdkrw.reindex(idx).ffill()
        usdjpy = usdjpy.reindex(idx).ffill()
        jpykrw = (usdkrw / usdjpy).rename('JPYKRW')
    else:
        jpykrw = pd.Series(dtype=float, name='JPYKRW')
    return usdkrw, usdjpy, jpykrw


def harmonize_calendar(df: pd.DataFrame, coverage: float = 0.9) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    df = df.reindex(idx).ffill()
    valid_ratio = df.count().div(len(df))
    keep_cols = valid_ratio[valid_ratio >= coverage].index
    return df[keep_cols] if len(keep_cols) > 0 else pd.DataFrame()


def align_bday_ffill(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    return df.reindex(idx).ffill()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def returns_pct(df: pd.DataFrame, n: int) -> pd.Series:
    if df.shape[0] <= n:
        return pd.Series(index=df.columns, dtype=float)
    dff = df.ffill()
    r = dff.pct_change(periods=n, fill_method=None).iloc[-1]
    return r


def ytd_return(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    dff = df.ffill()
    last = dff.index[-1]
    y0 = pd.Timestamp(datetime(last.year, 1, 1))
    start_idx = dff.index.get_indexer([y0], method='nearest')[0]
    return dff.iloc[-1] / dff.iloc[start_idx] - 1.0


def last_vol_annualized(df: pd.DataFrame, window: int = 20) -> pd.Series:
    rets = df.ffill().pct_change(fill_method=None).dropna()
    if rets.empty:
        return pd.Series(index=df.columns, dtype=float)
    vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252.0)
    return vol


def calculate_tradeability_filters(ohlc_data: pd.DataFrame, symbols: List[str]) -> Tuple[Dict[str, bool], Dict[str, str]]:
    disqualification: Dict[str, bool] = {}
    filter_reasons: Dict[str, str] = {}
    for symbol in symbols:
        try:
            if isinstance(ohlc_data.columns, pd.MultiIndex):
                if ((symbol, 'High') in ohlc_data.columns and (symbol, 'Low') in ohlc_data.columns and (symbol, 'Close') in ohlc_data.columns):
                    high = ohlc_data[(symbol, 'High')].dropna()
                    low = ohlc_data[(symbol, 'Low')].dropna()
                    close = ohlc_data[(symbol, 'Close')].dropna()
                else:
                    disqualification[symbol] = True
                    filter_reasons[symbol] = 'OHLC 데이터 부족'
                    continue
            else:
                if all(c in ohlc_data.columns for c in ['High', 'Low', 'Close']):
                    high = ohlc_data['High'].dropna()
                    low = ohlc_data['Low'].dropna()
                    close = ohlc_data['Close'].dropna()
                else:
                    disqualification[symbol] = True
                    filter_reasons[symbol] = 'OHLC 데이터 부족'
                    continue

            if len(close) < 63:
                disqualification[symbol] = True
                filter_reasons[symbol] = '데이터 기간 부족 (63일 미만)'
                continue

            prev_close = close.shift(1)
            true_range = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1, skipna=False)
            daily_true_range_vol = true_range / prev_close
            daily_downside_risk = (low / prev_close) - 1

            extreme_days = daily_true_range_vol.tail(63)
            extreme_days = extreme_days[extreme_days > 0.30]
            severe_days = daily_downside_risk.tail(20)
            severe_days = severe_days[severe_days < -0.07]

            reasons = []
            if len(extreme_days) > 0:
                reasons.append(f'치명적 변동성 ({len(extreme_days)}일 30% 초과)')
            if len(severe_days) >= 4:
                reasons.append(f'반복적 하방리스크 ({len(severe_days)}일 -7% 미만)')

            disqualification[symbol] = len(reasons) > 0
            filter_reasons[symbol] = '; '.join(reasons) if reasons else '정상'
        except Exception as e:
            disqualification[symbol] = True
            filter_reasons[symbol] = f'계산 오류: {e}'
    return disqualification, filter_reasons


def _mom_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                  ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    r_1m = returns_pct(prices_krw, 21)
    r_3m = returns_pct(prices_krw, 63)
    above_ema50 = {}
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            above_ema50[c] = np.nan
            continue
        e50 = ema(s, 50)
        above_ema50[c] = (s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan
    above_ema50 = pd.Series(above_ema50, name='AboveEMA50')
    vol20 = last_vol_annualized(prices_krw, 20).rename('Vol20(ann)')

    disqualification_flags: Dict[str, bool] = {}
    filter_reasons: Dict[str, str] = {}
    if ohlc_data is not None and symbols is not None:
        disqualification_flags, filter_reasons = calculate_tradeability_filters(ohlc_data, symbols)

    if reference_prices_krw is not None:
        ref_r_1m = returns_pct(reference_prices_krw, 21)
        ref_r_3m = returns_pct(reference_prices_krw, 63)
        ref_above_ema50 = {}
        for c in reference_prices_krw.columns:
            s = reference_prices_krw[c].dropna()
            if s.empty:
                ref_above_ema50[c] = np.nan
                continue
            e50 = ema(s, 50)
            ref_above_ema50[c] = (s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan
        ref_above_ema50 = pd.Series(ref_above_ema50, name='AboveEMA50')
        ref_vol20 = last_vol_annualized(reference_prices_krw, 20).rename('Vol20(ann)')

        def z_with_reference(x: pd.Series, ref_x: pd.Series) -> pd.Series:
            x = x.astype(float); ref_x = ref_x.astype(float)
            m = np.nanmean(ref_x); sd = np.nanstd(ref_x)
            return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0

        FMS = (0.4 * z_with_reference(r_1m, ref_r_1m)
               + 0.3 * z_with_reference(r_3m, ref_r_3m)
               + 0.2 * z_with_reference(above_ema50, ref_above_ema50)
               - 0.4 * z_with_reference(vol20.fillna(vol20.median()), ref_vol20.fillna(ref_vol20.median())))
    else:
        def z(x: pd.Series) -> pd.Series:
            x = x.astype(float); m = np.nanmean(x); sd = np.nanstd(x)
            return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0

        FMS = (0.4 * z(r_1m) + 0.3 * z(r_3m) + 0.2 * z(above_ema50) - 0.4 * z(vol20.fillna(vol20.median())))

    if disqualification_flags:
        for symbol in FMS.index:
            if symbol in disqualification_flags and disqualification_flags[symbol]:
                FMS[symbol] = -999.0

    filter_reasons_series = pd.Series(filter_reasons, name='Filter_Status').reindex(FMS.index, fill_value='정상')
    snap = pd.concat([r_1m.rename('R_1M'), r_3m.rename('R_3M'), above_ema50, vol20, FMS.rename('FMS'), filter_reasons_series], axis=1)
    return snap


def momentum_now_and_delta(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                           ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    now = _mom_snapshot(prices_krw, reference_prices_krw, ohlc_data, symbols)
    d1 = _mom_snapshot(prices_krw.iloc[:-1], reference_prices_krw, ohlc_data, symbols) if len(prices_krw) > 1 else now * np.nan
    d5 = _mom_snapshot(prices_krw.iloc[:-5], reference_prices_krw, ohlc_data, symbols) if len(prices_krw) > 5 else now * np.nan
    df = now.copy()
    df['ΔFMS_1D'] = df['FMS'] - d1['FMS']
    df['ΔFMS_5D'] = df['FMS'] - d5['FMS']
    df['R_1W'] = returns_pct(prices_krw, 5)
    df['R_6M'] = returns_pct(prices_krw, 126)
    df['R_YTD'] = ytd_return(prices_krw)
    return df.sort_values('FMS', ascending=False)


def build_prices_krw_from_symbols(period_key: str, symbols: List[str]) -> pd.DataFrame:
    period_map = {'3M': '6mo', '6M': '1y', '1Y': '2y', '2Y': '5y', '5Y': '10y'}
    yf_period = period_map.get(period_key, '1y')
    interval = '1d'
    usd_symbols = [str(s) for s in symbols if classify(s) == 'USA']
    krw_symbols = [str(s) for s in symbols if classify(s) == 'KOR']
    jpy_symbols = [str(s) for s in symbols if classify(s) == 'JPN']
    usdkrw, _, jpykrw = download_fx(yf_period, interval)
    usd_df, _ = download_prices(usd_symbols, yf_period, interval)
    krw_df, _ = download_prices(krw_symbols, yf_period, interval)
    jpy_df, _ = download_prices(jpy_symbols, yf_period, interval)
    frames: List[pd.DataFrame] = []
    if not usd_df.empty and not usdkrw.empty:
        usdkrw_matched = usdkrw.reindex(usd_df.index).ffill()
        frames.append(usd_df.mul(usdkrw_matched, axis=0))
    if not krw_df.empty:
        frames.append(krw_df)
    if not jpy_df.empty and not jpykrw.empty:
        jpykrw_matched = jpykrw.reindex(jpy_df.index).ffill()
        frames.append(jpy_df.mul(jpykrw_matched, axis=0))
    if not frames:
        return pd.DataFrame()
    prices_krw = pd.concat(frames, axis=1).sort_index()
    prices_krw = prices_krw.loc[:, ~prices_krw.columns.duplicated()]
    prices_krw = harmonize_calendar(prices_krw, coverage=0.9)
    return prices_krw


def calculate_fms_for_batch(symbols_batch: List[str], period_: str = '1y', interval: str = '1d',
                            reference_prices_krw: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if not symbols_batch:
        return pd.DataFrame()

    prices, _ = download_prices(symbols_batch, period_, interval)
    if prices.empty:
        return pd.DataFrame()

    usd_symbols = [s for s in symbols_batch if classify(s) == 'USA']
    if usd_symbols:
        usdkrw, _, _ = download_fx(period_, interval)
        if not usdkrw.empty:
            usdkrw_matched = usdkrw.reindex(prices.index).ffill()
            usd_prices = prices[[s for s in usd_symbols if s in prices.columns]]
            if not usd_prices.empty:
                prices[usd_prices.columns] = usd_prices.mul(usdkrw_matched, axis=0)

    jpy_symbols = [s for s in symbols_batch if classify(s) == 'JPN']
    if jpy_symbols:
        _, _, jpykrw = download_fx(period_, interval)
        if not jpykrw.empty:
            jpykrw_matched = jpykrw.reindex(prices.index).ffill()
            jpy_prices = prices[[s for s in jpy_symbols if s in prices.columns]]
            if not jpy_prices.empty:
                prices[jpy_prices.columns] = jpy_prices.mul(jpykrw_matched, axis=0)

    prices_krw = harmonize_calendar(prices, coverage=0.9)
    if prices_krw.empty:
        return pd.DataFrame()

    # 거래 적합성 필터 위한 OHLC
    ohlc_data, _ = download_ohlc_prices(symbols_batch, period_, interval)
    if ohlc_data.empty:
        ohlc_data = None

    df = momentum_now_and_delta(prices_krw, reference_prices_krw, ohlc_data, symbols_batch)
    return df.sort_values('FMS', ascending=False)


