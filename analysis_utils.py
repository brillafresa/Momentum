# -*- coding: utf-8 -*-
"""
공통 분석 유틸리티: 데이터 다운로드/전처리, FMS 계산, 거래 적합성 필터
app.py 및 run_scan_batch.py가 이 모듈만 참조하도록 표준화합니다.
"""

from datetime import datetime
import time
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress


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


def _yf_download_with_retry(tickers_or_symbol, period_: str, interval: str, threads: bool = True, max_retries: int = 10, initial_sleep: float = 0.1) -> pd.DataFrame:
    """
    yfinance 다운로드에 대한 지수 백오프 재시도 래퍼.
    - Too Many Requests(레이트리밋)만 재시도 사유로 간주
    - PricesMissing(상장폐지/데이터 없음)은 재시도하지 않고 빈 결과 반환
    """
    delay = max(initial_sleep, 0.0)
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers_or_symbol, period=period_, interval=interval, auto_adjust=False,
                group_by='column', progress=False, threads=threads
            )
            return df
        except Exception as e:
            msg = str(e)
            # Prices missing → 데이터 없음: 재시도하지 않고 빈 프레임 반환
            if 'YFPricesMissingError' in msg or 'No data found, symbol may be delisted' in msg:
                return pd.DataFrame()
            # 레이트리밋만 재시도
            if ('Too Many Requests' in msg) or ('Rate limited' in msg) or ('429' in msg):
                last_exc = e
                if delay > 0:
                    time.sleep(delay)
                delay = delay * 2 if delay > 0 else 0.1
                continue
            # 그 외 예외는 즉시 실패
            raise
    # 재시도 한계 초과 → 마지막 예외가 있으면 그에 준하여 빈 프레임 반환
    # 배치 상에서 빈 결과는 실패로 간주되어 정상/실패 구분 가능
    return pd.DataFrame()


def download_prices(tickers: List[str], period_: str = '1y', interval: str = '1d', chunk: int = 25, initial_sleep: float = 0.1) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        raw = _yf_download_with_retry(part, period_, interval, threads=True, initial_sleep=initial_sleep)
        adj = _extract_adj_close(raw, part) if not raw.empty else pd.DataFrame()
        if adj.empty or adj.isna().all().all():
            pframes = []
            for t in part:
                r = _yf_download_with_retry(t, period_, interval, threads=False, initial_sleep=initial_sleep)
                if r.empty:
                    missing.append(t)
                    continue
                a = _extract_adj_close(r, [t])
                pframes.append(a)
            if pframes:
                frames.append(pd.concat(pframes, axis=1))
        else:
            frames.append(adj)
        
        # 배치 간 대기 (API 제한 방지)
        if i + chunk < len(tickers):  # 마지막 배치가 아니면 대기
            time.sleep(0.1)
    
    if not frames:
        return pd.DataFrame(), missing
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    all_nan = out.columns[out.isna().all()]
    out = out.drop(columns=all_nan)
    return out, sorted(list(dict.fromkeys(list(missing) + list(all_nan))))


def download_ohlc_prices(tickers: List[str], period_: str = '1y', interval: str = '1d', chunk: int = 25, initial_sleep: float = 0.1) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        raw = _yf_download_with_retry(part, period_, interval, threads=True, initial_sleep=initial_sleep)
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
        
        # 배치 간 대기 (API 제한 방지)
        if i + chunk < len(tickers):  # 마지막 배치가 아니면 대기
            time.sleep(0.1)
    
    if not frames:
        return pd.DataFrame(), missing
    all_ohlc = pd.concat(frames, axis=1)
    all_ohlc = all_ohlc.loc[:, ~all_ohlc.columns.duplicated()].sort_index()
    return all_ohlc, sorted(list(dict.fromkeys(missing)))


def download_fx(period_: str = '1y', interval: str = '1d', initial_sleep: float = 0.1) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fx_krw, _ = download_prices(['KRW=X'], period_, interval, initial_sleep=initial_sleep)
    fx_jpy, _ = download_prices(['JPY=X'], period_, interval, initial_sleep=initial_sleep)
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


def r_squared_3m(prices_krw: pd.DataFrame) -> pd.Series:
    """
    3개월(63거래일) 로그 수익률 기반 결정계수(R²)를 계산합니다.
    R²는 추세의 매끄러움을 평가하며, 높을수록 안정적인 우상향 추세를 의미합니다.
    
    Args:
        prices_krw (pd.DataFrame): KRW 환산 가격 데이터 (컬럼: 종목, 인덱스: 날짜)
    
    Returns:
        pd.Series: 각 종목별 R² 값 (0~1 사이, NaN 가능)
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
        
        # 로그 수익률 계산
        log_returns = np.log(recent / recent.iloc[0])
        
        # 선형 회귀를 위한 인덱스 (0부터 시작하는 정수)
        x = np.arange(len(log_returns))
        y = log_returns.values
        
        try:
            # 선형 회귀 수행
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r2 = r_value ** 2
            r2_dict[col] = r2
        except Exception:
            r2_dict[col] = np.nan
    
    return pd.Series(r2_dict, name='R2_3M')


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
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            
            # 당일 고가/저가가 0인 경우 전일 데이터로 대체
            # 조건: 당일 고가=0 AND 당일 저가=0 AND 전일 종가>0 AND 전일 고가>0 AND 전일 저가>0
            invalid_high_low = (high == 0) & (low == 0) & (prev_close > 0) & (prev_high > 0) & (prev_low > 0)
            high_fixed = high.copy()
            low_fixed = low.copy()
            high_fixed[invalid_high_low] = prev_high[invalid_high_low]
            low_fixed[invalid_high_low] = prev_low[invalid_high_low]
            
            true_range = pd.concat([
                high_fixed - low_fixed,
                (high_fixed - prev_close).abs(),
                (low_fixed - prev_close).abs()
            ], axis=1).max(axis=1, skipna=False)
            daily_true_range_vol = true_range / prev_close
            daily_downside_risk = (low_fixed / prev_close) - 1

            extreme_days = daily_true_range_vol.tail(63)
            extreme_days_filtered = extreme_days[extreme_days > 0.30]
            severe_days = daily_downside_risk.tail(20)
            severe_days_filtered = severe_days[severe_days < -0.07]

            reasons = []
            if len(extreme_days_filtered) > 0:
                reasons.append(f'치명적 변동성 ({len(extreme_days_filtered)}일 30% 초과)')
            if len(severe_days_filtered) >= 4:
                reasons.append(f'반복적 하방리스크 ({len(severe_days_filtered)}일 -7% 미만)')

            disqualification[symbol] = len(reasons) > 0
            filter_reasons[symbol] = '; '.join(reasons) if reasons else '정상'
        except Exception as e:
            disqualification[symbol] = True
            filter_reasons[symbol] = f'계산 오류: {e}'
    return disqualification, filter_reasons


def get_filter_debug_info(ohlc_data: pd.DataFrame, symbol: str) -> Dict:
    """거래 적합성 필터 계산을 위한 디버그 정보 수집"""
    debug_info = {'symbol': symbol, 'has_ohlc': False, 'data_points': 0, 'last_date': None, 'error': None}
    
    try:
        if ohlc_data is None or ohlc_data.empty:
            debug_info['error'] = 'OHLC 데이터 없음'
            return debug_info
            
        if isinstance(ohlc_data.columns, pd.MultiIndex):
            if ((symbol, 'High') in ohlc_data.columns and (symbol, 'Low') in ohlc_data.columns and (symbol, 'Close') in ohlc_data.columns):
                high = ohlc_data[(symbol, 'High')].dropna()
                low = ohlc_data[(symbol, 'Low')].dropna()
                close = ohlc_data[(symbol, 'Close')].dropna()
                debug_info['has_ohlc'] = True
            else:
                debug_info['error'] = 'OHLC 컬럼 없음 (MultiIndex)'
                return debug_info
        else:
            if all(c in ohlc_data.columns for c in ['High', 'Low', 'Close']):
                high = ohlc_data['High'].dropna()
                low = ohlc_data['Low'].dropna()
                close = ohlc_data['Close'].dropna()
                debug_info['has_ohlc'] = True
            else:
                debug_info['error'] = 'OHLC 컬럼 없음'
                return debug_info

        debug_info['data_points'] = len(close)
        if len(close) > 0:
            debug_info['last_date'] = str(close.index[-1])

        if len(close) < 63:
            debug_info['error'] = f'데이터 부족: {len(close)}일 (63일 필요)'
            return debug_info

        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        
        # 당일 고가/저가가 0인 경우 전일 데이터로 대체
        # 조건: 당일 고가=0 AND 당일 저가=0 AND 전일 종가>0 AND 전일 고가>0 AND 전일 저가>0
        invalid_high_low = (high == 0) & (low == 0) & (prev_close > 0) & (prev_high > 0) & (prev_low > 0)
        high_fixed = high.copy()
        low_fixed = low.copy()
        high_fixed[invalid_high_low] = prev_high[invalid_high_low]
        low_fixed[invalid_high_low] = prev_low[invalid_high_low]
        
        true_range = pd.concat([
            high_fixed - low_fixed,
            (high_fixed - prev_close).abs(),
            (low_fixed - prev_close).abs()
        ], axis=1).max(axis=1, skipna=False)
        daily_true_range_vol = true_range / prev_close
        daily_downside_risk = (low_fixed / prev_close) - 1

        extreme_days = daily_true_range_vol.tail(63)
        extreme_days_filtered = extreme_days[extreme_days > 0.30]
        severe_days = daily_downside_risk.tail(20)
        severe_days_filtered = severe_days[severe_days < -0.07]

        # 최근 데이터 상세 정보 수집
        if len(close) >= 2:
            last_date = close.index[-1]
            prev_date = close.index[-2] if len(close) >= 2 else None
            # 수정된 고가/저가 사용 (0인 경우 전일 데이터로 대체된 값)
            debug_info['recent_data'] = {
                'last_date': str(last_date),
                'prev_date': str(prev_date) if prev_date else None,
                'last_close': float(close.iloc[-1]) if len(close) > 0 else None,
                'prev_close': float(close.iloc[-2]) if len(close) >= 2 else None,
                'last_high': float(high_fixed.iloc[-1]) if len(high_fixed) > 0 else None,
                'last_low': float(low_fixed.iloc[-1]) if len(low_fixed) > 0 else None,
                'last_high_original': float(high.iloc[-1]) if len(high) > 0 else None,  # 원본 값도 기록
                'last_low_original': float(low.iloc[-1]) if len(low) > 0 else None,  # 원본 값도 기록
                'last_true_range_vol_pct': float(daily_true_range_vol.iloc[-1] * 100) if len(daily_true_range_vol) > 0 and not pd.isna(daily_true_range_vol.iloc[-1]) else None,
                'high_low_fixed': bool(invalid_high_low.iloc[-1]) if len(invalid_high_low) > 0 else False,  # 수정 여부 표시
            }
            
            # 마지막 날짜의 True Range 구성 요소 (수정된 값 사용)
            if len(close) >= 2:
                last_high_low = float(high_fixed.iloc[-1] - low_fixed.iloc[-1]) if len(high_fixed) > 0 and len(low_fixed) > 0 else None
                last_high_gap = float(abs(high_fixed.iloc[-1] - close.iloc[-2])) if len(high_fixed) > 0 and len(close) >= 2 else None
                last_low_gap = float(abs(low_fixed.iloc[-1] - close.iloc[-2])) if len(low_fixed) > 0 and len(close) >= 2 else None
                debug_info['recent_data']['true_range_components'] = {
                    'high_minus_low': last_high_low,
                    'abs_high_minus_prev_close': last_high_gap,
                    'abs_low_minus_prev_close': last_low_gap,
                    'true_range': float(true_range.iloc[-1]) if len(true_range) > 0 and not pd.isna(true_range.iloc[-1]) else None,
                }
            
            # 30% 초과인 날짜들 상세 정보
            if len(extreme_days_filtered) > 0:
                extreme_details = []
                for date_idx in extreme_days_filtered.index:
                    date_str = str(date_idx)
                    vol_pct = float(extreme_days_filtered.loc[date_idx]) * 100
                    if date_idx in close.index:
                        date_pos = list(close.index).index(date_idx)
                        if date_pos > 0:
                            extreme_details.append({
                                'date': date_str,
                                'vol_pct': round(vol_pct, 2),
                                'close': float(close.loc[date_idx]) if date_idx in close.index else None,
                                'prev_close': float(close.iloc[date_pos-1]) if date_pos > 0 else None,
                                'high': float(high.loc[date_idx]) if date_idx in high.index else None,
                                'low': float(low.loc[date_idx]) if date_idx in low.index else None,
                            })
                debug_info['extreme_days_detail'] = extreme_details
                debug_info['extreme_days_count'] = len(extreme_days_filtered)
            
            debug_info['severe_days_count'] = len(severe_days_filtered)
            if len(severe_days_filtered) >= 4:
                severe_details = []
                for date_idx in severe_days_filtered.index:
                    date_str = str(date_idx)
                    downside_pct = float(severe_days_filtered.loc[date_idx]) * 100
                    if date_idx in close.index:
                        date_pos = list(close.index).index(date_idx)
                        if date_pos > 0:
                            severe_details.append({
                                'date': date_str,
                                'downside_pct': round(downside_pct, 2),
                                'close': float(close.loc[date_idx]) if date_idx in close.index else None,
                                'prev_close': float(close.iloc[date_pos-1]) if date_pos > 0 else None,
                                'low': float(low.loc[date_idx]) if date_idx in low.index else None,
                            })
                debug_info['severe_days_detail'] = severe_details
    except Exception as e:
        debug_info['error'] = f'계산 오류: {str(e)}'
        debug_info['exception_type'] = type(e).__name__
    
    return debug_info


def _mom_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                  ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    r_1m = returns_pct(prices_krw, 21)
    r_3m = returns_pct(prices_krw, 63)
    r2_3m = r_squared_3m(prices_krw).rename('R2_3M')
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

    # 거래 적합성 필터 먼저 확인
    disqualification_flags: Dict[str, bool] = {}
    filter_reasons: Dict[str, str] = {}
    if ohlc_data is not None and symbols is not None:
        disqualification_flags, filter_reasons = calculate_tradeability_filters(ohlc_data, symbols)
    
    # 실격 종목 추출 (prices_krw에 있는 종목만)
    disqualified_symbols = set()
    if disqualification_flags:
        disqualified_symbols = {sym for sym, is_disq in disqualification_flags.items() 
                               if is_disq and sym in prices_krw.columns}
    
    # 참조 데이터가 없는 경우(current 데이터만 있는 경우) 실격 종목 제외하고 Z-score 계산
    # 참조 데이터가 있는 경우는 참조 데이터 기준으로만 normalize하므로 실격 종목 제외 불필요

    if reference_prices_krw is not None:
        ref_r_1m = returns_pct(reference_prices_krw, 21)
        ref_r_3m = returns_pct(reference_prices_krw, 63)
        ref_r2_3m = r_squared_3m(reference_prices_krw).rename('R2_3M')
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
            # 평균/표준편차는 항상 ref_x 전체 기준
            m = np.nanmean(ref_x); sd = np.nanstd(ref_x)
            result = (x - m) / sd if sd and not np.isnan(sd) else x * 0.0
            # 실격 종목은 추후 -999 적용될 예정이므로, Z-score 계산 후에도 결과 반환
            return result

        FMS = (0.2 * z_with_reference(r_1m, ref_r_1m)
               + 0.3 * z_with_reference(r_3m, ref_r_3m)
               + 0.3 * z_with_reference(r2_3m.fillna(0.0), ref_r2_3m.fillna(0.0))
               + 0.2 * z_with_reference(above_ema50, ref_above_ema50)
               - 0.2 * z_with_reference(vol20.fillna(vol20.median()), ref_vol20.fillna(ref_vol20.median())))
    else:
        def z(x: pd.Series, exclude_disq: bool = False) -> pd.Series:
            x = x.astype(float)
            # 실격 종목 제외하고 평균/표준편차 계산
            if exclude_disq and disqualified_symbols:
                valid_idx = [idx for idx in x.index if idx not in disqualified_symbols]
                valid_x = x.loc[valid_idx] if valid_idx else x
            else:
                valid_x = x
            m = np.nanmean(valid_x); sd = np.nanstd(valid_x)
            return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0

        FMS = (0.2 * z(r_1m, exclude_disq=True) 
               + 0.3 * z(r_3m, exclude_disq=True) 
               + 0.3 * z(r2_3m.fillna(0.0), exclude_disq=True)
               + 0.2 * z(above_ema50, exclude_disq=True) 
               - 0.2 * z(vol20.fillna(vol20.median()), exclude_disq=True))

    # 실격 종목은 FMS = -999 적용
    if disqualification_flags:
        for symbol in FMS.index:
            if symbol in disqualification_flags and disqualification_flags[symbol]:
                FMS[symbol] = -999.0

    filter_reasons_series = pd.Series(filter_reasons, name='Filter_Status').reindex(FMS.index, fill_value='정상')
    snap = pd.concat([r_1m.rename('R_1M'), r_3m.rename('R_3M'), r2_3m, above_ema50, vol20, FMS.rename('FMS'), filter_reasons_series], axis=1)
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

