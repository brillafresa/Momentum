# -*- coding: utf-8 -*-
"""
공통 분석 유틸리티: 데이터 다운로드/전처리, FMS 계산, 거래 적합성 필터

app.py 및 run_scan_batch.py가 이 모듈만 참조하도록 표준화합니다.

Harness note
------------
``compute_fms_snapshot`` / ``momentum_now_and_delta`` live in ``core.fms`` and are
re-exported here for transitional callers (inject fixtures; do not call
``download_*`` inside unit tests). This module remains the I/O + orchestration facade.

Batch CLI prints (``[Batch]`` / ``[yf]`` rate-limit / outer progress) are
**intentional operator feedback** for long scans — not temporary debug noise.
"""

import time
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import shared as yf_shared
from yfinance.exceptions import YFRateLimitError
# Pure scoring helpers live in core/; re-exported here for transitional callers.
from core.indicators import (  # noqa: F401
    ema,
    last_vol_annualized,
    returns_pct,
    r_squared_3m,
    ytd_return,
)
from core.tradeability import calculate_tradeability_filters  # noqa: F401
from core.fms import (  # noqa: F401
    _mom_snapshot,
    compute_fms_snapshot,
    momentum_now_and_delta,
    score_fms_from_feature_frame,
)

# Batch-friendly Yahoo download defaults (time over failed coverage)
YF_CHUNK_SIZE_DEFAULT = 15
YF_CHUNK_SLEEP_DEFAULT = 0.5
YF_CHUNK_SLEEP_BATCH = 1.25
YF_RATE_LIMIT_INITIAL_SLEEP = 2.0
YF_RATE_LIMIT_MAX_SLEEP = 120.0
YF_MAX_RETRIES_DEFAULT = 12
YF_MAX_RETRIES_BATCH = 40
YF_OUTER_BATCH_SIZE = 120


def classify(sym: str) -> str:
    s = str(sym)
    # 코스피/코스닥은 Yahoo Finance에서 suffix가 다를 수 있음
    # - KOSPI: .KS
    # - KOSDAQ: .KQ
    if s.endswith('.KS') or s.endswith('.KQ'):
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


def _is_rate_limit_message(msg: str) -> bool:
    """Return True if an error string indicates Yahoo rate limiting."""
    m = str(msg)
    return (
        ('Too Many Requests' in m)
        or ('Rate limited' in m)
        or ('YFRateLimitError' in m)
        or ('429' in m)
    )


def _pop_yf_shared_rate_limited() -> List[str]:
    """
    yfinance multi-download swallows YFRateLimitError into shared._ERRORS.
    Detect those tickers so our wrapper can retry (exception path alone is insufficient).
    """
    errors = getattr(yf_shared, '_ERRORS', None) or {}
    hit = [t for t, err in errors.items() if _is_rate_limit_message(err)]
    return hit


def _clear_yf_shared_state() -> None:
    """Reset yfinance download shared buffers between attempts."""
    for attr in ('_ERRORS', '_TRACEBACKS', '_DFS', '_ISINS'):
        bucket = getattr(yf_shared, attr, None)
        if isinstance(bucket, dict):
            bucket.clear()


def _yf_download_with_retry(
    tickers_or_symbol,
    period_: str,
    interval: str,
    threads: bool = False,
    max_retries: int = YF_MAX_RETRIES_DEFAULT,
    initial_sleep: float = YF_RATE_LIMIT_INITIAL_SLEEP,
) -> pd.DataFrame:
    """
    yfinance 다운로드에 대한 지수 백오프 재시도 래퍼.

    - Too Many Requests: shared._ERRORS 및 예외 모두 감지 후 재시도
    - PricesMissing(상장폐지/데이터 없음): 재시도하지 않고 빈 결과 반환
    - threads 기본 False: 동시 요청으로 인한 레이트리밋 증폭 완화
    """
    delay = max(float(initial_sleep), YF_RATE_LIMIT_INITIAL_SLEEP)
    for attempt in range(max_retries):
        _clear_yf_shared_state()
        try:
            df = yf.download(
                tickers_or_symbol, period=period_, interval=interval, auto_adjust=False,
                group_by='column', progress=False, threads=threads
            )
        except YFRateLimitError as e:
            print(f"[yf] Rate limited (attempt {attempt + 1}/{max_retries}); sleep {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, YF_RATE_LIMIT_MAX_SLEEP)
            continue
        except Exception as e:
            msg = str(e)
            if 'YFPricesMissingError' in msg or 'No data found, symbol may be delisted' in msg:
                return pd.DataFrame()
            if _is_rate_limit_message(msg):
                print(f"[yf] Rate limited via exception (attempt {attempt + 1}/{max_retries}); sleep {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, YF_RATE_LIMIT_MAX_SLEEP)
                continue
            raise

        rate_hit = _pop_yf_shared_rate_limited()
        if rate_hit:
            print(
                f"[yf] Rate limited on {len(rate_hit)} ticker(s) "
                f"(attempt {attempt + 1}/{max_retries}); sleep {delay:.1f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, YF_RATE_LIMIT_MAX_SLEEP)
            continue

        return df if df is not None else pd.DataFrame()

    print(f"[yf] Giving up after {max_retries} rate-limit retries: {tickers_or_symbol!r}")
    return pd.DataFrame()


def _extract_ohlc_frame(raw: pd.DataFrame, tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract High/Low/Close panel from a yfinance download frame."""
    missing: List[str] = []
    if raw is None or raw.empty:
        return pd.DataFrame(), list(tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        ohlc_map: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            if ('High', t) in raw.columns and ('Low', t) in raw.columns and ('Close', t) in raw.columns:
                ohlc_map[t] = pd.DataFrame({
                    'High': raw[('High', t)],
                    'Low': raw[('Low', t)],
                    'Close': raw[('Close', t)],
                })
            else:
                missing.append(t)
        if not ohlc_map:
            return pd.DataFrame(), list(tickers)
        return pd.concat(ohlc_map, axis=1), missing

    if len(tickers) == 1 and all(c in raw.columns for c in ['High', 'Low', 'Close']):
        t = tickers[0]
        return pd.concat({t: raw[['High', 'Low', 'Close']].copy()}, axis=1), []

    return pd.DataFrame(), list(tickers)


def download_prices(
    tickers: List[str],
    period_: str = '1y',
    interval: str = '1d',
    chunk: int = YF_CHUNK_SIZE_DEFAULT,
    initial_sleep: float = YF_RATE_LIMIT_INITIAL_SLEEP,
    chunk_sleep: float = YF_CHUNK_SLEEP_DEFAULT,
    max_retries: int = YF_MAX_RETRIES_DEFAULT,
    threads: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download Adj Close panels (dividend-adjusted) with rate-limit-aware chunking."""
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    tickers = list(dict.fromkeys(tickers))
    n_chunks = max(1, (len(tickers) + chunk - 1) // chunk) if tickers else 0

    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        chunk_idx = i // chunk + 1
        raw = _yf_download_with_retry(
            part, period_, interval, threads=threads,
            max_retries=max_retries, initial_sleep=initial_sleep,
        )
        adj = _extract_adj_close(raw, part) if not raw.empty else pd.DataFrame()
        if adj.empty or adj.isna().all().all():
            pframes = []
            for t in part:
                r = _yf_download_with_retry(
                    t, period_, interval, threads=False,
                    max_retries=max_retries, initial_sleep=initial_sleep,
                )
                if r.empty:
                    missing.append(t)
                    continue
                a = _extract_adj_close(r, [t])
                if a.empty or a.isna().all().all():
                    missing.append(t)
                    continue
                pframes.append(a)
                time.sleep(max(chunk_sleep * 0.25, 0.05))
            if pframes:
                frames.append(pd.concat(pframes, axis=1))
        else:
            frames.append(adj)
            # Partial NaN columns → per-symbol retry for those only
            bad = [c for c in part if c not in adj.columns or adj[c].isna().all()]
            for t in bad:
                r = _yf_download_with_retry(
                    t, period_, interval, threads=False,
                    max_retries=max_retries, initial_sleep=initial_sleep,
                )
                if r.empty:
                    missing.append(t)
                    continue
                a = _extract_adj_close(r, [t])
                if a.empty or a.isna().all().all():
                    missing.append(t)
                else:
                    frames.append(a)

        if chunk_idx % 10 == 0 or chunk_idx == n_chunks:
            print(f"[yf] prices chunk {chunk_idx}/{n_chunks} ({len(part)} symbols)")

        if i + chunk < len(tickers):
            time.sleep(max(chunk_sleep, 0.0))

    if not frames:
        return pd.DataFrame(), missing
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    all_nan = out.columns[out.isna().all()]
    out = out.drop(columns=all_nan)
    return out, sorted(list(dict.fromkeys(list(missing) + list(all_nan))))


def download_ohlc_prices(
    tickers: List[str],
    period_: str = '1y',
    interval: str = '1d',
    chunk: int = YF_CHUNK_SIZE_DEFAULT,
    initial_sleep: float = YF_RATE_LIMIT_INITIAL_SLEEP,
    chunk_sleep: float = YF_CHUNK_SLEEP_DEFAULT,
    max_retries: int = YF_MAX_RETRIES_DEFAULT,
    threads: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download High/Low/Close panels with rate-limit-aware chunking."""
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    tickers = list(dict.fromkeys(tickers))
    n_chunks = max(1, (len(tickers) + chunk - 1) // chunk) if tickers else 0

    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        chunk_idx = i // chunk + 1
        raw = _yf_download_with_retry(
            part, period_, interval, threads=threads,
            max_retries=max_retries, initial_sleep=initial_sleep,
        )
        ohlc, miss = _extract_ohlc_frame(raw, part)
        if ohlc.empty:
            for t in part:
                r = _yf_download_with_retry(
                    t, period_, interval, threads=False,
                    max_retries=max_retries, initial_sleep=initial_sleep,
                )
                one, one_miss = _extract_ohlc_frame(r, [t])
                if one.empty or one_miss:
                    missing.append(t)
                else:
                    frames.append(one)
                time.sleep(max(chunk_sleep * 0.25, 0.05))
        else:
            frames.append(ohlc)
            missing.extend(miss)
            for t in miss:
                r = _yf_download_with_retry(
                    t, period_, interval, threads=False,
                    max_retries=max_retries, initial_sleep=initial_sleep,
                )
                one, one_miss = _extract_ohlc_frame(r, [t])
                if one.empty or one_miss:
                    if t not in missing:
                        missing.append(t)
                else:
                    frames.append(one)
                    if t in missing:
                        missing.remove(t)

        if chunk_idx % 10 == 0 or chunk_idx == n_chunks:
            print(f"[yf] ohlc chunk {chunk_idx}/{n_chunks} ({len(part)} symbols)")

        if i + chunk < len(tickers):
            time.sleep(max(chunk_sleep, 0.0))

    if not frames:
        return pd.DataFrame(), sorted(list(dict.fromkeys(missing)))
    all_ohlc = pd.concat(frames, axis=1)
    all_ohlc = all_ohlc.loc[:, ~all_ohlc.columns.duplicated()].sort_index()
    return all_ohlc, sorted(list(dict.fromkeys(missing)))


def download_fx(
    period_: str = '1y',
    interval: str = '1d',
    initial_sleep: float = YF_RATE_LIMIT_INITIAL_SLEEP,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fx_krw, _ = download_prices(['KRW=X'], period_, interval, initial_sleep=initial_sleep, chunk_sleep=0.0)
    fx_jpy, _ = download_prices(['JPY=X'], period_, interval, initial_sleep=initial_sleep, chunk_sleep=0.0)
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


def _calculate_fms_for_symbol_chunk(
    symbols_batch: List[str],
    period_: str,
    interval: str,
    reference_prices_krw: Optional[pd.DataFrame],
    *,
    market_data,
) -> pd.DataFrame:
    """Score one outer chunk: fetch Adj Close + OHLC via the injected port, then run FMS."""
    if not symbols_batch:
        return pd.DataFrame()

    prices, miss_px = market_data.get_prices(symbols_batch, period_, interval)
    if miss_px:
        print(f"[Batch] prices missing/delisted in chunk: {len(miss_px)} (e.g. {miss_px[:5]})")
    if prices.empty:
        return pd.DataFrame()

    usd_symbols = [s for s in symbols_batch if classify(s) == 'USA']
    jpy_symbols = [s for s in symbols_batch if classify(s) == 'JPN']
    need_fx = bool(usd_symbols or jpy_symbols)
    usdkrw = jpykrw = None
    if need_fx:
        usdkrw, _, jpykrw = market_data.get_fx(period_, interval)

    if usd_symbols and usdkrw is not None and not usdkrw.empty:
        usdkrw_matched = usdkrw.reindex(prices.index).ffill()
        usd_prices = prices[[s for s in usd_symbols if s in prices.columns]]
        if not usd_prices.empty:
            prices[usd_prices.columns] = usd_prices.mul(usdkrw_matched, axis=0)

    if jpy_symbols and jpykrw is not None and not jpykrw.empty:
        jpykrw_matched = jpykrw.reindex(prices.index).ffill()
        jpy_prices = prices[[s for s in jpy_symbols if s in prices.columns]]
        if not jpy_prices.empty:
            prices[jpy_prices.columns] = jpy_prices.mul(jpykrw_matched, axis=0)

    prices_krw = harmonize_calendar(prices, coverage=0.9)
    if prices_krw.empty:
        return pd.DataFrame()

    scored_symbols = [s for s in symbols_batch if s in prices_krw.columns]
    ohlc_data, miss_ohlc = market_data.get_ohlc(scored_symbols, period_, interval)
    if miss_ohlc:
        print(f"[Batch] OHLC missing in chunk: {len(miss_ohlc)} (e.g. {miss_ohlc[:5]})")
    if ohlc_data.empty:
        ohlc_data = None

    return momentum_now_and_delta(prices_krw, reference_prices_krw, ohlc_data, scored_symbols)


def calculate_fms_for_batch(
    symbols_batch: List[str],
    period_: str = '1y',
    interval: str = '1d',
    reference_prices_krw: Optional[pd.DataFrame] = None,
    outer_batch_size: int = YF_OUTER_BATCH_SIZE,
    chunk: int = YF_CHUNK_SIZE_DEFAULT,
    chunk_sleep: float = YF_CHUNK_SLEEP_BATCH,
    max_retries: int = YF_MAX_RETRIES_BATCH,
    market_data=None,
) -> pd.DataFrame:
    """
    Universe FMS scan with outer batching and rate-limit-aware downloads.

    Prefers complete coverage over speed: slower chunk sleeps + many 429 retries.
    Delisted / no-data tickers are skipped (logged) and do not abort the run.

    ``market_data`` accepts a ``MarketDataPort`` (see ``adapters.market_data``)
    so tests/harness can inject fixture panels; defaults to ``YFinanceAdapter``.
    """
    symbols_batch = list(dict.fromkeys(symbols_batch))
    if not symbols_batch:
        return pd.DataFrame()

    if market_data is None:
        # Local import: adapters.market_data imports this module's download helpers.
        from adapters.market_data import YFinanceAdapter
        market_data = YFinanceAdapter(
            chunk=chunk, chunk_sleep=chunk_sleep, max_retries=max_retries,
            initial_sleep=YF_RATE_LIMIT_INITIAL_SLEEP, threads=False,
        )

    outer = max(int(outer_batch_size), 1)
    parts: List[pd.DataFrame] = []
    n_outer = (len(symbols_batch) + outer - 1) // outer
    t0 = time.time()

    for i in range(0, len(symbols_batch), outer):
        part = symbols_batch[i:i + outer]
        idx = i // outer + 1
        print(
            f"[Batch] FMS outer {idx}/{n_outer}: {len(part)} symbols "
            f"(elapsed {time.time() - t0:.0f}s)"
        )
        try:
            df_part = _calculate_fms_for_symbol_chunk(
                part, period_, interval, reference_prices_krw,
                market_data=market_data,
            )
        except Exception as e:
            print(f"[Batch] outer {idx} failed: {e}; continuing with remaining symbols")
            continue
        if not df_part.empty:
            parts.append(df_part)
            print(f"[Batch] outer {idx} scored {len(df_part)} rows")
        else:
            print(f"[Batch] outer {idx} produced no rows")

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep='first')]
    return out.sort_values('FMS', ascending=False)

