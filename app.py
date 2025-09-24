# app.py
# -*- coding: utf-8 -*-
# KRW Momentum Radar - v3.0.7
# 
# 주요 기능:
# - FMS(Fast Momentum Score) 기반 모멘텀 분석
# - 다국가 시장 통합 분석 (미국, 한국, 일본)
# - 수익률-변동성 이동맵 (정적/애니메이션 모드)
# - 실시간 데이터 업데이트 및 시각화
# - 동적 관심종목 관리 및 신규 종목 탐색 엔진
#
# v3.0.4 개선사항:
# - FMS 전략 단일화: 안정 성장형 전략으로 통일하여 일관된 모멘텀 분석
# - 3M수익률 지표: 3개월(63거래일) 수익률을 통한 중기 모멘텀 평가
# - 변동성 가속도 지표: (20일 표준편차) / (120일 표준편차)로 급등 패턴 감지
# - 이벤트성 급등주 필터링: 변동성 가속도로 수직 폭등 종목 자동 제거
# - 안정적 추세 종목 발굴: 꾸준하고 지속 가능한 상승 추세 종목 우선 표시
# - 변동성 제어 강화: 변동성 페널티 4배 강화로 안정성 중시
#
# v3.0.3 개선사항:
# - UI/UX 개선: 페이징 컨트롤을 가로 배치로 변경 (⬅️➡️ 버튼 양쪽 끝 배치)
# - 사용자 경험: 관심종목 추가 시 불필요한 메시지 제거로 깔끔한 UI 제공
# - 페이징 안전성: 종목 추가 후 페이징이 깨지지 않도록 안전장치 추가
# - 유니버스 신선도 체크: Streamlit 웨이크업 시 파일 타임스탬프 변경 문제 해결
# - 재평가 UI 개선: 재평가 후 제거 제안 종목에서 휴지통 버튼 클릭 시 목록에서 즉시 제거
# - 버튼 비활성화: 오래 걸리는 작업 실행 중 관련 버튼들 자동 비활성화로 중복 실행 방지
# - 상태 표시: 작업 진행 중 버튼 텍스트 변경으로 현재 상태 명확히 표시
# - 유니버스 스크리닝 고도화: 추세 품질 중심 필터링으로 노이즈 종목 제거 및 안정적 모멘텀 종목 선별
# - 워치리스트 초기값 업데이트: 더 균형잡힌 글로벌 포트폴리오로 초기 관심종목 목록 개선
# - 스캔 완료 후 UI 정리: FMS 스캔 완료 시 스캔 중단 버튼이 자동으로 사라지도록 개선
# - 스캔 상태 표시 개선: 스캔 완료/중지 시 "스캔 중..." 표시가 정확히 사라지도록 개선
# - 변수명 개선: col1, col2, col3 → prev_col, spacer_col, next_col 등으로 명확화
# - 에러 처리: print 문을 주석으로 변경하여 콘솔 출력 정리

import os
os.environ.setdefault("CURL_CFFI_DISABLE_CACHE", "1")  # curl_cffi sqlite 캐시 비활성화

import warnings
import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import re
import streamlit as st
import yfinance as yf
from watchlist_utils import load_watchlist, save_watchlist, add_to_watchlist, remove_from_watchlist, export_watchlist_to_csv, import_watchlist_from_csv
from universe_utils import check_universe_file_freshness, update_universe_file, load_universe_file, save_scan_results, load_latest_scan_results, get_scan_results_info
from config import FMS_FORMULA, FMS_DESCRIPTION

warnings.filterwarnings("ignore", category=ResourceWarning)
KST = pytz.timezone("Asia/Seoul")

# ------------------------------
# 기본 유니버스 (관심종목 초기화용)
# ------------------------------
DEFAULT_USD_SYMBOLS = [
    'AAPL','ABBV','AMZN','ARKK','AVGO','BND','BRK-B','CAT','COST','CRM','CVX','DIA',
    'DIS','EEM','EFA','EWJ','GLD','GOOGL','HD','ICLN','INDA','IWM','IYT','JNJ','JPM',
    'KO','LLY','META','MRK','MSFT','NFLX','NVDA','PFE','PG','QQQ','SLV','SMH','SOXX',
    'SPY','T','TSLA','UNH','URA','V','VNQ','WMT','XLE','XLF','XLI','XLK','XLP','XLV','XLY'
]
DEFAULT_KRW_SYMBOLS = [
    '000660.KS','005930.KS'
]
DEFAULT_JPY_SYMBOLS = ['7203.T']



def classify(sym):
    # sym이 float이나 다른 타입일 경우를 대비해 str로 변환
    sym_str = str(sym)
    if sym_str.endswith(".KS"): return "KOR"
    if sym_str.endswith(".T"):  return "JPN"
    return "USA"

# ------------------------------
# 페이지/스타일
# ------------------------------
st.set_page_config(page_title="KRW Momentum Radar v3.0.7", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 0.8rem;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.75rem; margin-right:6px; background:#f1f3f5;}
.kpi {border:1px solid #eee; border-radius:16px; padding:10px 14px; box-shadow:0 1px 6px rgba(0,0,0,0.06);}
.small {font-size:0.8rem; color:#555;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 관심종목 초기화 (UI보다 먼저 실행)
# ------------------------------
if 'watchlist' not in st.session_state:
    default_symbols = DEFAULT_USD_SYMBOLS + DEFAULT_KRW_SYMBOLS + DEFAULT_JPY_SYMBOLS
    st.session_state.watchlist = load_watchlist(default_symbols)
    # 관심종목 초기화 완료

# 현재 관심종목을 기존 변수명으로 매핑 (하위 호환성)
USD_SYMBOLS = [str(s) for s in st.session_state.watchlist if classify(s) == "USA"]
KRW_SYMBOLS = [str(s) for s in st.session_state.watchlist if classify(s) == "KOR"]
JPY_SYMBOLS = [str(s) for s in st.session_state.watchlist if classify(s) == "JPN"]

# ------------------------------
# 데이터 다운로드 및 처리 함수들
# ------------------------------
def _extract_adj_close(df_chunk, tickers):
    if df_chunk is None or len(df_chunk)==0:
        return pd.DataFrame(columns=tickers, dtype=float)
    if isinstance(df_chunk.columns, pd.MultiIndex):
        if 'Adj Close' in df_chunk.columns.get_level_values(0):
            adj = df_chunk['Adj Close'].copy()
        elif 'Close' in df_chunk.columns.get_level_values(0):
            adj = df_chunk['Close'].copy()
        else:
            parts=[]
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

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_prices(tickers, period_="2y", interval="1d", chunk=25):
    frames=[]; missing=[]
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        try:
            raw = yf.download(part, period=period_, interval=interval, auto_adjust=False,
                              group_by='column', progress=False, threads=True)
            adj = _extract_adj_close(raw, part)
        except Exception as e:
            log(f"ERROR download chunk: {part[:3]}... -> {e}")
            adj = pd.DataFrame()
        if adj.empty or adj.isna().all().all():
            pframes=[]
            for t in part:
                try:
                    r = yf.download(t, period=period_, interval=interval, auto_adjust=False,
                                    group_by='column', progress=False, threads=False)
                    a = _extract_adj_close(r, [t])
                    pframes.append(a)
                except Exception as e:
                    log(f"ERROR download single: {t} -> {e}")
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
    if len(all_nan):
        log(f"DROP all-NaN columns: {list(all_nan)[:5]}{' ...' if len(all_nan)>5 else ''}")
    out = out.drop(columns=all_nan)
    return out, sorted(list(dict.fromkeys(list(missing)+list(all_nan))))

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_ohlc_prices(tickers, period_="2y", interval="1d", chunk=25):
    """
    거래 적합성 필터를 위한 OHLC 데이터를 다운로드합니다.
    
    Args:
        tickers (list): 다운로드할 티커 목록
        period_ (str): 데이터 기간
        interval (str): 데이터 간격
        chunk (int): 배치 크기
    
    Returns:
        tuple: (ohlc_data, missing_tickers)
    """
    frames=[]; missing=[]
    tickers = list(dict.fromkeys(tickers))
    
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        try:
            raw = yf.download(part, period=period_, interval=interval, auto_adjust=False,
                              group_by='column', progress=False, threads=True)
            
            if raw.empty:
                missing.extend(part)
                continue
                
            # OHLC 데이터 추출
            if isinstance(raw.columns, pd.MultiIndex):
                ohlc_data = {}
                for t in part:
                    if ('High', t) in raw.columns and ('Low', t) in raw.columns and ('Close', t) in raw.columns:
                        ohlc_data[t] = pd.DataFrame({
                            'High': raw[('High', t)],
                            'Low': raw[('Low', t)],
                            'Close': raw[('Close', t)]
                        })
                    else:
                        missing.append(t)
            else:
                # 단일 티커인 경우
                if len(part) == 1 and 'High' in raw.columns and 'Low' in raw.columns and 'Close' in raw.columns:
                    t = part[0]
                    ohlc_data[t] = pd.DataFrame({
                        'High': raw['High'],
                        'Low': raw['Low'],
                        'Close': raw['Close']
                    })
                else:
                    missing.extend(part)
                    continue
            
            if ohlc_data:
                frames.append(pd.concat(ohlc_data, axis=1))
                
        except Exception as e:
            log(f"ERROR download OHLC chunk: {part[:3]}... -> {e}")
            missing.extend(part)
    
    if not frames:
        return pd.DataFrame(), missing
    
    # 모든 OHLC 데이터를 하나의 DataFrame으로 합치기
    all_ohlc = pd.concat(frames, axis=1)
    all_ohlc = all_ohlc.loc[:, ~all_ohlc.columns.duplicated()].sort_index()
    
    return all_ohlc, sorted(list(dict.fromkeys(missing)))

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_fx(period_="2y", interval="1d"):
    fx_krw, miss1 = download_prices(["KRW=X"], period_, interval)
    fx_jpy, miss2 = download_prices(["JPY=X"], period_, interval)
    usdkrw = fx_krw.iloc[:,0].rename("USDKRW") if not fx_krw.empty else pd.Series(dtype=float, name="USDKRW")
    usdjpy = fx_jpy.iloc[:,0].rename("USDJPY") if not fx_jpy.empty else pd.Series(dtype=float, name="USDJPY")
    if not usdkrw.empty and not usdjpy.empty:
        start=min(usdkrw.index.min(), usdjpy.index.min())
        end=max(usdkrw.index.max(), usdjpy.index.max())
        idx = pd.date_range(start, end, freq='B')
        usdkrw = usdkrw.reindex(idx).ffill()
        usdjpy = usdjpy.reindex(idx).ffill()
        jpykrw = (usdkrw/usdjpy).rename("JPYKRW")
    else:
        jpykrw = pd.Series(dtype=float, name="JPYKRW")
    return usdkrw, usdjpy, jpykrw, (miss1+miss2)

def harmonize_calendar(df, coverage=0.9):
    if df.empty: return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    df = df.reindex(idx).ffill()
    # coverage 체크
    valid_ratio = df.count().div(len(df))
    keep_cols = valid_ratio[valid_ratio >= coverage].index
    return df[keep_cols] if len(keep_cols) > 0 else pd.DataFrame()

def align_bday_ffill(df):
    if df is None or len(df)==0: return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    return df.reindex(idx).ffill()

# ------------------------------
# 로깅 함수
# ------------------------------
if "LOG" not in st.session_state:
    st.session_state["LOG"] = []
def log(msg):
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["LOG"].append(f"[{ts}] {msg}")

# ------------------------------
# 유틸리티 함수들
# ------------------------------
def warn_to_log(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as wlist:
        result = fn(*args, **kwargs)
        for w in wlist:
            st.session_state["LOG"].append(f"WARNING: {w.category.__name__}: {str(w.message)}")
        return result

# ------------------------------
# 지표/점수 계산 함수들
# ------------------------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def returns_pct(df, n):
    if df.shape[0] <= n: 
        return pd.Series(index=df.columns, dtype=float)
    dff = df.ffill()
    r = warn_to_log(dff.pct_change, periods=n, fill_method=None).iloc[-1]
    return r

def ytd_return(df):
    if df.empty: return pd.Series(dtype=float)
    dff = df.ffill()
    last = dff.index[-1]
    y0 = pd.Timestamp(datetime(last.year, 1, 1))
    start_idx = dff.index.get_indexer([y0], method='nearest')[0]
    return dff.iloc[-1] / dff.iloc[start_idx] - 1.0

def last_vol_annualized(df, window=20):
    rets = warn_to_log(df.ffill().pct_change, fill_method=None).dropna()
    if rets.empty: return pd.Series(index=df.columns, dtype=float)
    vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252.0)
    return vol

def calculate_tradeability_filters(ohlc_data, symbols):
    """
    거래 적합성 실격 필터를 계산합니다.
    
    Args:
        ohlc_data (pd.DataFrame): OHLC 데이터 (MultiIndex columns)
        symbols (list): 심볼 목록
    
    Returns:
        dict: 각 심볼별 실격 여부 (True면 실격)
    """
    disqualification = {}
    
    for symbol in symbols:
        try:
            # OHLC 데이터 추출
            if isinstance(ohlc_data.columns, pd.MultiIndex):
                if ((symbol, 'High') in ohlc_data.columns and 
                    (symbol, 'Low') in ohlc_data.columns and 
                    (symbol, 'Close') in ohlc_data.columns):
                    high = ohlc_data[(symbol, 'High')].dropna()
                    low = ohlc_data[(symbol, 'Low')].dropna()
                    close = ohlc_data[(symbol, 'Close')].dropna()
                else:
                    disqualification[symbol] = True
                    continue
            else:
                # 단일 심볼인 경우
                if 'High' in ohlc_data.columns and 'Low' in ohlc_data.columns and 'Close' in ohlc_data.columns:
                    high = ohlc_data['High'].dropna()
                    low = ohlc_data['Low'].dropna()
                    close = ohlc_data['Close'].dropna()
                else:
                    disqualification[symbol] = True
                    continue
            
            if len(close) < 63:  # 최소 63거래일 데이터 필요
                disqualification[symbol] = True
                continue
            
            # 일일 변동폭 계산: (당일 고가 - 당일 저가) / 전일 종가
            daily_range = (high - low) / close.shift(1)
            
            # 일일 하방 리스크 계산: (당일 저가 / 전일 종가) - 1
            daily_downside_risk = (low / close.shift(1)) - 1
            
            # 필터 1: 치명적 변동성 필터 (63거래일 내 일일 변동폭 15% 초과)
            recent_63_days = daily_range.tail(63)
            extreme_volatility_days = recent_63_days[recent_63_days > 0.15]  # 원래 요청: 15%
            
            # 필터 2: 반복적 하방 리스크 필터 (20거래일 내 하방 리스크 -7% 미만 4일 이상)
            recent_20_days = daily_downside_risk.tail(20)
            severe_downside_days = recent_20_days[recent_20_days < -0.07]  # 원래 요청: -7%, 4일
            
            # 실격 조건 확인
            is_disqualified = (
                len(extreme_volatility_days) > 0 or  # 치명적 변동성 1일 이상 (15% 초과)
                len(severe_downside_days) >= 4      # 심각한 하방 리스크 4일 이상 (-7% 미만)
            )
            
            
            disqualification[symbol] = is_disqualified
            
        except Exception as e:
            log(f"거래 적합성 필터 계산 오류 {symbol}: {str(e)}")
            disqualification[symbol] = True
    
    return disqualification

def _mom_snapshot(prices_krw, reference_prices_krw=None, ohlc_data=None, symbols=None):
    """
    모멘텀 스냅샷을 계산합니다.
    
    Args:
        prices_krw (pd.DataFrame): KRW 환산 가격 데이터
        reference_prices_krw (pd.DataFrame, optional): Z-score 계산 기준이 되는 참조 데이터
        ohlc_data (pd.DataFrame, optional): OHLC 데이터 (거래 적합성 필터용)
        symbols (list, optional): 심볼 목록 (거래 적합성 필터용)
    
    Returns:
        pd.DataFrame: 모멘텀 지표들이 포함된 DataFrame
    """
    r_1m = returns_pct(prices_krw, 21)
    r_3m = returns_pct(prices_krw, 63)  # 3개월 수익률
    
    above_ema50 = {}
    
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            above_ema50[c] = np.nan
            continue
            
        e50 = ema(s, 50)
        above_ema50[c] = (s.iloc[-1]/e50.iloc[-1]-1.0) if e50.iloc[-1] > 0 else np.nan
    
    above_ema50 = pd.Series(above_ema50, name="AboveEMA50")
    vol20 = last_vol_annualized(prices_krw, 20).rename("Vol20(ann)")

    # 거래 적합성 실격 필터 적용
    disqualification_flags = {}
    if ohlc_data is not None and symbols is not None:
        disqualification_flags = calculate_tradeability_filters(ohlc_data, symbols)
    
    # Z-score 계산 기준 결정
    if reference_prices_krw is not None:
        # 참조 데이터가 있으면 참조 데이터로 Z-score 계산
        ref_r_1m = returns_pct(reference_prices_krw, 21)
        ref_r_3m = returns_pct(reference_prices_krw, 63)
        
        ref_above_ema50 = {}
        
        for c in reference_prices_krw.columns:
            s = reference_prices_krw[c].dropna()
            if s.empty:
                ref_above_ema50[c] = np.nan
                continue
                
            e50 = ema(s, 50)
            ref_above_ema50[c] = (s.iloc[-1]/e50.iloc[-1]-1.0) if e50.iloc[-1] > 0 else np.nan
        
        ref_above_ema50 = pd.Series(ref_above_ema50, name="AboveEMA50")
        ref_vol20 = last_vol_annualized(reference_prices_krw, 20).rename("Vol20(ann)")
        
        # 참조 데이터로 Z-score 계산
        def z_with_reference(x, ref_x):
            x = x.astype(float)
            ref_x = ref_x.astype(float)
            m = np.nanmean(ref_x); sd = np.nanstd(ref_x)
            return (x-m)/sd if sd and not np.isnan(sd) else x*0.0
        
        FMS = (0.4*z_with_reference(r_1m, ref_r_1m) + 
               0.3*z_with_reference(r_3m, ref_r_3m) + 
               0.2*z_with_reference(above_ema50, ref_above_ema50) 
               - 0.4*z_with_reference(vol20.fillna(vol20.median()), ref_vol20.fillna(ref_vol20.median())))
    else:
        # 기존 방식: 현재 데이터로 Z-score 계산
        def z(x):
            x = x.astype(float)
            m = np.nanmean(x); sd = np.nanstd(x)
            return (x-m)/sd if sd and not np.isnan(sd) else x*0.0

        FMS = (0.4*z(r_1m) + 0.3*z(r_3m) + 0.2*z(above_ema50) 
               - 0.4*z(vol20.fillna(vol20.median())))
    
    # 거래 적합성 실격 필터 적용: 실격된 종목은 FMS를 -999로 설정
    if disqualification_flags:
        for symbol in FMS.index:
            if symbol in disqualification_flags and disqualification_flags[symbol]:
                FMS[symbol] = -999.0
                log(f"거래 적합성 실격: {symbol} (FMS = -999)")
    
    # 결과 DataFrame 구성
    snap = pd.concat([r_1m.rename("R_1M"), r_3m.rename("R_3M"), above_ema50, 
                     vol20, FMS.rename("FMS")], axis=1)
    
    return snap

def momentum_now_and_delta(prices_krw, reference_prices_krw=None, ohlc_data=None, symbols=None):
    """
    모멘텀과 델타를 계산합니다.
    
    Args:
        prices_krw (pd.DataFrame): KRW 환산 가격 데이터
        reference_prices_krw (pd.DataFrame, optional): Z-score 계산 기준이 되는 참조 데이터
        ohlc_data (pd.DataFrame, optional): OHLC 데이터 (거래 적합성 필터용)
        symbols (list, optional): 심볼 목록 (거래 적합성 필터용)
    
    Returns:
        pd.DataFrame: 모멘텀 지표와 델타가 포함된 DataFrame
    """
    now = _mom_snapshot(prices_krw, reference_prices_krw, ohlc_data, symbols)
    d1 = _mom_snapshot(prices_krw.iloc[:-1], reference_prices_krw, ohlc_data, symbols) if len(prices_krw)>1 else now*np.nan
    d5 = _mom_snapshot(prices_krw.iloc[:-5], reference_prices_krw, ohlc_data, symbols) if len(prices_krw)>5 else now*np.nan
    df = now.copy()
    df["ΔFMS_1D"] = df["FMS"] - d1["FMS"]
    df["ΔFMS_5D"] = df["FMS"] - d5["FMS"]
    df["R_1W"] = returns_pct(prices_krw, 5)
    df["R_6M"] = returns_pct(prices_krw, 126)
    df["R_YTD"] = ytd_return(prices_krw)
    return df.sort_values("FMS", ascending=False)

# ------------------------------
# 유니버스 업데이트 함수들은 universe_utils.py로 이동
# ------------------------------

# ------------------------------
# 신규 종목 탐색 엔진 함수들
# ------------------------------
def calculate_fms_for_batch(symbols_batch, period_="1y", interval="1d", reference_prices_krw=None):
    """
    배치 단위로 FMS를 계산합니다.
    API 제한을 회피하기 위해 재시도 로직과 타임아웃을 포함합니다.
    거래 적합성 실격 필터가 기본적으로 적용됩니다.
    
    Args:
        symbols_batch (list): 계산할 심볼 목록
        period_ (str): 데이터 기간
        interval (str): 데이터 간격
        reference_prices_krw (pd.DataFrame, optional): Z-score 계산 기준이 되는 참조 데이터
        
    Returns:
        pd.DataFrame: FMS 계산 결과
    """
    if not symbols_batch:
        return pd.DataFrame()
    
    max_retries = 3
    retry_delay = 2  # 초
    
    for attempt in range(max_retries):
        try:
            # 가격 데이터 다운로드
            prices, missing = download_prices(symbols_batch, period_, interval)
            if prices.empty:
                if attempt < max_retries - 1:
                    log(f"배치 데이터 없음, 재시도 {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()
            
            # OHLC 데이터 다운로드 (거래 적합성 필터용)
            ohlc_data, ohlc_missing = download_ohlc_prices(symbols_batch, period_, interval)
            if ohlc_data.empty:
                ohlc_data = None
            
            # KRW 환산을 위한 FX 데이터 다운로드
            usd_symbols = [str(s) for s in symbols_batch if classify(s) == "USA"]
            if usd_symbols:
                usdkrw, _, _, _ = download_fx(period_, interval)
                if not usdkrw.empty:
                    usdkrw_matched = usdkrw.reindex(prices.index).ffill()
                    usd_prices = prices[[s for s in usd_symbols if s in prices.columns]]
                    if not usd_prices.empty:
                        prices[usd_prices.columns] = usd_prices.mul(usdkrw_matched, axis=0)
            
            # JPY 심볼 처리
            jpy_symbols = [str(s) for s in symbols_batch if classify(s) == "JPN"]
            if jpy_symbols:
                _, _, jpykrw, _ = download_fx(period_, interval)
                if not jpykrw.empty:
                    jpykrw_matched = jpykrw.reindex(prices.index).ffill()
                    jpy_prices = prices[[s for s in jpy_symbols if s in prices.columns]]
                    if not jpy_prices.empty:
                        prices[jpy_prices.columns] = jpy_prices.mul(jpykrw_matched, axis=0)
            
            # 캘린더 정규화
            prices_krw = harmonize_calendar(prices, coverage=0.9)
            if prices_krw.empty:
                if attempt < max_retries - 1:
                    log(f"캘린더 정규화 실패, 재시도 {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()
            
            # FMS 계산 (참조 데이터 및 거래 적합성 필터 사용)
            df = momentum_now_and_delta(prices_krw, reference_prices_krw, ohlc_data, symbols_batch)
            return df.sort_values("FMS", ascending=False)
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["rate limit", "too many requests", "429", "timeout"]):
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 지수 백오프
                    log(f"API 제한 감지, {wait_time}초 대기 후 재시도 {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    log(f"API 제한으로 인한 최종 실패: {str(e)}")
                    return pd.DataFrame()
            else:
                log(f"FMS 계산 중 오류 (시도 {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return pd.DataFrame()
    
    return pd.DataFrame()

def scan_market_for_new_opportunities():
    """
    유니버스 업데이트 후 FMS 스코어를 계산합니다.
    진행 상황을 실시간으로 모니터링할 수 있도록 개선되었습니다.
    
    Returns:
        tuple: (top_performers_df, message)
    """
    # 1단계: 유니버스 파일 신선도 확인 및 업데이트
    log("🔍 유니버스 파일 상태 확인 중...")
    
    is_fresh, last_modified, hours_since_update = check_universe_file_freshness()
    
    # 유니버스 업데이트 진행 상황 표시를 위한 컨테이너 생성
    universe_progress_container = st.empty()
    universe_status_container = st.empty()
    
    if is_fresh:
        universe_status_container.text(f"✅ 유니버스 파일이 최신입니다 (업데이트: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}, {hours_since_update:.1f}시간 전)")
        log(f"✅ 유니버스 파일이 최신입니다 (업데이트: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}, {hours_since_update:.1f}시간 전)")
    else:
        if last_modified:
            universe_status_container.text(f"⚠️ 유니버스 파일이 오래되었습니다 (업데이트: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}, {hours_since_update:.1f}시간 전)")
            log(f"⚠️ 유니버스 파일이 오래되었습니다 (업데이트: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}, {hours_since_update:.1f}시간 전)")
        else:
            universe_status_container.text("⚠️ 유니버스 파일이 없습니다")
            log("⚠️ 유니버스 파일이 없습니다")
        
        universe_status_container.text("🔄 유니버스 업데이트 시작...")
        log("🔄 유니버스 업데이트 시작...")
        
        # 진행률 콜백 함수 정의
        def progress_callback(progress, message):
            universe_progress_container.progress(progress, text=message)
            log(f"진행률 {progress*100:.0f}%: {message}")
        
        def status_callback(message):
            universe_status_container.text(message)
            log(message)
        
        # 유니버스 업데이트 실행 (진행 상황 표시 포함)
        update_success, update_message, symbol_count = update_universe_file(
            progress_callback=progress_callback,
            status_callback=status_callback
        )
        
        if not update_success:
            error_msg = f"유니버스 업데이트 실패: {update_message}"
            universe_status_container.text(f"❌ {error_msg}")
            log(error_msg)
            return pd.DataFrame(), error_msg
        
        universe_status_container.text(f"✅ {update_message}")
        log(f"✅ {update_message}")
    
    # 유니버스 업데이트 컨테이너 정리
    universe_progress_container.empty()
    universe_status_container.empty()
    
    # 2단계: 스크리닝된 유니버스 파일 로드
    try:
        success, master_list, load_message = load_universe_file()
        
        if not success:
            error_msg = f"유니버스 파일 로드 실패: {load_message}"
            log(error_msg)
            return pd.DataFrame(), error_msg
        
        log(f"📊 {load_message}")
        
    except Exception as e:
        error_msg = f"유니버스 파일 로딩 중 오류 발생: {e}"
        log(error_msg)
        return pd.DataFrame(), error_msg

    # 기존 관심종목 제외
    current_watchlist = st.session_state.get('watchlist', [])
    scan_targets = [s for s in master_list if s not in current_watchlist]
    
    if not scan_targets:
        return pd.DataFrame(), "알림: 현재 유망주 목록의 모든 종목이 이미 관심종목에 포함되어 있습니다."

    # 관심종목 데이터를 참조 데이터로 사용하기 위해 가져오기
    log("📊 관심종목 데이터를 참조 기준으로 로드 중...")
    try:
        reference_prices_krw, _ = build_prices_krw("1y", current_watchlist)
        if reference_prices_krw.empty:
            log("⚠️ 관심종목 데이터가 없어서 기존 방식으로 FMS 계산합니다.")
            reference_prices_krw = None
        else:
            log(f"✅ 관심종목 참조 데이터 로드 완료: {len(reference_prices_krw.columns)}개 종목")
    except Exception as e:
        log(f"⚠️ 관심종목 참조 데이터 로드 실패: {str(e)}, 기존 방식으로 FMS 계산합니다.")
        reference_prices_krw = None

    log(f"총 {len(scan_targets)}개 신규 종목을 스캔합니다...")
    
    # 진행 상황 모니터링을 위한 상태 초기화
    if 'scan_progress' not in st.session_state:
        st.session_state.scan_progress = {
            'total_batches': 0,
            'completed_batches': 0,
            'current_batch': 0,
            'successful_symbols': 0,
            'failed_symbols': 0,
            'start_time': None,
            'last_update': None
        }
    
    # 스캔 상태 초기화
    st.session_state.scan_progress.update({
        'total_batches': 0,
        'completed_batches': 0,
        'current_batch': 0,
        'successful_symbols': 0,
        'failed_symbols': 0,
        'start_time': datetime.now(KST),
        'last_update': datetime.now(KST)
    })
    
    # 배치 처리 설정 (API 제한 고려)
    batch_size = 20  # Yahoo Finance API 제한을 고려한 최적 배치 크기
    total_batches = (len(scan_targets) - 1) // batch_size + 1
    st.session_state.scan_progress['total_batches'] = total_batches
    
    log(f"배치 크기: {batch_size}개, 총 배치 수: {total_batches}개")
    
    all_results = []
    failed_batches = []
    
    # 진행 상황 표시를 위한 컨테이너 생성
    progress_container = st.empty()
    status_container = st.empty()
    
    try:
        for i in range(0, len(scan_targets), batch_size):
            batch_num = i // batch_size + 1
            batch = scan_targets[i:i+batch_size]
            
            # 진행 상황 업데이트
            st.session_state.scan_progress.update({
                'current_batch': batch_num,
                'last_update': datetime.now(KST)
            })
            
            # 진행률 계산
            progress = batch_num / total_batches
            elapsed_time = (datetime.now(KST) - st.session_state.scan_progress['start_time']).total_seconds()
            
            # 진행 상황 표시
            progress_container.progress(progress, text=f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 종목)")
            status_container.text(f"처리 중: {batch[0]} ~ {batch[-1]} | 경과시간: {elapsed_time:.0f}초")
            
            log(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 종목: {batch[0]} ~ {batch[-1]})")
            
            try:
                # 배치 처리 (참조 데이터 포함)
                batch_results = calculate_fms_for_batch(batch, reference_prices_krw=reference_prices_krw)
                
                if not batch_results.empty:
                    all_results.append(batch_results)
                    st.session_state.scan_progress['successful_symbols'] += len(batch_results)
                    log(f"✅ 배치 {batch_num} 완료: {len(batch_results)}개 종목 성공")
                else:
                    st.session_state.scan_progress['failed_symbols'] += len(batch)
                    failed_batches.append(batch_num)
                    log(f"⚠️ 배치 {batch_num} 실패: 데이터 없음")
                
            except Exception as e:
                st.session_state.scan_progress['failed_symbols'] += len(batch)
                failed_batches.append(batch_num)
                log(f"❌ 배치 {batch_num} 오류: {str(e)}")
                
                # yfinance API 제한 감지 시 잠시 대기
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    log("⏳ API 제한 감지, 5초 대기...")
                    time.sleep(5)
            
            st.session_state.scan_progress['completed_batches'] = batch_num
            
            # 배치 간 대기 (API 제한 방지)
            time.sleep(2)  # Yahoo Finance API 제한 방지를 위한 대기 시간
    
    except Exception as e:
        log(f"❌ 스캔 중 치명적 오류: {str(e)}")
        return pd.DataFrame(), f"스캔 중 오류 발생: {str(e)}"
    
    finally:
        # 진행 상황 컨테이너 정리
        progress_container.empty()
        status_container.empty()
    
    # 최종 결과 처리
    if not all_results:
        return pd.DataFrame(), "알림: 스캔 대상 종목에 대한 데이터를 가져오지 못했습니다."
    
    # 모든 결과 합치기
    combined_results = pd.concat(all_results)
    
    # FMS 순으로 정렬
    all_performers = combined_results.sort_values("FMS", ascending=False)
    
    # 최종 통계
    total_time = (datetime.now(KST) - st.session_state.scan_progress['start_time']).total_seconds()
    success_rate = (st.session_state.scan_progress['successful_symbols'] / 
                   (st.session_state.scan_progress['successful_symbols'] + st.session_state.scan_progress['failed_symbols'])) * 100
    
    scan_message = (f"✅ 스캔 완료! "
                   f"성공: {st.session_state.scan_progress['successful_symbols']}개, "
                   f"실패: {st.session_state.scan_progress['failed_symbols']}개, "
                   f"성공률: {success_rate:.1f}%, "
                   f"소요시간: {total_time:.0f}초")
    
    log(scan_message)
    
    # 실패한 배치가 있으면 경고
    if failed_batches:
        log(f"⚠️ 실패한 배치: {failed_batches[:10]}{'...' if len(failed_batches) > 10 else ''}")
    
    # FMS 2.0 이상인 종목만 저장
    fms_threshold = 2.0
    filtered_results = all_performers[all_performers['FMS'] >= fms_threshold]
    
    if not filtered_results.empty:
        # 스캔 결과 파일로 저장
        save_success, save_message, saved_count = save_scan_results(filtered_results, fms_threshold)
        if save_success:
            log(f"💾 {save_message}")
        else:
            log(f"⚠️ 저장 실패: {save_message}")
    
    # 스캔 완료 후 진행 상태 초기화
    if 'scan_progress' in st.session_state:
        del st.session_state.scan_progress
    
    return all_performers, scan_message

def get_dynamic_candidates(scan_results_df, current_watchlist, page_size=10, page_num=1):
    """
    현재 관심종목에 없는 종목들을 동적으로 반환합니다.
    페이징 처리를 통해 대량의 후보를 효율적으로 관리합니다.
    
    Args:
        scan_results_df (pd.DataFrame): 전체 스캔 결과
        current_watchlist (list): 현재 관심종목 목록
        page_size (int): 페이지당 표시할 종목 수
        page_num (int): 현재 페이지 번호 (1부터 시작)
    
    Returns:
        tuple: (candidates_df, total_pages, current_page)
    """
    if scan_results_df.empty:
        return pd.DataFrame(), 0, 1
    
    # 현재 관심종목에 없는 종목만 필터링
    available_candidates = scan_results_df[~scan_results_df.index.isin(current_watchlist)].copy()
    
    if available_candidates.empty:
        return pd.DataFrame(), 0, 1
    
    # FMS 순으로 정렬 (이미 정렬되어 있지만 확실히 하기 위해)
    available_candidates = available_candidates.sort_values("FMS", ascending=False)
    
    # 페이징 처리
    total_candidates = len(available_candidates)
    total_pages = (total_candidates - 1) // page_size + 1
    
    # 페이지 번호 유효성 검사
    page_num = max(1, min(page_num, total_pages))
    
    # 현재 페이지의 종목들 추출
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    current_page_candidates = available_candidates.iloc[start_idx:end_idx]
    
    return current_page_candidates, total_pages, page_num

# ------------------------------
# UI 관련 함수들
# ------------------------------
def get_button_states():
    """
    버튼 비활성화 상태를 반환합니다.
    
    Returns:
        tuple: (is_scanning, is_reassessing, button_disabled)
            - is_scanning (bool): 유니버스 스캔 진행 중 여부
            - is_reassessing (bool): 재평가 진행 중 여부
            - button_disabled (bool): 버튼 비활성화 여부
    """
    is_scanning = ('scan_progress' in st.session_state and 
                   st.session_state.scan_progress.get('total_batches', 0) > 0 and
                   st.session_state.scan_progress.get('completed_batches', 0) < st.session_state.scan_progress.get('total_batches', 0))
    is_reassessing = 'reassessing' in st.session_state and st.session_state.reassessing
    return is_scanning, is_reassessing, is_scanning or is_reassessing
def display_name(sym):
    """심볼을 표시용 이름으로 변환합니다."""
    if 'NAME_MAP' not in globals():
        return sym
    nm = NAME_MAP.get(sym, sym)
    return f"{nm} ({sym})" if nm and nm != sym else sym

def only_name(sym):
    """심볼의 이름만 반환합니다."""
    if 'NAME_MAP' not in globals():
        return sym
    nm = NAME_MAP.get(sym, sym)
    return nm if nm else sym

def update_candidates_after_addition(symbol_to_remove):
    """
    종목 추가 후 후보 리스트를 업데이트합니다.
    스캔 결과에서 해당 종목을 제거하여 UI에서 사라지도록 합니다.
    
    Args:
        symbol_to_remove (str): 제거할 종목 심볼
    
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # 현재 세션 상태의 스캔 결과에서 해당 종목 제거
        if 'scan_results' in st.session_state and st.session_state['scan_results'] is not None:
            current_results = st.session_state['scan_results']
            if symbol_to_remove in current_results.index:
                # 해당 종목 제거
                updated_results = current_results.drop(symbol_to_remove)
                st.session_state['scan_results'] = updated_results
                return True
        return False
    except Exception as e:
        log(f"후보 리스트 업데이트 중 오류: {str(e)}")
        return False

# ------------------------------
# 좌측 제어 - 깔끔하게 정리된 메뉴 구조
# ------------------------------

# 1. 분석 설정
with st.sidebar.expander("📊 분석 설정", expanded=True):
    period = st.selectbox("차트 기간", ["3M","6M","1Y","2Y","5Y"], index=0)
    
    rank_by = st.selectbox("정렬 기준", ["ΔFMS(1D)","ΔFMS(5D)","FMS(현재)","1M 수익률"], index=2)
    TOP_N = st.slider("Top N", 5, 60, 20, step=5)
    use_log_scale = st.checkbox("비교차트 로그 스케일", True)

# 2. 관심종목 관리
with st.sidebar.expander("📋 관심종목 관리", expanded=False):
    # 현재 관심종목 정보
    st.info(f"현재 관심종목: **{len(st.session_state.watchlist)}개**")
    
    # 파일 관리
    st.markdown("**📁 파일 관리**")
    
    # 다운로드
    if st.button("💾 관심종목 다운로드", help="현재 관심종목을 CSV 파일로 다운로드합니다."):
        csv_data = export_watchlist_to_csv(
            st.session_state.watchlist, 
            country_classifier=classify, 
            name_display=display_name
        )
        
        if csv_data:
            st.download_button(
                label="📥 CSV 파일 다운로드",
                data=csv_data,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ 다운로드 데이터 생성 중 오류가 발생했습니다.")
    
    # 업로드
    uploaded_watchlist = st.file_uploader(
        "📤 관심종목 업로드", 
        type=['csv'],
        help="CSV 파일을 업로드하여 관심종목을 교체합니다.",
        key="watchlist_uploader"
    )
    
    if uploaded_watchlist is not None and not st.session_state.get('upload_processed', False):
        try:
            csv_data = uploaded_watchlist.read().decode('utf-8-sig')
            new_symbols, message = import_watchlist_from_csv(csv_data)
            
            if new_symbols:
                st.session_state.watchlist = new_symbols
                st.cache_data.clear()
                st.success(message)
                st.session_state.upload_processed = True
                st.rerun()
            else:
                st.error(message)
                
        except Exception as e:
            st.error(f"❌ 파일 업로드 중 오류가 발생했습니다: {str(e)}")
    
    # 업로드 처리 완료 후 플래그 리셋
    if st.session_state.get('upload_processed', False):
        st.session_state.upload_processed = False
    
    # 구분선
    st.divider()
    
    # 재평가 기능
    st.markdown("**🔄 재평가**")
    is_scanning, is_reassessing, button_disabled = get_button_states()
    button_text = '⏳ 재평가 중...' if is_reassessing else '📊 재평가 실행'
    
    if st.button(button_text, disabled=button_disabled, help="현재 관심종목의 FMS를 재계산하여 저성과 종목을 식별합니다."):
        # 재평가 상태 설정
        st.session_state.reassessing = True
        
        with st.spinner("관심종목을 재평가 중입니다..."):
            watchlist_fms = calculate_fms_for_batch(st.session_state.watchlist, period_="1y")
            
            if not watchlist_fms.empty:
                fms_25th = watchlist_fms['FMS'].quantile(0.25)
                stale_candidates = watchlist_fms[watchlist_fms['FMS'] < fms_25th].sort_values('FMS')
                
                if not stale_candidates.empty:
                    st.warning(f"FMS 하위 25% 종목 ({len(stale_candidates)}개) 발견")
                    st.session_state['reassessment_results'] = stale_candidates
                else:
                    st.success("모든 관심종목이 양호한 상태입니다!")
                    st.session_state['reassessment_results'] = None
            else:
                st.error("재평가 데이터를 가져올 수 없습니다.")
                st.session_state['reassessment_results'] = None
        
        # 재평가 완료
        st.session_state.reassessing = False
    
    # 재평가 결과 표시
    if 'reassessment_results' in st.session_state and st.session_state['reassessment_results'] is not None:
        st.markdown("**📋 제거 제안 종목:**")
        stale_candidates = st.session_state['reassessment_results']
        
        for symbol in stale_candidates.index[:5]:
            col1, col2 = st.columns([3, 1])
            with col1:
                fms_score = stale_candidates.loc[symbol, 'FMS']
                st.write(f"**{symbol}** (FMS: {fms_score:.1f})")
            with col2:
                if st.button("🗑️", key=f"remove_{symbol}"):
                    # 관심종목에서 제거
                    st.session_state.watchlist = remove_from_watchlist(st.session_state.watchlist, [symbol])
                    
                    # 재평가 결과에서도 제거
                    if 'reassessment_results' in st.session_state and st.session_state['reassessment_results'] is not None:
                        if symbol in st.session_state['reassessment_results'].index:
                            st.session_state['reassessment_results'] = st.session_state['reassessment_results'].drop(symbol)
                    
                    st.cache_data.clear()
                    st.rerun()

# 3. 신규 종목 탐색
with st.sidebar.expander("🚀 신규 종목 탐색", expanded=False):
    
    # 스캔 실행
    st.markdown("**🔍 종목 스캔**")
    
    # 유니버스 파일 상태 표시
    is_fresh, last_modified, hours_since_update = check_universe_file_freshness()
    if is_fresh and last_modified:
        st.info(f"📊 유니버스 파일 최신 상태: {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({hours_since_update:.1f}시간 전)")
    elif last_modified:
        st.warning(f"⚠️ 유니버스 파일 오래됨: {last_modified.strftime('%Y-%m-%d %H:%M:%S')} ({hours_since_update:.1f}시간 전) - 업데이트 필요")
    else:
        st.error("❌ 유니버스 파일 없음 - 업데이트 필요")
    
    # 유니버스 파일 업로드
    uploaded_universe = st.file_uploader(
        "📤 유니버스 파일 업로드", 
        type=['csv'],
        help="CSV 파일을 업로드하여 유니버스를 교체합니다. (선택사항)",
        key="universe_uploader"
    )
    
    if uploaded_universe is not None:
        try:
            # 업로드된 파일을 메모리에서 읽기
            temp_universe = pd.read_csv(uploaded_universe)
            if 'Symbol' in temp_universe.columns:
                temp_universe.to_csv('screened_universe.csv', index=False)
                st.success(f"✅ 유니버스 파일이 업데이트되었습니다: {len(temp_universe)}개 종목")
                st.rerun()
            else:
                st.error("❌ CSV 파일에 'Symbol' 컬럼이 없습니다.")
        except Exception as e:
            st.error(f"❌ 파일 업로드 중 오류가 발생했습니다: {str(e)}")
    
    # 스캔 진행 상황 표시
    if 'scan_progress' in st.session_state and st.session_state.scan_progress['total_batches'] > 0:
        progress = st.session_state.scan_progress['completed_batches'] / st.session_state.scan_progress['total_batches']
        st.progress(progress, text=f"FMS 스캔 진행률: {st.session_state.scan_progress['completed_batches']}/{st.session_state.scan_progress['total_batches']} 배치")
        
        if st.session_state.scan_progress['start_time']:
            elapsed = (datetime.now(KST) - st.session_state.scan_progress['start_time']).total_seconds()
            st.caption(f"경과시간: {elapsed:.0f}초 | 성공: {st.session_state.scan_progress['successful_symbols']}개 | 실패: {st.session_state.scan_progress['failed_symbols']}개")
        
        # 스캔이 완료되지 않은 경우에만 중단 버튼 표시
        if st.session_state.scan_progress['completed_batches'] < st.session_state.scan_progress['total_batches']:
            if st.button('⏹️ 스캔 중단', help="진행 중인 스캔을 중단합니다."):
                if 'scan_progress' in st.session_state:
                    del st.session_state.scan_progress
                st.rerun()
    
    # FMS 임계값 설정
    fms_threshold = st.slider("FMS 임계값", 0.0, 5.0, 2.0, 0.1, help="이 값 이상의 FMS를 가진 종목만 표시됩니다.")
    
    # 저장된 스캔 결과 로드 버튼
    if st.button('📂 저장된 결과 로드', help="이전에 저장된 스캔 결과를 불러옵니다."):
        try:
            success, loaded_results, load_message = load_latest_scan_results(fms_threshold)
            if success and not loaded_results.empty:
                st.success(load_message)
                st.session_state['scan_results'] = loaded_results
                st.session_state['scan_page'] = 1  # 페이지 초기화
            else:
                st.warning(load_message)
        except Exception as e:
            st.error(f"결과 로드 중 오류: {str(e)}")
    
   
    # 스캔 실행 버튼
    is_scanning, is_reassessing, button_disabled = get_button_states()
    button_text = '⏳ 스캔 중...' if is_scanning else '🚀 유니버스 스캔'
    if st.button(button_text, type="primary", disabled=button_disabled, help="유니버스 업데이트 후 FMS 상위 종목을 탐색합니다. (실제 진행률은 콘솔에서 확인 가능)"):
        # 스캔 상태 초기화
        if 'scan_progress' in st.session_state:
            del st.session_state.scan_progress
        
        try:
            scan_results, scan_message = scan_market_for_new_opportunities()
            
            if not scan_results.empty:
                st.success(scan_message)
                # FMS 임계값 적용
                filtered_results = scan_results[scan_results['FMS'] >= fms_threshold]
                if not filtered_results.empty:
                    st.session_state['scan_results'] = filtered_results
                    st.session_state['scan_page'] = 1  # 페이지 초기화
                else:
                    st.warning(f"FMS {fms_threshold} 이상인 종목이 없습니다.")
                    st.session_state['scan_results'] = None
            else:
                st.error(f"스캔 결과가 없습니다: {scan_message}")
                st.session_state['scan_results'] = None
                
        except Exception as e:
            st.error(f"스캔 실행 중 오류가 발생했습니다: {str(e)}")
            st.session_state['scan_results'] = None
    
    # 스캔 결과 표시
    if 'scan_results' in st.session_state and st.session_state['scan_results'] is not None:
        st.markdown("**📋 발견된 종목:**")
        
        # 페이징 설정
        page_size = st.selectbox("페이지당 표시 종목 수", [5, 10, 15, 20, 25, 30], index=1)
        
        # 현재 페이지 초기화
        if 'scan_page' not in st.session_state:
            st.session_state['scan_page'] = 1
        
        # 동적 후보 리스트 생성
        current_watchlist = st.session_state.get('watchlist', [])
        candidates_df, total_pages, current_page = get_dynamic_candidates(
            st.session_state['scan_results'], 
            current_watchlist, 
            page_size, 
            st.session_state['scan_page']
        )
        
        if not candidates_df.empty:
            # 페이지 정보 표시
            st.info(f"📄 페이지 {current_page}/{total_pages}")
            
            # 페이징 컨트롤
            prev_col, spacer_col, next_col = st.columns([1, 2, 1])
            with prev_col:
                if st.button("⬅️", disabled=(current_page <= 1), key=f"prev_page_{current_page}"):
                    st.session_state['scan_page'] = max(1, current_page - 1)
                    st.rerun()
            with next_col:
                if st.button("➡️", disabled=(current_page >= total_pages), key=f"next_page_{current_page}"):
                    st.session_state['scan_page'] = min(total_pages, current_page + 1)
                    st.rerun()
            
            # 종목 목록 표시
            for idx, (symbol, row) in enumerate(candidates_df.iterrows()):
                info_col, button_col = st.columns([3, 1])
                with info_col:
                    fms_score = row['FMS']
                    fms_color = "🟢" if fms_score >= 3.0 else "🟡" if fms_score >= 2.0 else "🔴"
                    st.write(f"{fms_color} **{symbol}** (FMS: {fms_score:.1f})")
                with button_col:
                    if st.button("➕", key=f"add_{symbol}_{idx}"):
                        # 관심종목에 추가 (이미 있어도 중복 제거됨)
                        st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, [symbol])
                        
                        # 후보 리스트에서 제거 (성공/실패 관계없이)
                        update_candidates_after_addition(symbol)
                        
                        # 페이징 안전성 보장: 현재 페이지가 유효하지 않으면 첫 페이지로
                        if 'scan_results' in st.session_state and st.session_state['scan_results'] is not None:
                            remaining_candidates = st.session_state['scan_results'][~st.session_state['scan_results'].index.isin(st.session_state.watchlist)]
                            if not remaining_candidates.empty:
                                total_pages = (len(remaining_candidates) - 1) // page_size + 1
                                if st.session_state.get('scan_page', 1) > total_pages:
                                    st.session_state['scan_page'] = 1
                        
                        # 캐시 초기화 및 페이지 새로고침
                        st.cache_data.clear()
                        st.rerun()

        else:
            st.info("더 이상 추가할 수 있는 종목이 없습니다.")
            
            # 저장된 스캔 결과 파일 정보 표시
            scan_files_info = get_scan_results_info()
            if scan_files_info:
                st.markdown("**📁 저장된 스캔 결과 파일:**")
                for file_info in scan_files_info[:3]:  # 최근 3개만 표시
                    st.caption(f"📄 {file_info['formatted_time']} - {file_info['symbol_count']}개 종목")
    

# 4. 수동 관리 (간단한 추가/삭제)
with st.sidebar.expander("✏️ 수동 관리", expanded=False):
    # 티커 추가
    new_ticker = st.text_input("티커 추가 (예: AAPL)", "").upper().strip()
    if st.button("➕ 추가"):
        if new_ticker and new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, [new_ticker])
            st.success(f"'{new_ticker}' 추가됨")
            st.rerun()
        elif new_ticker in st.session_state.watchlist:
            st.warning(f"'{new_ticker}'는 이미 관심종목에 있습니다.")
        else:
            st.error("유효한 티커를 입력하세요.")

    # 티커 삭제
    if st.session_state.watchlist:
        ticker_to_remove = st.selectbox("삭제할 티커 선택", [""] + st.session_state.watchlist)
        if st.button("🗑️ 삭제"):
            if ticker_to_remove:
                st.session_state.watchlist = remove_from_watchlist(st.session_state.watchlist, [ticker_to_remove])
                st.success(f"'{ticker_to_remove}' 삭제됨")
                st.rerun()
            else:
                st.error("삭제할 종목을 선택해주세요.")

with st.sidebar.expander("🔧 도구 및 도움말", expanded=False):
    # FMS 설명
    st.markdown("**📊 FMS (Fast Momentum Score)**")
    
    st.markdown(f"""
    **FMS = {FMS_FORMULA}**
    
    • **추세 지속성**: 1M + 3M 수익률로 단기/중기 모멘텀 종합 평가
    • **안정성 중시**: 변동성 페널티 강화 (-0.4)로 급등 종목 필터링
    • **EMA 상대위치**: 50일 지수이동평균 대비 현재가 위치로 추세 강도 측정
    • **거래 적합성 필터**: 
      - 치명적 변동성: 63거래일 내 일일 변동폭 15% 초과 시 실격
      - 반복적 하방리스크: 20거래일 내 하방리스크 -7% 미만 4일 이상 시 실격
    • **목표**: 꾸준하고 지속 가능한 상승 추세 종목 발굴
    """)
    
    st.markdown("---")
    
    # 도구 버튼들
    if st.button("🗂️ 데이터 캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시 초기화 완료")
    
    is_scanning, is_reassessing, button_disabled = get_button_states()
    if st.button("🔄 관심종목 초기화", disabled=button_disabled):
        default_symbols = DEFAULT_USD_SYMBOLS + DEFAULT_KRW_SYMBOLS + DEFAULT_JPY_SYMBOLS
        st.session_state.watchlist = default_symbols
        save_watchlist(default_symbols)
        st.success("관심종목이 기본값으로 초기화되었습니다!")
        st.rerun()



@st.cache_data(ttl=60*60*6, show_spinner=True)
def build_prices_krw(period_key="6M", watchlist_symbols=None):
    period_map = {"3M":"6mo","6M":"1y","1Y":"2y","2Y":"5y","5Y":"10y"}
    yf_period = period_map.get(period_key, "1y")
    interval = "1d"

    # 관심종목 목록을 매개변수로 받아서 캐시 키에 포함
    if watchlist_symbols is None:
        watchlist_symbols = st.session_state.watchlist

    # 현재 관심종목에서 국가별로 분류
    usd_symbols = [str(s) for s in watchlist_symbols if classify(s) == "USA"]
    krw_symbols = [str(s) for s in watchlist_symbols if classify(s) == "KOR"]
    jpy_symbols = [str(s) for s in watchlist_symbols if classify(s) == "JPN"]

    usdkrw, usdjpy, jpykrw, fx_missing = download_fx(yf_period, interval)
    usd_df, miss_us = download_prices(usd_symbols, yf_period, interval)
    krw_df, miss_kr = download_prices(krw_symbols, yf_period, interval)
    jpy_df, miss_jp = download_prices(jpy_symbols, yf_period, interval)

    usd_df = align_bday_ffill(usd_df)
    krw_df = align_bday_ffill(krw_df)
    jpy_df = align_bday_ffill(jpy_df)
    usdkrw = align_bday_ffill(usdkrw.to_frame()).iloc[:,0] if not usdkrw.empty else usdkrw
    jpykrw = align_bday_ffill(jpykrw.to_frame()).iloc[:,0] if not jpykrw.empty else jpykrw

    frames=[]
    if not usd_df.empty and not usdkrw.empty:
        usdkrw_matched = usdkrw.reindex(usd_df.index).ffill()
        frames.append(usd_df.mul(usdkrw_matched, axis=0))
    if not krw_df.empty:
        frames.append(krw_df)
    if not jpy_df.empty and not jpykrw.empty:
        jpykrw_matched = jpykrw.reindex(jpy_df.index).ffill()
        frames.append(jpy_df.mul(jpykrw_matched, axis=0))

    if not frames:
        return pd.DataFrame(), {"fx_missing": fx_missing, "price_missing": miss_us+miss_kr+miss_jp}

    prices_krw = pd.concat(frames, axis=1).sort_index()
    prices_krw = prices_krw.loc[:, ~prices_krw.columns.duplicated()]
    prices_krw = harmonize_calendar(prices_krw, coverage=0.9)

    miss_dict = {
        "fx_missing": fx_missing,
        "price_missing": sorted(list(set(miss_us+miss_kr+miss_jp)))
    }
    last_row = prices_krw.iloc[-1]
    usa_cols = [c for c in prices_krw.columns if classify(c)=="USA"]
    na_usa = last_row[usa_cols].isna().sum()
    log(f"Final DF shape: {prices_krw.shape}; last row USA NaNs: {na_usa}/{len(usa_cols)}")
    return prices_krw, miss_dict

# ------------------------------
# 이름 캐시
# ------------------------------
@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_long_names(symbols):
    out = {}
    for s in symbols:
        name = None
        try:
            info = yf.Ticker(s).get_info()
            name = info.get("longName") or info.get("shortName")
        except Exception as e:
            log(f"INFO name fetch fail: {s} -> {e}")
        out[s] = name if name else s
    return out





# ------------------------------
# 데이터 로드 및 이름
# ------------------------------
with st.spinner("데이터 불러오는 중…"):
    prices_krw, miss = build_prices_krw(period, st.session_state.watchlist)
if prices_krw.empty:
    st.error("가격 데이터를 불러오지 못했습니다.")
    st.stop()

with st.spinner("종목명(풀네임) 로딩 중…(최초 1회만 다소 지연)"):
    NAME_MAP = fetch_long_names(list(prices_krw.columns))


st.title("⚡ KRW Momentum Radar v3.0.7")



# ------------------------------
# 모멘텀/가속 계산 (거래 적합성 필터 적용)
# ------------------------------
with st.spinner("모멘텀/가속 계산 중…"):
    # 관심종목의 OHLC 데이터 다운로드 (거래 적합성 필터용)
    watchlist_symbols = list(prices_krw.columns)
    period_map = {"3M":"6mo","6M":"1y","1Y":"2y","2Y":"5y","5Y":"10y"}
    ohlc_data, ohlc_missing = download_ohlc_prices(watchlist_symbols, period_map.get(period, "1y"), "1d")
    if ohlc_data.empty:
        ohlc_data = None
    
    mom = momentum_now_and_delta(prices_krw, ohlc_data=ohlc_data, symbols=watchlist_symbols)
rank_col = {"ΔFMS(1D)":"ΔFMS_1D","ΔFMS(5D)":"ΔFMS_5D","FMS(현재)":"FMS","1M 수익률":"R_1M"}[rank_by]
mom_ranked = mom.sort_values(rank_col, ascending=False)

# ------------------------------
# ① 가속 보드
# ------------------------------
st.subheader("가속 보드")
topN = mom_ranked.head(TOP_N)
bar = go.Figure([go.Bar(
    x=topN.index, y=topN[rank_col],
    customdata=np.array([only_name(s) for s in topN.index]),
    hovertemplate="%{customdata} (%{x})<br>"+rank_col+": %{y:.2f}<extra></extra>"
)])
bar.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_tickangle=-45, yaxis_title=rank_col)
st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

# ------------------------------
# ② 비교 차트 — Top N
# ------------------------------
st.subheader(f"비교 차트 — 상위 {TOP_N} (기준: {rank_col})")
sel_syms = list(topN.index)
df_view = prices_krw[sel_syms].dropna(how="all")
win_map={"3M":63,"6M":126,"1Y":252,"2Y":504,"5Y":1260}
win = win_map.get(period,126)
if df_view.shape[0]>win: df_view = df_view.iloc[-win:]
df_base = df_view/df_view.iloc[0]*100.0
fig_comp = go.Figure()
for c in df_base.columns:
    nm_only = only_name(c)
    fig_comp.add_trace(go.Scatter(
        x=df_base.index, y=df_base[c], mode="lines", name=c,
        customdata=np.array([nm_only]*len(df_base)),
        hovertemplate="%{customdata} ("+c+")<br>%{x|%Y-%m-%d}<br>Index:%{y:.2f}<extra></extra>"
    ))
fig_comp.update_layout(
    height=420, margin=dict(l=10,r=10,t=10,b=10),
    yaxis=dict(type="log" if use_log_scale else "linear", title="Rebased 100"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_comp, use_container_width=True)

# ==============================
# ③ 수익률–변동성 이동맵
# ==============================
st.subheader("수익률–변동성 이동맵 (최근 상태 → 어디서 왔는가)")

cc1, cc2, cc3, cc4 = st.columns([1.2,1.2,1.2,1.6])
with cc1:
    rv_window = st.selectbox("수익률/변동성 창(거래일)", [21, 42, 63], index=0, help="연율화: 252 기준")
with cc2:
    plot_n = st.selectbox("표시 종목 수", [10, 15, 20, 25, 30], index=2, help="상위 랭킹 기준으로 제한해 과밀도 완화")
with cc4:
    motion_mode = st.selectbox("모션(애니메이션)", ["끄기", "최근 10일", "최근 20일"], index=0,
                               help="프레임마다 현재 위치와 꼬리를 동시에 갱신")
with cc3:
    # 애니메이션 모드가 선택되면 꼬리 길이를 5로 자동 설정
    if motion_mode != "끄기":
        tail_days = st.selectbox("꼬리 길이(최근 n일 경로)", [0, 3, 5, 10], index=2, help="오늘 기준 과거 n거래일의 이동 경로를 점선으로 표시")
    else:
        tail_days = st.selectbox("꼬리 길이(최근 n일 경로)", [0, 3, 5, 10], index=0, help="오늘 기준 과거 n거래일의 이동 경로를 점선으로 표시")

def ann_vol(series, win):
    r = series.pct_change().dropna()
    if len(r) < max(5, int(win/3)): return np.nan
    r = r.tail(win);  return r.std() * np.sqrt(252.0)

def ann_cagr(series, win):
    s = series.dropna().tail(win+1)
    if len(s) < win+1: return np.nan
    start, end = s.iloc[-(win+1)], s.iloc[-1]
    if start <= 0 or end <= 0: return np.nan
    return (end/start)**(252.0/win) - 1.0

def scatter_points_for_date(df_prices, ref_loc, win):
    if ref_loc < 1: return pd.DataFrame()
    pts = {}
    for c in df_prices.columns:
        s = df_prices[c].iloc[:ref_loc+1]
        v = ann_vol(s, win); g = ann_cagr(s, win)
        pts[c] = (v, g)
    return pd.DataFrame(pts, index=["Vol","CAGR"]).T

plot_syms = list(mom_ranked.head(plot_n).index)
idx_all = prices_krw.index
today_loc = len(idx_all) - 1
yesterday_loc = max(0, today_loc - 1)
monthago_loc = max(0, today_loc - 21)

pts_today = scatter_points_for_date(prices_krw[plot_syms], today_loc, rv_window)
pts_yest  = scatter_points_for_date(prices_krw[plot_syms], yesterday_loc, rv_window)
pts_mago  = scatter_points_for_date(prices_krw[plot_syms], monthago_loc, rv_window)

# --- 꼬리 세그먼트 생성 유틸(연한→진한)
def add_tail_segments(fig, sym, tail_len, color="rgba(0,0,0,1)"):
    if tail_len <= 0: return
    # loc: today - tail_len ... today
    prev_xy = None
    for k in range(tail_len, -1, -1):
        loc = max(0, today_loc - k)
        p = scatter_points_for_date(prices_krw[[sym]], loc, rv_window)
        if p.empty or p.isna().any().any(): 
            prev_xy = None
            continue
        x = float(p.iloc[0,0])*100.0  # Vol%
        y = float(p.iloc[0,1])*100.0  # CAGR%
        if prev_xy is not None:
            age = (tail_len - k + 1)  # 최근일수록 작음
            alpha = max(0.15, min(0.5, age/(tail_len+1)))  # 0.15~0.5
            fig.add_trace(go.Scatter(
                x=[prev_xy[0], x], y=[prev_xy[1], y],
                mode="lines", line=dict(width=2, dash="dot"),
                showlegend=False, hoverinfo="skip", opacity=alpha, name=f"{sym}-tail"
            ))
        prev_xy = (x, y)

# --- 정적(3점 + 이동선 + 꼬리)
def make_static_scatter():
    fig = go.Figure()
    # 꼬리: 심볼별로 세그먼트 추가
    for c in plot_syms:
        add_tail_segments(fig, c, tail_days)

    # 꼬리 길이가 0일 때만 과거 시점들 표시
    if tail_days == 0:
        # 1M ago
        fig.add_trace(go.Scatter(
            x=pts_mago["Vol"]*100, y=pts_mago["CAGR"]*100, mode="markers",
            marker=dict(size=7, color="lightgray"),
            text=[display_name(s) for s in pts_mago.index],
            hovertemplate="%{text}<br>1M ago<br>Vol: %{x:.2f}% | CAGR: %{y:.2f}%<extra></extra>",
            name="1M ago", showlegend=True
        ))
        # Yesterday
        fig.add_trace(go.Scatter(
            x=pts_yest["Vol"]*100, y=pts_yest["CAGR"]*100, mode="markers",
            marker=dict(size=8, color="silver"),
            text=[display_name(s) for s in pts_yest.index],
            hovertemplate="%{text}<br>Yesterday<br>Vol: %{x:.2f}% | CAGR: %{y:.2f}%<extra></extra>",
            name="Yesterday", showlegend=True
        ))
    # Today
    fig.add_trace(go.Scatter(
        x=pts_today["Vol"]*100, y=pts_today["CAGR"]*100, mode="markers+text",
        marker=dict(size=9),
        text=[display_name(s) for s in pts_today.index],
        textposition="top center",
        hovertemplate="%{text}<br>Today<br>Vol: %{x:.2f}% | CAGR: %{y:.2f}%<extra></extra>",
        name="Today", showlegend=True
    ))
    # 꼬리 길이가 0일 때만 1M→Yest→Today 연결선 표시
    if tail_days == 0:
        for c in plot_syms:
            xs=[]; ys=[]
            for dfp in (pts_mago, pts_yest, pts_today):
                if c in dfp.index and not dfp.loc[c].isna().any():
                    xs.append(float(dfp.loc[c,"Vol"])*100)
                    ys.append(float(dfp.loc[c,"CAGR"])*100)
            if len(xs)>=2:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(width=1), opacity=0.35,
                                         name=f"path-{c}", showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        height=520, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="Volatility (ann, %)", yaxis_title="CAGR (ann, %)",
        hovermode="closest"
    )
    return fig

# --- 애니메이션: 프레임마다 현재 위치 + 각 종목 꼬리 동시 갱신
def make_motion_scatter(days):
    days = min(days, len(idx_all)-1)
    start_loc = max(0, today_loc - days)
    frames = []

    # 초기 프레임 데이터
    p0 = scatter_points_for_date(prices_krw[plot_syms], start_loc, rv_window)
    traces = []

    # 꼬리는 심볼별로 개별 trace를 사용(프레임마다 재계산)
    # 초기 꼬리
    tail_traces = []
    for c in plot_syms:
        # 빈 꼬리(초기)
        tail_traces.append(go.Scatter(x=[], y=[], mode="lines", line=dict(width=2, dash="dot"),
                                      showlegend=False, hoverinfo="skip", name=f"{c}-tail"))

    # 초기 포인트
    traces.extend(tail_traces)
    traces.append(go.Scatter(
        x=p0["Vol"]*100, y=p0["CAGR"]*100, mode="markers",
        marker=dict(size=9),
        text=[display_name(s) for s in p0.index],
        hovertemplate="%{text}<br>%{x:.2f}% / %{y:.2f}%<extra></extra>",
        name="Points", showlegend=False
    ))

    # 프레임 생성
    for loc in range(start_loc, today_loc+1):
        p = scatter_points_for_date(prices_krw[plot_syms], loc, rv_window)
        frame_data = []

        # 각 종목 꼬리 좌표 계산
        for c in plot_syms:
            xs=[]; ys=[]
            for k in range(tail_days, -1, -1):
                loc_k = max(0, loc - k)
                pk = scatter_points_for_date(prices_krw[[c]], loc_k, rv_window)
                if pk.empty or pk.isna().any().any(): 
                    continue
                xs.append(float(pk.iloc[0,0])*100.0)
                ys.append(float(pk.iloc[0,1])*100.0)
            frame_data.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2, dash="dot"),
                                         showlegend=False, hoverinfo="skip", name=f"{c}-tail"))
        # 포인트
        frame_data.append(go.Scatter(
            x=p["Vol"]*100, y=p["CAGR"]*100, mode="markers",
            marker=dict(size=9),
            text=[display_name(s) for s in p.index],
            hovertemplate="%{text}<br>%{x:.2f}% / %{y:.2f}%<extra></extra>",
            name="Points", showlegend=False
        ))

        frames.append(go.Frame(data=frame_data, name=str(prices_krw.index[loc].date())))

    fig = go.Figure(data=traces, frames=frames)
    # 애니메이션 모드에 따른 자동 재생 설정
    auto_play = motion_mode != "끄기"
    
    fig.update_layout(
        height=520, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="Volatility (ann, %)", yaxis_title="CAGR (ann, %)",
        updatemenus=[{
            "type": "buttons", "showactive": False,
            "buttons": [
                {"label":"▶ Play","method":"animate",
                 "args":[None, {"frame":{"duration":300, "redraw":True}, "fromcurrent":True, "transition":{"duration":0}}]},
                {"label":"⏸ Pause","method":"animate","args":[[None], {"frame":{"duration":0}, "mode":"immediate"}]}
            ]
        }],
        sliders=[{
            "steps":[{"label":f.name, "method":"animate", "args":[[f.name], {"mode":"immediate","frame":{"duration":0,"redraw":True}}]} for f in frames],
            "currentvalue":{"prefix":"Date: "}
        }]
    )
    
    # 애니메이션 모드가 선택되면 자동 재생
    if auto_play:
        fig.update_layout(
            updatemenus=[{
                "type": "buttons", "showactive": False,
                "buttons": [
                    {"label":"▶ Play","method":"animate",
                     "args":[None, {"frame":{"duration":300, "redraw":True}, "fromcurrent":True, "transition":{"duration":0}}]},
                    {"label":"⏸ Pause","method":"animate","args":[[None], {"frame":{"duration":0}, "mode":"immediate"}]}
                ]
            }]
        )
    return fig

# 렌더

if motion_mode == "끄기":
    fig_mv = make_static_scatter()
else:
    days = 10 if "10" in motion_mode else 20
    fig_mv = make_motion_scatter(days)
    
    # 애니메이션 모드일 때 자동 재생을 위한 JavaScript 추가
    if motion_mode != "끄기":
        st.markdown("""
        <script>
        setTimeout(function() {
            var plotlyDiv = document.querySelector('[data-testid="stPlotlyChart"] iframe');
            if (plotlyDiv) {
                plotlyDiv.onload = function() {
                    var plotlyFrame = plotlyDiv.contentWindow;
                    if (plotlyFrame && plotlyFrame.Plotly) {
                        setTimeout(function() {
                            plotlyFrame.Plotly.animate(plotlyFrame.document.querySelector('.plotly-graph-div'), null, {
                                frame: {duration: 300, redraw: true},
                                fromcurrent: true,
                                transition: {duration: 0}
                            });
                        }, 1000);
                    }
                };
            }
        }, 500);
        </script>
        """, unsafe_allow_html=True)

st.plotly_chart(fig_mv, use_container_width=True)
st.caption("설명: 각 점은 선택한 창(기본 21거래일)의 연율화 수익률(CAGR)·연율화 변동성입니다. "
           "‘꼬리 길이’는 오늘 기준 과거 n거래일 동안 좌표의 이동 경로를 점선으로 표시합니다. "
           "애니메이션 모드에서는 날짜가 바뀜에 따라 현재 위치와 꼬리가 함께 갱신됩니다.")

# ------------------------------
# ④ 세부 보기
# ------------------------------
st.subheader("세부 보기")
ordered_options = list(mom_ranked.index)
default_sym = (sel_syms[0] if sel_syms else ordered_options[0]) if ordered_options else prices_krw.columns[0]
detail_sym = st.selectbox("티커 선택", options=ordered_options,
                          index=ordered_options.index(default_sym) if default_sym in ordered_options else 0,
                          format_func=lambda s: display_name(s))
s = prices_krw[detail_sym].dropna()
e20,e50,e200 = ema(s,20), ema(s,50), ema(s,200)
fig_det = go.Figure()
fig_det.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="KRW"))
fig_det.add_trace(go.Scatter(x=e20.index, y=e20.values, mode="lines", name="EMA20"))
fig_det.add_trace(go.Scatter(x=e50.index, y=e50.values, mode="lines", name="EMA50"))
fig_det.add_trace(go.Scatter(x=e200.index, y=e200.values, mode="lines", name="EMA200"))
fig_det.update_layout(
    height=420, 
    margin=dict(l=10,r=10,t=10,b=10), 
    yaxis_title="KRW",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="center",
        x=0.5
    )
)
st.plotly_chart(fig_det, use_container_width=True, config={"displayModeBar": False})

roll_max = s.cummax()
dd = (s/roll_max - 1.0)*100.0
fig_dd = go.Figure([go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown(%)")])
fig_dd.update_layout(height=180, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="%")
st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

row = mom.loc[detail_sym]
badges=[]
if row["R_1M"]>0: badges.append("1M +")
if row["AboveEMA50"]>0: badges.append("EMA50 상회")

# FMS 지표 표시
if "R_3M" in row and row["R_3M"]>0: badges.append("3M +")

if row["ΔFMS_1D"]>0: badges.append("가속(1D+)")
if row["ΔFMS_5D"]>0: badges.append("가속(5D+)")
st.markdown(" ".join([f"<span class='badge'>{b}</span>" for b in badges]) or "<span class='small'>상태 배지 없음</span>", unsafe_allow_html=True)

# ------------------------------
# ⑤ 표 (컬럼 자동 재구성)
# ------------------------------
st.subheader("모멘텀 테이블 (가속/추세/수익률)")
disp = mom.copy()

# FMS 컬럼 표시
for c in ["R_1W","R_1M","R_3M","R_6M","R_YTD","AboveEMA50"]:
    if c in disp: disp[c] = (disp[c]*100).round(2)

for c in ["FMS","ΔFMS_1D","ΔFMS_5D"]:
    if c in disp: disp[c] = disp[c].round(2)

# 컬럼 자동 재구성: FMS 전략에 맞춰 동적 컬럼 순서 생성
def generate_dynamic_column_order(fms_formula, available_columns):
    """
    FMS 전략에 맞춰 동적 컬럼 순서를 생성합니다.
    
    Args:
        fms_formula (str): FMS 공식 문자열
        available_columns (list): 사용 가능한 컬럼 목록
    
    Returns:
        list: 재구성된 컬럼 순서
    """
    
    # 1. Symbol 컬럼 (가장 왼쪽)
    column_order = []
    if 'Symbol' in available_columns:
        column_order.append('Symbol')
    
    # 2. FMS 컬럼 (두 번째)
    if 'FMS' in available_columns:
        column_order.append('FMS')
    
    # 3. FMS 공식에서 사용된 변수들을 순서대로 추출
    fms_variables = []
    
    # 공식에서 변수명 추출 (정규식 사용)
    # 예: "0.4 * Z('1M수익률') + 0.3 * Z('3M수익률')" -> ['1M수익률', '3M수익률']
    variable_pattern = r"Z\('([^']+)'\)"
    matches = re.findall(variable_pattern, fms_formula)
    
    # 변수명을 실제 컬럼명으로 매핑
    variable_mapping = {
        '1M수익률': 'R_1M',
        '3M수익률': 'R_3M', 
        'EMA50상대위치': 'AboveEMA50',
        '20일변동성': 'Vol20(ann)'
    }
    
    for var_name in matches:
        if var_name in variable_mapping:
            col_name = variable_mapping[var_name]
            if col_name in available_columns and col_name not in column_order:
                fms_variables.append(col_name)
    
    # FMS 변수들을 공식에 나타난 순서대로 추가
    column_order.extend(fms_variables)
    
    # 4. 나머지 보조 변수들 추가
    remaining_columns = [col for col in available_columns if col not in column_order]
    
    # 보조 변수들을 우선순위에 따라 정렬
    priority_order = ['ΔFMS_1D', 'ΔFMS_5D', 'R_1W', 'R_6M', 'R_YTD']
    prioritized_remaining = []
    for priority_col in priority_order:
        if priority_col in remaining_columns:
            prioritized_remaining.append(priority_col)
            remaining_columns.remove(priority_col)
    
    # 나머지 컬럼들을 알파벳 순으로 정렬
    remaining_columns.sort()
    
    column_order.extend(prioritized_remaining)
    column_order.extend(remaining_columns)
    
    return column_order

# 현재 FMS 전략의 공식 가져오기
current_fms_formula = FMS_FORMULA

# 동적 컬럼 순서 생성
dynamic_column_order = generate_dynamic_column_order(current_fms_formula, list(disp.columns))

# 컬럼 순서 적용 (존재하는 컬럼만)
final_column_order = [col for col in dynamic_column_order if col in disp.columns]

# 데이터프레임 재구성
disp_reordered = disp[final_column_order]

# 정렬 적용
disp_reordered = disp_reordered.sort_values(rank_col if rank_col in disp_reordered.columns else "FMS", ascending=False)

st.dataframe(disp_reordered, use_container_width=True)

# ------------------------------
# ⑥ 디버그/진단
# ------------------------------
with st.expander("디버그 로그 / 진단 (복사해서 붙여넣기 가능)"):
    last_row = prices_krw.iloc[-1]
    usa_cols = [c for c in prices_krw.columns if classify(c)=="USA"]
    kor_cols = [c for c in prices_krw.columns if classify(c)=="KOR"]
    jpn_cols = [c for c in prices_krw.columns if classify(c)=="JPN"]
    diag = {
        "last_date": str(prices_krw.index[-1].date()),
        "cols_total": prices_krw.shape[1],
        "last_notna_total": int(last_row.notna().sum()),
        "last_notna_USA": int(last_row[usa_cols].notna().sum()),
        "last_notna_KOR": int(last_row[kor_cols].notna().sum()),
        "last_notna_JPN": int(last_row[jpn_cols].notna().sum()),
        "env_CURL_CFFI_DISABLE_CACHE": os.environ.get("CURL_CFFI_DISABLE_CACHE")
    }
    st.json(diag)
    st.text_area("LOG", value="\n".join(st.session_state["LOG"][-400:]), height=200)
