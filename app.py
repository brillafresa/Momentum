# app.py
# -*- coding: utf-8 -*-
# KRW Momentum Radar - v2.8
# 
# 주요 기능:
# - FMS(Fast Momentum Score) 기반 모멘텀 분석
# - 다국가 시장 통합 분석 (미국, 한국, 일본)
# - 수익률-변동성 이동맵 (정적/애니메이션 모드)
# - 실시간 데이터 업데이트 및 시각화
# - 동적 관심종목 관리 및 신규 종목 탐색 엔진
#
# v2.8 개선사항:
# - 관심종목 영구 저장 기능
# - 신규 종목 탐색 엔진 (전체 미국 ETF 시장 스캔)
# - 진부한 종목 자동 편출 제안
# - 동적 포트폴리오 관리

import os
os.environ.setdefault("CURL_CFFI_DISABLE_CACHE", "1")  # curl_cffi sqlite 캐시 비활성화

import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf
from watchlist_utils import load_watchlist, save_watchlist, add_to_watchlist, remove_from_watchlist

warnings.filterwarnings("ignore", category=ResourceWarning)
KST = pytz.timezone("Asia/Seoul")

# ------------------------------
# 기본 유니버스 (관심종목 초기화용)
# ------------------------------
DEFAULT_USD_SYMBOLS = [
    'JEPI','IAU','JEPQ','VOO','NLY','PAVE','ITA','INDA','MCHI','EWG','GREK','GOOGL',
    'URA','GDX','ENFR','MDST','VNM','FXU','SPY','DIA','QQQ','EWQ','EWU','EWJ','EWH',
    'EWA','EWZ','EIDO','TUR','VT','VEA','VWO','BND','BNDX','GLD','SLV','DBC','CPER',
    'VNQ','VNQI','DBA','CORN','WEAT','USO','UNG','QUAL','VLUE','MTUM','USMV','IJR',
    'VB','TIP','XLK','XLF','XLV','SOXX','EWC','EWT','EPOL','EWW','BOTZ','ICLN','IBB',
    'QYLD','XYLD','REM','MORT','AGNC','TLTW','ULTY','BIZD','BKLN','SRLN','FLOT',
    'NOBL','SCHD','KSA','EZA','EDEN','JETS','SRVR','REMX','UUP','IVOL','PFIX','AOR',
    'NVDA'
]
DEFAULT_KRW_SYMBOLS = [
    '005930.KS','102110.KS','474220.KS','441680.KS','289480.KS',
    '166400.KS','276970.KS','482730.KS','486290.KS','480020.KS'
]
DEFAULT_JPY_SYMBOLS = ['2563.T']



def classify(sym):
    if sym.endswith(".KS"): return "KOR"
    if sym.endswith(".T"):  return "JPN"
    return "USA"

# ------------------------------
# 페이지/스타일
# ------------------------------
st.set_page_config(page_title="KRW Momentum Radar v2.8", page_icon="⚡", layout="wide")
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
USD_SYMBOLS = [s for s in st.session_state.watchlist if classify(s) == "USA"]
KRW_SYMBOLS = [s for s in st.session_state.watchlist if classify(s) == "KOR"]
JPY_SYMBOLS = [s for s in st.session_state.watchlist if classify(s) == "JPN"]

# ------------------------------
# 좌측 제어
# ------------------------------
st.sidebar.header("⚡ KRW Momentum Radar v2.8")

# 관심종목 관리 섹션
st.sidebar.subheader("📋 관심종목 관리")
st.sidebar.info(f"현재 관심종목: **{len(st.session_state.watchlist)}개**")

# 티커 추가
new_ticker = st.sidebar.text_input("티커 추가 (예: AAPL)", "").upper().strip()
if st.sidebar.button("➕ 추가"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, [new_ticker])
        st.sidebar.success(f"'{new_ticker}' 추가 완료!")
        st.rerun()
    elif new_ticker in st.session_state.watchlist:
        st.sidebar.warning(f"'{new_ticker}'는 이미 관심종목에 있습니다.")
    else:
        st.sidebar.error("유효한 티커를 입력해주세요.")

# 티커 삭제
if st.session_state.watchlist:
    tickers_to_remove = st.sidebar.multiselect(
        "삭제할 티커 선택",
        options=st.session_state.watchlist,
        key="remove_tickers"
    )
    if st.sidebar.button("🗑️ 선택 삭제"):
        if tickers_to_remove:
            st.session_state.watchlist = remove_from_watchlist(st.session_state.watchlist, tickers_to_remove)
            st.sidebar.warning(f"{len(tickers_to_remove)}개 종목 삭제 완료!")
            st.rerun()
        else:
            st.sidebar.error("삭제할 종목을 선택해주세요.")

# 기본 설정 섹션
st.sidebar.subheader("⚙️ 분석 설정")
period = st.sidebar.selectbox("차트 기간", ["3M","6M","1Y","2Y","5Y"], index=0)
rank_by = st.sidebar.selectbox("정렬 기준", ["ΔFMS(1D)","ΔFMS(5D)","FMS(현재)","1M 수익률"], index=2)  # 기본 FMS
TOP_N = st.sidebar.slider("Top N", 5, 60, 20, step=5)
use_log_scale = st.sidebar.checkbox("비교차트 로그 스케일", True)

with st.sidebar.expander("🔧 도구", expanded=False):
    if st.button("🗂️ 데이터 캐시 초기화"):
        st.cache_data.clear()
        st.success("캐시 초기화 완료 → 상단 Rerun 클릭")
    
    if st.button("🔄 관심종목 초기화"):
        default_symbols = DEFAULT_USD_SYMBOLS + DEFAULT_KRW_SYMBOLS + DEFAULT_JPY_SYMBOLS
        st.session_state.watchlist = default_symbols
        save_watchlist(default_symbols)
        st.success("관심종목이 기본값으로 초기화되었습니다!")
        st.rerun()

# ------------------------------
# 로깅
# ------------------------------
if "LOG" not in st.session_state:
    st.session_state["LOG"] = []
def log(msg):
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["LOG"].append(f"[{ts}] {msg}")
def warn_to_log(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as wlist:
        result = fn(*args, **kwargs)
        for w in wlist:
            st.session_state["LOG"].append(f"WARNING: {w.category.__name__}: {str(w.message)}")
        return result



# ------------------------------
# 다운로드/전처리
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

def align_bday_ffill(df):
    if df is None or len(df)==0: return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    return df.reindex(idx).ffill()

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
    counts = df.notna().sum(axis=1)
    threshold = int(np.ceil(coverage * df.shape[1]))
    valid = counts[counts >= threshold]
    if valid.empty:
        log("WARN harmonize_calendar: no date meets coverage; using full ffilled DF")
        return df
    last_common_date = valid.index[-1]
    df = df.loc[:last_common_date].ffill()
    log(f"Calendar harmonized to {last_common_date.date()} (>= {threshold}/{df.shape[1]} cols present)")
    return df

@st.cache_data(ttl=60*60*6, show_spinner=True)
def build_prices_krw(period_key="6M"):
    period_map = {"3M":"6mo","6M":"1y","1Y":"2y","2Y":"5y","5Y":"10y"}
    yf_period = period_map.get(period_key, "1y")
    interval = "1d"

    usdkrw, usdjpy, jpykrw, fx_missing = download_fx(yf_period, interval)
    usd_df, miss_us = download_prices(USD_SYMBOLS, yf_period, interval)
    krw_df, miss_kr = download_prices(KRW_SYMBOLS, yf_period, interval)
    jpy_df, miss_jp = download_prices(JPY_SYMBOLS, yf_period, interval)

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
# 지표/점수
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

def log_slope_annualized(s, window=30):
    s = s.dropna().tail(window)
    if len(s) < 3: return np.nan
    y = np.log(s.values); x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return slope * 252.0

def last_vol_annualized(df, window=20):
    rets = warn_to_log(df.ffill().pct_change, fill_method=None).dropna()
    if rets.empty: return pd.Series(index=df.columns, dtype=float)
    vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252.0)
    return vol

def rolling_max(s, window): return s.rolling(window).max()

def _mom_snapshot(prices_krw):
    r_1m = returns_pct(prices_krw, 21)
    slope30 = {}
    above_ema50 = {}
    breakout120 = {}
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            slope30[c]=np.nan; above_ema50[c]=np.nan; breakout120[c]=np.nan; continue
        slope30[c] = log_slope_annualized(s, 30)
        e50 = ema(s, 50)
        above_ema50[c] = (s.iloc[-1]/e50.iloc[-1]-1.0) if e50.iloc[-1] > 0 else np.nan
        hi120 = rolling_max(s, 120).iloc[-1]
        breakout120[c] = (s.iloc[-1]/hi120-1.0) if hi120 and hi120>0 else np.nan
    slope30 = pd.Series(slope30, name="Slope30(ann)")
    above_ema50 = pd.Series(above_ema50, name="AboveEMA50")
    breakout120 = pd.Series(breakout120, name="Breakout120")
    vol20 = last_vol_annualized(prices_krw, 20).rename("Vol20(ann)")

    def z(x):
        x = x.astype(float)
        m = np.nanmean(x); sd = np.nanstd(x)
        return (x-m)/sd if sd and not np.isnan(sd) else x*0.0

    FMS = (0.5*z(r_1m) + 0.3*z(slope30) + 0.2*z(above_ema50) + 0.1*z(breakout120)
           - 0.1*z(vol20.fillna(vol20.median())))
    snap = pd.concat([r_1m.rename("R_1M"), above_ema50, breakout120, slope30, vol20, FMS.rename("FMS")], axis=1)
    return snap

def momentum_now_and_delta(prices_krw):
    now = _mom_snapshot(prices_krw)
    d1 = _mom_snapshot(prices_krw.iloc[:-1]) if len(prices_krw)>1 else now*np.nan
    d5 = _mom_snapshot(prices_krw.iloc[:-5]) if len(prices_krw)>5 else now*np.nan
    df = now.copy()
    df["ΔFMS_1D"] = df["FMS"] - d1["FMS"]
    df["ΔFMS_5D"] = df["FMS"] - d5["FMS"]
    df["R_1W"] = returns_pct(prices_krw, 5)
    df["R_3M"] = returns_pct(prices_krw, 63)
    df["R_6M"] = returns_pct(prices_krw, 126)
    df["R_YTD"] = ytd_return(prices_krw)
    return df.sort_values("FMS", ascending=False)

# ------------------------------
# 신규 종목 탐색 엔진
# ------------------------------
@st.cache_data(ttl=60*60*2, show_spinner=False)  # 2시간 캐시
def calculate_fms_for_batch(symbols_batch, period_="1y", interval="1d"):
    """
    배치 단위로 FMS를 계산합니다.
    
    Args:
        symbols_batch (list): 계산할 심볼 목록
        period_ (str): 데이터 기간
        interval (str): 데이터 간격
        
    Returns:
        pd.DataFrame: FMS 계산 결과
    """
    if not symbols_batch:
        return pd.DataFrame()
    
    try:
        # 가격 데이터 다운로드
        prices_df, missing = download_prices(symbols_batch, period_, interval)
        if prices_df.empty:
            return pd.DataFrame()
        
        # FX 데이터 다운로드 (USD 심볼만)
        usd_symbols = [s for s in symbols_batch if classify(s) == "USA"]
        if usd_symbols:
            usdkrw, _, _, _ = download_fx(period_, interval)
            if not usdkrw.empty:
                usdkrw_matched = usdkrw.reindex(prices_df.index).ffill()
                usd_prices = prices_df[[s for s in usd_symbols if s in prices_df.columns]]
                if not usd_prices.empty:
                    prices_df[usd_prices.columns] = usd_prices.mul(usdkrw_matched, axis=0)
        
        # JPY 심볼 처리
        jpy_symbols = [s for s in symbols_batch if classify(s) == "JPN"]
        if jpy_symbols:
            _, _, jpykrw, _ = download_fx(period_, interval)
            if not jpykrw.empty:
                jpykrw_matched = jpykrw.reindex(prices_df.index).ffill()
                jpy_prices = prices_df[[s for s in jpy_symbols if s in prices_df.columns]]
                if not jpy_prices.empty:
                    prices_df[jpy_prices.columns] = jpy_prices.mul(jpykrw_matched, axis=0)
        
        # FMS 계산
        prices_krw = harmonize_calendar(prices_df, coverage=0.8)
        if prices_krw.empty:
            return pd.DataFrame()
        
        mom_df = momentum_now_and_delta(prices_krw)
        return mom_df
        
    except Exception as e:
        log(f"ERROR calculate_fms_for_batch: {e}")
        return pd.DataFrame()

def scan_market_for_new_opportunities():
    """
    전체 미국 ETF 시장을 스캔하여 새로운 기회를 찾습니다.
    """
    try:
        # ETF 목록 로드
        etf_df = pd.read_csv('us_etf_list.csv')
        master_list = etf_df['Symbol'].tolist()
        
        # 현재 관심종목에서 제외
        scan_targets = [s for s in master_list if s not in st.session_state.watchlist]
        
        if not scan_targets:
            return pd.DataFrame(), "스캔할 새로운 종목이 없습니다."
        
        log(f"시장 스캔 시작: {len(scan_targets)}개 종목")
        
        # 배치 처리
        all_results = []
        batch_size = 25  # API 제한 고려
        total_batches = (len(scan_targets) + batch_size - 1) // batch_size
        
        progress_bar = st.progress(0, "스캔 준비 중...")
        
        for i in range(0, len(scan_targets), batch_size):
            batch = scan_targets[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            progress = batch_num / total_batches
            progress_bar.progress(progress, f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개)")
            
            # FMS 계산
            batch_results = calculate_fms_for_batch(batch)
            if not batch_results.empty:
                all_results.append(batch_results)
            
            # API 제한 고려한 지연
            if batch_num < total_batches:
                import time
                time.sleep(0.5)
        
        progress_bar.progress(1.0, "스캔 완료!")
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=False)
            # FMS 기준으로 정렬하고 상위 30개만 반환
            top_performers = combined_results.sort_values('FMS', ascending=False).head(30)
            return top_performers, f"스캔 완료: {len(combined_results)}개 종목 중 상위 30개 발견"
        else:
            return pd.DataFrame(), "스캔 결과가 없습니다."
            
    except Exception as e:
        log(f"ERROR scan_market_for_new_opportunities: {e}")
        return pd.DataFrame(), f"스캔 중 오류 발생: {str(e)}"

# ------------------------------
# 데이터 로드 및 이름
# ------------------------------
with st.spinner("데이터 불러오는 중…"):
    prices_krw, miss = build_prices_krw(period)
if prices_krw.empty:
    st.error("가격 데이터를 불러오지 못했습니다.")
    st.stop()

with st.spinner("종목명(풀네임) 로딩 중…(최초 1회만 다소 지연)"):
    NAME_MAP = fetch_long_names(list(prices_krw.columns))

def display_name(sym):
    nm = NAME_MAP.get(sym, sym)
    return f"{nm} ({sym})" if nm and nm != sym else sym
def only_name(sym):
    nm = NAME_MAP.get(sym, sym)
    return nm if nm else sym

st.title("⚡ KRW Momentum Radar v2.8")

# FMS 설명
st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
<h4>📊 FMS (Fast Momentum Score)란?</h4>
<p><strong>FMS</strong>는 다차원 모멘텀 지표를 종합한 점수로, 다음과 같이 계산됩니다:</p>
<div style="background-color: white; padding: 10px; border-radius: 5px; font-family: monospace; margin: 10px 0;">
FMS = 0.5×Z(1M수익률) + 0.3×Z(30일기울기) + 0.2×Z(EMA50상대위치) + 0.1×Z(120일돌파) - 0.1×Z(20일변동성)
</div>
<p style="margin: 5px 0; font-size: 0.9em; color: #666;">
• <strong>Z()</strong>: Z-score 정규화 (평균 0, 표준편차 1)<br>
• <strong>가중치</strong>: 수익률(50%) > 기울기(30%) > EMA50위치(20%) > 돌파(10%) > 변동성(-10%)<br>
• <strong>높은 FMS</strong>: 강한 상승 모멘텀과 낮은 변동성을 의미
</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# 신규 종목 탐색 엔진
# ------------------------------
st.subheader("🚀 신규 모멘텀 종목 탐색")
st.markdown("전체 미국 ETF 시장을 스캔하여 새로운 투자 기회를 발굴합니다.")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.button('🔍 전체 미국 ETF 시장 스캔 실행', type="primary"):
        with st.spinner("전체 시장을 스캔 중입니다. 수 분 이상 소요될 수 있습니다..."):
            scan_results, scan_message = scan_market_for_new_opportunities()
            
            if not scan_results.empty:
                st.success(scan_message)
                
                # 결과 표시
                st.markdown("##### 📈 FMS 상위 신규 종목")
                
                # 표시용 데이터 준비
                display_results = scan_results.copy()
                display_results['추가'] = False
                
                # 숫자 포맷팅
                for col in ['R_1M', 'R_1W', 'R_3M', 'R_6M', 'R_YTD', 'AboveEMA50', 'Breakout120']:
                    if col in display_results.columns:
                        display_results[col] = (display_results[col] * 100).round(2)
                
                for col in ['FMS', 'ΔFMS_1D', 'ΔFMS_5D']:
                    if col in display_results.columns:
                        display_results[col] = display_results[col].round(2)
                
                if 'Slope30(ann)' in display_results.columns:
                    display_results['Slope30(ann)'] = display_results['Slope30(ann)'].round(3)
                
                # 데이터 에디터로 표시
                edited_df = st.data_editor(
                    display_results[['FMS', 'R_1M', 'R_1W', 'R_3M', 'AboveEMA50', 'Breakout120', '추가']],
                    key='discovery_editor',
                    use_container_width=True,
                    num_rows="dynamic"
                )
                
                # 선택된 항목을 관심종목에 추가
                selected_to_add = edited_df[edited_df['추가'] == True].index.tolist()
                if selected_to_add:
                    if st.button("✅ 선택 항목을 관심종목에 추가"):
                        new_symbols = [symbol for symbol in selected_to_add if symbol not in st.session_state.watchlist]
                        if new_symbols:
                            st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, new_symbols)
                            st.success(f"{len(new_symbols)}개 종목이 관심종목에 추가되었습니다!")
                            st.rerun()
                        else:
                            st.warning("선택한 종목들이 이미 관심종목에 있습니다.")
            else:
                st.warning(scan_message)

with col2:
    st.metric("현재 관심종목", len(st.session_state.watchlist))
    
with col3:
    # 관심종목 재평가 버튼
    if st.button('🔄 관심종목 재평가'):
        with st.spinner("관심종목 재평가 중..."):
            # 현재 관심종목의 FMS 계산
            watchlist_results = calculate_fms_for_batch(st.session_state.watchlist)
            
            if not watchlist_results.empty:
                # 제외 기준 설정 (FMS 하위 25%)
                fms_quantile_25 = watchlist_results['FMS'].quantile(0.25)
                stale_candidates = watchlist_results[watchlist_results['FMS'] < fms_quantile_25].copy()
                
                if not stale_candidates.empty:
                    st.markdown("##### 📉 관심종목 재평가 (제외 추천)")
                    
                    # 표시용 데이터 준비
                    display_stale = stale_candidates.copy()
                    display_stale['제거'] = False
                    
                    # 숫자 포맷팅
                    for col in ['R_1M', 'R_1W', 'R_3M', 'R_6M', 'R_YTD', 'AboveEMA50', 'Breakout120']:
                        if col in display_stale.columns:
                            display_stale[col] = (display_stale[col] * 100).round(2)
                    
                    for col in ['FMS', 'ΔFMS_1D', 'ΔFMS_5D']:
                        if col in display_stale.columns:
                            display_stale[col] = display_stale[col].round(2)
                    
                    # 데이터 에디터로 표시
                    edited_stale_df = st.data_editor(
                        display_stale[['FMS', 'R_1M', 'R_1W', 'R_3M', 'AboveEMA50', 'Breakout120', '제거']],
                        key='pruning_editor',
                        use_container_width=True
                    )
                    
                    # 선택된 항목을 관심종목에서 제거
                    selected_to_remove = edited_stale_df[edited_stale_df['제거'] == True].index.tolist()
                    if selected_to_remove:
                        if st.button("🗑️ 선택 항목을 관심종목에서 제거"):
                            st.session_state.watchlist = remove_from_watchlist(st.session_state.watchlist, selected_to_remove)
                            st.success(f"{len(selected_to_remove)}개 종목이 관심종목에서 제거되었습니다!")
                            st.rerun()
                else:
                    st.info("제외할 만한 저성과 종목이 없습니다.")
            else:
                st.error("관심종목 재평가 중 오류가 발생했습니다.")

# ------------------------------
# 모멘텀/가속 계산
# ------------------------------
with st.spinner("모멘텀/가속 계산 중…"):
    mom = momentum_now_and_delta(prices_krw)
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
if row["Breakout120"]>=-0.01: badges.append("120D 신고가 근접")
if row["ΔFMS_1D"]>0: badges.append("가속(1D+)")
if row["ΔFMS_5D"]>0: badges.append("가속(5D+)")
st.markdown(" ".join([f"<span class='badge'>{b}</span>" for b in badges]) or "<span class='small'>상태 배지 없음</span>", unsafe_allow_html=True)

# ------------------------------
# ⑤ 표
# ------------------------------
st.subheader("모멘텀 테이블 (가속/추세/수익률)")
disp = mom.copy()
for c in ["R_1W","R_1M","R_3M","R_6M","R_YTD","AboveEMA50","Breakout120"]:
    if c in disp: disp[c] = (disp[c]*100).round(2)
if "Slope30(ann)" in disp: disp["Slope30(ann)"] = disp["Slope30(ann)"].round(3)
for c in ["FMS","ΔFMS_1D","ΔFMS_5D"]:
    if c in disp: disp[c] = disp[c].round(2)
disp = disp.sort_values(rank_col if rank_col in disp.columns else "FMS", ascending=False)
st.dataframe(disp, use_container_width=True)

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
