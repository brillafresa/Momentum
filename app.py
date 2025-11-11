# app.py
# -*- coding: utf-8 -*-
# KRW Momentum Radar - v3.6.0
# 
# 주요 기능:
# - FMS(Fast Momentum Score) 기반 모멘텀 분석
# - 다국가 시장 통합 분석 (미국, 한국, 일본)
# - 수익률-변동성 이동맵 (정적/애니메이션 모드)
# - 실시간 데이터 업데이트 및 시각화
# - 동적 관심종목 관리 및 배치 스캔 결과 확인
# - True Range 기반 거래 적합성 필터

import os
os.environ.setdefault("CURL_CFFI_DISABLE_CACHE", "1")  # curl_cffi sqlite 캐시 비활성화

import warnings
from datetime import datetime
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import re
import streamlit as st
import yfinance as yf
from watchlist_utils import load_watchlist, save_watchlist, add_to_watchlist, remove_from_watchlist, export_watchlist_to_csv, import_watchlist_from_csv
from config import FMS_FORMULA
from analysis_utils import (
    calculate_tradeability_filters as _au_trade_filters,
    momentum_now_and_delta as _au_momentum_now_and_delta,
    calculate_fms_for_batch as _au_calculate_fms_for_batch,
)

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
st.set_page_config(page_title="KRW Momentum Radar v3.6.0", page_icon="⚡", layout="wide")
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

# ------------------------------
# 중앙화된 FMS/필터 로직으로 오버라이드
# ------------------------------
calculate_tradeability_filters = _au_trade_filters
momentum_now_and_delta = _au_momentum_now_and_delta

def calculate_fms_for_batch(symbols_batch, period_="1y", interval="1d", reference_prices_krw=None):
    return _au_calculate_fms_for_batch(symbols_batch, period_, interval, reference_prices_krw)

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
    is_scanning = False  # 배치 스캔은 별도 프로세스로 실행됨
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
    # --- [신규] 배치 스캔 관리 ---
    st.markdown("**📦 배치 스캔 관리**")
    import subprocess
    import psutil
    import os as _os
    from datetime import datetime as _dt

    latest_scan_file = "scan_results/latest_scan_results.csv"
    status_text = "배치 스캔 내역 없음"
    if _os.path.exists(latest_scan_file):
        last_mod_time = _dt.fromtimestamp(_os.path.getmtime(latest_scan_file))
        time_diff_hours = (_dt.now() - last_mod_time).total_seconds() / 3600
        if time_diff_hours <= 24:
            status_text = f"✅ 최신 데이터: {last_mod_time.strftime('%Y-%m-%d %H:%M')}"
        else:
            status_text = f"⚠️ 오래된 데이터: {last_mod_time.strftime('%Y-%m-%d %H:%M')}"
    st.info(status_text)

    BATCH_SCRIPT_NAME = "run_scan_batch.py"
    is_batch_running = False
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            cmdline = proc.info.get('cmdline')
            if cmdline and BATCH_SCRIPT_NAME in " ".join(map(str, cmdline)):
                is_batch_running = True
                break
    except Exception:
        is_batch_running = False

    if is_batch_running:
        st.warning("⏳ 배치 스캔이 백그라운드에서 실행 중입니다...")

    if st.button("🔄 지금 배치 강제 실행", help="백그라운드에서 전체 유니버스 스캔을 실행합니다. (기존 스캔은 강제 종료)"):
        if is_batch_running:
            try:
                for proc in psutil.process_iter(['name', 'cmdline']):
                    cmdline = proc.info.get('cmdline')
                    if cmdline and BATCH_SCRIPT_NAME in " ".join(map(str, cmdline)):
                        proc.kill()
                        st.toast(f"기존 스캔(PID: {proc.pid})을 중지했습니다.")
            except Exception as e:
                st.error(f"기존 스캔 중지 실패: {e}")

        try:
            subprocess.Popen(["cmd", "/c", "start", "run_batch_manual.bat"], shell=True)
            st.toast("새로운 배치 스캔을 시작합니다! (새 콘솔 창 확인)")
            st.rerun()
        except Exception as e:
            st.error(f"배치 스캔 시작 실패: {e}")
    
    # 배치 스캔 결과 표시
    if _os.path.exists(latest_scan_file):
        st.divider()
        st.markdown("**📋 배치 스캔 결과**")
        
        try:
            scan_results_df = pd.read_csv(latest_scan_file, index_col=0)
            
            # FMS 임계값 필터링 및 이미 관심종목에 추가된 종목 제외
            fms_threshold_scan = st.slider("FMS 임계값", 0.0, 5.0, 0.0, 0.1, key="scan_fms_threshold")
            filtered_results = scan_results_df[
                (scan_results_df['FMS'] >= fms_threshold_scan) & 
                (~scan_results_df.index.isin(st.session_state.watchlist))
            ].sort_values('FMS', ascending=False)
            
            if not filtered_results.empty:
                st.info(f"총 {len(filtered_results)}개 종목 (FMS ≥ {fms_threshold_scan})")
                
                # 페이지당 표시 개수
                items_per_page = st.selectbox("페이지당 표시", [5, 10, 20, 30], index=1, key="scan_items_per_page")
                
                # 페이징 계산
                total_pages = max(1, (len(filtered_results) + items_per_page - 1) // items_per_page)
                current_page = st.session_state.get('scan_page', 1)
                if current_page > total_pages:
                    current_page = 1
                    st.session_state.scan_page = 1
                
                start_idx = (current_page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_results = filtered_results.iloc[start_idx:end_idx]
                
                # 페이징 컨트롤
                prev_col, info_col, next_col = st.columns([0.5, 1, 0.5])
                with prev_col:
                    if st.button("⬅️", disabled=(current_page <= 1), key="scan_prev"):
                        st.session_state.scan_page = max(1, current_page - 1)
                        st.rerun()
                with info_col:
                    st.caption(f"{current_page}/{total_pages}")
                with next_col:
                    if st.button("➡️", disabled=(current_page >= total_pages), key="scan_next"):
                        st.session_state.scan_page = min(total_pages, current_page + 1)
                        st.rerun()
                
                # 결과 표시
                for symbol in page_results.index:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        fms_score = page_results.loc[symbol, 'FMS']
                        st.write(f"**{symbol}** (FMS: {fms_score:.2f})")
                    with col2:
                        if st.button("➕", key=f"add_scan_{symbol}"):
                            st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, [symbol])
                            st.rerun()
            else:
                st.info("조건에 맞는 종목이 없습니다.")
                
        except Exception as e:
            st.error(f"스캔 결과 로드 실패: {str(e)}")

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
      - 치명적 변동성: 63거래일 내 일일 변동폭 30% 초과 시 실격
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
# 이름 캐시 (영구 파일 기반)
# ------------------------------
SYMBOL_NAMES_CACHE_FILE = "symbol_names_cache.json"

def load_symbol_names_cache():
    """
    종목명 캐시 파일을 로드합니다.
    
    Returns:
        dict: {symbol: name} 형태의 딕셔너리
    """
    try:
        if os.path.exists(SYMBOL_NAMES_CACHE_FILE):
            with open(SYMBOL_NAMES_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log(f"WARNING: Failed to load symbol names cache: {e}")
    return {}

def save_symbol_names_cache(cache_dict):
    """
    종목명 캐시를 파일에 저장합니다.
    
    Args:
        cache_dict (dict): {symbol: name} 형태의 딕셔너리
    """
    try:
        with open(SYMBOL_NAMES_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"WARNING: Failed to save symbol names cache: {e}")

def load_korean_stock_names():
    """
    korean_universe.csv 파일에서 한국 종목명을 로드합니다.
    
    Returns:
        dict: {symbol: name} 형태의 딕셔너리
    """
    korean_names = {}
    try:
        if os.path.exists('korean_universe.csv'):
            df = pd.read_csv('korean_universe.csv')
            if 'Symbol' in df.columns and 'Name' in df.columns:
                for _, row in df.iterrows():
                    symbol = str(row['Symbol']).strip()
                    name = str(row['Name']).strip() if pd.notna(row['Name']) else None
                    if symbol and name:
                        korean_names[symbol] = name
    except Exception as e:
        log(f"WARNING: Failed to load Korean stock names: {e}")
    return korean_names

@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_long_names(symbols):
    """
    종목명을 가져옵니다. 한국 종목은 korean_universe.csv에서 읽고, 
    다른 종목은 영구 캐시 파일 또는 yfinance를 사용합니다.
    
    Args:
        symbols (list): 종목 심볼 목록
    
    Returns:
        dict: {symbol: name} 형태의 딕셔너리
    """
    # 한국 종목명 파일에서 로드
    korean_names = load_korean_stock_names()
    
    # 캐시 파일에서 기존 종목명 로드 (한국 종목 제외)
    cache = load_symbol_names_cache()
    
    # 결과 딕셔너리 초기화
    out = {}
    new_symbols = []
    
    # 종목별로 처리
    for symbol in symbols:
        # 한국 종목(.KS)은 korean_universe.csv에서 우선 확인
        if symbol.endswith('.KS'):
            if symbol in korean_names:
                out[symbol] = korean_names[symbol]
            else:
                # 파일에 없으면 종목코드 사용
                out[symbol] = symbol
                log(f"WARNING: Korean stock name not found in korean_universe.csv: {symbol}")
        else:
            # 한국 종목이 아닌 경우 캐시 확인
            if symbol in cache:
                out[symbol] = cache[symbol]
            else:
                new_symbols.append(symbol)
    
    # 캐시에 없는 비한국 종목만 yfinance에서 가져오기
    if new_symbols:
        log(f"Fetching names for {len(new_symbols)} non-Korean symbols from yfinance...")
        for symbol in new_symbols:
            name = None
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.get_info()
                raw_name = info.get("longName") or info.get("shortName")
                
                if raw_name:
                    name = raw_name
                else:
                    name = symbol
                    
            except Exception as e:
                log(f"INFO name fetch fail: {symbol} -> {e}")
                name = symbol
            
            # 결과 저장
            out[symbol] = name
            # 캐시에도 저장
            cache[symbol] = name
        
        # 캐시 파일 업데이트
        save_symbol_names_cache(cache)
    
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


st.title("⚡ KRW Momentum Radar v3.6.0")



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
    
    mom = momentum_now_and_delta(prices_krw, reference_prices_krw=prices_krw, ohlc_data=ohlc_data, symbols=watchlist_symbols)
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

# ------------------------------
# ③ 세부 보기
# ------------------------------
st.subheader("세부 보기")
ordered_options = list(mom_ranked.index)

if not ordered_options:
    st.session_state.detail_symbol_index = 0
    st.info("표시할 종목이 없습니다. 필터 조건을 조정하거나 데이터를 새로 고침해 주세요.")
else:
    default_candidates = [sym for sym in sel_syms if sym in ordered_options]
    default_sym = default_candidates[0] if default_candidates else ordered_options[0]

    if "detail_symbol_index" not in st.session_state:
        st.session_state.detail_symbol_index = ordered_options.index(default_sym)

    # 현재 선택된 인덱스가 유효한 범위를 벗어났는지 확인
    if st.session_state.detail_symbol_index >= len(ordered_options):
        st.session_state.detail_symbol_index = len(ordered_options) - 1
    if st.session_state.detail_symbol_index < 0:
        st.session_state.detail_symbol_index = 0

    option_count = max(1, len(ordered_options))
    index_width = max(2, len(str(option_count)))
    option_labels = {
        sym: f"[{idx:0{index_width}d}] {display_name(sym)}"
        for idx, sym in enumerate(ordered_options, 1)
    }

    # selectbox와 네비게이션 버튼을 한 줄에 배치
    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns([11, 1, 1, 1])

    # 버튼 클릭 처리
    with detail_col2:
        # 이전 버튼 (▲)
        prev_disabled = st.session_state.detail_symbol_index <= 0
        if st.button("▲", disabled=prev_disabled, key="detail_prev", help="이전 종목", use_container_width=True):
            if st.session_state.detail_symbol_index > 0:
                st.session_state.detail_symbol_index -= 1
                st.rerun()

    with detail_col3:
        # 다음 버튼 (▼)
        next_disabled = st.session_state.detail_symbol_index >= len(ordered_options) - 1
        if st.button("▼", disabled=next_disabled, key="detail_next", help="다음 종목", use_container_width=True):
            if st.session_state.detail_symbol_index < len(ordered_options) - 1:
                st.session_state.detail_symbol_index += 1
                st.rerun()

    with detail_col4:
        # 맨 끝으로 가기 버튼 (⏬)
        end_disabled = st.session_state.detail_symbol_index >= len(ordered_options) - 1
        if st.button("⏬", disabled=end_disabled, key="detail_end", help="맨 끝으로 이동", use_container_width=True):
            if st.session_state.detail_symbol_index < len(ordered_options) - 1:
                st.session_state.detail_symbol_index = len(ordered_options) - 1
                st.rerun()

    with detail_col1:
        # selectbox의 key를 인덱스 기반으로 동적 생성하여 버튼 클릭 시 새로운 상태로 인식
        selectbox_key = f"detail_selectbox_{st.session_state.detail_symbol_index}"

        detail_sym = st.selectbox(
            "",
            options=ordered_options,
            index=st.session_state.detail_symbol_index,
            format_func=lambda s: option_labels.get(s, display_name(s)),
            key=selectbox_key,
            label_visibility="collapsed",
        )

        # selectbox 변경 시 인덱스 업데이트 (사용자가 직접 선택한 경우)
        if detail_sym in ordered_options:
            new_index = ordered_options.index(detail_sym)
            if new_index != st.session_state.detail_symbol_index:
                st.session_state.detail_symbol_index = new_index
                st.rerun()

    s = prices_krw[detail_sym].dropna()
    e20, e50, e200 = ema(s, 20), ema(s, 50), ema(s, 200)
    fig_det = go.Figure()
    fig_det.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="KRW"))
    fig_det.add_trace(go.Scatter(x=e20.index, y=e20.values, mode="lines", name="EMA20"))
    fig_det.add_trace(go.Scatter(x=e50.index, y=e50.values, mode="lines", name="EMA50"))
    fig_det.add_trace(go.Scatter(x=e200.index, y=e200.values, mode="lines", name="EMA200"))
    fig_det.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="KRW",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )
    st.plotly_chart(fig_det, use_container_width=True, config={"displayModeBar": False})

    roll_max = s.cummax()
    dd = (s / roll_max - 1.0) * 100.0
    fig_dd = go.Figure([go.Scatter(x=dd.index, y=dd.values, mode="lines", name="Drawdown(%)")])
    fig_dd.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
    st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

    row = mom.loc[detail_sym]
    badges = []
    if row["R_1M"] > 0:
        badges.append("1M +")
    if row["AboveEMA50"] > 0:
        badges.append("EMA50 상회")

    # FMS 지표 표시
    if "R_3M" in row and row["R_3M"] > 0:
        badges.append("3M +")

    if row["ΔFMS_1D"] > 0:
        badges.append("가속(1D+)")
    if row["ΔFMS_5D"] > 0:
        badges.append("가속(5D+)")
    st.markdown(
        " ".join([f"<span class='badge'>{b}</span>" for b in badges]) or "<span class='small'>상태 배지 없음</span>",
        unsafe_allow_html=True,
    )

# ==============================
# ④ 수익률–변동성 이동맵
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
        y_raw = float(p.iloc[0,1])*100.0  # CAGR%
        y = max(y_raw + 100.0, 0.1)  # 로그 스케일을 위한 offset 적용 (100 = 0%, 최소값 0.1 보장)
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
        # 1M ago (데이터가 있는 경우만 표시)
        if not pts_mago.empty and "CAGR" in pts_mago.columns:
            cagr_mago = pts_mago["CAGR"]*100
            fig.add_trace(go.Scatter(
                x=pts_mago["Vol"]*100, y=(cagr_mago + 100).clip(lower=0.1), mode="markers",
                marker=dict(size=7, color="lightgray"),
                text=[display_name(s) for s in pts_mago.index],
                customdata=cagr_mago.values,
                hovertemplate="%{text}<br>1M ago<br>Vol: %{x:.2f}% | CAGR: %{customdata:.2f}%<extra></extra>",
                name="1M ago", showlegend=True
            ))
        # Yesterday (데이터가 있는 경우만 표시)
        if not pts_yest.empty and "CAGR" in pts_yest.columns:
            cagr_yest = pts_yest["CAGR"]*100
            fig.add_trace(go.Scatter(
                x=pts_yest["Vol"]*100, y=(cagr_yest + 100).clip(lower=0.1), mode="markers",
                marker=dict(size=8, color="silver"),
                text=[display_name(s) for s in pts_yest.index],
                customdata=cagr_yest.values,
                hovertemplate="%{text}<br>Yesterday<br>Vol: %{x:.2f}% | CAGR: %{customdata:.2f}%<extra></extra>",
                name="Yesterday", showlegend=True
            ))
    # Today (데이터가 있는 경우만 표시)
    if not pts_today.empty and "CAGR" in pts_today.columns:
        cagr_today = pts_today["CAGR"]*100
        fig.add_trace(go.Scatter(
            x=pts_today["Vol"]*100, y=(cagr_today + 100).clip(lower=0.1), mode="markers+text",
            marker=dict(size=9),
            text=[display_name(s) for s in pts_today.index],
            textposition="top center",
            customdata=cagr_today.values,
            hovertemplate="%{text}<br>Today<br>Vol: %{x:.2f}% | CAGR: %{customdata:.2f}%<extra></extra>",
            name="Today", showlegend=True
        ))
    # 꼬리 길이가 0일 때만 1M→Yest→Today 연결선 표시 (데이터가 있는 경우만)
    if tail_days == 0:
        for c in plot_syms:
            xs=[]; ys=[]
            for dfp in (pts_mago, pts_yest, pts_today):
                if not dfp.empty and "CAGR" in dfp.columns and c in dfp.index and not dfp.loc[c].isna().any():
                    xs.append(float(dfp.loc[c,"Vol"])*100)
                    y_val = float(dfp.loc[c,"CAGR"])*100 + 100
                    ys.append(max(y_val, 0.1))  # offset 적용 (최소값 0.1 보장)
            if len(xs)>=2:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(width=1), opacity=0.35,
                                         name=f"path-{c}", showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        height=520, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="Volatility (ann, %)", yaxis_title="CAGR (ann, %), log scale (100 = 0%)",
        yaxis=dict(type="log"),
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

    # 초기 포인트 (데이터가 있는 경우만)
    traces.extend(tail_traces)
    if not p0.empty and "CAGR" in p0.columns:
        cagr_p0 = p0["CAGR"]*100
        traces.append(go.Scatter(
            x=p0["Vol"]*100, y=(cagr_p0 + 100).clip(lower=0.1), mode="markers+text",
            marker=dict(size=9),
            text=[display_name(s) for s in p0.index],
            textposition="top center",
            customdata=cagr_p0.values,
            hovertemplate="%{text}<br>Vol: %{x:.2f}% | CAGR: %{customdata:.2f}%<extra></extra>",
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
                y_val = float(pk.iloc[0,1])*100.0 + 100.0
                ys.append(max(y_val, 0.1))  # offset 적용 (최소값 0.1 보장)
            frame_data.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2, dash="dot"),
                                         showlegend=False, hoverinfo="skip", name=f"{c}-tail"))
        # 포인트 (데이터가 있는 경우만)
        if not p.empty and "CAGR" in p.columns:
            cagr_p = p["CAGR"]*100
            frame_data.append(go.Scatter(
                x=p["Vol"]*100, y=(cagr_p + 100).clip(lower=0.1), mode="markers+text",
                marker=dict(size=9),
                text=[display_name(s) for s in p.index],
                textposition="top center",
                customdata=cagr_p.values,
                hovertemplate="%{text}<br>Vol: %{x:.2f}% | CAGR: %{customdata:.2f}%<extra></extra>",
                name="Points", showlegend=False
            ))

        frames.append(go.Frame(data=frame_data, name=str(prices_krw.index[loc].date())))

    fig = go.Figure(data=traces, frames=frames)
    # 애니메이션 모드에 따른 자동 재생 설정
    auto_play = motion_mode != "끄기"
    
    fig.update_layout(
        height=520, margin=dict(l=10,r=10,t=10,b=10),
        xaxis_title="Volatility (ann, %)", yaxis_title="CAGR (ann, %), log scale (100 = 0%)",
        yaxis=dict(type="log"),
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
