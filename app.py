# app.py
# -*- coding: utf-8 -*-
# KRW Momentum Radar - v5.0.5
# 
# 주요 기능:
# - FMS(Fast Momentum Score) 기반 모멘텀 분석 (v5.0 alive_pullback nonlinear)
# - 다국가 시장 통합 분석 (미국, 한국, 일본)
# - 수익률-변동성 이동맵 (정적/애니메이션 모드)
# - 실시간 데이터 업데이트 및 시각화
# - 동적 관심종목 관리 및 배치 스캔 결과 확인
# - True Range 기반 거래 적합성 필터
# - 거래 적합성 필터 디버그 로깅 (모든 국가 종목 지원)

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
from typing import Optional, Tuple
import yfinance as yf
from watchlist_utils import (
    load_watchlist, save_watchlist, add_to_watchlist, remove_from_watchlist, 
    export_watchlist_to_csv, import_watchlist_from_csv, MODE_FREE, MODE_IRP
)
from config import FMS_FORMULA
from analysis_utils import (
    align_bday_ffill,
    calculate_tradeability_filters as _au_trade_filters,
    calculate_fms_for_batch as _au_calculate_fms_for_batch,
    get_filter_debug_info,
    harmonize_calendar,
    momentum_now_and_delta as _au_momentum_now_and_delta,
)
from adapters.ui_data_bundle import (
    DETAIL_ATOM_CACHE_KEY,
    DETAIL_INDEX_KEY,
    DETAIL_SELECT_KEY,
    SESSION_BUNDLE_FP_KEY,
    SESSION_BUNDLE_KEY,
    build_detail_view_atom,
    clear_ui_session_caches,
    reconcile_detail_selection,
    watchlist_bundle_key,
)
from calibration_utils import (
    create_snapshot_id,
    save_snapshot,
    load_snapshot,
    list_sessions,
    save_session,
    load_session,
    init_sort_state,
    get_next_comparison,
    apply_sort_choice,
    build_review_queue_from_final,
    get_next_review_pair,
    apply_review_choice,
)

warnings.filterwarnings("ignore", category=ResourceWarning)
KST = pytz.timezone("Asia/Seoul")

# ------------------------------
# 기본 유니버스 (세션 최초 로드·모드 전환 폴백)
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

def _rebase_100(s: pd.Series) -> pd.Series:
    """
    시계열을 처음 값을 기준으로 100으로 리베이스합니다.
    (Rebased 100 차트 및 고정 y-range 계산에 사용)
    """
    s = s.dropna()
    if s.empty:
        return s
    base = float(s.iloc[0])
    if base <= 0 or pd.isna(base):
        return pd.Series(dtype=float, index=s.index)
    return (s / base) * 100.0


def classify(sym):
    # sym이 float이나 다른 타입일 경우를 대비해 str로 변환
    sym_str = str(sym)
    # 코스피/코스닥은 Yahoo Finance에서 suffix가 다를 수 있음
    # - KOSPI: .KS
    # - KOSDAQ: .KQ
    if sym_str.endswith(".KS") or sym_str.endswith(".KQ"):
        return "KOR"
    if sym_str.endswith(".HK"):
        return "HKG"
    if sym_str.endswith(".T"):  return "JPN"
    return "USA"

# ------------------------------
# 페이지/스타일
# ------------------------------
st.set_page_config(page_title="KRW Momentum Radar v5.0.5", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 0.8rem;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.75rem; margin-right:6px; background:#f1f3f5;}
.kpi {border:1px solid #eee; border-radius:16px; padding:10px 14px; box-shadow:0 1px 6px rgba(0,0,0,0.06);}
.small {font-size:0.8rem; color:#555;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 계좌 모드 초기화 (UI보다 먼저 실행)
# ------------------------------
if 'account_mode' not in st.session_state:
    st.session_state.account_mode = MODE_FREE

# ------------------------------
# 관심종목 세션 로드 (모드별)
# ------------------------------
if 'watchlist' not in st.session_state:
    default_symbols = DEFAULT_USD_SYMBOLS + DEFAULT_KRW_SYMBOLS + DEFAULT_JPY_SYMBOLS
    st.session_state.watchlist = load_watchlist(default_symbols, mode=st.session_state.account_mode)

# ------------------------------
# 데이터 다운로드 및 처리 함수들
# ------------------------------
@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_prices(tickers, period_="2y", interval="1d", chunk=25):
    """Download Adj Close via disk-backed CachingMarketDataAdapter (batch-shared)."""
    from adapters.price_cache import make_caching_yfinance_adapter
    tickers = list(dict.fromkeys(tickers or []))
    if not tickers:
        return pd.DataFrame(), []
    md = make_caching_yfinance_adapter(chunk=chunk, threads=True)
    out, missing = md.get_prices(tickers, period_, interval)
    if not out.empty:
        all_nan = out.columns[out.isna().all()]
        if len(all_nan):
            log(f"DROP all-NaN columns: {list(all_nan)[:5]}{' ...' if len(all_nan)>5 else ''}")
            out = out.drop(columns=list(all_nan))
            missing = sorted(list(dict.fromkeys(list(missing) + list(all_nan))))
    return out, missing

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_ohlc_prices(tickers, period_="2y", interval="1d", chunk=25):
    """OHLC download via disk cache + last-bar probe (shared with batch)."""
    from adapters.price_cache import make_caching_yfinance_adapter
    tickers = list(dict.fromkeys(tickers or []))
    if not tickers:
        return pd.DataFrame(), []
    md = make_caching_yfinance_adapter(chunk=chunk, threads=True)
    return md.get_ohlc(tickers, period_, interval)

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_fx(period_="2y", interval="1d"):
    """FX download via caching adapter (write-through to disk)."""
    from adapters.price_cache import make_caching_yfinance_adapter
    md = make_caching_yfinance_adapter(threads=True)
    usdkrw, usdjpy, jpykrw, hkdkrw = md.get_fx(period_, interval)
    miss = []
    if usdkrw is None or usdkrw.empty:
        miss.append("KRW=X")
        usdkrw = pd.Series(dtype=float, name="USDKRW")
    else:
        usdkrw = usdkrw.rename("USDKRW")
    if usdjpy is None or usdjpy.empty:
        miss.append("JPY=X")
        usdjpy = pd.Series(dtype=float, name="USDJPY")
    else:
        usdjpy = usdjpy.rename("USDJPY")
    if jpykrw is None or jpykrw.empty:
        jpykrw = pd.Series(dtype=float, name="JPYKRW")
    else:
        jpykrw = jpykrw.rename("JPYKRW")
    if hkdkrw is None or hkdkrw.empty:
        miss.append("HKD=X")
        hkdkrw = pd.Series(dtype=float, name="HKDKRW")
    else:
        hkdkrw = hkdkrw.rename("HKDKRW")
    return usdkrw, usdjpy, jpykrw, hkdkrw, miss

# ------------------------------
# 로깅 함수
# ------------------------------
if "LOG" not in st.session_state:
    st.session_state["LOG"] = []
def log(msg):
    ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["LOG"].append(f"[{ts}] {msg}")

# ------------------------------
# 지표 헬퍼 (차트 렌더링용; FMS 스코어는 analysis_utils / core.fms SSOT)
# ------------------------------
def ema(s, span):
    """EMA for detail-chart overlays (not used by production FMS scoring)."""
    return s.ewm(span=span, adjust=False).mean()


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

# 0. 계좌 모드 선택 (최상단)
with st.sidebar.expander("🏦 계좌 모드 선택", expanded=True):
    mode_options = {
        "자유투자계좌": MODE_FREE,
        "퇴직연금IRP": MODE_IRP
    }
    
    selected_mode_label = st.radio(
        "계좌 모드",
        options=list(mode_options.keys()),
        index=0 if st.session_state.account_mode == MODE_FREE else 1,
        help="자유투자계좌: 미국+한국+홍콩 주식 | 퇴직연금IRP: 국내상장 ETF 전 종목"
    )
    
    selected_mode = mode_options[selected_mode_label]
    
    # 모드 변경 감지 및 처리
    if selected_mode != st.session_state.account_mode:
        st.session_state.account_mode = selected_mode
        # 모드 변경 시 관심종목 재로드
        default_symbols = DEFAULT_USD_SYMBOLS + DEFAULT_KRW_SYMBOLS + DEFAULT_JPY_SYMBOLS
        st.session_state.watchlist = load_watchlist(default_symbols, mode=selected_mode)
        # 캐시 초기화 (Streamlit + UI 세션 번들/DetailViewAtom)
        st.cache_data.clear()
        clear_ui_session_caches(st.session_state)
        st.rerun()

# 1. 분석 설정
with st.sidebar.expander("📊 분석 설정", expanded=True):
    period = st.selectbox("차트 기간", ["1M","3M","6M","1Y","2Y"], index=1)
    
    rank_by = st.selectbox("정렬 기준", ["ΔFMS(1D)","ΔFMS(5D)","FMS(현재)","1M 수익률"], index=2)
    TOP_N = st.slider("Top N", 5, 60, 20, step=5)
    # 수익률-변동성 이동맵 설정 (데이터 로드 시점에 필요)
    st.divider()
    st.markdown("**수익률-변동성 이동맵 설정**")
    rv_window = st.selectbox("수익률/변동성 창(거래일)", [21, 42, 63], index=0, help="연율화: 252 기준", key="sidebar_rv_window")
    tail_days = st.selectbox("꼬리 길이(최근 n일 경로)", [0, 3, 5, 10], index=0, help="오늘 기준 과거 n거래일의 이동 경로를 점선으로 표시", key="sidebar_tail_days")

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
            new_symbols, message = import_watchlist_from_csv(csv_data, mode=st.session_state.account_mode)
            
            if new_symbols:
                st.session_state.watchlist = new_symbols
                st.cache_data.clear()
                clear_ui_session_caches(st.session_state)
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
                    st.session_state.watchlist = remove_from_watchlist(st.session_state.watchlist, [symbol], mode=st.session_state.account_mode)
                    
                    # 재평가 결과에서도 제거
                    if 'reassessment_results' in st.session_state and st.session_state['reassessment_results'] is not None:
                        if symbol in st.session_state['reassessment_results'].index:
                            st.session_state['reassessment_results'] = st.session_state['reassessment_results'].drop(symbol)
                    
                    st.cache_data.clear()
                    clear_ui_session_caches(st.session_state)
                    st.rerun()

# 3. 신규 종목 탐색
with st.sidebar.expander("🚀 신규 종목 탐색", expanded=False):
    # --- [신규] 배치 스캔 관리 ---
    st.markdown("**📦 배치 스캔 관리**")
    import subprocess
    import psutil
    import os as _os
    from datetime import datetime as _dt

    # 모드별 스캔 결과 파일 경로
    from universe_utils import load_latest_scan_results
    current_mode = st.session_state.account_mode
    latest_scan_file = f"scan_results/latest_scan_results_{current_mode.lower()}.csv"
    
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
            # 현재 모드를 배치 파일에 전달
            current_mode_for_batch = st.session_state.account_mode
            # start 명령어는 첫 번째 인자가 창 제목이므로 빈 문자열을 사용하고, 배치 파일과 인자를 전달
            mode_label = "자유투자계좌" if current_mode_for_batch == MODE_FREE else "퇴직연금IRP"
            subprocess.Popen(["cmd", "/c", "start", f"KRW Momentum Batch Scan ({mode_label})", "cmd", "/c", f"run_batch_manual.bat {current_mode_for_batch}"], shell=True)
            st.toast(f"새로운 배치 스캔을 시작합니다! ({mode_label}, 새 콘솔 창 확인)")
            st.rerun()
        except Exception as e:
            st.error(f"배치 스캔 시작 실패: {e}")
    
    # 배치 스캔 결과 표시 (모드별)
    success, scan_results_df, load_msg = load_latest_scan_results(fms_threshold=0.0, mode=current_mode)
    if success and not scan_results_df.empty:
        st.divider()
        mode_label = "자유투자계좌" if current_mode == MODE_FREE else "퇴직연금IRP"
        st.markdown(f"**📋 배치 스캔 결과 ({mode_label})**")
        
        try:
            
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
                
                # 메시지 표시 영역 (페이징 컨트롤 위에 표시)
                # 모든 종목의 메시지를 확인하여 표시 (현재 페이지에 없어도 표시)
                for symbol in scan_results_df.index:
                    message_key = f"scan_message_{symbol}"
                    if message_key in st.session_state and st.session_state[message_key] is not None:
                        message_type = st.session_state[message_key]['type']
                        message_text = st.session_state[message_key]['text']
                        if message_type == 'warning':
                            st.warning(message_text)
                        elif message_type == 'success':
                            st.success(message_text)
                        elif message_type == 'error':
                            st.error(message_text)
                        # 메시지 표시 후 초기화 (다음 렌더링에서 사라지도록)
                        st.session_state[message_key] = None
                
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
                    message_key = f"scan_message_{symbol}"
                    
                    with col1:
                        fms_score = page_results.loc[symbol, 'FMS']
                        st.write(f"**{symbol}** (FMS: {fms_score:.2f})")
                    
                    with col2:
                        if st.button("➕", key=f"add_scan_{symbol}"):
                            # 이미 관심종목에 있는지 체크
                            if symbol in st.session_state.watchlist:
                                st.session_state[message_key] = {
                                    'type': 'warning',
                                    'text': f"'{symbol}'는 이미 관심종목에 있습니다."
                                }
                            else:
                                try:
                                    # 관심종목에 추가
                                    st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, [symbol], mode=st.session_state.account_mode)
                                    # 추가 성공 확인
                                    if symbol in st.session_state.watchlist:
                                        st.session_state[message_key] = {
                                            'type': 'success',
                                            'text': f"'{symbol}' 추가됨"
                                        }
                                    else:
                                        st.session_state[message_key] = {
                                            'type': 'error',
                                            'text': f"'{symbol}' 추가 실패: 알 수 없는 이유로 추가되지 않았습니다."
                                        }
                                except Exception as e:
                                    st.session_state[message_key] = {
                                        'type': 'error',
                                        'text': f"'{symbol}' 추가 실패: {str(e)}"
                                    }
                            st.rerun()
            else:
                st.info("조건에 맞는 종목이 없습니다.")
                
        except Exception as e:
            st.error(f"스캔 결과 로드 실패: {str(e)}")
    elif not success:
        st.info(f"배치 스캔 결과가 없습니다. ({load_msg})")

# 4. 수동 관리 (간단한 추가/삭제)
with st.sidebar.expander("✏️ 수동 관리", expanded=False):
    # 티커 추가
    new_ticker = st.text_input("티커 추가 (예: AAPL)", "").upper().strip()
    if st.button("➕ 추가"):
        if not new_ticker:
            st.error("유효한 티커를 입력하세요.")
        elif new_ticker in st.session_state.watchlist:
            st.warning(f"'{new_ticker}'는 이미 관심종목에 있습니다.")
        else:
            try:
                st.session_state.watchlist = add_to_watchlist(st.session_state.watchlist, [new_ticker], mode=st.session_state.account_mode)
                st.success(f"'{new_ticker}' 추가됨")
                st.rerun()
            except Exception as e:
                st.error(f"'{new_ticker}' 추가 실패: {str(e)}")

    # 티커 삭제
    if st.session_state.watchlist:
        ticker_to_remove = st.selectbox("삭제할 티커 선택", [""] + st.session_state.watchlist)
        if st.button("🗑️ 삭제"):
            if ticker_to_remove:
                st.session_state.watchlist = remove_from_watchlist(st.session_state.watchlist, [ticker_to_remove], mode=st.session_state.account_mode)
                st.success(f"'{ticker_to_remove}' 삭제됨")
                st.rerun()
            else:
                st.error("삭제할 종목을 선택해주세요.")

with st.sidebar.expander("🔧 도구 및 도움말", expanded=False):
    # FMS 설명 (SSOT: core/fms.py · config.FMS_FORMULA — UI 문구는 여기와 동기화)
    st.markdown("**📊 FMS (Fast Momentum Score)**")
    
    st.markdown(f"""
    **현재 적용 상태:**
    - 아래 설명은 앱·배치가 실제 사용하는 **production v5.0.0 FMS**입니다.
    - **alive_pullback** 비선형 수식(비중첩 SEG_* + 잔차 피처)이며,
      관심종목 상대 Z-score / 현금성 게이트는 사용하지 않습니다(절대 경로 점수).

    **개요:**
    - FMS는 사용자가 본 **최근 3개월(63거래일) 가격 경로**에서 최근 회복·이전 추세 지지·
      절대수익 바닥·정체/단발급등 패널티를 평가합니다.
    - 정답셋 `cal_fms_20260730_190637`에서 NL→비선형 MC로 피팅 후 승격한 수식입니다.

    **가산 요인**
    - **SEG_RET_0_5 / alive**: 최근 1주 수익·회복(이전 지지가 있을 때 가산)
    - **MID_DIP_RECOVERY**: 중간 구간 조정 후 최근 회복 연속성
    - **SEG_RET_21_63 / PRIOR_SUPPORT_SIGN**: 이전 추세 지지
    - **R_3M softplus floor**: 절대 3개월 수익이 바닥(~1.2%) 아래면 강하게 억제
    - **RECENT_UP_DAYS_5D / grind**: 최근 상승일 폭·고R²·비스파이크 경로 소폭 가산
    - **TREND_EFFICIENCY_REWARD_15D**: 최근 경로 효율

    **감점 요인**
    - **STALE_AFTER_RUN**: 최근이 약할 때만 적용되는 대상승 후 정체 패널티
    - **RECENT_JUMP_SHARE_5D**: 최근 5일 상승이 단발 급등에 치우치면 감점

    **정규화**
    - 절대 피처 점수(watchlist 상대 Z 없음). 관심종목 구성이 바뀌어도
      같은 가격 경로면 FMS는 동일합니다. 배치는 후보를 같은 절대 수식으로 점수화합니다.

    **추가 필터 (거래 적합성)**  
    - True Range 기반 **치명적 변동성 30% 초과** 또는  
      **20일 내 -7% 미만 하락 4일 이상**이면 FMS = -999 로 실격 처리합니다.

    **한 줄 요약:**
    - **“이전 지지 + 최근 회복이 확인된 연속 상승”을 선호하고,
      “절대 저수익·죽은 대상승 정체·단발 급등”을 억제하는 비선형 점수입니다.**
    """)
    
    st.markdown("---")
    
    # 도구 버튼들
    if st.button("🗂️ 데이터 캐시 초기화"):
        st.cache_data.clear()
        clear_ui_session_caches(st.session_state)
        from adapters.price_cache import clear_disk_market_cache
        clear_disk_market_cache()
        st.success("캐시 초기화 완료 (세션·Streamlit·디스크)")


def calculate_minimum_data_period(rv_window=63, tail_days=10):
    """
    모든 기능이 정상 작동하기 위한 최소 데이터 기간을 계산합니다.
    
    각 기능별 최소 거래일 요구사항:
    - FMS 계산: R_3M(63일) + ΔFMS_5D(5일) = 68일
    - 거래 적합성 필터: 63일
    - R_4M: 84일
    - 수익률-변동성 이동맵: rv_window + tail_days (최대 73일)
    - YTD Return: 연초부터 (1년 데이터면 충분)
    
    Args:
        rv_window (int): 수익률-변동성 이동맵 창 크기 (기본값: 최대값 63)
        tail_days (int): 꼬리 길이 (기본값: 최대값 10)
    
    Returns:
        str: yfinance period 문자열 ("3mo", "6mo", "1y", "2y", "5y")
    """
    # 각 기능별 최소 거래일 요구사항
    requirements = []
    
    # 1. FMS 계산: R_3M(63일) + ΔFMS_5D(5일) = 68일
    requirements.append(68)
    
    # 2. 거래 적합성 필터: 63일
    requirements.append(63)
    
    # 3. R_4M: 84일
    requirements.append(84)
    
    # 4. 수익률-변동성 이동맵: rv_window + tail_days
    requirements.append(rv_window + tail_days)
    
    # 5. YTD Return: 연초부터 (1년 데이터면 충분)
    # 실제로는 연초부터만 필요하지만, 안전하게 1년 데이터를 다운로드
    requirements.append(252)
    
    # 최소 필요 거래일 계산 (여유분 10% 추가하여 휴일/데이터 누락 대비)
    min_trading_days = int(max(requirements) * 1.1)
    
    # 거래일을 yfinance period로 변환 (1년 = 약 252거래일 기준)
    # 안전하게 항상 최소 1년 데이터를 다운로드
    if min_trading_days <= 252:  # 약 1년
        return "1y"
    elif min_trading_days <= 504:  # 약 2년
        return "2y"
    else:  # 2년 이상
        return "5y"

@st.cache_data(ttl=60*60*6, show_spinner=True)
def build_prices_krw(period_key="6M", watchlist_symbols=None, min_data_period=None):
    """
    KRW 환산 가격 데이터를 다운로드하고 구성합니다.
    
    Args:
        period_key (str): 차트 표시 기간 ("1M", "3M", "6M", "1Y", "2Y") - 캐시 키의 일부로 사용됨
        watchlist_symbols (list): 관심종목 목록
        min_data_period (str): 계산에 필요한 최소 데이터 기간 (None이면 자동 계산)
    
    Note:
        period_key는 캐시 키의 일부로 사용되지만, 실제 데이터 다운로드에는 min_data_period가 사용됩니다.
    """
    # 계산에 필요한 최소 데이터 기간 (기본값: 안전하게 1년)
    if min_data_period is None:
        min_data_period = "1y"
    
    interval = "1d"

    # 관심종목 목록을 매개변수로 받아서 캐시 키에 포함
    if watchlist_symbols is None:
        watchlist_symbols = st.session_state.watchlist

    # 현재 관심종목에서 국가별로 분류
    usd_symbols = [str(s) for s in watchlist_symbols if classify(s) == "USA"]
    krw_symbols = [str(s) for s in watchlist_symbols if classify(s) == "KOR"]
    jpy_symbols = [str(s) for s in watchlist_symbols if classify(s) == "JPN"]
    hkg_symbols = [str(s) for s in watchlist_symbols if classify(s) == "HKG"]

    # 계산에 필요한 최소 데이터 기간 사용 (차트 표시 기간이 아닌)
    usdkrw, usdjpy, jpykrw, hkdkrw, fx_missing = download_fx(min_data_period, interval)
    usd_df, miss_us = download_prices(usd_symbols, min_data_period, interval)
    krw_df, miss_kr = download_prices(krw_symbols, min_data_period, interval)
    jpy_df, miss_jp = download_prices(jpy_symbols, min_data_period, interval)
    hkg_df, miss_hk = download_prices(hkg_symbols, min_data_period, interval)

    usd_df = align_bday_ffill(usd_df)
    krw_df = align_bday_ffill(krw_df)
    jpy_df = align_bday_ffill(jpy_df)
    usdkrw = align_bday_ffill(usdkrw.to_frame()).iloc[:,0] if not usdkrw.empty else usdkrw
    jpykrw = align_bday_ffill(jpykrw.to_frame()).iloc[:,0] if not jpykrw.empty else jpykrw
    hkg_df = align_bday_ffill(hkg_df)
    hkdkrw = align_bday_ffill(hkdkrw.to_frame()).iloc[:,0] if not hkdkrw.empty else hkdkrw

    frames=[]
    if not usd_df.empty and not usdkrw.empty:
        usdkrw_matched = usdkrw.reindex(usd_df.index).ffill()
        frames.append(usd_df.mul(usdkrw_matched, axis=0))
    if not krw_df.empty:
        frames.append(krw_df)
    if not jpy_df.empty and not jpykrw.empty:
        jpykrw_matched = jpykrw.reindex(jpy_df.index).ffill()
        frames.append(jpy_df.mul(jpykrw_matched, axis=0))
    if not hkg_df.empty and not hkdkrw.empty:
        hkdkrw_matched = hkdkrw.reindex(hkg_df.index).ffill()
        frames.append(hkg_df.mul(hkdkrw_matched, axis=0))

    if not frames:
        return pd.DataFrame(), {"fx_missing": fx_missing, "price_missing": miss_us+miss_kr+miss_jp+miss_hk}

    prices_krw = pd.concat(frames, axis=1).sort_index()
    prices_krw = prices_krw.loc[:, ~prices_krw.columns.duplicated()]
    
    # harmonize_calendar 전에 원본 컬럼 수 기록
    original_cols = set(prices_krw.columns)
    # coverage 임계값을 0.5로 낮춰서 최소한의 데이터가 있으면 포함
    # 0.9는 너무 엄격하여 신규 상장 종목이나 데이터가 부족한 종목이 제외될 수 있음
    prices_krw = harmonize_calendar(prices_krw, coverage=0.5)
    # harmonize_calendar 후 제외된 종목 확인
    excluded_cols = original_cols - set(prices_krw.columns)
    if excluded_cols:
        log(f"Excluded from prices_krw (low coverage): {sorted(list(excluded_cols))}")

    miss_dict = {
        "fx_missing": fx_missing,
        "price_missing": sorted(list(set(miss_us+miss_kr+miss_jp+miss_hk)))
    }
    
    # 빈 DataFrame 체크 (harmonize_calendar가 모든 컬럼을 제외한 경우)
    if prices_krw.empty:
        log(f"Final DF shape: {prices_krw.shape}; all columns excluded by coverage threshold")
        # 관심종목 중 prices_krw에 없는 종목 확인
        watchlist_missing = set(watchlist_symbols) - set(prices_krw.columns)
        if watchlist_missing:
            miss_dict["watchlist_missing"] = sorted(list(watchlist_missing))
        return prices_krw, miss_dict
    
    last_row = prices_krw.iloc[-1]
    usa_cols = [c for c in prices_krw.columns if classify(c)=="USA"]
    na_usa = last_row[usa_cols].isna().sum()
    log(f"Final DF shape: {prices_krw.shape}; last row USA NaNs: {na_usa}/{len(usa_cols)}")
    
    # 관심종목 중 prices_krw에 없는 종목 확인
    watchlist_missing = set(watchlist_symbols) - set(prices_krw.columns)
    if watchlist_missing:
        log(f"Watchlist symbols missing from prices_krw: {sorted(list(watchlist_missing))}")
        # 누락된 종목을 miss_dict에 추가
        miss_dict["watchlist_missing"] = sorted(list(watchlist_missing))
    
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
    한국 종목명(한글 표시용)을 csv에서 로드합니다.
    
    Returns:
        dict: {symbol: name} 형태의 딕셔너리
    """
    korean_names = {}
    try:
        def _load_one(path: str) -> None:
            if not os.path.exists(path):
                return
            df = pd.read_csv(path)
            if 'Symbol' not in df.columns or 'Name' not in df.columns:
                return
            for _, row in df.iterrows():
                symbol = str(row['Symbol']).strip()
                name = str(row['Name']).strip() if pd.notna(row['Name']) else None
                if not symbol or not name:
                    continue
                korean_names[symbol] = name

                # csv에는 대개 `.KS`가 들어있지만 실제 표시는 `.KQ`를 쓸 수 있음
                # 6자리 코드 기반으로 `.KQ` 별칭을 함께 등록
                if symbol.endswith('.KS') or symbol.endswith('.KQ'):
                    base = symbol.split('.', 1)[0].strip()
                    if base.isdigit() and len(base) == 6:
                        korean_names[f'{base}.KQ'] = name

        _load_one('korean_universe.csv')
        _load_one('korean_etf_univers.csv')
    except Exception as e:
        log(f"WARNING: Failed to load Korean stock names: {e}")
    return korean_names

@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_long_names(symbols):
    """
    종목명을 가져옵니다. 우선 순위는 다음과 같습니다.
    1) 영구 캐시(`symbol_names_cache.json`)
    2) 한국 종목인 경우 csv(`korean_universe.csv` + `korean_etf_univers.csv`)
    3) yfinance API (마지막 수단, 결과는 캐시에 저장)
    
    Args:
        symbols (list): 종목 심볼 목록
    
    Returns:
        dict: {symbol: name} 형태의 딕셔너리
    """
    cache = load_symbol_names_cache()
    korean_names = None  # 필요할 때만 로드
    out = {}
    missing_for_yfinance = []
    cache_updated = False

    for symbol in symbols:
        name = None
        symbol_str = str(symbol)
        is_korean = symbol_str.endswith('.KS') or symbol_str.endswith('.KQ')

        cached_name = cache.get(symbol)
        if cached_name:
            name = cached_name
        elif is_korean:
            if korean_names is None:
                korean_names = load_korean_stock_names()
            name = korean_names.get(symbol_str)

        if name is not None:
            out[symbol] = name
            cache[symbol] = name
            cache_updated = True
            continue

        missing_for_yfinance.append(symbol)

    if missing_for_yfinance:
        log(f"Fetching names for {len(missing_for_yfinance)} symbols from yfinance...")
        for symbol in missing_for_yfinance:
            name = symbol
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.get_info()
                raw_name = info.get("longName") or info.get("shortName")
                if raw_name:
                    name = raw_name
            except Exception as e:
                log(f"INFO name fetch fail: {symbol} -> {e}")
            out[symbol] = name
            cache[symbol] = name
            cache_updated = True

    if cache_updated:
        save_symbol_names_cache(cache)

    # 캐시/유니버스/yfinance 어디에서도 찾지 못한 경우 심볼 자체를 반환
    for symbol in symbols:
        if symbol not in out:
            out[symbol] = symbol

    return out





# ------------------------------
# 데이터 로드 및 이름
# ------------------------------
# 최소 데이터 기간 계산 (사용자가 사이드바에서 선택한 값 사용)
# rv_window와 tail_days는 사이드바에서 선택되므로 여기서 사용 가능
min_data_period = calculate_minimum_data_period(
    rv_window=rv_window,
    tail_days=tail_days
)

with st.spinner("데이터 불러오는 중…"):
    prices_krw, miss = build_prices_krw(period, st.session_state.watchlist, min_data_period=min_data_period)
if prices_krw.empty:
    st.error("가격 데이터를 불러오지 못했습니다.")
    st.stop()

# 관심종목 중 데이터가 없는 종목 확인 및 경고
watchlist_missing = miss.get("watchlist_missing", [])
if watchlist_missing:
    missing_symbols_str = ", ".join(watchlist_missing)
    st.warning(f"⚠️ 다음 종목은 데이터 부족으로 표시되지 않습니다: {missing_symbols_str}")

with st.spinner("종목명(풀네임) 로딩 중…(최초 1회만 다소 지연)"):
    NAME_MAP = fetch_long_names(list(prices_krw.columns))


st.title("⚡ KRW Momentum Radar v5.0.5")



# ------------------------------
# 모멘텀/가속 계산 (거래 적합성 필터 적용) — 세션 번들 메모
# ------------------------------
watchlist_symbols = list(prices_krw.columns)
# 차트 기간과 무관하게 계산에 필요한 최소 기간 사용
ohlc_data, ohlc_missing = download_ohlc_prices(watchlist_symbols, min_data_period, "1d")
if ohlc_data.empty:
    ohlc_data = None

_bundle_fp = watchlist_bundle_key(
    st.session_state.account_mode,
    list(st.session_state.watchlist),
    min_data_period,
    prices_krw,
    ohlc_data,
)
_reuse_bundle = (
    st.session_state.get(SESSION_BUNDLE_FP_KEY) == _bundle_fp
    and isinstance(st.session_state.get(SESSION_BUNDLE_KEY), dict)
    and st.session_state[SESSION_BUNDLE_KEY].get("mom") is not None
)
if _reuse_bundle:
    mom = st.session_state[SESSION_BUNDLE_KEY]["mom"]
else:
    with st.spinner("모멘텀/가속 계산 중…"):
        mom = momentum_now_and_delta(
            prices_krw,
            reference_prices_krw=prices_krw,
            ohlc_data=ohlc_data,
            symbols=watchlist_symbols,
        )
    st.session_state[SESSION_BUNDLE_KEY] = {
        "prices_krw": prices_krw,
        "ohlc_data": ohlc_data,
        "mom": mom,
        "miss": miss,
        "name_map": NAME_MAP,
    }
    st.session_state[SESSION_BUNDLE_FP_KEY] = _bundle_fp
    if DETAIL_ATOM_CACHE_KEY in st.session_state:
        del st.session_state[DETAIL_ATOM_CACHE_KEY]
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
win_map={"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504}
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
    yaxis=dict(type="log", title="Rebased 100"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------------
# ③ 세부 보기 (DetailViewAtom: 심볼↔차트 원자 결합)
# ------------------------------
st.subheader("세부 보기")
ordered_options = list(mom_ranked.index)

if not ordered_options:
    st.session_state[DETAIL_INDEX_KEY] = 0
    st.info("표시할 종목이 없습니다. 필터 조건을 조정하거나 데이터를 새로 고침해 주세요.")
else:
    default_candidates = [sym for sym in sel_syms if sym in ordered_options]
    default_sym = default_candidates[0] if default_candidates else ordered_options[0]

    # 선택 심볼 SSOT — index는 파생. 위젯 세션값과 reconcile.
    _widget_sym = st.session_state.get(DETAIL_SELECT_KEY)
    _current_sym = _widget_sym if _widget_sym in ordered_options else st.session_state.get(DETAIL_INDEX_KEY)
    if isinstance(_current_sym, int):
        _idx = _current_sym
        _current_sym = ordered_options[_idx] if 0 <= _idx < len(ordered_options) else None
    detail_sym, detail_idx = reconcile_detail_selection(
        ordered_options, _current_sym if isinstance(_current_sym, str) else None, default_sym
    )
    st.session_state[DETAIL_INDEX_KEY] = detail_idx
    st.session_state[DETAIL_SELECT_KEY] = detail_sym

    option_count = max(1, len(ordered_options))
    index_width = max(2, len(str(option_count)))
    option_labels = {
        sym: f"[{idx:0{index_width}d}] {display_name(sym)}"
        for idx, sym in enumerate(ordered_options, 1)
    }

    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns([11, 1, 1, 1])

    with detail_col2:
        prev_disabled = detail_idx <= 0
        if st.button("▲", disabled=prev_disabled, key="detail_prev", help="이전 종목", use_container_width=True):
            if detail_idx > 0:
                new_sym = ordered_options[detail_idx - 1]
                st.session_state[DETAIL_SELECT_KEY] = new_sym
                st.session_state[DETAIL_INDEX_KEY] = detail_idx - 1
                st.rerun()

    with detail_col3:
        next_disabled = detail_idx >= len(ordered_options) - 1
        if st.button("▼", disabled=next_disabled, key="detail_next", help="다음 종목", use_container_width=True):
            if detail_idx < len(ordered_options) - 1:
                new_sym = ordered_options[detail_idx + 1]
                st.session_state[DETAIL_SELECT_KEY] = new_sym
                st.session_state[DETAIL_INDEX_KEY] = detail_idx + 1
                st.rerun()

    with detail_col4:
        end_disabled = detail_idx >= len(ordered_options) - 1
        if st.button("⏬", disabled=end_disabled, key="detail_end", help="맨 끝으로 이동", use_container_width=True):
            if detail_idx < len(ordered_options) - 1:
                new_sym = ordered_options[-1]
                st.session_state[DETAIL_SELECT_KEY] = new_sym
                st.session_state[DETAIL_INDEX_KEY] = len(ordered_options) - 1
                st.rerun()

    with detail_col1:
        detail_sym = st.selectbox(
            "종목 선택",
            options=ordered_options,
            index=detail_idx,
            format_func=lambda s: option_labels.get(s, display_name(s)),
            key=DETAIL_SELECT_KEY,
            label_visibility="collapsed",
        )
        # Streamlit already reruns on select change; sync derived index only.
        if detail_sym in ordered_options:
            st.session_state[DETAIL_INDEX_KEY] = ordered_options.index(detail_sym)

    # Atom: symbol + display_name + series bound together (never index-sliced)
    _atom_cache = st.session_state.setdefault(DETAIL_ATOM_CACHE_KEY, {})
    _atom_key = (detail_sym, period, _bundle_fp)
    atom = _atom_cache.get(_atom_key)
    if atom is None or not atom.is_consistent() or atom.symbol != detail_sym:
        atom = build_detail_view_atom(
            detail_sym, prices_krw, name_map=NAME_MAP, mom=mom, period_key=period
        )
        if atom is not None:
            _atom_cache.clear()
            _atom_cache[_atom_key] = atom

    if atom is None or not atom.is_consistent() or atom.symbol != detail_sym:
        st.error(
            "세부 보기 데이터 불일치: 선택한 종목과 차트 시계열을 안전하게 결합하지 못했습니다. "
            "캐시를 초기화한 뒤 다시 시도해 주세요."
        )
    else:
        st.caption(f"**{atom.symbol}** · {atom.display_name}")
        s_full = atom.series_krw.dropna()
        win_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
        win = win_map.get(period, 126)
        s = s_full.iloc[-win:] if s_full.shape[0] > win else s_full

        def _global_rebased_log_range(prices: pd.DataFrame, period_key: str):
            if prices.empty:
                return None
            win_map_local = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
            win_local = win_map_local.get(period_key, 126)
            sub = prices.iloc[-win_local:] if prices.shape[0] > win_local else prices
            rebased_list = []
            for col in sub.columns:
                s_col = sub[col].dropna()
                if s_col.empty:
                    continue
                rebased_list.append(_rebase_100(s_col))
            if not rebased_list:
                return None
            all_vals = pd.concat(rebased_list, axis=0).dropna()
            all_vals = all_vals[all_vals > 0]
            if all_vals.empty:
                return None
            mn = float(all_vals.min())
            mx = float(all_vals.max())
            mn = max(mn, 1e-6)
            mx = max(mx, mn * 1.000001)
            return (float(np.log10(mn)), float(np.log10(mx)))

        def _global_drawdown_range(prices: pd.DataFrame, period_key: str):
            if prices.empty:
                return None
            win_map_local = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
            win_local = win_map_local.get(period_key, 126)
            dd_list = []
            for col in prices.columns:
                s_col = prices[col].dropna()
                if s_col.empty:
                    continue
                s_win = s_col.iloc[-win_local:] if s_col.shape[0] > win_local else s_col
                roll_max = s_win.cummax()
                dd_list.append((s_win / roll_max - 1.0) * 100.0)
            if not dd_list:
                return None
            dd_vals = pd.concat(dd_list, axis=0).dropna()
            if dd_vals.empty:
                return None
            mn = float(dd_vals.min())
            mx = float(dd_vals.max())
            if mn == mx:
                return (mn - 1.0, mx + 1.0)
            pad = (mx - mn) * 0.05
            return (mn - pad, mx + pad)

        valid_for_scale = [
            sym for sym in ordered_options
            if sym in mom.index and not pd.isna(mom.loc[sym, "FMS"]) and mom.loc[sym, "FMS"] != -999.0
        ]
        base_prices = prices_krw[valid_for_scale] if valid_for_scale else prices_krw[ordered_options]
        y_global_log = _global_rebased_log_range(base_prices, period)
        y_dd_global = _global_drawdown_range(base_prices, period)

        hover_label = f"{atom.display_name} ({atom.symbol})"
        s100 = _rebase_100(s)
        ma5 = s100.rolling(window=5, min_periods=5).mean()
        e20, e50, e200 = ema(s100, 20), ema(s100, 50), ema(s100, 200)
        fig_det = go.Figure()
        fig_det.add_trace(go.Scatter(
            x=s100.index, y=s100.values, mode="lines", name="Rebased(100)",
            customdata=np.array([hover_label] * len(s100)),
            hovertemplate="%{customdata}<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))
        fig_det.add_trace(
            go.Scatter(
                x=ma5.index,
                y=ma5.values,
                mode="lines",
                name="5일 MA",
                line=dict(color="#c8c8c8", width=1),
                customdata=np.array([hover_label] * len(ma5)),
                hovertemplate="%{customdata}<br>5일 MA<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            )
        )
        fig_det.add_trace(go.Scatter(
            x=e20.index, y=e20.values, mode="lines", name="EMA20",
            customdata=np.array([hover_label] * len(e20)),
            hovertemplate="%{customdata}<br>EMA20<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))
        fig_det.add_trace(go.Scatter(
            x=e50.index, y=e50.values, mode="lines", name="EMA50",
            customdata=np.array([hover_label] * len(e50)),
            hovertemplate="%{customdata}<br>EMA50<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))
        fig_det.add_trace(go.Scatter(
            x=e200.index, y=e200.values, mode="lines", name="EMA200",
            customdata=np.array([hover_label] * len(e200)),
            hovertemplate="%{customdata}<br>EMA200<br>%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        ))
        fig_det.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Rebased 100 (log)",
            yaxis=dict(type="log"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
            ),
        )
        if y_global_log is not None:
            fig_det.update_yaxes(range=[y_global_log[0], y_global_log[1]])
        st.plotly_chart(fig_det, use_container_width=True, config={"displayModeBar": False})

        roll_max = s.cummax()
        dd = (s / roll_max - 1.0) * 100.0
        fig_dd = go.Figure([go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name="Drawdown(%)",
            customdata=np.array([hover_label] * len(dd)),
            hovertemplate="%{customdata}<br>Drawdown<br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        )])
        fig_dd.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
        if y_dd_global is not None:
            fig_dd.update_yaxes(range=[y_dd_global[0], y_dd_global[1]])
        st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

        row = atom.fms_row
        badges = []
        if row is not None:
            if "R_1M" in row and row["R_1M"] > 0:
                badges.append("1M +")
            if "AboveEMA50" in row and row["AboveEMA50"] > 0:
                badges.append("EMA50 상회")
            if "R_3M" in row and row["R_3M"] > 0:
                badges.append("3M +")
            if "ΔFMS_1D" in row and row["ΔFMS_1D"] > 0:
                badges.append("가속(1D+)")
            if "ΔFMS_5D" in row and row["ΔFMS_5D"] > 0:
                badges.append("가속(5D+)")
            if badges:
                st.markdown(
                    " ".join('<span class="badge">%s</span>' % b for b in badges),
                    unsafe_allow_html=True,
                )
# ==============================
# ④ 수익률–변동성 이동맵
# ==============================
st.subheader("수익률–변동성 이동맵 (최근 상태 → 어디서 왔는가)")

cc1, cc2 = st.columns([1, 1])
with cc1:
    plot_n = st.selectbox("표시 종목 수", [10, 20, 30, 40, 50, 60], index=1, help="상위 랭킹 기준으로 제한해 과밀도 완화")
with cc2:
    motion_mode = st.selectbox("모션(애니메이션)", ["끄기", "최근 10일", "최근 20일"], index=0,
                               help="프레임마다 현재 위치와 꼬리를 동시에 갱신")

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

# ------------------------------
# ⑤ 표 (컬럼 자동 재구성)
# ------------------------------
st.subheader("모멘텀 테이블 (가속/추세/수익률)")
disp = mom.copy()

# FMS 컬럼 표시
for c in ["R_1W","R_1M","R_3M","R_4M","R_YTD","AboveEMA50"]:
    if c in disp:
        disp[c] = (disp[c]*100).round(2)

# R2_3M 컬럼 표시 (0~1 사이 값이므로 100 곱하지 않음, 소수점 3자리)
if "R2_3M" in disp:
    disp["R2_3M"] = disp["R2_3M"].round(3)

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
    
    # 공식에서 Z(...) 변수명을 순서대로 추출
    variable_pattern = r"Z\(([^)]+)\)"
    matches = re.findall(variable_pattern, fms_formula)
    
    # 변수명을 실제 컬럼명으로 매핑
    variable_mapping = {
        '1M수익률': 'R_1M',
        '3M수익률': 'R_3M',
        '3M_R2': 'R2_3M',
        'EMA50상대위치': 'AboveEMA50',
        '20일변동성': 'Vol20(ann)',
        'R2_3M': 'R2_3M',
        'DD_RECOVERY': 'DD_RECOVERY',
        'TREND_QUALITY_21D': 'TREND_QUALITY_21D',
        'JUMP_DISCONTINUITY_3M': 'JUMP_DISCONTINUITY_3M',
        'UNDER_EMA20_DAYS': 'UNDER_EMA20_DAYS',
        'R_3M': 'R_3M',
        'STALE_AGE': 'STALE_AGE',
        'UP_STREAK_5D': 'UP_STREAK_5D',
        'TREND_EFFICIENCY_REWARD_15D': 'TREND_EFFICIENCY_REWARD_15D',
        'RANGE_COMPRESSION_20D': 'RANGE_COMPRESSION_20D',
    }
    
    for var_name in matches:
        var_name = var_name.strip().strip("'\"")
        if var_name in variable_mapping:
            col_name = variable_mapping[var_name]
            if col_name in available_columns and col_name not in column_order:
                fms_variables.append(col_name)
    
    # FMS 변수들을 공식에 나타난 순서대로 추가
    column_order.extend(fms_variables)
    
    # 4. 나머지 보조 변수들 추가
    remaining_columns = [col for col in available_columns if col not in column_order]
    
    # 보조 변수들을 우선순위에 따라 정렬
    priority_order = ['ΔFMS_1D', 'ΔFMS_5D', 'R_1W', 'R_4M', 'R_YTD']
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

# ==============================
# ⑤-1 FMS 재보정 (A/B 그래프 비교, 스냅샷 고정)
# ==============================
st.subheader("FMS 재보정")
st.info(
    "이 UI는 고정 snapshot에서 사람의 A/B 정답 순서를 수집합니다. "
    "재보정은 saved_at 기준 최신 완료 세션 하나만 사용해 0점에서 별도 후보를 피팅하며, "
    "후보 생성만으로 현재 production FMS가 변경되지는 않습니다."
)

def _slice_series_by_period(s_full: pd.Series, period_key: str) -> pd.Series:
    win_map_local = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
    win_local = win_map_local.get(period_key, 126)
    if s_full.shape[0] > win_local:
        return s_full.iloc[-win_local:]
    return s_full

def _make_detail_price_fig(
    s: pd.Series,
    *,
    title: str,
    y_range_log10: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """
    A/B 비교용 가격 차트.
    - 시작값을 100으로 리베이스
    - 로그 스케일(yaxis.type='log')
    - y축 범위는 log10 공간(range=[log10(min), log10(max)])에서 동일하게 고정
    """
    s100 = _rebase_100(s)
    e20_, e50_, e200_ = ema(s100, 20), ema(s100, 50), ema(s100, 200)
    fig_ = go.Figure()
    fig_.add_trace(go.Scatter(x=s100.index, y=s100.values, mode="lines", name="Rebased(100)"))
    fig_.add_trace(go.Scatter(x=e20_.index, y=e20_.values, mode="lines", name="EMA20"))
    fig_.add_trace(go.Scatter(x=e50_.index, y=e50_.values, mode="lines", name="EMA50"))
    fig_.add_trace(go.Scatter(x=e200_.index, y=e200_.values, mode="lines", name="EMA200"))
    layout_kwargs = dict(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Rebased 100 (log)",
        title=dict(text=title, x=0.0, xanchor="left", y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig_.update_layout(**layout_kwargs)
    fig_.update_yaxes(type="log")
    if y_range_log10 is not None and all(v is not None for v in y_range_log10):
        fig_.update_yaxes(range=[y_range_log10[0], y_range_log10[1]])
    return fig_

def _make_drawdown_fig(s: pd.Series, *, y_range: Optional[Tuple[float, float]] = None) -> go.Figure:
    roll_max_ = s.cummax()
    dd_ = (s / roll_max_ - 1.0) * 100.0
    fig_ = go.Figure([go.Scatter(x=dd_.index, y=dd_.values, mode="lines", name="Drawdown(%)")])
    fig_.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%")
    if y_range is not None:
        fig_.update_yaxes(range=[y_range[0], y_range[1]])
    return fig_

def _calc_shared_y_range_log10(series_a: pd.Series, series_b: pd.Series) -> Optional[Tuple[float, float]]:
    a100 = _rebase_100(series_a)
    b100 = _rebase_100(series_b)
    vals = pd.concat([a100, b100], axis=0).dropna()
    if vals.empty:
        return None
    # log axis는 양수만 허용
    vals = vals[vals > 0]
    if vals.empty:
        return None
    mn = float(vals.min())
    mx = float(vals.max())
    if mn == mx:
        pad = 1.0 if mn == 0 else abs(mn) * 0.05
        mn_p, mx_p = (mn - pad), (mx + pad)
    else:
        pad = (mx - mn) * 0.05
        mn_p, mx_p = (mn - pad), (mx + pad)
    # 안정적으로 양수 유지
    mn_p = max(mn_p, 1e-6)
    mx_p = max(mx_p, mn_p * 1.000001)
    return (float(np.log10(mn_p)), float(np.log10(mx_p)))

def _calc_shared_dd_range(series_a: pd.Series, series_b: pd.Series) -> Optional[Tuple[float, float]]:
    def _dd(s_):
        rm = s_.cummax()
        return (s_ / rm - 1.0) * 100.0
    dd_vals = pd.concat([_dd(series_a), _dd(series_b)], axis=0).dropna()
    if dd_vals.empty:
        return None
    mn = float(dd_vals.min())
    mx = float(dd_vals.max())
    # Drawdown은 보통 0~음수 범위. 위아래 약간 패딩.
    if mn == mx:
        return (mn - 1.0, mx + 1.0)
    pad = (mx - mn) * 0.05
    return (mn - pad, mx + pad)

if "fms_calibration" not in st.session_state:
    st.session_state.fms_calibration = {"active_session_id": None, "state": None}

cal = st.session_state.fms_calibration

with st.expander("🧪 재보정 세션 관리 (시작/불러오기/저장)", expanded=False):
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("**새 세션 시작 (현재 화면의 데이터로 스냅샷 고정)**")
        session_name = st.text_input("세션 이름(선택)", value="", help="비워두면 자동 이름을 사용합니다.")
        start_btn = st.button("✅ 새 세션 시작", use_container_width=True)

    with col_b:
        st.markdown("**기존 세션 불러오기**")
        existing_sessions = list_sessions()
        pick = st.selectbox("세션 선택", options=[""] + existing_sessions, index=0)
        load_btn = st.button("📂 불러오기", disabled=(pick == ""), use_container_width=True)

    if start_btn:
        # 스냅샷 생성: 현재 prices_krw 기준(작업 중 갱신 금지)
        snapshot_id = create_snapshot_id("fms")
        # 비교 대상: 현재 관심종목 중 prices_krw에 존재하는 종목만
        symbols_all = [str(s) for s in st.session_state.watchlist]
        available = [s for s in symbols_all if s in prices_krw.columns]
        snapshot_prices = prices_krw[available].copy()

        meta = {
            "account_mode": st.session_state.account_mode,
            "chart_period": period,
            "symbols": available,
            "note": "FMS recalibration snapshot. Do not refresh during session.",
        }
        save_snapshot(snapshot_id, prices_krw=snapshot_prices, ohlc_data=None, meta=meta)

        session_id = session_name.strip() or f"cal_{snapshot_id}"
        state = init_sort_state(available, snapshot_id=snapshot_id, meta=meta)
        save_session(session_id, state)

        cal["active_session_id"] = session_id
        cal["state"] = state
        st.success(f"세션 시작됨: {session_id} (snapshot: {snapshot_id})")
        st.rerun()

    if load_btn and pick:
        state = load_session(pick)
        cal["active_session_id"] = pick
        cal["state"] = state
        st.success(f"세션 불러옴: {pick}")
        st.rerun()

active_session_id = cal.get("active_session_id")
state = cal.get("state")

if active_session_id and state:
    # 항상 스냅샷에서 로드 (작업 중 최신 갱신 금지)
    snap_id = state.get("snapshot_id")
    try:
        snap_prices, _, snap_meta = load_snapshot(snap_id)
    except Exception as e:
        st.error(f"스냅샷 로드 실패: {e}")
        snap_prices, snap_meta = pd.DataFrame(), {}

    # 저장 버튼 (수동)
    top_col1, top_col2, top_col3 = st.columns([1, 1, 2])
    with top_col1:
        if st.button("💾 지금 저장", use_container_width=True):
            save_session(active_session_id, state)
            st.toast("저장 완료")
    with top_col2:
        if st.button("⛔ 세션 종료(메모리만)", use_container_width=True):
            st.session_state.fms_calibration = {"active_session_id": None, "state": None}
            st.rerun()
    with top_col3:
        st.caption(
            f"세션: {active_session_id} | snapshot: {snap_id} | "
            f"phase: {state.get('phase')} | saved_at: {state.get('saved_at', '—')}"
        )

    if state.get("phase") == "done":
        st.warning(
            "완료 세션을 다시 저장하면 saved_at이 갱신되어 "
            "'최신 완료 세션' 피팅 대상이 바뀔 수 있습니다."
        )

    # 진행률 표시 (정렬 단계에서만)
    if state.get("phase") == "sorting":
        done = int(state.get("comparisons_done", 0))
        est = int(state.get("comparisons_est_max", 1)) or 1
        st.progress(min(1.0, done / est), text=f"정렬 진행: 비교 {done}/{est} (최대치 기준)")

        pair = get_next_comparison(state)
        if pair is None:
            # 내부 상태가 review로 넘어갔을 수 있음
            save_session(active_session_id, state)
            st.rerun()
        sym_a, sym_b = pair

        # 데이터 준비(기간 슬라이스는 스냅샷의 chart_period 사용)
        chart_period = (state.get("meta") or {}).get("chart_period") or period
        sA_full = snap_prices[sym_a].dropna() if sym_a in snap_prices.columns else pd.Series(dtype=float)
        sB_full = snap_prices[sym_b].dropna() if sym_b in snap_prices.columns else pd.Series(dtype=float)
        sA = _slice_series_by_period(sA_full, chart_period)
        sB = _slice_series_by_period(sB_full, chart_period)

        y_shared = _calc_shared_y_range_log10(sA, sB)
        dd_shared = _calc_shared_dd_range(sA, sB)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(_make_detail_price_fig(sA, title=f"A: {display_name(sym_a)}", y_range_log10=y_shared),
                            use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(_make_drawdown_fig(sA, y_range=dd_shared),
                            use_container_width=True, config={"displayModeBar": False})
        with right:
            st.plotly_chart(_make_detail_price_fig(sB, title=f"B: {display_name(sym_b)}", y_range_log10=y_shared),
                            use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(_make_drawdown_fig(sB, y_range=dd_shared),
                            use_container_width=True, config={"displayModeBar": False})

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("⬅️ A가 더 우수함", use_container_width=True):
                apply_sort_choice(state, sym_a, ts=datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"))
                save_session(active_session_id, state)
                st.rerun()
        with btn2:
            if st.button("➡️ B가 더 우수함", use_container_width=True):
                apply_sort_choice(state, sym_b, ts=datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"))
                save_session(active_session_id, state)
                st.rerun()

    elif state.get("phase") == "review":
        if not state.get("review_queue"):
            build_review_queue_from_final(state, fraction=0.10)
            save_session(active_session_id, state)
            st.rerun()

        pair = get_next_review_pair(state)
        if pair is None:
            save_session(active_session_id, state)
            st.rerun()
        sym_a, sym_b = pair

        st.info("최종 인접 등수 재검토(약 10%). 이전 선택과 다르면 불일치로 기록되고, 인접 순서를 조정합니다.")

        chart_period = (state.get("meta") or {}).get("chart_period") or period
        sA_full = snap_prices[sym_a].dropna() if sym_a in snap_prices.columns else pd.Series(dtype=float)
        sB_full = snap_prices[sym_b].dropna() if sym_b in snap_prices.columns else pd.Series(dtype=float)
        sA = _slice_series_by_period(sA_full, chart_period)
        sB = _slice_series_by_period(sB_full, chart_period)

        y_shared = _calc_shared_y_range_log10(sA, sB)
        dd_shared = _calc_shared_dd_range(sA, sB)

        left, right = st.columns(2)
        with left:
            st.plotly_chart(_make_detail_price_fig(sA, title=f"A: {display_name(sym_a)}", y_range_log10=y_shared),
                            use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(_make_drawdown_fig(sA, y_range=dd_shared),
                            use_container_width=True, config={"displayModeBar": False})
        with right:
            st.plotly_chart(_make_detail_price_fig(sB, title=f"B: {display_name(sym_b)}", y_range_log10=y_shared),
                            use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(_make_drawdown_fig(sB, y_range=dd_shared),
                            use_container_width=True, config={"displayModeBar": False})

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("⬅️ (재검토) A가 더 우수함", use_container_width=True):
                res = apply_review_choice(state, sym_a, ts=datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"))
                save_session(active_session_id, state)
                if state.get("inconsistencies"):
                    st.warning("불일치가 기록되었습니다. (세션 상단에서 저장 후 최종 결과를 확인하세요.)")
                if res.get("status") == "done":
                    st.success("재검토 완료")
                st.rerun()
        with btn2:
            if st.button("➡️ (재검토) B가 더 우수함", use_container_width=True):
                res = apply_review_choice(state, sym_b, ts=datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"))
                save_session(active_session_id, state)
                if state.get("inconsistencies"):
                    st.warning("불일치가 기록되었습니다. (세션 상단에서 저장 후 최종 결과를 확인하세요.)")
                if res.get("status") == "done":
                    st.success("재검토 완료")
                st.rerun()

    elif state.get("phase") == "done":
        ranking = state.get("final_ranking") or []
        st.success("정렬/재검토가 완료되었습니다.")

        if state.get("inconsistencies"):
            st.warning(f"불일치 감지: {len(state.get('inconsistencies'))}건 (인접 재검토에서 초기 판단과 달랐음)")
            with st.expander("불일치 상세 보기", expanded=False):
                st.json(state.get("inconsistencies"))

        if ranking:
            df_rank = pd.DataFrame({"rank": range(1, len(ranking) + 1), "symbol": ranking})
            st.dataframe(df_rank, use_container_width=True, hide_index=True)
            csv = df_rank.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("📥 최종 순서 CSV 다운로드", data=csv, file_name=f"{active_session_id}_final_ranking.csv", mime="text/csv")

        st.info(
            "다음 단계: 이 정답 순서로 CLI에서 **후보**를 생성합니다. "
            "산출물은 candidate-only이며 production FMS는 자동으로 변경되지 않습니다.\n\n"
            "1. `python fms_recalib_build_features.py`\n"
            "2. `python -m calibration.fms_recalib_inspect_patterns`\n"
            "3. `python fms_recalib_nonlinear_mc.py`\n"
            "4. `python -m calibration.fms_recalib_plot_residuals`"
        )

# ------------------------------
# ⑥ 디버그/진단
# ------------------------------
with st.expander("디버그 로그 / 진단 (복사해서 붙여넣기 가능)"):
    # 빈 DataFrame 체크
    if prices_krw.empty:
        diag = {
            "current_time_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "error": "prices_krw is empty (all columns excluded by coverage threshold)",
            "cols_total": 0,
            "env_CURL_CFFI_DISABLE_CACHE": os.environ.get("CURL_CFFI_DISABLE_CACHE")
        }
    else:
        last_row = prices_krw.iloc[-1]
        usa_cols = [c for c in prices_krw.columns if classify(c)=="USA"]
        kor_cols = [c for c in prices_krw.columns if classify(c)=="KOR"]
        jpn_cols = [c for c in prices_krw.columns if classify(c)=="JPN"]
        
        # 현재 시간 (KST)
        current_time_kst = datetime.now(KST)
        
        diag = {
            "current_time_kst": current_time_kst.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "last_date": str(prices_krw.index[-1].date()),
            "cols_total": prices_krw.shape[1],
            "last_notna_total": int(last_row.notna().sum()),
            "last_notna_USA": int(last_row[usa_cols].notna().sum()) if usa_cols else 0,
            "last_notna_KOR": int(last_row[kor_cols].notna().sum()) if kor_cols else 0,
            "last_notna_JPN": int(last_row[jpn_cols].notna().sum()) if jpn_cols else 0,
            "env_CURL_CFFI_DISABLE_CACHE": os.environ.get("CURL_CFFI_DISABLE_CACHE")
        }
    
    # OHLC 데이터 상태
    if ohlc_data is not None and not ohlc_data.empty:
        diag["ohlc_status"] = {
            "available": True,
            "total_symbols": len(watchlist_symbols),
            "ohlc_missing_count": len(ohlc_missing) if ohlc_missing else 0,
            "ohlc_missing_symbols": list(ohlc_missing) if ohlc_missing else [],
            "ohlc_shape": list(ohlc_data.shape) if ohlc_data is not None else None,
            "ohlc_index_range": [str(ohlc_data.index[0]), str(ohlc_data.index[-1])] if ohlc_data is not None and len(ohlc_data) > 0 else None,
        }
    else:
        diag["ohlc_status"] = {
            "available": False,
            "error": "OHLC 데이터 없음 또는 비어있음"
        }
    
    # 실격된 종목 상세 디버그 정보 (모든 국가)
    if 'Filter_Status' in mom.columns:
        disqualified_all = mom[mom['Filter_Status'] != '정상']
        if len(disqualified_all) > 0:
            filter_debug_details = {}
            for symbol in disqualified_all.index:
                if ohlc_data is not None:
                    debug_info = get_filter_debug_info(ohlc_data, symbol)
                    filter_debug_details[symbol] = debug_info
            
            diag["disqualified_stocks"] = {
                "count": len(disqualified_all),
                "symbols": list(disqualified_all.index),
                "filter_status": {symbol: str(disqualified_all.loc[symbol, 'Filter_Status']) for symbol in disqualified_all.index},
                "detailed_debug": filter_debug_details
            }
            
            # 실격된 종목 요약 표시 (읽기 쉽게)
            st.markdown("### 🚫 실격된 종목 목록")
            summary_data = []
            for symbol in disqualified_all.index:
                filter_status = str(disqualified_all.loc[symbol, 'Filter_Status'])
                debug_info = filter_debug_details.get(symbol, {})
                
                # 국가 분류 추가
                country = classify(symbol)
                country_name = {"USA": "🇺🇸 미국", "KOR": "🇰🇷 한국", "JPN": "🇯🇵 일본"}.get(country, country)
                
                # 요약 정보 추출
                summary_info = {
                    "종목": symbol,
                    "국가": country_name,
                    "필터 상태": filter_status,
                    "OHLC 데이터": "✅" if debug_info.get('has_ohlc') else "❌",
                    "데이터 포인트": debug_info.get('data_points', 0),
                    "마지막 날짜": debug_info.get('last_date', 'N/A'),
                }
                
                # 30% 초과 날짜 수 추가
                if 'extreme_days_count' in debug_info:
                    summary_info["30% 초과 일수"] = debug_info.get('extreme_days_count', 0)
                
                # 최근 True Range 변동률 추가
                if 'recent_data' in debug_info and debug_info['recent_data'].get('last_true_range_vol_pct') is not None:
                    vol_pct = debug_info['recent_data']['last_true_range_vol_pct']
                    summary_info["최근 변동률(%)"] = f"{vol_pct:.2f}"
                
                # 오류 정보 추가
                if debug_info.get('error'):
                    summary_info["오류"] = debug_info.get('error', '')
                
                summary_data.append(summary_info)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # 각 종목별 상세 정보 (expander로 접기 가능)
            st.markdown("### 📋 종목별 상세 디버그 정보")
            for symbol in disqualified_all.index:
                debug_info = filter_debug_details.get(symbol, {})
                filter_status = str(disqualified_all.loc[symbol, 'Filter_Status'])
                
                # 국가 분류 추가
                country = classify(symbol)
                country_name = {"USA": "🇺🇸 미국", "KOR": "🇰🇷 한국", "JPN": "🇯🇵 일본"}.get(country, country)
                
                with st.expander(f"🔍 {symbol} ({country_name}) - {filter_status}", expanded=False):
                    # 기본 정보
                    st.markdown("#### 기본 정보")
                    basic_info = {
                        "종목": symbol,
                        "국가": country_name,
                        "필터 상태": filter_status,
                        "OHLC 데이터": "✅" if debug_info.get('has_ohlc') else "❌",
                        "데이터 포인트": debug_info.get('data_points', 0),
                        "마지막 날짜": debug_info.get('last_date', 'N/A'),
                    }
                    if debug_info.get('error'):
                        basic_info["오류"] = debug_info.get('error', '')
                    st.json(basic_info)
                    
                    # 최근 데이터 정보
                    if 'recent_data' in debug_info and debug_info['recent_data']:
                        st.markdown("#### 최근 데이터 상세")
                        recent = debug_info['recent_data']
                        recent_display = {
                            "날짜": recent.get('last_date', 'N/A'),
                            "전일 종가": recent.get('prev_close'),
                            "당일 종가": recent.get('last_close'),
                            "당일 고가": recent.get('last_high'),
                            "당일 저가": recent.get('last_low'),
                            "True Range 변동률(%)": recent.get('last_true_range_vol_pct'),
                        }
                        if 'true_range_components' in recent:
                            recent_display["True Range 구성 요소"] = recent['true_range_components']
                        st.json(recent_display)
                    
                    # 30% 초과 날짜 상세
                    if 'extreme_days_detail' in debug_info and debug_info['extreme_days_detail']:
                        st.markdown(f"#### ⚠️ 치명적 변동성 (30% 초과) - 총 {debug_info.get('extreme_days_count', 0)}일")
                        extreme_df = pd.DataFrame(debug_info['extreme_days_detail'])
                        st.dataframe(extreme_df, use_container_width=True, hide_index=True)
                    
                    # 하방리스크 상세
                    if 'severe_days_detail' in debug_info and debug_info['severe_days_detail']:
                        st.markdown(f"#### ⚠️ 반복적 하방리스크 (-7% 미만) - 총 {debug_info.get('severe_days_count', 0)}일")
                        severe_df = pd.DataFrame(debug_info['severe_days_detail'])
                        st.dataframe(severe_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
    
    # 전체 진단 정보 (JSON 형태로 복사 가능)
    st.markdown("### 📊 전체 진단 정보 (JSON - 복사 가능)")
    st.json(diag)
    st.text_area("LOG", value="\n".join(st.session_state["LOG"][-400:]), height=200)
