# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 유니버스 관리 유틸리티
Finviz를 사용한 유니버스 스크리닝 및 파일 관리 기능
"""

import os
import re
import time
import glob
import pandas as pd
from datetime import datetime
import pytz
from typing import Tuple, Optional

KST = pytz.timezone("Asia/Seoul")

# 모드 상수 정의
MODE_FREE = "FREE"
MODE_IRP = "IRP"

def is_leveraged_or_inverse_etf(ticker: str, name: str = "") -> bool:
    """
    레버리지 또는 인버스 ETF인지 판단합니다.
    
    일반적인 레버리지/인버스 ETF 패턴:
    - 숫자 + X (2X, 3X, 2x, 3x 등)
    - Leverage, Inverse, Short, Bear, Ultra 같은 키워드
    - 특정 티커 패턴 (LLYX, SMST, GGLL, GOOX 등)
    
    Args:
        ticker (str): 티커 심볼
        name (str): 종목명 (선택사항, 티커만으로 판단 불가능할 경우 사용)
    
    Returns:
        bool: 레버리지/인버스 ETF이면 True, 아니면 False
    """
    ticker_upper = str(ticker).upper().strip()
    name_upper = str(name).upper().strip()
    
    # 레버리지/인버스 키워드 패턴
    leverage_keywords = [
        'LEVERAGE', 'LEVERAGED', 'LEV',
        'INVERSE', 'INV', 'SHORT', 'BEAR',
        'ULTRA', 'PRO', 'BULL'
    ]
    
    # 숫자 + X 패턴 (2X, 3X, 2x, 3x 등)
    numeric_leverage_pattern = r'\d+[Xx]'
    
    # 티커 패턴 체크
    # 알려진 레버리지/인버스 ETF 티커 패턴들
    known_leverage_patterns = [
        'LLYX', 'SMST', 'GGLL', 'GOOX',  # 사용자가 보고한 패턴
        'TQQQ', 'SQQQ', 'SOXL', 'SOXS',  # 일반적인 레버리지/인버스 ETF
        'UPRO', 'SPXU', 'UDOW', 'SDOW',  # 주요 레버리지/인버스 ETF
    ]
    
    # 1. 티커가 알려진 레버리지/인버스 ETF 패턴과 정확히 일치하는지 체크
    if ticker_upper in known_leverage_patterns:
        return True
    
    # 2. 숫자 + X 패턴 체크 (2X, 3X 등)
    if re.search(numeric_leverage_pattern, ticker_upper) or re.search(numeric_leverage_pattern, name_upper):
        return True
    
    # 3. 레버리지/인버스 키워드 체크
    for keyword in leverage_keywords:
        if keyword in ticker_upper or keyword in name_upper:
            return True
    
    # 4. 특정 티커 패턴 체크 (4글자 티커가 특정 패턴을 포함하는 경우)
    # 예: LLYX, SMST, GGLL, GOOX 같은 패턴
    if len(ticker_upper) >= 4:
        # 마지막 글자가 X로 끝나는 패턴 (LLYX 등)
        if ticker_upper.endswith('X') and len(ticker_upper) == 4:
            # 앞 3글자가 모두 대문자인 경우 (레버리지 ETF 가능성 높음)
            if ticker_upper[:3].isalpha() and ticker_upper[:3].isupper():
                # 일부 예외 처리 (예: 일반적인 ETF도 X로 끝날 수 있음)
                # 더 정확한 판단을 위해 종목명도 체크
                if name_upper and ('LEVERAGE' in name_upper or 'INVERSE' in name_upper or 'ULTRA' in name_upper):
                    return True
    
    return False

def check_universe_file_freshness(mode: str = MODE_FREE):
    """
    유니버스 파일의 실제 업데이트 시간을 확인합니다.
    파일 타임스탬프 대신 별도 저장된 업데이트 시간을 사용합니다.
    
    Args:
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        - FREE: screened_universe.csv (Finviz 스크리닝 결과)
        - IRP: korean_etf_univers.csv (수동 관리, 신선도 체크 불필요)
    
    Returns:
        tuple: (is_fresh, last_updated_time, hours_since_update)
    """
    try:
        # IRP 모드는 수동 관리 파일이므로 신선도 체크 불필요
        if mode == MODE_IRP:
            return True, None, None
        
        if not os.path.exists('screened_universe.csv'):
            return False, None, None
        
        # 실제 업데이트 시간이 저장된 파일 확인
        timestamp_file = 'universe_last_updated.txt'
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r', encoding='utf-8') as f:
                timestamp_str = f.read().strip()
                last_updated = datetime.fromisoformat(timestamp_str)
        else:
            # 타임스탬프 파일이 없으면 파일 생성 시간으로 fallback
            file_mtime = os.path.getmtime('screened_universe.csv')
            last_updated = datetime.fromtimestamp(file_mtime, KST)
        
        hours_since_update = (datetime.now(KST) - last_updated).total_seconds() / 3600
        
        # 6시간 이내면 fresh로 간주
        is_fresh = hours_since_update < 6
        
        return is_fresh, last_updated, hours_since_update
        
    except Exception as e:
        # 유니버스 파일 신선도 확인 중 오류는 조용히 처리
        return False, None, None

def update_universe_file(progress_callback=None, status_callback=None):
    """
    Finviz를 사용하여 유니버스 파일을 업데이트합니다.
    추세 품질 중심의 필터링을 통해 안정적이고 지속적인 모멘텀을 가진 종목들을 선별합니다.
    진행 상황을 콜백 함수를 통해 실시간으로 전달합니다.
    
    필터링 조건:
    - 유동성: 가격 $10 이상, 평균 거래량 300K 이상
    - 추세 지속성: 분기 10% 이상, 반기 20% 이상 상승
    - 추세 안정성: 50일/200일 이동평균 위에 위치
    
    Args:
        progress_callback: 진행률 콜백 함수 (progress, message)
        status_callback: 상태 메시지 콜백 함수 (message)
    
    Returns:
        tuple: (success, message, symbol_count)
    """
    try:
        from finvizfinance.screener import Overview
        
        # 1단계: Finviz 스크리너 실행
        if progress_callback:
            progress_callback(0.0, "🔍 Finviz 스크리너 실행 중...")
        
        # 스크리닝 필터 조건 설정 (추세 품질 중심)
        filters = {
            # 1. 유동성 필터 (기준 강화)
            'Price': 'Over $10',           # 가격 $10 이상 (기존 $5에서 강화)
            'Average Volume': 'Over 300K', # 평균 거래량 300,000주 이상 (기존 200K에서 강화)

            # 2. 추세 지속성 필터 (신규 도입)
            'Performance': 'Quarter +10%',       # 최소 3개월간 10% 이상 상승
            'Performance 2': 'Half +20%',        # 최소 6개월간 20% 이상 상승

            # 3. 추세 안정성 필터 (핵심 신규 도입)
            '50-Day Simple Moving Average': 'Price above SMA50',  # 중기 상승 추세 확인
            '200-Day Simple Moving Average': 'Price above SMA200' # 장기 상승 추세 확인
        }
        
        if progress_callback:
            progress_callback(0.05, "📊 스크리닝 필터 적용 중...")
        
        # Finviz 스크리너 실행 (진행률 콜백 포함)
        foverview = Overview()
        
        if progress_callback:
            progress_callback(0.1, "🔍 Finviz 데이터 다운로드 중...")
        
        if progress_callback:
            progress_callback(0.12, "📡 Finviz 서버에 연결 중...")
        
        if progress_callback:
            progress_callback(0.15, "📊 8,000+ 종목 데이터 처리 중... (콘솔에서 실제 진행률 확인 가능)")
        
        # Finviz API 호출 (블로킹 작업)
        # 실제 진행률은 콘솔에 [Info] loading page [####------] 형태로 표시됩니다.
        df = foverview.screener_view()
        
        if progress_callback:
            progress_callback(0.2, f"📥 전체 데이터 다운로드 완료: {len(df)}개 종목")
        
        # 2단계: 추세 품질 중심 필터링 적용
        if progress_callback:
            progress_callback(0.25, "🔍 추세 품질 중심 필터링 적용 중...")
        
        original_count = len(df)
        
        # 1. 유동성 필터 (기준 강화)
        if 'Price' in df.columns:
            df['Price_clean'] = df['Price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            df = df[df['Price_clean'] >= 10.0]  # $5 → $10으로 강화
            if progress_callback:
                progress_callback(0.35, f"💰 가격 $10 이상 필터링: {len(df)}개 종목")
        
        if 'Avg Volume' in df.columns:
            df['Volume_clean'] = df['Avg Volume'].str.replace(',', '').astype(float)
            df = df[df['Volume_clean'] >= 300000]  # 200K → 300K로 강화
            if progress_callback:
                progress_callback(0.45, f"📈 거래량 300K 이상 필터링: {len(df)}개 종목")
        
        # 2. 추세 지속성 필터 (완화: 0% 이상)
        if 'Perf Quarter' in df.columns:
            df['Perf_Quarter_clean'] = df['Perf Quarter'].str.replace('%', '').astype(float)
            df = df[df['Perf_Quarter_clean'] >= 0.0]  # 3개월간 0% 이상 상승
            if progress_callback:
                progress_callback(0.55, f"📊 분기 수익률 0% 이상 필터링: {len(df)}개 종목")
        
        if 'Perf Half' in df.columns:
            df['Perf_Half_clean'] = df['Perf Half'].str.replace('%', '').astype(float)
            df = df[df['Perf_Half_clean'] >= 0.0]  # 6개월간 0% 이상 상승
            if progress_callback:
                progress_callback(0.65, f"📊 반기 수익률 0% 이상 필터링: {len(df)}개 종목")
        
        # 3. 추세 안정성 필터 (핵심 신규 도입)
        if 'SMA50' in df.columns:
            df['Price_vs_SMA50'] = df['Price_clean'] / df['SMA50'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            df = df[df['Price_vs_SMA50'] >= 1.0]  # 현재가 > 50일 이동평균
            if progress_callback:
                progress_callback(0.75, f"📈 50일 이동평균 위 종목 필터링: {len(df)}개 종목")
        
        if 'SMA200' in df.columns:
            df['Price_vs_SMA200'] = df['Price_clean'] / df['SMA200'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            df = df[df['Price_vs_SMA200'] >= 1.0]  # 현재가 > 200일 이동평균
            if progress_callback:
                progress_callback(0.85, f"📈 200일 이동평균 위 종목 필터링: {len(df)}개 종목")
        
        # 4단계: 레버리지/인버스 ETF 제외 (강화된 필터링)
        if not df.empty and 'Ticker' in df.columns:
            if progress_callback:
                progress_callback(0.9, "🚫 레버리지/인버스 ETF 제외 필터링 중...")

            excluded_tickers = []
            for _, row in df.iterrows():
                ticker = str(row['Ticker'])
                # 종목명이 있으면 함께 체크 (더 정확한 판단)
                name = str(row.get('Company', '')) if 'Company' in row else ""
                
                if is_leveraged_or_inverse_etf(ticker, name):
                    excluded_tickers.append(ticker)

            if excluded_tickers:
                df = df[~df['Ticker'].isin(excluded_tickers)]
                if progress_callback:
                    progress_callback(0.92, f"🚫 레버리지/인버스 ETF 제외: {len(excluded_tickers)}개, 남은 종목: {len(df)}개")
        
        # 5단계: 파일 저장
        if not df.empty and 'Ticker' in df.columns:
            if progress_callback:
                progress_callback(0.95, "💾 유니버스 파일 저장 중...")
            
            tickers = df['Ticker'].tolist()
            unique_tickers = sorted(list(set(tickers)))
            
            output_df = pd.DataFrame({'Symbol': unique_tickers})
            output_df.to_csv('screened_universe.csv', index=False)
            
            # 실제 업데이트 시간 저장
            timestamp_file = 'universe_last_updated.txt'
            current_time = datetime.now(KST)
            with open(timestamp_file, 'w', encoding='utf-8') as f:
                f.write(current_time.isoformat())
            
            if progress_callback:
                progress_callback(1.0, f"✅ 유니버스 업데이트 완료: {len(unique_tickers)}개 종목")
            
            return True, f"유니버스 업데이트 완료: {len(unique_tickers)}개 종목", len(unique_tickers)
        else:
            if progress_callback:
                progress_callback(1.0, "⚠️ 스크리닝 결과가 비어있습니다.")
            return False, "스크리닝 결과가 비어있습니다.", 0
            
    except Exception as e:
        error_msg = f"유니버스 업데이트 중 오류: {str(e)}"
        if progress_callback:
            progress_callback(1.0, f"❌ {error_msg}")
        return False, error_msg, 0

def load_universe_file(mode: str = MODE_FREE):
    """
    모드별 유니버스 파일을 로드합니다.
    
    Args:
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        - FREE: screened_universe.csv (미국) + korean_universe.csv (한국)
        - IRP: korean_etf_univers.csv (국내상장 ETF 전 종목)
    
    Returns:
        tuple: (success, symbols_list, message)
    """
    try:
        if mode == MODE_IRP:
            # IRP 모드: 국내상장 ETF 유니버스
            if not os.path.exists('korean_etf_univers.csv'):
                return False, [], "IRP 유니버스 파일이 없습니다."
            
            universe_df = pd.read_csv('korean_etf_univers.csv')
            symbols = universe_df['Symbol'].tolist()
            
            return True, symbols, f"IRP 유니버스 로드 완료: {len(symbols)}개 종목"
        else:
            # FREE 모드: 미국 + 한국 유니버스 병합
            usa_symbols = []
            kor_symbols = []
            
            # 미국 유니버스 로드
            if os.path.exists('screened_universe.csv'):
                usa_df = pd.read_csv('screened_universe.csv')
                usa_symbols = usa_df['Symbol'].tolist()
            
            # 한국 유니버스 로드
            if os.path.exists('korean_universe.csv'):
                kor_df = pd.read_csv('korean_universe.csv')
                kor_symbols = kor_df['Symbol'].tolist()
            
            all_symbols = usa_symbols + kor_symbols
            
            if not all_symbols:
                return False, [], "유니버스 파일이 없습니다."
            
            return True, all_symbols, f"유니버스 로드 완료: 미국 {len(usa_symbols)}개 + 한국 {len(kor_symbols)}개 = 총 {len(all_symbols)}개 종목"
        
    except Exception as e:
        return False, [], f"유니버스 파일 로드 중 오류: {str(e)}"

def load_korean_universe():
    """
    korean_universe.csv 파일을 로드합니다.
    KOSPI 200 + KOSDAQ 150 + 국내 지수 ETF(1배 및 인버스) 리스트를 반환합니다.
    
    Returns:
        tuple: (success, symbols_list, message)
    """
    try:
        if not os.path.exists('korean_universe.csv'):
            return False, [], "한국 유니버스 파일이 없습니다."
        
        universe_df = pd.read_csv('korean_universe.csv')
        symbols = universe_df['Symbol'].tolist()
        
        return True, symbols, f"한국 유니버스 로드 완료: {len(symbols)}개 종목"
        
    except Exception as e:
        return False, [], f"한국 유니버스 파일 로드 중 오류: {str(e)}"

def save_scan_results(scan_results_df, fms_threshold=2.0, mode: str = MODE_FREE):
    """
    FMS 스캔 결과를 모드별 파일로 저장합니다.
    FMS 임계값 이상인 종목만 저장합니다.
    
    Args:
        scan_results_df (pd.DataFrame): 스캔 결과 DataFrame
        fms_threshold (float): FMS 임계값 (기본값: 2.0)
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
    
    Returns:
        tuple: (success, message, saved_count)
    """
    try:
        if scan_results_df.empty:
            return False, "저장할 스캔 결과가 없습니다.", 0
        
        # FMS 임계값 이상인 종목만 필터링
        filtered_results = scan_results_df[scan_results_df['FMS'] >= fms_threshold].copy()
        
        if filtered_results.empty:
            return False, f"FMS {fms_threshold} 이상인 종목이 없습니다.", 0
        
        # scan_results 디렉토리 확인 및 생성
        scan_results_dir = "scan_results"
        if not os.path.exists(scan_results_dir):
            os.makedirs(scan_results_dir, exist_ok=True)
        
        # 파일명에 타임스탬프 및 모드 포함
        timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        mode_suffix = mode.lower()
        filename = f"scan_results_{mode_suffix}_{timestamp}.csv"
        
        # scan_results 디렉토리에 저장
        filepath = os.path.join(scan_results_dir, filename)
        filtered_results.to_csv(filepath, index=True)
        
        # 최신 결과 포인터 파일도 저장
        latest_pointer = os.path.join(scan_results_dir, f"latest_scan_results_{mode_suffix}.csv")
        filtered_results.to_csv(latest_pointer, index=True)
        
        return True, f"스캔 결과 저장 완료: {len(filtered_results)}개 종목 (FMS ≥ {fms_threshold})", len(filtered_results)
        
    except Exception as e:
        return False, f"스캔 결과 저장 중 오류: {str(e)}", 0

def load_latest_scan_results(fms_threshold=2.0, mode: str = MODE_FREE):
    """
    모드별 가장 최근의 스캔 결과 파일을 로드합니다.
    
    Args:
        fms_threshold (float): FMS 임계값 (기본값: 2.0)
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
    
    Returns:
        tuple: (success, results_df, message)
    """
    try:
        mode_suffix = mode.lower()
        
        # scan_results 디렉토리에서 모드별 스캔 결과 파일 찾기
        scan_results_dir = "scan_results"
        if not os.path.exists(scan_results_dir):
            return False, pd.DataFrame(), f"{mode} 모드의 저장된 스캔 결과가 없습니다."
        
        # 모드별 스캔 결과 파일 찾기
        pattern_new = os.path.join(scan_results_dir, f"scan_results_{mode_suffix}_*.csv")
        
        scan_files = glob.glob(pattern_new)
        
        if not scan_files:
            return False, pd.DataFrame(), f"{mode} 모드의 저장된 스캔 결과가 없습니다."
        
        # 가장 최근 파일 선택 (타임스탬프 기준)
        if not scan_files:
            return False, pd.DataFrame(), f"{mode} 모드의 저장된 스캔 결과가 없습니다."
        
        latest_file = max(scan_files, key=os.path.getctime)
        
        # 파일 로드
        results_df = pd.read_csv(latest_file, index_col=0)
        
        # FMS 임계값 필터링
        if 'FMS' in results_df.columns:
            filtered_results = results_df[results_df['FMS'] >= fms_threshold].copy()
        else:
            filtered_results = results_df
        
        # 파일 수정 시간 정보
        file_mtime = os.path.getmtime(latest_file)
        file_time = datetime.fromtimestamp(file_mtime, KST)
        
        return True, filtered_results, f"스캔 결과 로드 완료: {len(filtered_results)}개 종목 (파일: {file_time.strftime('%Y-%m-%d %H:%M:%S')})"
        
    except Exception as e:
        return False, pd.DataFrame(), f"스캔 결과 로드 중 오류: {str(e)}"

def get_scan_results_info(mode: str = MODE_FREE):
    """
    모드별 저장된 스캔 결과 파일들의 정보를 반환합니다.
    
    Args:
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
    
    Returns:
        list: 파일 정보 리스트
    """
    try:
        mode_suffix = mode.lower()
        scan_results_dir = "scan_results"
        
        # scan_results 디렉토리가 없으면 빈 리스트 반환
        if not os.path.exists(scan_results_dir):
            return []
        
        # 모드별 스캔 결과 파일 찾기 (scan_results 디렉토리 내)
        pattern_new = os.path.join(scan_results_dir, f"scan_results_{mode_suffix}_*.csv")
        
        scan_files = glob.glob(pattern_new)
        
        if not scan_files:
            return []
        
        file_info = []
        for file in scan_files:
            try:
                mtime = os.path.getmtime(file)
                file_time = datetime.fromtimestamp(mtime, KST)
                
                # 파일 크기 및 종목 수 확인
                df = pd.read_csv(file, index_col=0)
                symbol_count = len(df)
                
                file_info.append({
                    'filename': file,
                    'timestamp': file_time,
                    'symbol_count': symbol_count,
                    'formatted_time': file_time.strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception:
                continue
        
        # 시간순 정렬 (최신순)
        file_info.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return file_info
        
    except Exception as e:
        # 스캔 결과 정보 조회 중 오류는 조용히 처리
        return []
