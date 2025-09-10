# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 유니버스 관리 유틸리티
Finviz를 사용한 유니버스 스크리닝 및 파일 관리 기능
"""

import os
import time
import pandas as pd
from datetime import datetime
import pytz
from typing import Tuple, Optional

KST = pytz.timezone("Asia/Seoul")

def check_universe_file_freshness():
    """
    screened_universe.csv 파일의 최근 업데이트 시간을 확인합니다.
    
    Returns:
        tuple: (is_fresh, last_modified_time, hours_since_update)
    """
    try:
        if not os.path.exists('screened_universe.csv'):
            return False, None, None
        
        file_mtime = os.path.getmtime('screened_universe.csv')
        last_modified = datetime.fromtimestamp(file_mtime, KST)
        hours_since_update = (datetime.now(KST) - last_modified).total_seconds() / 3600
        
        # 6시간 이내면 fresh로 간주
        is_fresh = hours_since_update < 6
        
        return is_fresh, last_modified, hours_since_update
        
    except Exception as e:
        print(f"파일 타임스탬프 확인 중 오류: {str(e)}")
        return False, None, None

def update_universe_file(progress_callback=None, status_callback=None):
    """
    Finviz를 사용하여 유니버스 파일을 업데이트합니다.
    진행 상황을 콜백 함수를 통해 실시간으로 전달합니다.
    
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
        
        # 스크리닝 필터 조건 설정
        filters = {
            'Price': 'Over $5',           # 가격 $5 이상
            'Average Volume': 'Over 200K', # 평균 거래량 200,000주 이상
            'Performance': 'Month +>0%',  # 1개월 수익률 0% 이상
            'Relative Volume': 'Over 1.5' # 최근 거래량 평소의 1.5배 이상
        }
        
        if progress_callback:
            progress_callback(0.05, "📊 스크리닝 필터 적용 중...")
        
        # Finviz 스크리너 실행 (진행률 콜백 포함)
        foverview = Overview()
        
        if progress_callback:
            progress_callback(0.1, "🔍 Finviz 데이터 다운로드 중...")
        
        df = foverview.screener_view()
        
        if progress_callback:
            progress_callback(0.2, f"📥 전체 데이터 다운로드 완료: {len(df)}개 종목")
        
        # 2단계: 필터링 적용
        if progress_callback:
            progress_callback(0.25, "🔍 필터링 적용 중...")
        
        original_count = len(df)
        
        # 가격 필터
        if 'Price' in df.columns:
            df['Price_clean'] = df['Price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            df = df[df['Price_clean'] >= 5.0]
            if progress_callback:
                progress_callback(0.35, f"💰 가격 $5 이상 필터링: {len(df)}개 종목")
        
        # 거래량 필터
        if 'Avg Volume' in df.columns:
            df['Volume_clean'] = df['Avg Volume'].str.replace(',', '').astype(float)
            df = df[df['Volume_clean'] >= 200000]
            if progress_callback:
                progress_callback(0.45, f"📈 거래량 200K 이상 필터링: {len(df)}개 종목")
        
        # 수익률 필터
        if 'Perf Month' in df.columns:
            df['Perf_Month_clean'] = df['Perf Month'].str.replace('%', '').astype(float)
            df = df[df['Perf_Month_clean'] >= 0.0]
            if progress_callback:
                progress_callback(0.55, f"📊 1개월 수익률 0% 이상 필터링: {len(df)}개 종목")
        
        # 상대 거래량 필터
        if 'Rel Volume' in df.columns:
            df['Rel_Volume_clean'] = df['Rel Volume'].astype(float)
            df = df[df['Rel_Volume_clean'] >= 1.5]
            if progress_callback:
                progress_callback(0.65, f"🔄 상대 거래량 1.5배 이상 필터링: {len(df)}개 종목")
        
        # 3단계: 레버리지/인버스 ETF 제외
        if not df.empty and 'Ticker' in df.columns:
            if progress_callback:
                progress_callback(0.7, "🚫 레버리지/인버스 ETF 제외 필터링 중...")
            
            leverage_patterns = [
                '2X', '3X', '2x', '3x', '2X', '3X',
                '2배', '3배', '2X배', '3X배',
                '1.5X', '1.75X', '1.5x', '1.75x',
                'Inverse', 'Short', 'Bear',
                '-1X', '-2X', '-3X', '-1x', '-2x', '-3x',
                'Leveraged', 'Ultra', 'ProShares',
                'ULTRA', 'ULTRA SHORT', 'ULTRA LONG',
                'AAPB', 'AAPU', 'SPXU', 'UPRO', 'TQQQ', 'SQQQ',
                'TMF', 'TMV', 'FAS', 'FAZ', 'ERX', 'ERY',
                'TNA', 'TZA', 'LABU', 'LABD', 'CURE', 'RXL',
                'BOIL', 'KOLD', 'NUGT', 'DUST', 'JNUG', 'JDST',
                'UVXY', 'SVXY', 'TVIX', 'XIV', 'YINN', 'YANG',
                'KWEB', 'CQQQ', 'TECL', 'TECS', 'SOXL', 'SOXS',
                'TBT', 'UBT', 'TYD', 'TYO', 'UST', 'PST'
            ]
            
            excluded_tickers = []
            for ticker in df['Ticker'].tolist():
                ticker_upper = str(ticker).upper()
                for pattern in leverage_patterns:
                    if pattern.upper() in ticker_upper:
                        excluded_tickers.append(ticker)
                        break
            
            if excluded_tickers:
                df = df[~df['Ticker'].isin(excluded_tickers)]
                if progress_callback:
                    progress_callback(0.8, f"🚫 레버리지/인버스 ETF 제외: {len(excluded_tickers)}개, 남은 종목: {len(df)}개")
        
        # 4단계: 파일 저장
        if not df.empty and 'Ticker' in df.columns:
            if progress_callback:
                progress_callback(0.9, "💾 유니버스 파일 저장 중...")
            
            tickers = df['Ticker'].tolist()
            unique_tickers = sorted(list(set(tickers)))
            
            output_df = pd.DataFrame({'Symbol': unique_tickers})
            output_df.to_csv('screened_universe.csv', index=False)
            
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

def load_universe_file():
    """
    screened_universe.csv 파일을 로드합니다.
    
    Returns:
        tuple: (success, symbols_list, message)
    """
    try:
        if not os.path.exists('screened_universe.csv'):
            return False, [], "유니버스 파일이 없습니다."
        
        universe_df = pd.read_csv('screened_universe.csv')
        symbols = universe_df['Symbol'].tolist()
        
        return True, symbols, f"유니버스 로드 완료: {len(symbols)}개 종목"
        
    except Exception as e:
        return False, [], f"유니버스 파일 로드 중 오류: {str(e)}"
