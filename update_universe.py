# update_universe.py
# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 유망주 유니버스 스크리닝 스크립트
Finviz.com의 주식 스크리너를 이용해 최소 조건을 만족하는 티커 목록을 추출합니다.

사용법:
    python update_universe.py

출력:
    screened_universe.csv - 사전 필터링된 유망주 목록
"""

from finvizfinance.screener import Overview
import pandas as pd
import os
from datetime import datetime
import pytz

KST = pytz.timezone("Asia/Seoul")

def main():
    """메인 스크리닝 함수"""
    print("=" * 60)
    print("🚀 KRW Momentum Radar - 유망주 유니버스 스크리닝")
    print("=" * 60)
    print(f"실행 시간: {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')}")
    print()
    
    try:
        # 스크리닝 필터 조건 설정
        print("📊 스크리닝 필터 조건:")
        filters = {
            'Price': 'Over $5',           # 가격 $5 이상
            'Average Volume': 'Over 200K', # 평균 거래량 200,000주 이상
            'Performance': 'Month +>0%',  # 1개월 수익률 0% 이상
            'Relative Volume': 'Over 1.5' # 최근 거래량 평소의 1.5배 이상
        }
        
        for key, value in filters.items():
            print(f"  • {key}: {value}")
        print()
        
        print("🔍 Finviz 스크리너 실행 중...")
        foverview = Overview()
        
        # Finviz 전체 데이터 가져오기
        print("📥 데이터 다운로드 중...")
        df = foverview.screener_view()
        
        print(f"📊 전체 데이터: {len(df)}개 종목")
        
        # Pandas로 필터링 적용
        print("🔍 필터링 적용 중...")
        
        # 가격 $5 이상 필터링
        if 'Price' in df.columns:
            df['Price_clean'] = df['Price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
            df = df[df['Price_clean'] >= 5.0]
            print(f"  • 가격 $5 이상: {len(df)}개")
        
        # 평균 거래량 200K 이상 필터링
        if 'Avg Volume' in df.columns:
            df['Volume_clean'] = df['Avg Volume'].str.replace(',', '').astype(float)
            df = df[df['Volume_clean'] >= 200000]
            print(f"  • 거래량 200K 이상: {len(df)}개")
        
        # 1개월 수익률 0% 이상 필터링
        if 'Perf Month' in df.columns:
            df['Perf_Month_clean'] = df['Perf Month'].str.replace('%', '').astype(float)
            df = df[df['Perf_Month_clean'] >= 0.0]
            print(f"  • 1개월 수익률 0% 이상: {len(df)}개")
        
        # 상대 거래량 1.5배 이상 필터링 (Rel Volume 컬럼이 있는 경우)
        if 'Rel Volume' in df.columns:
            df['Rel_Volume_clean'] = df['Rel Volume'].astype(float)
            df = df[df['Rel_Volume_clean'] >= 1.5]
            print(f"  • 상대 거래량 1.5배 이상: {len(df)}개")
        
        # 레버리지/인버스 ETF 제외 필터링
        if not df.empty and 'Ticker' in df.columns:
            print("🚫 레버리지/인버스 ETF 제외 필터링 중...")
            
            # 레버리지/인버스 패턴 정의 (더 포괄적으로 확장)
            leverage_patterns = [
                # 명시적 레버리지 패턴
                '2X', '3X', '2x', '3x', '2X', '3X',  # 레버리지
                '2배', '3배', '2X배', '3X배',           # 한국어 레버리지
                '1.5X', '1.75X', '1.5x', '1.75x',     # 소수점 레버리지
                
                # 인버스 패턴
                'Inverse', 'Short', 'Bear',            # 인버스
                '-1X', '-2X', '-3X', '-1x', '-2x', '-3x',  # 인버스 레버리지
                
                # ProShares 관련 패턴 (더 포괄적)
                'Leveraged', 'Ultra', 'ProShares',     # 기타 레버리지 관련 키워드
                'ULTRA', 'ULTRA SHORT', 'ULTRA LONG',  # Ultra 시리즈
                
                # 알려진 레버리지 ETF 티커 패턴
                'AAPB', 'AAPU',  # ProShares UltraShort/Ultra AAPL
                'SPXU', 'UPRO',  # ProShares S&P 500 레버리지
                'TQQQ', 'SQQQ',  # ProShares NASDAQ 레버리지
                'TMF', 'TMV',    # ProShares 20년+ 국채 레버리지
                'FAS', 'FAZ',    # ProShares 금융 섹터 레버리지
                'ERX', 'ERY',    # ProShares 에너지 섹터 레버리지
                'TNA', 'TZA',    # ProShares 소형주 레버리지
                'LABU', 'LABD',  # ProShares 바이오테크 레버리지
                'CURE', 'RXL',   # ProShares 헬스케어 레버리지
                'BOIL', 'KOLD',  # ProShares 천연가스 레버리지
                'NUGT', 'DUST',  # ProShares 금광업 레버리지
                'JNUG', 'JDST',  # ProShares 주니어 금광업 레버리지
                'UVXY', 'SVXY',  # ProShares VIX 레버리지
                'TVIX', 'XIV',   # ProShares VIX 레버리지
                'YINN', 'YANG',  # ProShares 중국 레버리지
                'KWEB', 'CQQQ',  # ProShares 중국 인터넷 레버리지
                'TECL', 'TECS',  # ProShares 기술 섹터 레버리지
                'SOXL', 'SOXS',  # ProShares 반도체 레버리지
                'TBT', 'UBT',    # ProShares 국채 레버리지
                'TYD', 'TYO',    # ProShares 7-10년 국채 레버리지
                'UST', 'PST',    # ProShares 단기 국채 레버리지
            ]
            
            # 제외할 종목들 식별
            excluded_tickers = []
            for ticker in df['Ticker'].tolist():
                ticker_upper = str(ticker).upper()
                for pattern in leverage_patterns:
                    if pattern.upper() in ticker_upper:
                        excluded_tickers.append(ticker)
                        break
            
            # 제외할 종목들 제거
            if excluded_tickers:
                df = df[~df['Ticker'].isin(excluded_tickers)]
                print(f"  • 레버리지/인버스 ETF 제외: {len(excluded_tickers)}개")
                print(f"  • 제외된 종목 예시: {excluded_tickers[:10]}{'...' if len(excluded_tickers) > 10 else ''}")
                print(f"  • 남은 종목: {len(df)}개")
            else:
                print(f"  • 레버리지/인버스 ETF 없음: {len(df)}개")
        
        if not df.empty and 'Ticker' in df.columns:
            tickers = df['Ticker'].tolist()
            
            # 중복 제거 및 정렬
            unique_tickers = sorted(list(set(tickers)))
            
            output_filename = 'screened_universe.csv'
            output_df = pd.DataFrame({'Symbol': unique_tickers})
            output_df.to_csv(output_filename, index=False)
            
            print("✅ 성공!")
            print(f"📈 발견된 유망 종목: {len(unique_tickers)}개")
            print(f"💾 저장 위치: {output_filename}")
            print()
            
            # 상위 10개 종목 미리보기
            if len(unique_tickers) > 0:
                print("📋 상위 10개 종목 미리보기:")
                for i, ticker in enumerate(unique_tickers[:10], 1):
                    print(f"  {i:2d}. {ticker}")
                if len(unique_tickers) > 10:
                    print(f"  ... 및 {len(unique_tickers) - 10}개 더")
            
        else:
            print("⚠️  경고: 스크리닝 결과가 비어있습니다.")
            print("💡 팁: 필터 조건이 너무 엄격할 수 있습니다. 조건을 완화해보세요.")
            
            # 빈 파일이라도 생성 (앱에서 파일 존재 여부로 판단)
            output_filename = 'screened_universe.csv'
            pd.DataFrame({'Symbol': []}).to_csv(output_filename, index=False)
            print(f"📄 빈 파일 생성: {output_filename}")

    except Exception as e:
        print(f"❌ 스크립트 실행 중 오류 발생: {e}")
        print()
        print("🔧 문제 해결 방법:")
        print("  1. finvizfinance 라이브러리가 최신 버전인지 확인: pip install --upgrade finvizfinance")
        print("  2. 인터넷 연결 상태 확인")
        print("  3. Finviz.com 사이트 접근 가능 여부 확인")
        print("  4. 방화벽이나 프록시 설정 확인")
        
        # 오류 발생 시에도 빈 파일 생성
        try:
            output_filename = 'screened_universe.csv'
            pd.DataFrame({'Symbol': []}).to_csv(output_filename, index=False)
            print(f"📄 오류 복구: 빈 파일 생성 - {output_filename}")
        except:
            print("❌ 복구 파일 생성도 실패")

    print()
    print("=" * 60)
    print("🏁 스크리닝 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
