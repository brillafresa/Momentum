# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 관심종목 관리 유틸리티
관심종목의 영구 저장 및 관리를 담당하는 모듈
"""

import pandas as pd
import os
import shutil
from typing import List, Tuple
from datetime import datetime
from io import StringIO

# 모드 상수 정의
MODE_FREE = "FREE"
MODE_IRP = "IRP"

def get_watchlist_file_path(mode: str = MODE_FREE) -> str:
    """
    모드에 따른 관심종목 파일 경로를 반환합니다.
    
    Args:
        mode (str): 계좌 모드 ("FREE" 또는 "IRP")
    
    Returns:
        str: 파일 경로
    """
    if mode == MODE_IRP:
        return "watchlist_irp.csv"
    else:
        return "watchlist_free.csv"

def migrate_legacy_watchlist():
    """
    기존 watchlist.csv 파일을 watchlist_free.csv로 마이그레이션합니다.
    한 번만 실행되도록 처리합니다.
    """
    legacy_file = "watchlist.csv"
    free_file = get_watchlist_file_path(MODE_FREE)
    
    if os.path.exists(legacy_file) and not os.path.exists(free_file):
        try:
            shutil.copy2(legacy_file, free_file)
            # 마이그레이션 완료 후 기존 파일은 삭제하지 않음 (안전을 위해)
        except Exception:
            pass  # 마이그레이션 실패 시 조용히 처리

def load_watchlist(default_symbols: List[str], mode: str = MODE_FREE) -> List[str]:
    """
    모드별 관심종목 파일에서 심볼 목록을 로드합니다. 
    파일이 없으면 기본 목록을 반환합니다 (저장하지 않음).
    
    Args:
        default_symbols (List[str]): 기본 심볼 목록
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        
    Returns:
        List[str]: 로드된 심볼 목록
    """
    # 기존 파일 마이그레이션 (한 번만 실행)
    if mode == MODE_FREE:
        migrate_legacy_watchlist()
    
    try:
        watchlist_file = get_watchlist_file_path(mode)
        if os.path.exists(watchlist_file):
            df = pd.read_csv(watchlist_file)
            if 'symbol' in df.columns and not df.empty:
                return df["symbol"].tolist()
        
        # 파일이 없거나 비어있으면 기본 목록 반환 (저장하지 않음)
        return default_symbols
        
    except Exception as e:
        # 관심종목 로드 중 오류는 조용히 처리
        # 오류 발생 시 기본 목록 반환 (저장하지 않음)
        return default_symbols

def save_watchlist(symbols: List[str], mode: str = MODE_FREE) -> bool:
    """
    심볼 목록을 모드별 관심종목 파일에 저장합니다. 
    중복을 제거하고 정렬합니다.
    
    Args:
        symbols (List[str]): 저장할 심볼 목록
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
    
    Returns:
        bool: 저장 성공 여부
    """
    try:
        # 중복 제거 및 정렬
        unique_symbols = sorted(list(set(symbols)))
        df = pd.DataFrame({"symbol": unique_symbols})
        watchlist_file = get_watchlist_file_path(mode)
        df.to_csv(watchlist_file, index=False, encoding='utf-8-sig')
        return True
        
    except Exception as e:
        # 관심종목 저장 중 오류는 조용히 처리
        # 디버깅 필요 시 아래 주석을 해제하여 예외 정보 확인 가능
        # import traceback
        # print(f"[watchlist_utils] save_watchlist 오류 (mode={mode}): {e}")
        # traceback.print_exc()
        return False

def add_to_watchlist(symbols: List[str], new_symbols: List[str], mode: str = MODE_FREE) -> List[str]:
    """
    관심종목에 새로운 심볼을 추가합니다.
    
    Args:
        symbols (List[str]): 기존 관심종목 목록
        new_symbols (List[str]): 추가할 심볼 목록
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        
    Returns:
        List[str]: 업데이트된 관심종목 목록
    """
    try:
        updated_symbols = list(set(symbols + new_symbols))
        save_success = save_watchlist(updated_symbols, mode)
        if not save_success:
            # 저장 실패 시에도 세션 상태는 업데이트 (사용자 경험 유지)
            pass
        return updated_symbols
    except Exception as e:
        # 오류 발생 시에도 업데이트된 목록은 반환 (세션 상태 유지)
        # 디버깅 필요 시 아래 주석을 해제하여 예외 정보 확인 가능
        # import traceback
        # print(f"[watchlist_utils] add_to_watchlist 오류 (mode={mode}): {e}")
        # traceback.print_exc()
        return list(set(symbols + new_symbols))

def remove_from_watchlist(symbols: List[str], symbols_to_remove: List[str], mode: str = MODE_FREE) -> List[str]:
    """
    관심종목에서 심볼을 제거합니다.
    
    Args:
        symbols (List[str]): 기존 관심종목 목록
        symbols_to_remove (List[str]): 제거할 심볼 목록
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        
    Returns:
        List[str]: 업데이트된 관심종목 목록
    """
    try:
        updated_symbols = [s for s in symbols if s not in symbols_to_remove]
        save_success = save_watchlist(updated_symbols, mode)
        if not save_success:
            # 저장 실패 시에도 세션 상태는 업데이트 (사용자 경험 유지)
            pass
        return updated_symbols
    except Exception as e:
        # 오류 발생 시에도 업데이트된 목록은 반환 (세션 상태 유지)
        # 디버깅 필요 시 아래 주석을 해제하여 예외 정보 확인 가능
        # import traceback
        # print(f"[watchlist_utils] remove_from_watchlist 오류 (mode={mode}): {e}")
        # traceback.print_exc()
        return [s for s in symbols if s not in symbols_to_remove]

def get_watchlist_stats(mode: str = MODE_FREE) -> dict:
    """
    관심종목 통계 정보를 반환합니다.
    
    Args:
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
    
    Returns:
        dict: 통계 정보
    """
    try:
        watchlist_file = get_watchlist_file_path(mode)
        if os.path.exists(watchlist_file):
            df = pd.read_csv(watchlist_file)
            return {
                "total_count": len(df),
                "file_exists": True,
                "last_modified": os.path.getmtime(watchlist_file)
            }
        else:
            return {
                "total_count": 0,
                "file_exists": False,
                "last_modified": None
            }
    except Exception as e:
        return {
            "total_count": 0,
            "file_exists": False,
            "last_modified": None,
            "error": str(e)
        }

def export_watchlist_to_csv(symbols: List[str], country_classifier=None, name_display=None) -> str:
    """
    관심종목을 CSV 형식으로 내보냅니다.
    
    Args:
        symbols (List[str]): 내보낼 심볼 목록
        country_classifier: 국가 분류 함수 (선택사항)
        name_display: 이름 표시 함수 (선택사항)
        
    Returns:
        str: CSV 데이터 (문자열)
    """
    try:
        # 기본 데이터프레임 생성
        data = {'symbol': symbols}
        
        # 국가 분류 함수가 있으면 추가
        if country_classifier:
            data['country'] = [country_classifier(s) for s in symbols]
        
        # 이름 표시 함수가 있으면 추가
        if name_display:
            data['name'] = [name_display(s) for s in symbols]
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False, encoding='utf-8-sig')
        
    except Exception as e:
        # 관심종목 내보내기 중 오류는 조용히 처리
        return ""

def import_watchlist_from_csv(csv_data: str, mode: str = MODE_FREE) -> Tuple[List[str], str]:
    """
    CSV 데이터에서 관심종목을 가져옵니다.
    
    Args:
        csv_data (str): CSV 데이터 (문자열)
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        
    Returns:
        Tuple[List[str], str]: (심볼 목록, 결과 메시지)
    """
    try:
        # CSV 데이터를 DataFrame으로 변환
        df = pd.read_csv(StringIO(csv_data), encoding='utf-8-sig')
        
        # symbol 컬럼 확인
        if 'symbol' not in df.columns:
            return [], "❌ 'symbol' 컬럼이 필요합니다."
        
        # 유효한 심볼만 추출
        new_symbols = []
        for symbol in df['symbol'].astype(str):
            if symbol and symbol.strip() and symbol != 'nan':
                new_symbols.append(symbol.strip())
        
        if not new_symbols:
            return [], "❌ 유효한 심볼이 없습니다."
        
        # 관심종목 업데이트 (파일 저장 포함)
        save_success = save_watchlist(new_symbols, mode)
        if not save_success:
            return [], "❌ 관심종목 파일 저장에 실패했습니다."
        
        return new_symbols, f"✅ 관심종목이 업데이트되었습니다! ({len(new_symbols)}개 종목)"
        
    except Exception as e:
        return [], f"❌ 파일 업로드 중 오류가 발생했습니다: {str(e)}"

def replace_watchlist(symbols: List[str], mode: str = MODE_FREE) -> Tuple[List[str], str]:
    """
    관심종목을 완전히 교체합니다.
    
    Args:
        symbols (List[str]): 새로운 관심종목 목록
        mode (str): 계좌 모드 ("FREE" 또는 "IRP", 기본값: "FREE")
        
    Returns:
        Tuple[List[str], str]: (업데이트된 심볼 목록, 결과 메시지)
    """
    try:
        # 관심종목 저장 (파일 업데이트 포함)
        save_success = save_watchlist(symbols, mode)
        if not save_success:
            return [], "❌ 관심종목 파일 저장에 실패했습니다."
        return symbols, f"✅ 관심종목이 업데이트되었습니다! ({len(symbols)}개 종목)"
        
    except Exception as e:
        return [], f"❌ 관심종목 업데이트 중 오류가 발생했습니다: {str(e)}"
