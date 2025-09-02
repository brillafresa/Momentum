# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 관심종목 관리 유틸리티
관심종목의 영구 저장 및 관리를 담당하는 모듈
"""

import pandas as pd
import os
from typing import List

WATCHLIST_FILE = "watchlist.csv"

def load_watchlist(default_symbols: List[str]) -> List[str]:
    """
    watchlist.csv에서 심볼 목록을 로드합니다. 
    파일이 없으면 기본 목록으로 생성합니다.
    
    Args:
        default_symbols (List[str]): 기본 심볼 목록
        
    Returns:
        List[str]: 로드된 심볼 목록
    """
    try:
        if os.path.exists(WATCHLIST_FILE):
            df = pd.read_csv(WATCHLIST_FILE)
            if 'symbol' in df.columns and not df.empty:
                return df["symbol"].tolist()
        
        # 파일이 없거나 비어있으면 기본 목록으로 생성
        save_watchlist(default_symbols)
        return default_symbols
        
    except Exception as e:
        print(f"관심종목 로드 중 오류 발생: {e}")
        # 오류 발생 시 기본 목록으로 생성
        save_watchlist(default_symbols)
        return default_symbols

def save_watchlist(symbols: List[str]) -> None:
    """
    심볼 목록을 watchlist.csv에 저장합니다. 
    중복을 제거하고 정렬합니다.
    
    Args:
        symbols (List[str]): 저장할 심볼 목록
    """
    try:
        # 중복 제거 및 정렬
        unique_symbols = sorted(list(set(symbols)))
        df = pd.DataFrame({"symbol": unique_symbols})
        df.to_csv(WATCHLIST_FILE, index=False)
        print(f"관심종목 {len(unique_symbols)}개 저장 완료")
        
    except Exception as e:
        print(f"관심종목 저장 중 오류 발생: {e}")

def add_to_watchlist(symbols: List[str], new_symbols: List[str]) -> List[str]:
    """
    관심종목에 새로운 심볼을 추가합니다.
    
    Args:
        symbols (List[str]): 기존 관심종목 목록
        new_symbols (List[str]): 추가할 심볼 목록
        
    Returns:
        List[str]: 업데이트된 관심종목 목록
    """
    updated_symbols = list(set(symbols + new_symbols))
    save_watchlist(updated_symbols)
    return updated_symbols

def remove_from_watchlist(symbols: List[str], symbols_to_remove: List[str]) -> List[str]:
    """
    관심종목에서 심볼을 제거합니다.
    
    Args:
        symbols (List[str]): 기존 관심종목 목록
        symbols_to_remove (List[str]): 제거할 심볼 목록
        
    Returns:
        List[str]: 업데이트된 관심종목 목록
    """
    updated_symbols = [s for s in symbols if s not in symbols_to_remove]
    save_watchlist(updated_symbols)
    return updated_symbols

def get_watchlist_stats() -> dict:
    """
    관심종목 통계 정보를 반환합니다.
    
    Returns:
        dict: 통계 정보
    """
    try:
        if os.path.exists(WATCHLIST_FILE):
            df = pd.read_csv(WATCHLIST_FILE)
            return {
                "total_count": len(df),
                "file_exists": True,
                "last_modified": os.path.getmtime(WATCHLIST_FILE)
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
