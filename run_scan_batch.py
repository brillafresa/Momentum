#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 배치 스캔 실행기 (CLI)

강제 유니버스 스캔 → 관심종목 기준으로 참조 가격 구성 →
analysis_utils의 단일 FMS/필터 로직으로 결과 산출 및 저장.

사용법:
    python run_scan_batch.py [--mode FREE|IRP] [--skip-universe-update]

    --mode: 계좌 모드 선택 (기본값: FREE)
        FREE: 자유투자계좌 (미국+한국+홍콩 주식)
        IRP: 퇴직연금IRP (국내상장 ETF 전 종목)
    --skip-universe-update: Finviz screened_universe.csv 갱신 생략 (기존 CSV 사용)
"""

import os
import sys
import argparse
import pandas as pd

from universe_utils import (
    update_universe_file,
    load_universe_file,
    save_scan_results,
    get_scan_results_info,
    MODE_FREE,
    MODE_IRP,
)
from watchlist_utils import load_watchlist, MODE_FREE as WL_MODE_FREE, MODE_IRP as WL_MODE_IRP
from analysis_utils import (
    build_prices_krw_from_symbols,
    calculate_fms_for_batch,
)
from config import DEFAULT_FMS_THRESHOLD


def main() -> int:
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description='KRW Momentum Radar 배치 스캔 실행기')
    parser.add_argument('--mode', type=str, choices=['FREE', 'IRP'], default='FREE',
                        help='계좌 모드 선택 (FREE: 자유투자계좌, IRP: 퇴직연금IRP)')
    parser.add_argument(
        '--skip-universe-update',
        action='store_true',
        help='FREE 모드에서 Finviz 유니버스 갱신 생략 (screened_universe.csv 등 기존 파일 사용)',
    )
    args = parser.parse_args()
    
    mode = args.mode
    mode_label = "자유투자계좌" if mode == MODE_FREE else "퇴직연금IRP"
    print(f"[Batch] 🏦 모드: {mode_label} ({mode})")
    
    # FREE 모드: 기본은 Finviz 유니버스 갱신. --skip-universe-update 시 기존 CSV 사용.
    if mode == MODE_FREE and not args.skip_universe_update:
        print("[Batch] 🔄 Updating universe with relaxed filters...")
        success, message, symbol_count = update_universe_file()
        print(f"[Batch] Universe update: {message} (symbols: {symbol_count})")
        if not success:
            return 1
    elif mode == MODE_FREE:
        print("[Batch] ℹ️ Skipping Finviz universe update; using existing universe CSV files.")
    else:
        print("[Batch] ℹ️ IRP 모드: 수동 관리 유니버스 파일 사용 (korean_etf_univers.csv)")

    print(f"[Batch] 📥 Loading watchlist ({mode_label}) and building reference prices...")
    watchlist = load_watchlist([], mode=mode)
    
    # Watchlist에서 실격 종목 필터링
    if watchlist:
        from analysis_utils import download_ohlc_prices, calculate_tradeability_filters
        watchlist_ohlc, _ = download_ohlc_prices(watchlist, '1y', '1d')
        if not watchlist_ohlc.empty:
            watchlist_flags, _ = calculate_tradeability_filters(watchlist_ohlc, watchlist)
            # 실격되지 않은 종목만 사용
            valid_watchlist = [s for s in watchlist if s in watchlist_flags and not watchlist_flags[s]]
            if len(valid_watchlist) != len(watchlist):
                print(f"[Batch] ⚠️ Filtered {len(watchlist) - len(valid_watchlist)} disqualified symbols from watchlist")
                watchlist = valid_watchlist
        else:
            print("[Batch] ⚠️ Failed to download watchlist OHLC data; skipping disqualification filtering for reference data")
    
    ref_prices = build_prices_krw_from_symbols("1Y", watchlist)
    if ref_prices.empty:
        print("[Batch] ⚠️ Reference watchlist prices are empty; proceeding without reference baseline.")
        ref_prices = None

    print(f"[Batch] 📂 Loading universe symbols ({mode_label})...")
    ok, all_symbols, msg = load_universe_file(mode=mode)
    if not ok or not all_symbols:
        print(f"[Batch] ⚠️ Failed to load universe: {msg}")
        all_symbols = []
    
    print(f"[Batch] 📊 Loaded {len(all_symbols)} total symbols")
    
    if not all_symbols:
        print(f"[Batch] ❌ No universe symbols to scan.")
        return 1

    scan_targets = [s for s in all_symbols if s not in watchlist]
    if not scan_targets:
        print("[Batch] ℹ️ No new symbols to scan.")
        return 0

    print(f"[Batch] 🚀 Calculating FMS for {len(scan_targets)} symbols (with tradeability filters)...")
    print("[Batch] ℹ️ 'no data / delisted' messages are normal skips; rate limits are retried with backoff.")
    results = calculate_fms_for_batch(scan_targets, reference_prices_krw=ref_prices)
    if results.empty:
        print("[Batch] ❌ No results were produced.")
        return 1

    print(f"[Batch] ✅ Scored {len(results)} symbols; saving FMS ≥ {DEFAULT_FMS_THRESHOLD} ({mode_label})...")
    save_success, save_msg, saved_count = save_scan_results(
        results, fms_threshold=DEFAULT_FMS_THRESHOLD, mode=mode
    )
    print(f"[Batch] {save_msg}")
    
    if save_success:
        latest_pointer_path = os.path.join("scan_results", f"latest_scan_results_{mode.lower()}.csv")
        print(f"[Batch] ✅ Saved latest to {latest_pointer_path}")

    print(f"[Batch] ✅ Done ({mode_label}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


