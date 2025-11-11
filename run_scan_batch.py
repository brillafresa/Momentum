#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 배치 스캔 실행기 (CLI)

강제 유니버스 스캔 → 관심종목 기준으로 참조 가격 구성 →
analysis_utils의 단일 FMS/필터 로직으로 결과 산출 및 저장.
"""

import os
import sys
import pandas as pd

from universe_utils import (
    update_universe_file,
    load_universe_file,
    load_korean_universe,
    save_scan_results,
    get_scan_results_info,
)
from watchlist_utils import load_watchlist
from analysis_utils import (
    build_prices_krw_from_symbols,
    calculate_fms_for_batch,
)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main() -> int:
    print("[Batch] 🔄 Updating universe with relaxed filters...")
    success, message, symbol_count = update_universe_file()
    print(f"[Batch] Universe update: {message} (symbols: {symbol_count})")
    if not success:
        return 1

    print("[Batch] 📥 Loading watchlist and building reference prices...")
    watchlist = load_watchlist([])
    
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

    print("[Batch] 📂 Loading screened universe symbols (USA + Korea)...")
    ok, usa_symbols, msg = load_universe_file()
    if not ok or not usa_symbols:
        print(f"[Batch] ⚠️ Failed to load USA universe: {msg}")
        usa_symbols = []
    
    # 한국 유니버스 로드 (KOSPI 200 + KOSDAQ 150 + 국내 지수 ETF 1배/인버스)
    ok_kor, kor_symbols, msg_kor = load_korean_universe()
    if not ok_kor or not kor_symbols:
        print(f"[Batch] ⚠️ Failed to load Korean universe: {msg_kor}")
        kor_symbols = []
    
    # USA + 한국 유니버스 병합
    all_symbols = usa_symbols + kor_symbols
    print(f"[Batch] 📊 Loaded {len(usa_symbols)} USA + {len(kor_symbols)} Korean = {len(all_symbols)} total symbols")
    
    if not all_symbols:
        print(f"[Batch] ❌ No universe symbols to scan.")
        return 1

    scan_targets = [s for s in all_symbols if s not in watchlist]
    if not scan_targets:
        print("[Batch] ℹ️ No new symbols to scan.")
        return 0

    print(f"[Batch] 🚀 Calculating FMS for {len(scan_targets)} symbols (with tradeability filters)...")
    results = calculate_fms_for_batch(scan_targets, reference_prices_krw=ref_prices)
    if results.empty:
        print("[Batch] ❌ No results were produced.")
        return 1

    print("[Batch] 💾 Saving results (FMS ≥ 0.0) to timestamped file and latest pointer...")
    save_success, save_msg, saved_count = save_scan_results(results, fms_threshold=0.0)
    print(f"[Batch] {save_msg}")

    ensure_dir("scan_results")
    info_list = get_scan_results_info()
    if info_list:
        latest_path = info_list[0]['filename']
        try:
            df_latest = pd.read_csv(latest_path, index_col=0)
            timestamp_name = os.path.basename(latest_path)
            target_timestamp_path = os.path.join("scan_results", timestamp_name)
            df_latest.to_csv(target_timestamp_path, index=True)
            latest_pointer_path = os.path.join("scan_results", "latest_scan_results.csv")
            df_latest.to_csv(latest_pointer_path, index=True)
            print(f"[Batch] ✅ Saved latest to {latest_pointer_path}")
        except Exception as e:
            print(f"[Batch] ⚠️ Failed to write scan_results copies: {e}")

    print("[Batch] ✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


