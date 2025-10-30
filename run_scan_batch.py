#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 배치 스캔 실행기 (CLI)

강제 유니버스 스캔 → 관심종목 기준으로 참조 가격 구성 →
analysis_utils의 단일 FMS/필터 로직으로 결과 산출 및 저장.
"""

import os
import sys
from datetime import datetime
import pytz
import pandas as pd

from universe_utils import (
    update_universe_file,
    load_universe_file,
    save_scan_results,
    get_scan_results_info,
)
from watchlist_utils import load_watchlist
from analysis_utils import (
    build_prices_krw_from_symbols,
    calculate_fms_for_batch,
)

KST = pytz.timezone("Asia/Seoul")


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
    ref_prices = build_prices_krw_from_symbols("1Y", watchlist)
    if ref_prices.empty:
        print("[Batch] ⚠️ Reference watchlist prices are empty; proceeding without reference baseline.")
        ref_prices = None

    print("[Batch] 📂 Loading screened universe symbols...")
    ok, symbols, msg = load_universe_file()
    if not ok or not symbols:
        print(f"[Batch] ❌ Failed to load universe: {msg}")
        return 1

    scan_targets = [s for s in symbols if s not in watchlist]
    if not scan_targets:
        print("[Batch] ℹ️ No new symbols to scan.")
        return 0

    print(f"[Batch] 🚀 Calculating FMS for {len(scan_targets)} symbols (with tradeability filters)...")
    results = calculate_fms_for_batch(scan_targets, reference_prices_krw=ref_prices)
    if results.empty:
        print("[Batch] ❌ No results were produced.")
        return 1

    print("[Batch] 💾 Saving results (FMS ≥ 2.0) to timestamped file and latest pointer...")
    save_success, save_msg, saved_count = save_scan_results(results, fms_threshold=2.0)
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 배치 스캔 실행기 (CLI)

윈도우 작업 스케줄러 등에서 정기적으로 실행하여:
1) 유니버스 업데이트 (완화된 필터)
2) 관심종목을 참조 기준으로 FMS 계산
3) 결과 저장: 프로젝트 루트와 scan_results/에 타임스탬프/최신 파일 생성
"""

import os
import sys
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf

from universe_utils import (
    update_universe_file,
    load_universe_file,
    save_scan_results,
    get_scan_results_info,
)
from watchlist_utils import load_watchlist

KST = pytz.timezone("Asia/Seoul")


def classify(sym: str) -> str:
    sym_str = str(sym)
    if sym_str.endswith(".KS"):
        return "KOR"
    if sym_str.endswith(".T"):
        return "JPN"
    return "USA"


def _extract_adj_close(df_chunk: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df_chunk is None or len(df_chunk) == 0:
        return pd.DataFrame(columns=tickers, dtype=float)
    if isinstance(df_chunk.columns, pd.MultiIndex):
        if 'Adj Close' in df_chunk.columns.get_level_values(0):
            adj = df_chunk['Adj Close'].copy()
        elif 'Close' in df_chunk.columns.get_level_values(0):
            adj = df_chunk['Close'].copy()
        else:
            parts = []
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


def download_prices(tickers: list[str], period_: str = "1y", interval: str = "1d", chunk: int = 25) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    missing: list[str] = []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, period=period_, interval=interval, auto_adjust=False,
                              group_by='column', progress=False, threads=True)
            adj = _extract_adj_close(raw, part)
        except Exception:
            adj = pd.DataFrame()
        if adj.empty or adj.isna().all().all():
            pframes = []
            for t in part:
                try:
                    r = yf.download(t, period=period_, interval=interval, auto_adjust=False,
                                    group_by='column', progress=False, threads=False)
                    a = _extract_adj_close(r, [t])
                    pframes.append(a)
                except Exception:
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
    out = out.drop(columns=all_nan)
    return out, sorted(list(dict.fromkeys(list(missing) + list(all_nan))))


def download_fx(period_: str = "1y", interval: str = "1d"):
    fx_krw, _ = download_prices(["KRW=X"], period_, interval)
    fx_jpy, _ = download_prices(["JPY=X"], period_, interval)
    usdkrw = fx_krw.iloc[:, 0].rename("USDKRW") if not fx_krw.empty else pd.Series(dtype=float, name="USDKRW")
    usdjpy = fx_jpy.iloc[:, 0].rename("USDJPY") if not fx_jpy.empty else pd.Series(dtype=float, name="USDJPY")
    if not usdkrw.empty and not usdjpy.empty:
        start = min(usdkrw.index.min(), usdjpy.index.min())
        end = max(usdkrw.index.max(), usdjpy.index.max())
        idx = pd.date_range(start, end, freq='B')
        usdkrw = usdkrw.reindex(idx).ffill()
        usdjpy = usdjpy.reindex(idx).ffill()
        jpykrw = (usdkrw / usdjpy).rename("JPYKRW")
    else:
        jpykrw = pd.Series(dtype=float, name="JPYKRW")
    return usdkrw, usdjpy, jpykrw


def download_ohlc_prices(tickers: list[str], period_: str = "1y", interval: str = "1d", chunk: int = 25) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    missing: list[str] = []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, period=period_, interval=interval, auto_adjust=False,
                              group_by='column', progress=False, threads=True)
            if raw.empty:
                missing.extend(part)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                ohlc_map: dict[str, pd.DataFrame] = {}
                for t in part:
                    if ('High', t) in raw.columns and ('Low', t) in raw.columns and ('Close', t) in raw.columns:
                        ohlc_map[t] = pd.DataFrame({
                            'High': raw[('High', t)],
                            'Low': raw[('Low', t)],
                            'Close': raw[('Close', t)]
                        })
                if ohlc_map:
                    frames.append(pd.concat(ohlc_map, axis=1))
            else:
                if len(part) == 1 and all(c in raw.columns for c in ['High', 'Low', 'Close']):
                    t = part[0]
                    frames.append(pd.concat({t: raw[['High', 'Low', 'Close']].copy()}, axis=1))
                else:
                    missing.extend(part)
        except Exception:
            missing.extend(part)
    if not frames:
        return pd.DataFrame(), missing
    all_ohlc = pd.concat(frames, axis=1)
    all_ohlc = all_ohlc.loc[:, ~all_ohlc.columns.duplicated()].sort_index()
    return all_ohlc, sorted(list(dict.fromkeys(missing)))


def calculate_tradeability_filters(ohlc_data: pd.DataFrame, symbols: list[str]) -> tuple[dict, dict]:
    disqualification: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    if ohlc_data is None or ohlc_data.empty:
        return disqualification, reasons
    for symbol in symbols:
        try:
            if isinstance(ohlc_data.columns, pd.MultiIndex):
                cols = ohlc_data.columns
                if (symbol, 'High') in cols and (symbol, 'Low') in cols and (symbol, 'Close') in cols:
                    high = ohlc_data[(symbol, 'High')].dropna()
                    low = ohlc_data[(symbol, 'Low')].dropna()
                    close = ohlc_data[(symbol, 'Close')].dropna()
                else:
                    disqualification[symbol] = True
                    reasons[symbol] = 'OHLC 데이터 부족'
                    continue
            else:
                if all(c in ohlc_data.columns for c in ['High', 'Low', 'Close']):
                    high = ohlc_data['High'].dropna()
                    low = ohlc_data['Low'].dropna()
                    close = ohlc_data['Close'].dropna()
                else:
                    disqualification[symbol] = True
                    reasons[symbol] = 'OHLC 데이터 부족'
                    continue

            if len(close) < 63:
                disqualification[symbol] = True
                reasons[symbol] = '데이터 기간 부족 (63일 미만)'
                continue

            prev_close = close.shift(1)
            true_range = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1, skipna=False)
            daily_true_range_vol = true_range / prev_close
            daily_downside_risk = (low / prev_close) - 1

            recent_63 = daily_true_range_vol.tail(63)
            extreme_days = recent_63[recent_63 > 0.30]

            recent_20 = daily_downside_risk.tail(20)
            severe_days = recent_20[recent_20 < -0.07]

            rs: list[str] = []
            if len(extreme_days) > 0:
                rs.append(f'치명적 변동성 ({len(extreme_days)}일 30% 초과)')
            if len(severe_days) >= 4:
                rs.append(f'반복적 하방리스크 ({len(severe_days)}일 -7% 미만)')

            disqualification[symbol] = len(rs) > 0
            reasons[symbol] = '; '.join(rs) if rs else '정상'
        except Exception as e:
            disqualification[symbol] = True
            reasons[symbol] = f'계산 오류: {e}'
    return disqualification, reasons

def harmonize_calendar(df: pd.DataFrame, coverage: float = 0.9) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    df = df.reindex(idx).ffill()
    valid_ratio = df.count().div(len(df))
    keep_cols = valid_ratio[valid_ratio >= coverage].index
    return df[keep_cols] if len(keep_cols) > 0 else pd.DataFrame()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def returns_pct(df: pd.DataFrame, n: int) -> pd.Series:
    if df.shape[0] <= n:
        return pd.Series(index=df.columns, dtype=float)
    dff = df.ffill()
    r = dff.pct_change(periods=n, fill_method=None).iloc[-1]
    return r


def ytd_return(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    dff = df.ffill()
    last = dff.index[-1]
    y0 = pd.Timestamp(datetime(last.year, 1, 1))
    start_idx = dff.index.get_indexer([y0], method='nearest')[0]
    return dff.iloc[-1] / dff.iloc[start_idx] - 1.0


def last_vol_annualized(df: pd.DataFrame, window: int = 20) -> pd.Series:
    rets = df.ffill().pct_change(fill_method=None).dropna()
    if rets.empty:
        return pd.Series(index=df.columns, dtype=float)
    vol = rets.rolling(window).std().iloc[-1] * np.sqrt(252.0)
    return vol


def _mom_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: pd.DataFrame | None = None) -> pd.DataFrame:
    r_1m = returns_pct(prices_krw, 21)
    r_3m = returns_pct(prices_krw, 63)
    above_ema50 = {}
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            above_ema50[c] = np.nan
            continue
        e50 = ema(s, 50)
        above_ema50[c] = (s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan
    above_ema50 = pd.Series(above_ema50, name="AboveEMA50")
    vol20 = last_vol_annualized(prices_krw, 20).rename("Vol20(ann)")

    if reference_prices_krw is not None:
        ref_r_1m = returns_pct(reference_prices_krw, 21)
        ref_r_3m = returns_pct(reference_prices_krw, 63)
        ref_above_ema50 = {}
        for c in reference_prices_krw.columns:
            s = reference_prices_krw[c].dropna()
            if s.empty:
                ref_above_ema50[c] = np.nan
                continue
            e50 = ema(s, 50)
            ref_above_ema50[c] = (s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan
        ref_above_ema50 = pd.Series(ref_above_ema50, name="AboveEMA50")
        ref_vol20 = last_vol_annualized(reference_prices_krw, 20).rename("Vol20(ann)")

        def z_with_reference(x: pd.Series, ref_x: pd.Series) -> pd.Series:
            x = x.astype(float)
            ref_x = ref_x.astype(float)
            m = np.nanmean(ref_x)
            sd = np.nanstd(ref_x)
            return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0

        FMS = (
            0.4 * z_with_reference(r_1m, ref_r_1m)
            + 0.3 * z_with_reference(r_3m, ref_r_3m)
            + 0.2 * z_with_reference(above_ema50, ref_above_ema50)
            - 0.4 * z_with_reference(vol20.fillna(vol20.median()), ref_vol20.fillna(ref_vol20.median()))
        )
    else:
        def z(x: pd.Series) -> pd.Series:
            x = x.astype(float)
            m = np.nanmean(x)
            sd = np.nanstd(x)
            return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0

        FMS = (
            0.4 * z(r_1m)
            + 0.3 * z(r_3m)
            + 0.2 * z(above_ema50)
            - 0.4 * z(vol20.fillna(vol20.median()))
        )

    snap = pd.concat([
        r_1m.rename("R_1M"),
        r_3m.rename("R_3M"),
        above_ema50,
        vol20,
        FMS.rename("FMS"),
    ], axis=1)
    return snap


def momentum_now_and_delta(prices_krw: pd.DataFrame, reference_prices_krw: pd.DataFrame | None = None) -> pd.DataFrame:
    now = _mom_snapshot(prices_krw, reference_prices_krw)
    d1 = _mom_snapshot(prices_krw.iloc[:-1], reference_prices_krw) if len(prices_krw) > 1 else now * np.nan
    d5 = _mom_snapshot(prices_krw.iloc[:-5], reference_prices_krw) if len(prices_krw) > 5 else now * np.nan
    df = now.copy()
    df["ΔFMS_1D"] = df["FMS"] - d1["FMS"]
    df["ΔFMS_5D"] = df["FMS"] - d5["FMS"]
    df["R_1W"] = returns_pct(prices_krw, 5)
    df["R_6M"] = returns_pct(prices_krw, 126)
    df["R_YTD"] = ytd_return(prices_krw)
    return df.sort_values("FMS", ascending=False)


def build_prices_krw_for_watchlist(watchlist_symbols: list[str], period_key: str = "6M") -> pd.DataFrame:
    period_map = {"3M": "6mo", "6M": "1y", "1Y": "2y", "2Y": "5y", "5Y": "10y"}
    yf_period = period_map.get(period_key, "1y")
    interval = "1d"

    usd_symbols = [str(s) for s in watchlist_symbols if classify(s) == "USA"]
    krw_symbols = [str(s) for s in watchlist_symbols if classify(s) == "KOR"]
    jpy_symbols = [str(s) for s in watchlist_symbols if classify(s) == "JPN"]

    usdkrw, _, jpykrw = download_fx(yf_period, interval)
    usd_df, _ = download_prices(usd_symbols, yf_period, interval)
    krw_df, _ = download_prices(krw_symbols, yf_period, interval)
    jpy_df, _ = download_prices(jpy_symbols, yf_period, interval)

    frames: list[pd.DataFrame] = []
    if not usd_df.empty and not usdkrw.empty:
        usdkrw_matched = usdkrw.reindex(usd_df.index).ffill()
        frames.append(usd_df.mul(usdkrw_matched, axis=0))
    if not krw_df.empty:
        frames.append(krw_df)
    if not jpy_df.empty and not jpykrw.empty:
        jpykrw_matched = jpykrw.reindex(jpy_df.index).ffill()
        frames.append(jpy_df.mul(jpykrw_matched, axis=0))

    if not frames:
        return pd.DataFrame()

    prices_krw = pd.concat(frames, axis=1).sort_index()
    prices_krw = prices_krw.loc[:, ~prices_krw.columns.duplicated()]
    prices_krw = harmonize_calendar(prices_krw, coverage=0.9)
    return prices_krw


def calculate_fms_for_batch(symbols_batch: list[str], reference_prices_krw: pd.DataFrame | None = None) -> pd.DataFrame:
    if not symbols_batch:
        return pd.DataFrame()

    # 가격 다운로드 및 KRW 환산
    prices, _ = download_prices(symbols_batch, "1y", "1d")
    if prices.empty:
        return pd.DataFrame()

    usd_symbols = [s for s in symbols_batch if classify(s) == "USA"]
    if usd_symbols:
        usdkrw, _, _ = download_fx("1y", "1d")
        if not usdkrw.empty:
            usdkrw_matched = usdkrw.reindex(prices.index).ffill()
            usd_prices = prices[[s for s in usd_symbols if s in prices.columns]]
            if not usd_prices.empty:
                prices[usd_prices.columns] = usd_prices.mul(usdkrw_matched, axis=0)

    jpy_symbols = [s for s in symbols_batch if classify(s) == "JPN"]
    if jpy_symbols:
        _, _, jpykrw = download_fx("1y", "1d")
        if not jpykrw.empty:
            jpykrw_matched = jpykrw.reindex(prices.index).ffill()
            jpy_prices = prices[[s for s in jpy_symbols if s in prices.columns]]
            if not jpy_prices.empty:
                prices[jpy_prices.columns] = jpy_prices.mul(jpykrw_matched, axis=0)

    prices_krw = harmonize_calendar(prices, coverage=0.9)
    if prices_krw.empty:
        return pd.DataFrame()

    # 거래 적합성 필터 계산을 위해 OHLC 다운로드
    ohlc_data, _ = download_ohlc_prices(symbols_batch, "1y", "1d")
    disq, reasons = calculate_tradeability_filters(ohlc_data, symbols_batch) if not ohlc_data.empty else ({}, {})

    df = momentum_now_and_delta(prices_krw, reference_prices_krw)

    # 실격 종목은 FMS = -999 적용
    if disq:
        for sym, bad in disq.items():
            if bad and sym in df.index:
                df.at[sym, 'FMS'] = -999.0
    # 필터 상태 컬럼 추가(참고용)
    if reasons:
        df['Filter_Status'] = pd.Series(reasons)
    return df.sort_values("FMS", ascending=False)


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
    ref_prices = build_prices_krw_for_watchlist(watchlist, period_key="1Y")
    if ref_prices.empty:
        print("[Batch] ⚠️ Reference watchlist prices are empty; proceeding without reference baseline.")
        ref_prices = None

    print("[Batch] 📂 Loading screened universe symbols...")
    ok, symbols, msg = load_universe_file()
    if not ok or not symbols:
        print(f"[Batch] ❌ Failed to load universe: {msg}")
        return 1

    scan_targets = [s for s in symbols if s not in watchlist]
    if not scan_targets:
        print("[Batch] ℹ️ No new symbols to scan.")
        return 0

    print(f"[Batch] 🚀 Calculating FMS for {len(scan_targets)} symbols...")
    results = calculate_fms_for_batch(scan_targets, reference_prices_krw=ref_prices)
    if results.empty:
        print("[Batch] ❌ No results were produced.")
        return 1

    print("[Batch] 💾 Saving results (FMS ≥ 2.0) to timestamped file and latest pointer...")
    # 먼저 기존 유틸로 저장 (루트)
    save_success, save_msg, saved_count = save_scan_results(results, fms_threshold=2.0)
    print(f"[Batch] {save_msg}")

    # scan_results 디렉터리에도 복사 및 latest 생성
    ensure_dir("scan_results")
    # 가장 최근 파일 탐색
    info_list = get_scan_results_info()
    if info_list:
        latest_path = info_list[0]['filename']
        try:
            df_latest = pd.read_csv(latest_path, index_col=0)
            timestamp_name = os.path.basename(latest_path)
            target_timestamp_path = os.path.join("scan_results", timestamp_name)
            df_latest.to_csv(target_timestamp_path, index=True)
            # latest 덮어쓰기 파일 생성
            latest_pointer_path = os.path.join("scan_results", "latest_scan_results.csv")
            df_latest.to_csv(latest_pointer_path, index=True)
            print(f"[Batch] ✅ Saved latest to {latest_pointer_path}")
        except Exception as e:
            print(f"[Batch] ⚠️ Failed to write scan_results copies: {e}")

    print("[Batch] ✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


