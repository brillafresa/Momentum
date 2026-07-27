# -*- coding: utf-8 -*-
"""
Live diagnostic for an extreme FMS outlier (manual, not for CI).

Purpose
-------
When a single ticker produces an obviously abnormal FMS during a batch scan
(e.g. due to Yahoo Adj Close glitches), this script helps inspect:
- raw close/price hygiene (min/max, negatives/zeros)
- abrupt jumps in the recent daily return series
- resulting feature values and final FMS snapshot
- tradeability filter flags (optional OHLC)

Important
---------
This runner may call live market APIs (via ``analysis_utils``). Do NOT use it
in automated tests / CI. It is placed under ``harness/`` to keep the
production-vs-validation boundary explicit.

Usage
------
From repo root:
    python -m harness.diagnose_fms_outlier 381560.KS
    python -m harness.diagnose_fms_outlier 381560.KS --mode IRP
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_utils import (  # noqa: E402
    build_prices_krw_from_symbols,
    calculate_tradeability_filters,
    download_ohlc_prices,
    momentum_now_and_delta,
)
from core.fms import (  # noqa: E402
    HORIZON_DAYS_4M,
    _mom_snapshot,
    production_fms_score_params,
)
from core.indicators import (  # noqa: E402
    ema,
    last_vol_annualized,
    returns_pct,
    r_squared_3m,
)
from watchlist_utils import load_watchlist  # noqa: E402


def _feature_row(prices: pd.DataFrame, symbol: str) -> dict:
    s = prices[symbol].dropna()
    out = {
        "n_bars": int(len(s)),
        "last": float(s.iloc[-1]) if len(s) else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "zeros": int((s == 0).sum()) if len(s) else 0,
        "negatives": int((s < 0).sum()) if len(s) else 0,
        "R_1M": float(returns_pct(prices[[symbol]], 21).iloc[0]),
        "R_3M": float(returns_pct(prices[[symbol]], 63).iloc[0]),
        "R_4M": float(returns_pct(prices[[symbol]], HORIZON_DAYS_4M).iloc[0]),
        "R_10D": float(returns_pct(prices[[symbol]], 10).iloc[0]),
        "R_5D": float(returns_pct(prices[[symbol]], 5).iloc[0]),
        "R2_3M": float(r_squared_3m(prices[[symbol]]).iloc[0]),
        "Vol20": float(last_vol_annualized(prices[[symbol]], 20).iloc[0]),
    }

    if not s.empty:
        e50 = ema(s, 50)
        out["AboveEMA50"] = (
            float(s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan
        )
        roll_max = s.cummax()
        out["MaxDD_Pct"] = float(((s / roll_max - 1.0) * 100.0).min())

        rets = s.pct_change().dropna()
        out["max_daily_ret"] = float(rets.max()) if len(rets) else np.nan
        out["min_daily_ret"] = float(rets.min()) if len(rets) else np.nan
        out["abs_ret_gt_50pct_days"] = int((rets.abs() > 0.5).sum()) if len(rets) else 0

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", nargs="?", default="381560.KS")
    parser.add_argument("--mode", default="FREE", choices=["FREE", "IRP"])
    args = parser.parse_args()
    symbol = str(args.symbol).strip()

    watchlist = load_watchlist([], mode=args.mode)
    print(f"[diag] target={symbol} mode={args.mode} watchlist_n={len(watchlist)}")

    symbols = list(dict.fromkeys([symbol] + list(watchlist)))
    prices = build_prices_krw_from_symbols("1Y", symbols)
    if prices.empty or symbol not in prices.columns:
        print(f"[diag] FAIL: could not build prices for {symbol}")
        print(f"[diag] columns={list(prices.columns)[:20]} ... total={prices.shape}")
        return 1

    print(f"[diag] prices shape={prices.shape} target_bars={prices[symbol].notna().sum()}")
    feat = _feature_row(prices, symbol)
    print("[diag] target feature dump:")
    for k, v in feat.items():
        print(f"  {k}: {v}")

    s = prices[symbol].dropna()
    print("[diag] last 15 closes:")
    print(s.tail(15).to_string())

    rets = s.pct_change()
    big = rets[rets.abs() > 0.3]
    if not big.empty:
        print(f"[diag] daily |ret|>30% days ({len(big)}):")
        for dt, r in big.items():
            print(f"  {dt.date()}: {r:.4f}  close={s.loc[dt]:.6g}")

    ohlc, miss = download_ohlc_prices([symbol], "1y", "1d")
    print(f"[diag] ohlc missing={miss} empty={ohlc.empty}")

    ref = prices[[c for c in watchlist if c in prices.columns]]
    if ref.empty:
        ref = None
        print("[diag] WARNING: empty reference panel; scoring peer-only")
    else:
        print(f"[diag] reference panel n={ref.shape[1]}")

    scored_syms = [symbol] + [c for c in (ref.columns if ref is not None else []) if c != symbol]

    snap = _mom_snapshot(
        prices[[symbol]],
        reference_prices_krw=ref,
        ohlc_data=ohlc if not ohlc.empty else None,
        symbols=[symbol],
    )
    print("[diag] snapshot (target alone vs ref):")
    print(snap.T.to_string())

    full = momentum_now_and_delta(
        prices[[c for c in scored_syms if c in prices.columns]],
        reference_prices_krw=ref,
        ohlc_data=None,
        symbols=[c for c in scored_syms if c in prices.columns],
    )
    if symbol in full.index:
        row = full.loc[symbol]
        print(f"[diag] FMS={row['FMS']:.4f}  Filter={row.get('Filter_Status', '')}")
        print(f"[diag] FMS rank 1/{len(full)} max={full['FMS'].max():.4f} median={full['FMS'].median():.4f}")
        top = full["FMS"].sort_values(ascending=False).head(5)
        print("[diag] top5 FMS:")
        print(top.to_string())

    if ref is not None and not ref.empty:
        p = production_fms_score_params()
        axes = {
            "R_1M": returns_pct(ref, 21),
            "R_3M": returns_pct(ref, 63),
            "R_4M": returns_pct(ref, HORIZON_DAYS_4M),
            "R2_3M": r_squared_3m(ref),
            "Vol20": last_vol_annualized(ref, 20),
        }
        print("[diag] reference axis mean/std:")
        for name, ser in axes.items():
            print(f"  {name}: mean={np.nanmean(ser):.6g} std={np.nanstd(ser):.6g} n={ser.notna().sum()}")
        print(f"[diag] params sample: w_r3={p.w_r3} w_r4={p.w_r4} w_ema_shape={p.w_ema_shape}")

    if not ohlc.empty:
        flags, reasons = calculate_tradeability_filters(ohlc, [symbol])
        print(f"[diag] tradeability: flag={flags.get(symbol)} reason={reasons.get(symbol)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

