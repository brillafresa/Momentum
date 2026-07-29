# -*- coding: utf-8 -*-
"""
Live diagnostic: relative FMS ranks within the current watchlist.

Purpose
-------
Quickly inspect whether selected symbols look over/under-ranked relative to
peers on the production FMS (v4.6 sparse-linear). Useful after formula changes
or when reviewing a live watchlist composition.

Important
---------
Calls live market APIs via ``analysis_utils``. Do NOT use in CI / pytest.
Kept under ``harness/`` to preserve the production-vs-validation boundary.

Usage (from repo root)
----------------------
    python -m harness.check_relative_ranks
    python -m harness.check_relative_ranks --symbols KMI SU 488210.KS PBR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_utils import (  # noqa: E402
    build_prices_krw_from_symbols,
    download_ohlc_prices,
    momentum_now_and_delta,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["KMI", "SU", "488210.KS", "PBR"],
        help="Focus symbols to print (must be in the loaded watchlist).",
    )
    parser.add_argument(
        "--watchlist",
        default="watchlist_free.csv",
        help="Watchlist CSV path (Symbol/symbol column).",
    )
    args = parser.parse_args(argv)

    watch_df = pd.read_csv(args.watchlist)
    col = "symbol" if "symbol" in watch_df.columns else "Symbol"
    watch = watch_df[col].dropna().astype(str).tolist()
    prices_krw = build_prices_krw_from_symbols("6M", watch)
    ohlc, _ = download_ohlc_prices(watch, period_="1y", interval="1d")
    df = momentum_now_and_delta(
        prices_krw,
        reference_prices_krw=prices_krw,
        ohlc_data=ohlc if not ohlc.empty else None,
        symbols=watch,
    )

    present = [s for s in args.symbols if s in df.index]
    if not present:
        print("Focus symbols not found in computed dataframe.")
        return 1

    cols = [
        c
        for c in [
            "FMS",
            "R_3M",
            "R2_3M",
            "DD_RECOVERY",
            "TREND_QUALITY_21D",
            "STALE_AGE",
            "Filter_Status",
        ]
        if c in df.columns
    ]
    sub = df.loc[present, cols].copy()
    sub["Rank_in_watchlist"] = (
        df["FMS"].rank(ascending=False, method="min").loc[sub.index].astype(int)
    )

    print("Watchlist size:", len(df))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(sub.sort_values("Rank_in_watchlist").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
