"""
관심종목(또는 지정 리스트) 기준으로 FMS 상대 순위를 확인합니다.
- 목적: "매끈함(R²) 때문에 상대적으로 과대평가되는가?"를 빠르게 점검
"""

import pandas as pd

from analysis_utils import build_prices_krw_from_symbols, download_ohlc_prices, momentum_now_and_delta


def main() -> None:
    watch = pd.read_csv("watchlist_free.csv")["symbol"].dropna().astype(str).tolist()
    prices_krw = build_prices_krw_from_symbols("6M", watch)
    ohlc, _ = download_ohlc_prices(watch, period_="1y", interval="1d")
    df = momentum_now_and_delta(
        prices_krw,
        reference_prices_krw=prices_krw,
        ohlc_data=ohlc if not ohlc.empty else None,
        symbols=watch,
    )

    focus = ["KMI", "SU", "488210.KS", "PBR"]
    present = [s for s in focus if s in df.index]
    if not present:
        print("Focus symbols not found in computed dataframe.")
        return

    sub = df.loc[
        present,
        ["FMS", "R_3M", "R_6M", "R2_3M", "AboveEMA50", "Vol20(ann)", "MaxDD_Pct", "Filter_Status"],
    ].copy()
    sub["Rank_in_watchlist"] = df["FMS"].rank(ascending=False, method="min").loc[sub.index].astype(int)

    print("Watchlist size:", len(df))
    sub_sorted = sub.sort_values("Rank_in_watchlist")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(sub_sorted.to_string())


if __name__ == "__main__":
    main()

