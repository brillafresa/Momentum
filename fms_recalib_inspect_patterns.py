"""
현재 fms_recalib_features.csv(새 정답셋)를 간단히 요약해
- 상/중/하위 그룹별 평균 지표
- 각 지표와 rank의 상관관계
를 출력하는 보조 스크립트입니다.
"""

import pandas as pd
import numpy as np


def main() -> None:
    df = pd.read_csv("fms_recalib_features.csv", index_col=0)
    cols = ["R_1M", "R_3M", "R_6M", "R2_3M", "AboveEMA50", "Vol20_Ann", "MaxDD_Pct"]

    n = len(df)
    third = max(1, n // 3)

    top = df.nsmallest(third, "rank")
    mid = df.iloc[third : 2 * third]
    bot = df.nlargest(third, "rank")

    print("N =", n, "third =", third)
    print("\n=== mean by group ===")
    for name, g in [("TOP", top), ("MID", mid), ("BOT", bot)]:
        print("\n", name)
        print(g[cols].mean())

    print("\n=== corr with rank (positive => 더 하위로 갈수록 증가) ===")
    print(df[cols + ["rank"]].corr()["rank"])


if __name__ == "__main__":
    main()

