"""
FMS 재보정: 수정 전 vs 수정 후만 비교.
- 입력: fms_recalib_features.csv
- 출력: current(현재 적용) vs proposed(수정 제안)의 역전 비율

Harness note
------------
``f_current`` / ``f_proposed`` must call ``core.fms.score_fms_from_feature_frame``
(production SSOT). Independent formula forks are forbidden (HARNESS_RULES §2.5).
New candidates belong in tune scripts until promoted into ``core/fms.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.fms import score_fms_from_feature_frame


FEATURE_CSV = "fms_recalib_features.csv"


def z(series: pd.Series) -> pd.Series:
    """Legacy helper retained for ad-hoc notebooks; prefer core scoring."""
    s = series.astype(float)
    m = np.nanmean(s)
    sd = np.nanstd(s)
    if not sd or np.isnan(sd):
        return s * 0.0
    return (s - m) / sd


def pairwise_inversion_rate(true_rank: pd.Series, score: pd.Series) -> float:
    """정답 순서 대비 역전된 순서쌍 비율. 낮을수록 좋음."""
    df = pd.concat([true_rank, score], axis=1).dropna()
    df.columns = ["true_rank", "score"]
    n = len(df)
    if n <= 1:
        return 0.0
    df_sorted = df.sort_values("true_rank", ascending=True)
    scores = df_sorted["score"].to_numpy()
    inv = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if scores[i] < scores[j]:
                inv += 1
    return inv / total if total else 0.0


def f_current(df: pd.DataFrame) -> pd.Series:
    """Production FMS via ``core.fms`` (single source of truth)."""
    return score_fms_from_feature_frame(df)


def f_proposed(df: pd.DataFrame) -> pd.Series:
    """Proposed FMS for A/B.

    Until a new candidate is promoted into ``core/fms.py``, proposed == current
    (production). Tune scripts may explore alternate weights offline; winners
    must be merged into core before becoming ``f_proposed``.
    """
    return score_fms_from_feature_frame(df)


def main() -> None:
    import os

    if not os.path.exists(FEATURE_CSV):
        print(f"{FEATURE_CSV}가 없습니다. python fms_recalib_build_features.py 를 먼저 실행하세요.")
        return
    df = pd.read_csv(FEATURE_CSV, index_col=0)
    true_rank = df["rank"]

    inv_current = pairwise_inversion_rate(true_rank, f_current(df))
    inv_proposed = pairwise_inversion_rate(true_rank, f_proposed(df))
    print(f"inversion_rate  current={inv_current:.4f}  proposed={inv_proposed:.4f}")
    if inv_proposed < inv_current:
        print("→ proposed가 current보다 역전 비율이 낮습니다 (개선).")
    elif inv_proposed > inv_current:
        print("→ proposed가 current보다 역전 비율이 높습니다 (악화).")
    else:
        print("→ current와 proposed 역전 비율이 동일합니다 (proposed==production).")


if __name__ == "__main__":
    main()

