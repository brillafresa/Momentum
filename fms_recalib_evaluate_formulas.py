"""
FMS 재보정: 수정 전 vs 수정 후만 비교.
- 입력: fms_recalib_features.csv
- 출력: current(현재 적용) vs proposed(수정 제안)의 역전 비율
"""

import numpy as np
import pandas as pd


FEATURE_CSV = "fms_recalib_features.csv"


def z(series: pd.Series) -> pd.Series:
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
    """현재 적용된 FMS (analysis_utils._mom_snapshot와 동일 로직, R² 추세상승 게이트 포함)."""
    r1 = df["R_1M"]
    r3 = df["R_3M"]
    r6 = df["R_6M"]
    r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
    ema50 = df["AboveEMA50"].clip(lower=-0.5, upper=1.5)
    vol20 = df["Vol20_Ann"]
    maxdd = df["MaxDD_Pct"]

    r2_effect = np.where(
        r2 < 0.7, 0.2 * r2,
        np.where(r2 < 0.9, 0.6 * r2, 1.2 * r2),
    )
    r2_gate = (r3 > 0.05).astype(float) * (r6 > 0.08).astype(float)
    r2_effect_gated = pd.Series(r2_effect * r2_gate, index=df.index)
    r2_term = z(r2_effect_gated)

    dd_mag = (-maxdd).clip(lower=0.0)
    dd_soft = dd_mag.clip(upper=30.0)
    dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
    dd_combined = dd_soft + dd_hard
    dd_penalty = z(pd.Series(dd_combined, index=df.index))

    v = vol20.clip(lower=0.0)
    q = np.nanpercentile(v, 60) if not v.dropna().empty else 0.0
    v_soft = v.clip(upper=q)
    v_hard = (v - q).clip(lower=0.0) ** 2
    v_combined = v_soft + v_hard
    vol_penalty = z(pd.Series(v_combined, index=df.index))

    r3_term = z(r3)
    r6_term = z(r6)
    ema_term = z(ema50)

    quality_mask = (r2 > 0.85) & (r3 > 0.3) & (r6 > 0.5)
    r1_good = pd.Series(np.where(quality_mask, r1, 0.0), index=df.index)
    r1_bad = pd.Series(np.where(~quality_mask & (r1 > 0.3), r1, 0.0), index=df.index)
    r1_pos = z(r1_good)
    r1_neg = z(r1_bad)

    pos = 0.45 * r3_term + 0.35 * r6_term + 0.5 * r2_term + 0.3 * ema_term + 0.15 * r1_pos
    neg = 0.5 * dd_penalty + 0.35 * vol_penalty + 0.15 * r1_neg
    return pos - neg


def f_proposed(df: pd.DataFrame) -> pd.Series:
    """
    수정 제안: FMS 변경 시 여기에 새 로직 구현 후 검증.
    현재는 f_current와 동일.
    """
    return f_current(df)


def main() -> None:
    import os
    if not os.path.exists(FEATURE_CSV):
        print(f"{FEATURE_CSV}가 없습니다. python fms_recalib_build_features.py 를 먼저 실행하세요.")
        return
    df = pd.read_csv(FEATURE_CSV, index_col=0)
    true_rank = df["rank"]

    inv_current = pairwise_inversion_rate(true_rank, f_current(df))
    inv_proposed = pairwise_inversion_rate(true_rank, f_proposed(df))

    print("=== FMS 수정 전 vs 수정 후 (역전 비율) ===")
    print(f"current:  inversion_rate={inv_current:.4f}")
    print(f"proposed: inversion_rate={inv_proposed:.4f}")
    if inv_proposed < inv_current:
        print(f">> 개선됨 (역전 비율 {inv_current:.4f} -> {inv_proposed:.4f})")
    else:
        print(f">> 개선되지 않음. 로직 재검토 필요.")


if __name__ == "__main__":
    main()
