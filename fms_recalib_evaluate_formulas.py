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
    """현재 적용된 FMS (analysis_utils._mom_snapshot와 동일 로직)."""
    r1 = df["R_1M"]
    r3 = df["R_3M"]
    r6 = df["R_6M"]
    r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
    ema50 = df["AboveEMA50"].clip(lower=-0.5, upper=1.5)
    vol20 = df["Vol20_Ann"]
    maxdd = df["MaxDD_Pct"]

    # Iteration 5 (튜닝 결과): 가중치/전이폭 파라미터
    P_W_R3 = 0.435991
    P_W_R6 = 0.319466
    P_W_R2 = 0.615106
    P_W_EMA = 0.284587
    P_W_R1_POS = 0.186529
    P_W_DD = 0.363645
    P_W_VOL = 0.377713
    P_W_R1_NEG = 0.165261
    P_R2_TRANSITION_W = 0.029645
    P_GATE_R3_W = 0.028663
    P_GATE_R6_W = 0.013226
    P_LEVEL_R3_HI = 0.123071
    P_LEVEL_R6_HI = 0.340733
    P_R2_FLOOR = 0.631902

    def smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
        if edge1 == edge0:
            return pd.Series(0.0, index=x.index)
        t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
        return t * t * (3.0 - 2.0 * t)

    w_mid = smoothstep(r2, 0.70 - P_R2_TRANSITION_W, 0.70 + P_R2_TRANSITION_W)
    w_high = smoothstep(r2, 0.90 - P_R2_TRANSITION_W, 0.90 + P_R2_TRANSITION_W)
    r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high
    r2_effect = r2_mult * r2

    r2_gate = smoothstep(r3, 0.05 - P_GATE_R3_W, 0.05 + P_GATE_R3_W) * smoothstep(r6, 0.08 - P_GATE_R6_W, 0.08 + P_GATE_R6_W)
    r2_level = smoothstep(r3, 0.05, P_LEVEL_R3_HI) * smoothstep(r6, 0.08, P_LEVEL_R6_HI)
    r2_strength = r2_gate * (P_R2_FLOOR + (1.0 - P_R2_FLOOR) * r2_level)
    r2_effect_gated = pd.Series(r2_effect * r2_strength, index=df.index)
    r2_term = z(r2_effect_gated)

    dd_mag = (-maxdd).clip(lower=0.0)
    dd_soft = dd_mag.clip(upper=30.0)
    dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
    dd_combined = dd_soft + dd_hard
    dd_penalty = z(pd.Series(dd_combined, index=df.index))

    v = vol20.clip(lower=0.0)
    q = np.nanpercentile(v, 70) if not v.dropna().empty else 0.0
    v_soft = v.clip(upper=q)
    v_hard = (v - q).clip(lower=0.0) ** 1.5
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

    pos = P_W_R3 * r3_term + P_W_R6 * r6_term + P_W_R2 * r2_term + P_W_EMA * ema_term + P_W_R1_POS * r1_pos
    neg = P_W_DD * dd_penalty + P_W_VOL * vol_penalty + P_W_R1_NEG * r1_neg
    return pos - neg


def f_proposed(df: pd.DataFrame) -> pd.Series:
    """
    수정 제안:
    - FMS 변경 실험을 이 함수에 구현한 뒤,
      `fms_recalib_evaluate_formulas.py` / `fms_recalib_rank_metrics.py`로 current vs proposed를 비교합니다.
    - 정답셋이 바뀌면 지표 절대값은 의미가 없으므로, **이번 정답셋 내부에서** 개선 여부만 판단합니다.
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
