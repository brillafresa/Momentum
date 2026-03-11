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

    # Iteration 5 (새 정답셋 기준 튜닝 결과): 가중치/전이폭 파라미터
    P_W_R3 = 0.46869
    P_W_R6 = 0.417409
    P_W_R2 = 0.505669
    P_W_EMA = 0.323264
    P_W_R1_POS = 0.213603
    P_W_DD = 0.28298
    P_W_VOL = 0.291973
    P_W_R1_NEG = 0.174149
    P_R2_TRANSITION_W = 0.04552
    P_GATE_R3_W = 0.019359
    P_GATE_R6_W = 0.006355
    P_LEVEL_R3_HI = 0.205305
    P_LEVEL_R6_HI = 0.430268
    P_R2_FLOOR = 0.734629

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
    새 구조 FMS 후보.

    - 기존 축: R_1M, R_3M, R_6M, R2_3M, AboveEMA50, Vol20_Ann, MaxDD_Pct
    - 추가 축: R_10D, R_5D, EMA20_SLOPE_10D, EMA20_CURV_20D,
              UNDER_EMA20_DEPTH, UNDER_EMA20_DAYS, DOWN_STREAK_5D
    """
    r1 = df["R_1M"]
    r3 = df["R_3M"]
    r6 = df["R_6M"]
    r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
    ema50 = df["AboveEMA50"].clip(lower=-0.5, upper=1.5)
    vol20 = df["Vol20_Ann"]
    maxdd = df["MaxDD_Pct"]
    r10 = df["R_10D"]
    r5 = df["R_5D"]
    e20_slope10 = df["EMA20_SLOPE_10D"]
    e20_curv20 = df["EMA20_CURV_20D"]
    under_depth = df["UNDER_EMA20_DEPTH"]
    under_days = df["UNDER_EMA20_DAYS"]
    down5 = df["DOWN_STREAK_5D"]

    # R² 연속 가중/게이트 (현재 f_current 구조를 기본으로 존중)
    P_R2_TRANSITION_W = 0.03
    P_GATE_R3_W = 0.02
    P_GATE_R6_W = 0.02
    P_LEVEL_R3_HI = 0.20
    P_LEVEL_R6_HI = 0.40
    P_R2_FLOOR = 0.75

    def smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
        if edge1 == edge0:
            return pd.Series(0.0, index=x.index)
        t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
        return t * t * (3.0 - 2.0 * t)

    # R² piecewise multiplier
    w_mid = smoothstep(r2, 0.70 - P_R2_TRANSITION_W, 0.70 + P_R2_TRANSITION_W)
    w_high = smoothstep(r2, 0.90 - P_R2_TRANSITION_W, 0.90 + P_R2_TRANSITION_W)
    r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high
    r2_effect = r2_mult * r2

    # 추세 게이트 + 레벨 램프
    gate_on = smoothstep(r3, 0.05 - P_GATE_R3_W, 0.05 + P_GATE_R3_W) * smoothstep(
        r6, 0.08 - P_GATE_R6_W, 0.08 + P_GATE_R6_W
    )
    level = smoothstep(r3, 0.05, P_LEVEL_R3_HI) * smoothstep(
        r6, 0.08, P_LEVEL_R6_HI
    )
    r2_strength = gate_on * (P_R2_FLOOR + (1.0 - P_R2_FLOOR) * level)
    r2_term = z(pd.Series(r2_effect * r2_strength, index=df.index))

    # Drawdown penalty (기존 형태 유지)
    dd_mag = (-maxdd).clip(lower=0.0)
    dd_soft = dd_mag.clip(upper=30.0)
    dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
    dd_penalty = z(pd.Series(dd_soft + dd_hard, index=df.index))

    # Vol penalty (기존 70%/1.5 power 매핑 유지)
    v = vol20.clip(lower=0.0)
    q = np.nanpercentile(v, 70) if not v.dropna().empty else 0.0
    v_soft = v.clip(upper=q)
    v_hard = (v - q).clip(lower=0.0) ** 1.5
    vol_penalty = z(pd.Series(v_soft + v_hard, index=df.index))

    # 장·중기 추세 + 위치
    r3_term = z(r3)
    r6_term = z(r6)
    ema_term = z(ema50)

    # EMA20 기울기/곡률: 상승·아래로 볼록 선호, 위로 볼록 과열은 약한 패널티
    slope_term = z(e20_slope10)
    # 곡률: 양수(위로 볼록)는 패널티, 음수(아래로 볼록)는 약한 가산
    curv_penalty_raw = e20_curv20.clip(lower=0.0)
    curv_reward_raw = (-e20_curv20).clip(lower=0.0)
    curv_penalty = z(curv_penalty_raw)
    curv_reward = z(curv_reward_raw)
    ema_shape_term = 0.7 * slope_term + 0.3 * curv_reward - 0.3 * curv_penalty

    # EMA20 아래 이탈: 얕고 짧으면 거의 무시, 깊고 길면 비선형 패널티
    depth_term = z(under_depth)  # 이미 음수(깊을수록 더 음수)
    days_term = z(under_days.astype(float))

    # 단기 추세 유지/붕괴: long-good + 10D/5D
    quality_mask = (r2 > 0.85) & (r3 > 0.3) & (r6 > 0.5)
    r10_good_raw = pd.Series(
        np.where(quality_mask & (r10 > 0.0), r10, 0.0), index=df.index
    )
    r10_break_raw = pd.Series(
        np.where(quality_mask & (r10 < 0.0), -r10, 0.0), index=df.index
    )
    r5_good_raw = pd.Series(
        np.where(quality_mask & (r5 > 0.0), r5, 0.0), index=df.index
    )
    recent_accel_term = z(r10_good_raw + 0.5 * r5_good_raw)
    recent_break_term = z(r10_break_raw)

    # 최근 5일 연속 하락
    down5_term = z(down5.astype(float))

    # 조건부 1M 수익률
    r1_good = pd.Series(np.where(quality_mask, r1, 0.0), index=df.index)
    r1_bad = pd.Series(
        np.where(~quality_mask & (r1 > 0.3), r1, 0.0), index=df.index
    )
    r1_pos = z(r1_good)
    r1_neg = z(r1_bad)

    # 가중치 (초기값: 직관 기반, 이후 튜너에서 조정)
    W_R3 = 0.45
    W_R6 = 0.40
    W_R2 = 0.55
    W_EMA = 0.30
    W_EMA_SHAPE = 0.25
    W_RECENT = 0.25
    W_R1_POS = 0.20

    W_DD = 0.30
    W_VOL = 0.30
    W_R1_NEG = 0.18
    W_BREAK = 0.22
    W_DOWN5 = 0.18
    W_UNDER_DEPTH = 0.20
    W_UNDER_DAYS = 0.10

    pos = (
        W_R3 * r3_term
        + W_R6 * r6_term
        + W_R2 * r2_term
        + W_EMA * ema_term
        + W_EMA_SHAPE * ema_shape_term
        + W_RECENT * recent_accel_term
        + W_R1_POS * r1_pos
    )
    neg = (
        W_DD * dd_penalty
        + W_VOL * vol_penalty
        + W_R1_NEG * r1_neg
        + W_BREAK * recent_break_term
        + W_DOWN5 * down5_term
        + W_UNDER_DEPTH * depth_term
        + W_UNDER_DAYS * days_term
    )
    return pos - neg


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
