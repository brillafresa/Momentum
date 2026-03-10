"""
FMS 재보정을 위한 후보 수식 비교 스크립트.

- 입력: fms_recalib_features.csv
- 출력: 각 후보 수식의 순서쌍 역전 비율(inversion_rate)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

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
    """
    true_rank: 낮을수록 상위 (1위, 2위, ...)
    score: 높을수록 상위가 되도록 해석되는 점수
    """
    df = pd.concat([true_rank, score], axis=1).dropna()
    df.columns = ["true_rank", "score"]
    n = len(df)
    if n <= 1:
        return 0.0

    # true 순서대로 인덱스 정렬
    df_sorted = df.sort_values("true_rank", ascending=True)
    scores = df_sorted["score"].to_numpy()

    inv = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            # i는 true 상위, j는 true 하위
            if scores[i] < scores[j]:
                # 점수 기준으로 j가 더 상위로 평가됨 → 역전
                inv += 1
    return inv / total if total else 0.0


@dataclass
class FMSCandidate:
    name: str
    func: Callable[[pd.DataFrame], pd.Series]


def build_candidates() -> List[FMSCandidate]:
    def f_old_linear(df: pd.DataFrame) -> pd.Series:
        """
        기존 FMS 선형 공식:
        FMS = 0.2*Z(R_1M) + 0.3*Z(R_3M) + 0.3*Z(R2_3M) + 0.2*Z(AboveEMA50) - 0.2*Z(Vol20_Ann)
        (MaxDD는 사용하지 않음)
        """
        r1 = df["R_1M"]
        r3 = df["R_3M"]
        r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
        ema50 = df["AboveEMA50"]
        vol20 = df["Vol20_Ann"]

        return (
            0.2 * z(r1)
            + 0.3 * z(r3)
            + 0.3 * z(r2.fillna(0.0))
            + 0.2 * z(ema50)
            - 0.2 * z(vol20.fillna(vol20.median()))
        )

    def f_linear(df: pd.DataFrame) -> pd.Series:
        """
        기본 가중합 버전 (현재 FMS와 유사하지만 일부 조정).
        """
        r1 = df["R_1M"]
        r3 = df["R_3M"]
        r6 = df["R_6M"]
        r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
        ema50 = df["AboveEMA50"].clip(lower=-0.5, upper=1.5)
        vol20 = df["Vol20_Ann"]
        maxdd = df["MaxDD_Pct"]

        # 드로우다운은 음수 → 부호 반전 후 클리핑
        dd_penalty = (-maxdd).clip(lower=0.0, upper=80.0)

        # 단기 1M 수익률 가중치는 작게 두고, 3M/6M을 더 중시
        pos = (
            0.5 * z(r3)
            + 0.3 * z(r6)
            + 0.4 * z(r2)
            + 0.3 * z(ema50)
            + 0.1 * z(r1)
        )
        neg = 0.4 * z(dd_penalty) + 0.3 * z(vol20)
        return pos - neg

    def f_nonlinear(df: pd.DataFrame) -> pd.Series:
        """
        비선형/조건적 버전:
        - R2가 0.7 이하일 땐 거의 가산 안 하고, 0.9 이상에서 강한 가산.
        - MaxDD는 -30% 이후부터 급격히 패널티 증가.
        - Vol20은 중간까지는 허용, 높은 구간에서만 강하게 패널티.
        - R_1M은 R2와 3M/6M 추세가 받쳐줄 때만 가산, 그렇지 않으면 감점 트리거.
        """
        r1 = df["R_1M"]
        r3 = df["R_3M"]
        r6 = df["R_6M"]
        r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
        ema50 = df["AboveEMA50"].clip(lower=-0.5, upper=1.5)
        vol20 = df["Vol20_Ann"]
        maxdd = df["MaxDD_Pct"]

        # R2 piecewise 가중: 0.7 이하는 약하게, 0.7~0.9 중간, 0.9 이상 강하게
        r2_effect = np.where(
            r2 < 0.7,
            0.2 * r2,
            np.where(r2 < 0.9, 0.6 * r2, 1.2 * r2),
        )
        r2_term = z(pd.Series(r2_effect, index=df.index))

        # MaxDD 패널티: -30까지는 완만, 이후는 제곱으로 강화
        dd_mag = (-maxdd).clip(lower=0.0)  # 0~100 정도
        dd_soft = dd_mag.clip(upper=30.0)
        dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0**2) * 70.0
        dd_combined = dd_soft + dd_hard
        dd_penalty = z(pd.Series(dd_combined, index=df.index))

        # Vol20 패널티: 중간까지는 완만, 상위 구간에서 제곱
        v = vol20.clip(lower=0.0)
        v_soft = v.clip(upper=np.nanpercentile(v, 60))
        v_hard = (v - np.nanpercentile(v, 60)).clip(lower=0.0) ** 2
        v_combined = v_soft + v_hard
        vol_penalty = z(pd.Series(v_combined, index=df.index))

        # R3/R6: 장·중기 추세
        r3_term = z(r3)
        r6_term = z(r6)

        # EMA50: 위에 얼마나 떠 있는지
        ema_term = z(ema50)

        # R1: R2와 3M/6M이 좋을 때만 가산, 아니면 감점
        quality_mask = (r2 > 0.85) & (r3 > 0.3) & (r6 > 0.5)
        r1_good = pd.Series(np.where(quality_mask, r1, 0.0), index=df.index)
        r1_bad = pd.Series(np.where(~quality_mask & (r1 > 0.3), r1, 0.0), index=df.index)
        r1_pos = z(r1_good)
        r1_neg = z(r1_bad)

        pos = (
            0.45 * r3_term
            + 0.35 * r6_term
            + 0.5 * r2_term
            + 0.3 * ema_term
            + 0.15 * r1_pos
        )
        neg = 0.5 * dd_penalty + 0.35 * vol_penalty + 0.15 * r1_neg

        return pos - neg

    return [
        FMSCandidate("old_linear", f_old_linear),
        FMSCandidate("linear_v1", f_linear),
        FMSCandidate("nonlinear_v1", f_nonlinear),
    ]


def main() -> None:
    df = pd.read_csv(FEATURE_CSV, index_col=0)
    true_rank = df["rank"]

    cands = build_candidates()
    results: Dict[str, float] = {}

    for cand in cands:
        score = cand.func(df)
        inv = pairwise_inversion_rate(true_rank, score)
        results[cand.name] = inv
        print(f"{cand.name}: inversion_rate={inv:.4f}")

    best_name = min(results, key=results.get)
    print("BEST:", best_name, "inversion_rate=", f"{results[best_name]:.4f}")


if __name__ == "__main__":
    main()

