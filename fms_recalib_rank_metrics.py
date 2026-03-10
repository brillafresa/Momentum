"""
FMS 재보정 후보 수식의 랭크 품질 지표 계산 스크립트.

- 입력: fms_recalib_features.csv
- 출력: 각 FMS 후보에 대한 Spearman 상관계수와 쌍별 순위차 오차
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fms_recalib_evaluate_formulas import build_candidates


FEATURE_CSV = "fms_recalib_features.csv"


def compute_pairwise_rank_delta_error(true_rank: pd.Series, model_rank: pd.Series) -> float:
    """
    쌍별 순위차 오차:
    - 각 쌍 (i, j)에 대해, 정답 순위차: d_true = rank_j - rank_i
      (rank는 낮을수록 상위라고 가정)
    - 모델 순위차: d_model = model_rank_j - model_rank_i
    - 오차: |d_true - d_model|
    - 모든 쌍에 대해 평균을 취한 값을 반환
    """
    df = pd.concat([true_rank, model_rank], axis=1).dropna()
    df.columns = ["true_rank", "model_rank"]
    n = len(df)
    if n <= 1:
        return 0.0

    # 정답 순서대로 정렬 (1,2,3,...)
    df_sorted = df.sort_values("true_rank", ascending=True)
    r_true = df_sorted["true_rank"].to_numpy()
    r_model = df_sorted["model_rank"].to_numpy()

    total_err = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_true = r_true[j] - r_true[i]
            d_model = r_model[j] - r_model[i]
            total_err += abs(d_true - d_model)
            pairs += 1
    return total_err / pairs if pairs else 0.0


def main() -> None:
    df = pd.read_csv(FEATURE_CSV, index_col=0)
    true_rank = df["rank"]

    cands = {c.name: c for c in build_candidates()}
    names = ["old_linear", "nonlinear_v1"]

    # 공통: 정답 순서대로 정렬한 인덱스
    base = true_rank.dropna().sort_values(ascending=True)
    idx = base.index.to_list()

    for name in names:
        score = cands[name].func(df)
        # 점수가 높을수록 상위이므로, 내림차순 정렬 기준으로 rank 부여
        order = score.sort_values(ascending=False).index.to_list()
        rank_map = {sym: i + 1 for i, sym in enumerate(order)}
        model_rank = pd.Series({sym: rank_map.get(sym, np.nan) for sym in df.index})

        # Spearman 상관 (정답 rank vs 모델 rank)
        common = pd.concat([true_rank, model_rank], axis=1).dropna()
        rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])

        # 쌍별 순위차 오차
        pair_err = compute_pairwise_rank_delta_error(true_rank, model_rank)

        print(f"{name}: spearman_rho={rho:.4f}, pair_delta_error={pair_err:.4f}")


if __name__ == "__main__":
    main()

