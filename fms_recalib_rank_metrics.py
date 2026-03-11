"""
FMS 재보정: 수정 전 vs 수정 후만 비교.
- Spearman ρ, 쌍별 순위차 오차
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fms_recalib_evaluate_formulas import f_current, f_proposed


FEATURE_CSV = "fms_recalib_features.csv"


def compute_pairwise_rank_delta_error(true_rank: pd.Series, model_rank: pd.Series) -> float:
    """쌍별 순위차 오차. 낮을수록 좋음."""
    df = pd.concat([true_rank, model_rank], axis=1).dropna()
    df.columns = ["true_rank", "model_rank"]
    n = len(df)
    if n <= 1:
        return 0.0
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


def score_to_model_rank(score: pd.Series) -> pd.Series:
    """점수 -> 모델 순위 (1=최상위)."""
    order = score.sort_values(ascending=False).index.to_list()
    rank_map = {sym: i + 1 for i, sym in enumerate(order)}
    return pd.Series({sym: rank_map.get(sym, np.nan) for sym in score.index})


def main() -> None:
    import os
    if not os.path.exists(FEATURE_CSV):
        print(f"{FEATURE_CSV}가 없습니다. python fms_recalib_build_features.py 를 먼저 실행하세요.")
        return
    df = pd.read_csv(FEATURE_CSV, index_col=0)
    true_rank = df["rank"]

    for name, score_func in [("current", f_current), ("proposed", f_proposed)]:
        score = score_func(df)
        model_rank = score_to_model_rank(score)
        common = pd.concat([true_rank, model_rank], axis=1).dropna()
        rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
        pair_err = compute_pairwise_rank_delta_error(true_rank, model_rank)
        print(f"{name}: spearman_rho={rho:.4f}, pair_delta_error={pair_err:.4f}")

    sc_current = f_current(df)
    sc_proposed = f_proposed(df)
    mr_c = score_to_model_rank(sc_current)
    mr_p = score_to_model_rank(sc_proposed)
    rho_c, _ = spearmanr(true_rank, mr_c)
    rho_p, _ = spearmanr(true_rank, mr_p)
    err_c = compute_pairwise_rank_delta_error(true_rank, mr_c)
    err_p = compute_pairwise_rank_delta_error(true_rank, mr_p)

    print("\n=== 개선 여부 ===")
    if rho_p > rho_c and err_p < err_c:
        print("proposed가 current보다 우수 (Spearman↑, pair_delta_error↓)")
    else:
        print("proposed가 일부 지표에서 current보다 나쁨. 로직 재검토 필요.")


if __name__ == "__main__":
    main()
