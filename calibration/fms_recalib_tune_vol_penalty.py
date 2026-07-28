"""
정답셋(fms_recalib_features.csv) 기준으로 Vol20 패널티 형태의 파라미터를 탐색합니다.

원칙:
- 다른 항/가중치는 고정, Vol20 패널티 'mapping'만 변경 (Iteration 3 단일 변경)
- 결과는 inversion_rate↓, spearman_rho↑, pair_delta_error↓ 를 우선으로 비교
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURE_CSV = "fms_recalib_features.csv"


def pairwise_inversion_rate(true_rank: pd.Series, score: pd.Series) -> float:
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


def score_to_model_rank(score: pd.Series) -> pd.Series:
    order = score.sort_values(ascending=False).index.to_list()
    rank_map = {sym: i + 1 for i, sym in enumerate(order)}
    return pd.Series({sym: rank_map.get(sym, np.nan) for sym in score.index})


def compute_pairwise_rank_delta_error(true_rank: pd.Series, model_rank: pd.Series) -> float:
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


def fms_score_with_vol_params(df: pd.DataFrame, *, q_pct: float, hard_power: float, hard_scale: float) -> pd.Series:
    """Offline Vol20-mapping search only — delegates to ``core.fms``.

    Baseline must use ``f_current`` (``core.fms``). This helper keeps the full
    production scoring body and overrides only the Vol20 mapping knobs:
    ``vol_q_pct``, ``vol_hard_power``, ``vol_hard_scale``.
    """
    from dataclasses import asdict

    from core.fms import production_fms_score_params, score_fms_from_feature_frame

    params = asdict(production_fms_score_params())
    params["vol_q_pct"] = float(q_pct)
    params["vol_hard_power"] = float(hard_power)
    params["vol_hard_scale"] = float(hard_scale)
    return score_fms_from_feature_frame(df, params=params)


@dataclass(frozen=True)
class Metrics:
    inv: float
    rho: float
    pair_err: float


def compute_metrics(df: pd.DataFrame, score: pd.Series) -> Metrics:
    true_rank = df["rank"]
    inv = pairwise_inversion_rate(true_rank, score)
    model_rank = score_to_model_rank(score)
    common = pd.concat([true_rank, model_rank], axis=1).dropna()
    rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
    pair_err = compute_pairwise_rank_delta_error(true_rank, model_rank)
    return Metrics(inv=float(inv), rho=float(rho), pair_err=float(pair_err))


def dominates(a: Metrics, b: Metrics) -> bool:
    """All three strictly better."""
    return (a.inv < b.inv) and (a.rho > b.rho) and (a.pair_err < b.pair_err)


def main() -> None:
    import os

    if not os.path.exists(FEATURE_CSV):
        print(f"{FEATURE_CSV}가 없습니다. python fms_recalib_build_features.py 를 먼저 실행하세요.")
        return

    df = pd.read_csv(FEATURE_CSV, index_col=0)

    from fms_recalib_evaluate_formulas import f_current

    # Baseline = production SSOT (core.fms). Vol-mapping search is offline only.
    base = compute_metrics(df, f_current(df))
    print("=== Baseline (production / core.fms) ===")
    print(f"inversion_rate={base.inv:.4f}  spearman_rho={base.rho:.4f}  pair_delta_error={base.pair_err:.4f}")

    q_pcts = [50.0, 55.0, 60.0, 65.0, 70.0]
    hard_powers = [1.5, 1.75, 2.0, 2.25, 2.5]
    hard_scales = [0.25, 0.5, 0.75, 1.0, 1.25]

    best = None
    best_params = None
    improved = []

    for q_pct, p, s in itertools.product(q_pcts, hard_powers, hard_scales):
        score = fms_score_with_vol_params(df, q_pct=q_pct, hard_power=p, hard_scale=s)
        m = compute_metrics(df, score)
        if dominates(m, base):
            improved.append((q_pct, p, s, m))
        # pick a best by lexicographic: inv asc, rho desc, pair_err asc
        key = (m.inv, -m.rho, m.pair_err)
        if best is None or key < (best.inv, -best.rho, best.pair_err):
            best = m
            best_params = (q_pct, p, s)

    print("\n=== Best (by inv↓, rho↑, err↓ ordering) ===")
    q_pct, p, s = best_params
    print(f"params: q_pct={q_pct:.1f}, hard_power={p:.2f}, hard_scale={s:.2f}")
    print(f"inversion_rate={best.inv:.4f}  spearman_rho={best.rho:.4f}  pair_delta_error={best.pair_err:.4f}")

    print("\n=== Strictly dominating candidates (all three improved vs baseline) ===")
    if not improved:
        print("none")
        return
    # show top 10 by inv then rho then err
    improved_sorted = sorted(improved, key=lambda x: (x[3].inv, -x[3].rho, x[3].pair_err))[:10]
    for q_pct, p, s, m in improved_sorted:
        print(f"q={q_pct:.0f} p={p:.2f} s={s:.2f} | inv={m.inv:.4f} rho={m.rho:.4f} err={m.pair_err:.4f}")


if __name__ == "__main__":
    main()

