"""
정답셋(fms_recalib_features.csv) 기준으로 Vol20 패널티 형태의 파라미터를 탐색합니다.

원칙:
- 다른 항/가중치는 고정, Vol20 패널티 'mapping'만 변경 (Iteration 3 단일 변경)
- 결과는 inversion_rate↓, spearman_rho↑, pair_delta_error↓ 를 우선으로 비교
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURE_CSV = "fms_recalib_features.csv"


def z(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    m = np.nanmean(s)
    sd = np.nanstd(s)
    if not sd or np.isnan(sd):
        return s * 0.0
    return (s - m) / sd


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


def smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
    if edge1 == edge0:
        return pd.Series(0.0, index=x.index)
    t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
    return t * t * (3.0 - 2.0 * t)


def fms_score_with_vol_params(df: pd.DataFrame, *, q_pct: float, hard_power: float, hard_scale: float) -> pd.Series:
    # --- current logic copied from fms_recalib_evaluate_formulas.f_current ---
    r1 = df["R_1M"]
    r3 = df["R_3M"]
    r6 = df["R_6M"]
    r2 = df["R2_3M"].clip(lower=0.0, upper=1.0)
    ema50 = df["AboveEMA50"].clip(lower=-0.5, upper=1.5)
    vol20 = df["Vol20_Ann"]
    maxdd = df["MaxDD_Pct"]

    w_mid = smoothstep(r2, 0.70 - 0.02, 0.70 + 0.02)
    w_high = smoothstep(r2, 0.90 - 0.02, 0.90 + 0.02)
    r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high
    r2_effect = r2_mult * r2

    r2_gate = smoothstep(r3, 0.05 - 0.01, 0.05 + 0.01) * smoothstep(r6, 0.08 - 0.01, 0.08 + 0.01)
    r2_level = smoothstep(r3, 0.05, 0.15) * smoothstep(r6, 0.08, 0.25)
    r2_strength = r2_gate * (0.80 + 0.20 * r2_level)
    r2_term = z(pd.Series(r2_effect * r2_strength, index=df.index))

    dd_mag = (-maxdd).clip(lower=0.0)
    dd_soft = dd_mag.clip(upper=30.0)
    dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
    dd_combined = dd_soft + dd_hard
    dd_penalty = z(pd.Series(dd_combined, index=df.index))

    # --- the only part we tune: Vol20 mapping ---
    v = vol20.clip(lower=0.0)
    q = np.nanpercentile(v, q_pct) if not v.dropna().empty else 0.0
    v_soft = v.clip(upper=q)
    # hard tail: scale * (excess^power)
    excess = (v - q).clip(lower=0.0)
    v_hard = hard_scale * (excess ** hard_power)
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

    # Baseline = current params (q=60, hard_power=2, hard_scale=1)
    base_score = fms_score_with_vol_params(df, q_pct=60.0, hard_power=2.0, hard_scale=1.0)
    base = compute_metrics(df, base_score)
    print("=== Baseline (current vol mapping) ===")
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

