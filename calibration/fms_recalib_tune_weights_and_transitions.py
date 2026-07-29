"""
정답셋(fms_recalib_features.csv) 기준으로 FMS 파라미터(가중치 + 전이폭) 튜닝.

원칙 (Iteration 5):
- 수식 구조(항의 종류, 비선형 형태)는 고정
- 변경 대상은 '가중치'와 'smoothstep 전이폭' 같은 파라미터만
- 점수 비교는 inversion_rate↓, spearman_rho↑, pair_delta_error↓ (3지표 동시개선 우선)

주의:
- 이 스크립트는 정답셋에 대한 휴리스틱 최적화이므로 과적합 위험이 있습니다.
- 결과 채택 후에는 다른 샘플/구간에서도 상식적으로 납득되는지 재확인이 필요합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURE_CSV = "fms_recalib_features.csv"


def smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
    if edge1 == edge0:
        return pd.Series(0.0, index=x.index)
    t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
    return t * t * (3.0 - 2.0 * t)


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
    return (a.inv < b.inv) and (a.rho > b.rho) and (a.pair_err < b.pair_err)


def fms_score(df: pd.DataFrame, p: Dict[str, float]) -> pd.Series:
    """Parameterized FMS for Monte-Carlo search — delegates to ``core.fms``.

    Baseline comparisons must use ``fms_recalib_evaluate_formulas.f_current``
    (default production params). Promote winning ``p`` into
    ``production_fms_score_params()`` before shipping.
    """
    from core.fms import score_legacy_fms_from_feature_frame

    return score_legacy_fms_from_feature_frame(df, params=p)


def baseline_params() -> Dict[str, float]:
    """Search baseline = production SSOT weights (``core.fms``)."""
    from dataclasses import asdict

    from core.fms import production_fms_score_params

    return asdict(production_fms_score_params())


def sample_params(rng: np.random.Generator, base: Dict[str, float]) -> Dict[str, float]:
    p = dict(base)

    # Weights: lognormal-ish perturbation around base, then clip to sane ranges
    def jitter(key: str, sigma: float, lo: float, hi: float) -> None:
        val = base[key]
        # multiplicative noise around 1
        mult = float(np.exp(rng.normal(0.0, sigma)))
        p[key] = float(np.clip(val * mult, lo, hi))

    jitter("w_r3", 0.15, 0.15, 0.80)
    jitter("w_r4", 0.15, 0.10, 0.70)
    jitter("w_r2", 0.15, 0.10, 0.90)
    jitter("w_ema", 0.20, 0.05, 0.60)
    jitter("w_ema_shape", 0.20, 0.05, 0.60)
    jitter("w_recent", 0.20, 0.00, 0.60)
    jitter("w_r1_pos", 0.25, 0.00, 0.40)
    jitter("w_dd", 0.15, 0.10, 0.80)
    jitter("w_vol", 0.15, 0.05, 0.70)
    jitter("w_r1_neg", 0.25, 0.00, 0.40)
    jitter("w_break", 0.20, 0.00, 0.60)
    jitter("w_down5", 0.20, 0.00, 0.60)
    jitter("w_under_depth", 0.20, 0.00, 0.60)
    jitter("w_under_days", 0.20, 0.00, 0.60)

    # Transition widths: sample in small continuous ranges
    p["r2_transition_w"] = float(rng.uniform(0.005, 0.05))
    p["gate_r3_w"] = float(rng.uniform(0.005, 0.03))
    p["gate_r4_w"] = float(rng.uniform(0.005, 0.04))

    # Ramp hi bounds (keep > low edges)
    p["level_r3_hi"] = float(rng.uniform(0.10, 0.25))
    p["level_r4_hi"] = float(rng.uniform(0.18, 0.60))

    # floor: keep close-ish to 0.8 but allow explore
    p["r2_floor"] = float(rng.uniform(0.60, 0.90))

    return p


def params_key(p: Dict[str, float]) -> str:
    # stable JSON for logging
    return json.dumps({k: round(float(v), 6) for k, v in sorted(p.items())}, ensure_ascii=False)


def main() -> None:
    import os

    if not os.path.exists(FEATURE_CSV):
        print(f"{FEATURE_CSV}가 없습니다. python fms_recalib_build_features.py 를 먼저 실행하세요.")
        return

    df = pd.read_csv(FEATURE_CSV, index_col=0)
    from fms_recalib_evaluate_formulas import f_current

    # Baseline = production SSOT (core.fms). fms_score(params) searches the same body.
    base_m = compute_metrics(df, f_current(df))
    base = baseline_params()

    print("=== Baseline (production / core.fms) ===")
    print(f"inversion_rate={base_m.inv:.4f}  spearman_rho={base_m.rho:.4f}  pair_delta_error={base_m.pair_err:.4f}")

    rng = np.random.default_rng(20260311)

    iters = 1500
    best_p = base
    best_m = base_m

    dominating: List[Tuple[Metrics, Dict[str, float]]] = []

    for _ in range(iters):
        cand = sample_params(rng, base)
        m = compute_metrics(df, fms_score(df, cand))
        key = (m.inv, -m.rho, m.pair_err)
        best_key = (best_m.inv, -best_m.rho, best_m.pair_err)
        if key < best_key:
            best_p, best_m = cand, m
        if dominates(m, base_m):
            dominating.append((m, cand))

    print("\n=== Best candidate (lexicographic inv↓, rho↑, err↓) ===")
    print(f"inversion_rate={best_m.inv:.4f}  spearman_rho={best_m.rho:.4f}  pair_delta_error={best_m.pair_err:.4f}")
    print("params:", params_key(best_p))

    print("\n=== Strictly dominating vs baseline (top 5) ===")
    if not dominating:
        print("none")
    else:
        dominating_sorted = sorted(dominating, key=lambda x: (x[0].inv, -x[0].rho, x[0].pair_err))[:5]
        for m, p in dominating_sorted:
            print(f"inv={m.inv:.4f} rho={m.rho:.4f} err={m.pair_err:.4f} | {params_key(p)}")


if __name__ == "__main__":
    main()

