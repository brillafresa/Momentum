# -*- coding: utf-8 -*-
"""Pattern summary + natural-language ranking rules for scratch refit.

Reads the latest feature table, prints TOP/MID/BOT group means for high-resolution
and non-overlapping segment features, and writes
``fms_recalib_natural_language_rules.json`` for the nonlinear MC pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from core.fms import score_fms_from_feature_frame

FEATURE_CSV = "fms_recalib_features.csv"
OUT_JSON = "fms_recalib_natural_language_rules.json"

KEY_COLS = [
    "R_3M",
    "R2_3M",
    "SEG_RET_0_3",
    "SEG_RET_0_5",
    "SEG_RET_5_21",
    "SEG_RET_21_63",
    "SEG_SLOPE_0_5",
    "PRIOR_SUPPORT_SIGN",
    "TREND_QUALITY_21D",
    "JUMP_DISCONTINUITY_3M",
    "STALE_AGE",
    "Vol20_Ann",
    "RECENT_UP_DAYS_5D",
    "RECENT_JUMP_SHARE_5D",
    "MID_DIP_RECOVERY",
    "STALE_AFTER_RUN",
]


def _default_rules(n: int, low_r3m_bot_share: float) -> List[dict]:
    """Human-readable rules seeded from the 2026-08-02 ground-truth review.

    Rules are intentionally free of asset-class / ticker exceptions.
    """
    return [
        {
            "id": "abs_return_floor",
            "text": (
                "최근 3개월 절대 수익률이 매우 낮으면(대략 +2% 미만) R²·변동성 모양과 "
                "무관하게 하위권에 둔다. 저수익만으로도 하위권을 강하게 설명한다."
            ),
            "evidence": f"n={n}, R_3M<2% 종목의 하위권 집중도≈{low_r3m_bot_share:.2f}",
        },
        {
            "id": "recent_high_resolution",
            "text": (
                "최근 3일·1주 수익률/기울기가 양이고 클수록 상위 후보로 본다. "
                "단발 점프보다 여러 날에 걸쳐 쌓인 상승을 선호한다."
            ),
            "evidence": "TOP 3분위 SEG_RET_0_5 / R_3D_LOG 평균이 MID·BOT보다 큼",
        },
        {
            "id": "prior_support_gate",
            "text": (
                "최근 상승은 1개월~3개월 전 구간(비중첩)에서도 방향이 맞을 때 "
                "(PRIOR_SUPPORT_SIGN=1) 더 신뢰한다. 지지의 크기보다 방향 일치가 중요하다."
            ),
            "evidence": "SEG_RET_21_63은 TOP≈MID이고 BOT만 크게 낮음 → 이진 지지 게이트",
        },
        {
            "id": "continuation_over_v_bounce",
            "text": (
                "이전 구간이 하락/보합인데 최근만 반등한 V자보다, 이전에도 상승하던 "
                "연속 상승 경로를 같은 최근 수익률이라도 더 높게 둔다."
            ),
            "evidence": "사용자 의도 + 비중첩 구간 분리 목적",
        },
        {
            "id": "regime_dependent_recent_weight",
            "text": (
                "절대 최근 1개월 수익률이 약한 구간에서는 장기 꾸준함보다 단기 상승률 "
                "가중치를 키운다. 절대 수익이 충분하면 이전 추세 연속성 비중을 키운다."
            ),
            "evidence": "지시서 자연어 예시 + regime_switch_recent family",
        },
        {
            "id": "soft_cash_without_labels",
            "text": (
                "초저수익∧초저변동이 동시에 나타나면 자산군 라벨 없이 하위 점수로 민다. "
                "고R² 조건에만 의존하지 않는다."
            ),
            "evidence": "채권·예금성 ETF가 R² 게이트를 피해 고득점하던 production 실패",
        },
        {
            "id": "jump_stale_penalty",
            "text": (
                "급등 후 follow-through가 없거나(STALE_AGE·JUMP) 정체면 최근 절대수익이 "
                "커도 상위권에서 내린다."
            ),
            "evidence": "TOP에서 JUMP/STALE 평균이 BOT보다 낮음",
        },
        {
            "id": "r2_not_overpenalize_strong_abs",
            "text": (
                "3개월 절대수익과 최근 구간이 충분히 강하면 R²가 피어보다 다소 낮다는 "
                "이유만으로 과도 감점하지 않는다."
            ),
            "evidence": "GSHD류: 강한 모멘텀인데 R² Z 감점으로 production 순위 붕괴",
        },
        {
            "id": "mid_dip_continuation",
            "text": (
                "이전 추세가 있는 종목이 중간 구간에서 잠시 조정을 받아도, 최근 구간이 "
                "회복하면(MID_DIP_RECOVERY) 연속 상승으로 본다. V자 단독 반등과 구분한다."
            ),
            "evidence": "round-1 under-rank: RLI/NEXN/HOMB류 중간 조정 후 회복",
        },
        {
            "id": "stale_after_large_run",
            "text": (
                "과거 절대수익이 커도 최근이 정체(STALE_AFTER_RUN)하거나 최근 5일 상승이 "
                "단발 급등 비중(RECENT_JUMP_SHARE_5D)으로 설명되면 상위권을 억제한다."
            ),
            "evidence": "round-1 over-rank: MNPR/MBX/CORT류 대상승+정체/스파이크",
        },
        {
            "id": "recent_breadth",
            "text": (
                "최근 5거래일 중 상승일 수(RECENT_UP_DAYS_5D)가 넓을수록 같은 누적수익이라도 "
                "더 신뢰한다."
            ),
            "evidence": "단발 급등 vs 분산 상승 구분",
        },
    ]


def main() -> None:
    df = pd.read_csv(FEATURE_CSV, index_col=0)
    if "rank" not in df.columns:
        raise RuntimeError(f"{FEATURE_CSV} missing rank column")

    cols = [c for c in KEY_COLS if c in df.columns]
    n = len(df)
    third = max(1, n // 3)
    top = df.nsmallest(third, "rank")
    mid = df.iloc[third : 2 * third]
    bot = df.nlargest(third, "rank")

    print("N =", n, "third =", third)
    print("\n=== mean by group ===")
    for name, group in [("TOP", top), ("MID", mid), ("BOT", bot)]:
        print("\n", name)
        print(group[cols].mean())

    baseline = score_fms_from_feature_frame(df)
    from calibration.ranking_metrics import compute_metrics

    metrics = compute_metrics(df["rank"], baseline)
    print("\n=== production benchmark ===")
    print(metrics)

    low = df["R_3M"] < 0.02
    bot_cut = df["rank"] >= (n - third + 1)
    share = float((low & bot_cut).sum() / max(int(low.sum()), 1))
    rules = _default_rules(n, share)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_symbols": n,
        "feature_csv": FEATURE_CSV,
        "production_benchmark": {
            "inversion_rate": metrics.inv,
            "spearman_rho": metrics.rho,
            "pair_delta_error": metrics.pair_err,
        },
        "group_means": {
            "TOP": top[cols].mean().to_dict(),
            "MID": mid[cols].mean().to_dict(),
            "BOT": bot[cols].mean().to_dict(),
        },
        "rules": rules,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("Wrote", OUT_JSON, "rules", len(rules))


if __name__ == "__main__":
    main()
