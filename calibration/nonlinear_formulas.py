# -*- coding: utf-8 -*-
"""Interpretable nonlinear FMS candidate formulas for scratch refit.

Each family encodes natural-language ranking rules as a continuous response
function. Parameters are fitted by Monte Carlo search in
``fms_recalib_nonlinear_mc``; production promotion remains a separate step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd


def _col(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return frame[name].astype(float).replace([np.inf, -np.inf], np.nan).fillna(default)


def softplus(x: pd.Series | np.ndarray, beta: float = 8.0) -> pd.Series:
    """Smooth ReLU; keeps gradients usable near the floor."""
    arr = np.asarray(x, dtype=float)
    out = np.log1p(np.exp(np.clip(beta * arr, -40.0, 40.0))) / beta
    return pd.Series(out, index=getattr(x, "index", None), dtype=float)


def smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
    if edge1 == edge0:
        return pd.Series(0.0, index=x.index, dtype=float)
    t = ((x.astype(float) - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(float)


@dataclass(frozen=True)
class FormulaFamily:
    """One nonlinear score family with a natural-language intent."""

    name: str
    natural_language: str
    param_names: Sequence[str]
    sample_params: Callable[[np.random.Generator], Dict[str, float]]
    score: Callable[[pd.DataFrame, Dict[str, float]], pd.Series]


def _sample_positive(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _floor_gated_momentum_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "r3m_floor": float(rng.uniform(0.005, 0.04)),
        "r3m_full": float(rng.uniform(0.04, 0.15)),
        "w_recent": _sample_positive(rng, 0.3, 3.0),
        "w_mid": _sample_positive(rng, 0.1, 2.0),
        "w_prior": _sample_positive(rng, 0.05, 1.5),
        "w_abs": _sample_positive(rng, 0.2, 2.5),
        "w_r2": _sample_positive(rng, 0.05, 1.2),
        "w_jump": _sample_positive(rng, 0.2, 3.0),
        "w_stale": _sample_positive(rng, 0.1, 2.0),
        "recent_pow": float(rng.uniform(0.6, 1.8)),
        "support_boost": float(rng.uniform(0.5, 2.0)),
    }


def _score_floor_gated_momentum(
    frame: pd.DataFrame, p: Dict[str, float]
) -> pd.Series:
    """Absolute-return floor × recent high-res momentum × prior support."""
    r3m = _col(frame, "R_3M")
    recent = _col(frame, "SEG_RET_0_5")
    mid = _col(frame, "SEG_RET_5_21")
    prior = _col(frame, "SEG_RET_21_63")
    support = _col(frame, "PRIOR_SUPPORT_SIGN")
    r2 = _col(frame, "R2_3M", 0.5)
    jump = _col(frame, "JUMP_DISCONTINUITY_3M")
    stale = _col(frame, "STALE_AGE")

    floor_gate = softplus(r3m - p["r3m_floor"])
    abs_drive = smoothstep(r3m, p["r3m_floor"], p["r3m_full"])
    recent_term = np.sign(recent) * (np.abs(recent) ** p["recent_pow"])
    mid_term = mid
    prior_term = prior * (1.0 + p["support_boost"] * support)
    quality = 1.0 + p["w_r2"] * r2
    penalty = 1.0 + p["w_jump"] * jump.clip(lower=0.0) + p["w_stale"] * stale.clip(lower=0.0)
    core = (
        p["w_recent"] * recent_term
        + p["w_mid"] * mid_term
        + p["w_prior"] * prior_term
        + p["w_abs"] * abs_drive * r3m
    )
    score = (floor_gate * quality * core) / penalty
    return score.astype(float).rename("score")


def _regime_switch_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "weak_edge": float(rng.uniform(0.02, 0.08)),
        "strong_edge": float(rng.uniform(0.12, 0.35)),
        "a_recent_weak": _sample_positive(rng, 0.8, 4.0),
        "a_recent_strong": _sample_positive(rng, 0.2, 2.0),
        "a_prior_strong": _sample_positive(rng, 0.3, 2.5),
        "a_smooth": _sample_positive(rng, 0.1, 1.5),
        "b_vol": _sample_positive(rng, 0.2, 3.0),
        "c_jump": _sample_positive(rng, 0.2, 3.0),
        "pow_recent": float(rng.uniform(0.5, 1.6)),
    }


def _score_regime_switch(frame: pd.DataFrame, p: Dict[str, float]) -> pd.Series:
    """When absolute 3M return is weak, emphasize recent; when strong, continuity."""
    r3m = _col(frame, "R_3M")
    recent = _col(frame, "SEG_RET_0_5")
    mid = _col(frame, "SEG_RET_5_21")
    prior = _col(frame, "SEG_RET_21_63")
    support = _col(frame, "PRIOR_SUPPORT_SIGN")
    r2 = _col(frame, "R2_3M", 0.5)
    vol = _col(frame, "Vol20_Ann", 0.2)
    jump = _col(frame, "JUMP_DISCONTINUITY_3M")

    weak = 1.0 - smoothstep(r3m, p["weak_edge"] * 0.5, p["weak_edge"])
    strong = smoothstep(r3m, p["weak_edge"], p["strong_edge"])
    recent_nl = np.sign(recent) * (np.abs(recent) ** p["pow_recent"])
    body = (
        weak * p["a_recent_weak"] * recent_nl
        + strong
        * (
            p["a_recent_strong"] * recent_nl
            + p["a_prior_strong"] * prior * (0.5 + support)
            + 0.5 * mid
        )
        + p["a_smooth"] * r2 * strong
    )
    # Low absolute return + ultra-low vol → soft floor without asset-class labels.
    cash_like = (1.0 - smoothstep(r3m, 0.005, 0.03)) * (
        1.0 - smoothstep(vol, 0.005, 0.04)
    )
    score = softplus(r3m - 0.005) * body / (
        1.0 + p["b_vol"] * cash_like + p["c_jump"] * jump.clip(lower=0.0)
    )
    return score.astype(float).rename("score")


def _product_confirmation_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "floor": float(rng.uniform(0.005, 0.03)),
        "a_recent": float(rng.uniform(0.4, 1.8)),
        "a_mid": float(rng.uniform(0.2, 1.2)),
        "a_prior": float(rng.uniform(0.2, 1.4)),
        "a_abs": float(rng.uniform(0.3, 1.6)),
        "eps": float(rng.uniform(0.01, 0.08)),
        "w_jump": _sample_positive(rng, 0.3, 4.0),
        "w_stale": _sample_positive(rng, 0.2, 3.0),
        "w_eff": _sample_positive(rng, 0.1, 2.0),
    }


def _score_product_confirmation(
    frame: pd.DataFrame, p: Dict[str, float]
) -> pd.Series:
    """Multiplicative confirmation across non-overlapping segments."""
    recent = softplus(_col(frame, "SEG_RET_0_5") - 0.0)
    mid = softplus(_col(frame, "SEG_RET_5_21"))
    prior = softplus(_col(frame, "SEG_RET_21_63") * _col(frame, "PRIOR_SUPPORT_SIGN"))
    abs_ret = softplus(_col(frame, "R_3M") - p["floor"])
    eff = softplus(_col(frame, "TREND_EFFICIENCY_REWARD_15D"))
    jump = _col(frame, "JUMP_DISCONTINUITY_3M").clip(lower=0.0)
    stale = _col(frame, "STALE_AGE").clip(lower=0.0)
    eps = p["eps"]
    numer = (
        ((recent + eps) ** p["a_recent"])
        * ((mid + eps) ** p["a_mid"])
        * ((prior + eps) ** p["a_prior"])
        * ((abs_ret + eps) ** p["a_abs"])
        * (1.0 + p["w_eff"] * eff)
    )
    denom = (1.0 + p["w_jump"] * jump) * (1.0 + p["w_stale"] * stale)
    return (numer / denom).astype(float).rename("score")


def _hybrid_sqrt_log_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "c0": float(rng.uniform(0.2, 2.0)),
        "c1": float(rng.uniform(0.2, 2.5)),
        "c2": float(rng.uniform(0.1, 2.0)),
        "c3": float(rng.uniform(0.1, 1.5)),
        "c4": float(rng.uniform(0.2, 2.0)),
        "floor": float(rng.uniform(0.005, 0.03)),
        "pow_den": float(rng.uniform(0.5, 1.5)),
    }


def _score_hybrid_sqrt_log(frame: pd.DataFrame, p: Dict[str, float]) -> pd.Series:
    """Creative nonlinear blend: quadratic recent + sqrt mid × prior − log jump."""
    recent = _col(frame, "SEG_RET_0_5")
    mid = _col(frame, "SEG_RET_5_21")
    prior = _col(frame, "SEG_RET_21_63")
    support = _col(frame, "PRIOR_SUPPORT_SIGN")
    r3m = _col(frame, "R_3M")
    jump = _col(frame, "JUMP_DISCONTINUITY_3M").clip(lower=0.0)
    stale = _col(frame, "STALE_AGE").clip(lower=0.0)
    numer = (
        p["c0"] * (recent ** 2) * np.sign(recent)
        + np.sqrt(np.maximum(p["c1"] + mid, 1e-8)) * (prior * (0.5 + support))
        + p["c2"] * softplus(r3m - p["floor"])
        - p["c3"] * np.log1p(1.23 * jump)
    )
    denom = (1.0 + p["c4"] * stale) ** p["pow_den"]
    return (softplus(r3m - p["floor"] * 0.5) * numer / denom).astype(float).rename(
        "score"
    )


def _pullback_continuation_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "floor": float(rng.uniform(0.008, 0.035)),
        "w_recent": _sample_positive(rng, 0.5, 3.5),
        "w_mid_pos": _sample_positive(rng, 0.2, 2.0),
        "w_mid_neg_forgive": _sample_positive(rng, 0.1, 2.5),
        "w_prior": _sample_positive(rng, 0.2, 2.0),
        "w_abs": _sample_positive(rng, 0.2, 2.0),
        "w_breadth": _sample_positive(rng, 0.05, 1.5),
        "w_stale_run": _sample_positive(rng, 0.3, 4.0),
        "w_jump_share": _sample_positive(rng, 0.3, 4.0),
        "w_eff": _sample_positive(rng, 0.1, 2.0),
        "pow_recent": float(rng.uniform(0.6, 1.7)),
    }


def _score_pullback_continuation(
    frame: pd.DataFrame, p: Dict[str, float]
) -> pd.Series:
    """Allow mid-band dips when prior support + recent recovery confirm continuation."""
    recent = _col(frame, "SEG_RET_0_5")
    mid = _col(frame, "SEG_RET_5_21")
    prior = _col(frame, "SEG_RET_21_63")
    support = _col(frame, "PRIOR_SUPPORT_SIGN")
    r3m = _col(frame, "R_3M")
    breadth = _col(frame, "RECENT_UP_DAYS_5D") / 5.0
    dip_rec = _col(frame, "MID_DIP_RECOVERY")
    stale_run = _col(frame, "STALE_AFTER_RUN").clip(lower=0.0)
    jump_share = _col(frame, "RECENT_JUMP_SHARE_5D").clip(lower=0.0, upper=1.0)
    eff = _col(frame, "TREND_EFFICIENCY_REWARD_15D")

    recent_nl = np.sign(recent) * (np.abs(recent) ** p["pow_recent"])
    mid_term = p["w_mid_pos"] * softplus(mid) + p["w_mid_neg_forgive"] * dip_rec * support
    body = (
        p["w_recent"] * recent_nl
        + mid_term
        + p["w_prior"] * prior * (0.5 + support)
        + p["w_abs"] * softplus(r3m - p["floor"])
        + p["w_breadth"] * breadth
        + p["w_eff"] * softplus(eff)
    )
    penalty = (
        1.0
        + p["w_stale_run"] * stale_run
        + p["w_jump_share"] * softplus(jump_share - 0.55)
    )
    return (softplus(r3m - p["floor"] * 0.5) * body / penalty).astype(float).rename(
        "score"
    )


def _anti_stale_run_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "floor": float(rng.uniform(0.01, 0.04)),
        "a_abs": float(rng.uniform(0.4, 1.6)),
        "a_recent": float(rng.uniform(0.6, 2.0)),
        "a_prior": float(rng.uniform(0.2, 1.2)),
        "a_r2": float(rng.uniform(0.1, 1.0)),
        "stale_edge0": float(rng.uniform(0.15, 0.45)),
        "stale_edge1": float(rng.uniform(0.6, 1.4)),
        "jump_edge0": float(rng.uniform(0.45, 0.65)),
        "jump_edge1": float(rng.uniform(0.75, 0.98)),
        "w_mid": _sample_positive(rng, 0.1, 1.5),
        "eps": float(rng.uniform(0.01, 0.06)),
    }


def _score_anti_stale_run(frame: pd.DataFrame, p: Dict[str, float]) -> pd.Series:
    """Large absolute runs only count when recent path is alive and not one-day dominated."""
    r3m = _col(frame, "R_3M")
    recent = _col(frame, "SEG_RET_0_5")
    mid = _col(frame, "SEG_RET_5_21")
    prior = _col(frame, "SEG_RET_21_63")
    support = _col(frame, "PRIOR_SUPPORT_SIGN")
    r2 = _col(frame, "R2_3M", 0.5)
    stale_run = _col(frame, "STALE_AFTER_RUN").clip(lower=0.0)
    jump_share = _col(frame, "RECENT_JUMP_SHARE_5D").clip(lower=0.0, upper=1.0)
    breadth = _col(frame, "RECENT_UP_DAYS_5D") / 5.0

    # Gate large historical gains when the path looks finished or spike-only.
    alive = 1.0 - smoothstep(stale_run, p["stale_edge0"], p["stale_edge1"])
    not_spike = 1.0 - smoothstep(jump_share, p["jump_edge0"], p["jump_edge1"])
    abs_term = softplus(r3m - p["floor"]) ** p["a_abs"]
    recent_term = softplus(recent) ** p["a_recent"]
    prior_term = (softplus(prior) * (0.5 + support) + p["eps"]) ** p["a_prior"]
    mid_term = 1.0 + p["w_mid"] * softplus(mid)
    score = (
        abs_term
        * recent_term
        * prior_term
        * mid_term
        * (1.0 + p["a_r2"] * r2)
        * (0.4 + 0.6 * breadth)
        * (0.25 + 0.75 * alive)
        * (0.25 + 0.75 * not_spike)
    )
    return score.astype(float).rename("score")


def _alive_pullback_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "floor": float(rng.uniform(0.008, 0.035)),
        "w_recent": _sample_positive(rng, 0.5, 3.5),
        "w_mid_pos": _sample_positive(rng, 0.15, 1.8),
        "w_mid_neg_forgive": _sample_positive(rng, 0.2, 3.0),
        "w_prior": _sample_positive(rng, 0.2, 2.2),
        "w_abs": _sample_positive(rng, 0.15, 1.8),
        "w_breadth": _sample_positive(rng, 0.05, 1.8),
        "w_grind": _sample_positive(rng, 0.05, 1.5),
        "w_stale_run": _sample_positive(rng, 0.5, 5.0),
        "w_jump_share": _sample_positive(rng, 0.5, 5.0),
        "w_eff": _sample_positive(rng, 0.05, 1.5),
        "pow_recent": float(rng.uniform(0.55, 1.6)),
        "alive_boost": float(rng.uniform(0.3, 1.8)),
    }


def _score_alive_pullback(frame: pd.DataFrame, p: Dict[str, float]) -> pd.Series:
    """Delegate to core SSOT so MC search cannot drift from production math."""
    from core.fms_features import score_alive_pullback_from_params

    return score_alive_pullback_from_params(frame, p).rename("score")


FORMULA_FAMILIES: List[FormulaFamily] = [
    FormulaFamily(
        name="floor_gated_momentum",
        natural_language=(
            "3개월 절대수익이 바닥(약 0.5~4%) 아래면 강하게 억제하고, "
            "최근 1주·1주~1개월·1~3개월 비중첩 구간 수익과 이전 추세 지지 부호를 "
            "가중합한 뒤 점프·정체로 나눈다."
        ),
        param_names=tuple(_floor_gated_momentum_params(np.random.default_rng(0)).keys()),
        sample_params=_floor_gated_momentum_params,
        score=_score_floor_gated_momentum,
    ),
    FormulaFamily(
        name="regime_switch_recent",
        natural_language=(
            "절대 3개월 수익이 약하면 최근 구간 비중을 키우고, 충분하면 "
            "이전 추세 연속성·R²를 더 반영한다. 저수익·초저변동은 자산군 라벨 없이 "
            "동시 발생 시 감점한다."
        ),
        param_names=tuple(_regime_switch_params(np.random.default_rng(0)).keys()),
        sample_params=_regime_switch_params,
        score=_score_regime_switch,
    ),
    FormulaFamily(
        name="product_confirmation",
        natural_language=(
            "최근/중간/이전 비중첩 구간과 절대수익이 동시에 확인할 때만 점수가 "
            "커지는 곱셈형 확인 구조. 점프·정체는 분모 패널티."
        ),
        param_names=tuple(_product_confirmation_params(np.random.default_rng(0)).keys()),
        sample_params=_product_confirmation_params,
        score=_score_product_confirmation,
    ),
    FormulaFamily(
        name="hybrid_sqrt_log",
        natural_language=(
            "최근 수익 제곱 + sqrt(중간)×이전지지 − log(점프) 형태의 창의적 비선형 결합을 "
            "절대수익 softplus 게이트로 조절한다."
        ),
        param_names=tuple(_hybrid_sqrt_log_params(np.random.default_rng(0)).keys()),
        sample_params=_hybrid_sqrt_log_params,
        score=_score_hybrid_sqrt_log,
    ),
    FormulaFamily(
        name="pullback_continuation",
        natural_language=(
            "이전 추세가 있는 종목이 중간 구간에서 잠시 조정해도, 최근이 회복하면 "
            "연속 상승으로 본다. 대신 대상승 후 정체(STALE_AFTER_RUN)와 "
            "최근 5일 단발 급등 비중은 감점한다."
        ),
        param_names=tuple(_pullback_continuation_params(np.random.default_rng(0)).keys()),
        sample_params=_pullback_continuation_params,
        score=_score_pullback_continuation,
    ),
    FormulaFamily(
        name="anti_stale_run",
        natural_language=(
            "과거 절대수익이 커도 최근이 살아 있고 단발 급등이 아니며 정체 강도가 낮을 때만 "
            "그 절대수익을 인정한다. 고수익·정체형 과대평가를 억제한다."
        ),
        param_names=tuple(_anti_stale_run_params(np.random.default_rng(0)).keys()),
        sample_params=_anti_stale_run_params,
        score=_score_anti_stale_run,
    ),
    FormulaFamily(
        name="alive_pullback",
        natural_language=(
            "중간 조정 후 회복(alive)과 이전 지지를 가산하고, 최근이 죽은 대상승 정체만 "
            "STALE_AFTER_RUN으로 감점한다. 또한 적정 절대수익+고R²+비스파이크 경로는 "
            "smooth grind로 소폭 가산한다(자산군 라벨 없음)."
        ),
        param_names=tuple(_alive_pullback_params(np.random.default_rng(0)).keys()),
        sample_params=_alive_pullback_params,
        score=_score_alive_pullback,
    ),
]


def family_by_name(name: str) -> FormulaFamily:
    for family in FORMULA_FAMILIES:
        if family.name == name:
            return family
    raise KeyError(name)
