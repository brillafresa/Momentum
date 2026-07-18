# -*- coding: utf-8 -*-
"""FMS snapshot / momentum scoring — pure logic, no network I/O.

Migrated from ``analysis_utils`` (``_mom_snapshot``, ``compute_fms_snapshot``,
``momentum_now_and_delta``). The transitional facade re-exports the public
entrypoints; prefer importing from ``core.fms`` in new code.

See HARNESS_RULES.md §2.5 (single source of truth).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.indicators import (
    ema,
    last_vol_annualized,
    returns_pct,
    r_squared_3m,
    ytd_return,
)
from core.tradeability import calculate_tradeability_filters


# Iteration 5 production weights / transition widths (reference-panel path).
_P_W_R3 = 0.46869
_P_W_R6 = 0.417409
_P_W_R2 = 0.505669
_P_W_EMA = 0.323264
_P_W_DD = 0.28298
_P_W_VOL = 0.291973
_P_R2_TRANSITION_W = 0.04552
_P_GATE_R3_W = 0.019359
_P_GATE_R6_W = 0.006355
_P_LEVEL_R3_HI = 0.205305
_P_LEVEL_R6_HI = 0.430268
_P_R2_FLOOR = 0.734629


def _smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
    """0..1로 부드럽게 전이 (C1 연속)."""
    if edge1 == edge0:
        return pd.Series(0.0, index=x.index)
    t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
    return t * t * (3.0 - 2.0 * t)


def _z_peer(x: pd.Series, mask_exclude: Optional[set] = None) -> pd.Series:
    """Z-score within the peer set, optionally excluding disqualified symbols from μ/σ."""
    x = x.astype(float)
    if mask_exclude:
        valid_idx = [idx for idx in x.index if idx not in mask_exclude]
        valid_x = x.loc[valid_idx] if valid_idx else x
    else:
        valid_x = x
    m = np.nanmean(valid_x)
    sd = np.nanstd(valid_x)
    return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0


def _z_ref(x: pd.Series, ref_x: pd.Series) -> pd.Series:
    x = x.astype(float)
    ref_x = ref_x.astype(float)
    m = np.nanmean(ref_x)
    sd = np.nanstd(ref_x)
    return (x - m) / sd if sd and not np.isnan(sd) else x * 0.0


def _normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Align recalib CSV names with production snapshot column names."""
    out = df.copy()
    if "Vol20(ann)" in out.columns and "Vol20_Ann" not in out.columns:
        out["Vol20_Ann"] = out["Vol20(ann)"]
    if "Vol20_Ann" in out.columns and "Vol20(ann)" not in out.columns:
        out["Vol20(ann)"] = out["Vol20_Ann"]
    return out


def score_fms_from_feature_frame(
    features: pd.DataFrame,
    *,
    reference_features: Optional[pd.DataFrame] = None,
    disqualified_symbols: Optional[set] = None,
) -> pd.Series:
    """Score FMS from a precomputed feature table (recalib / harness entrypoint).

    Implements the **production reference-panel formula** used when
    ``compute_fms_snapshot(..., reference_prices_krw=...)`` runs. When
    ``reference_features`` is omitted, the target frame is used as its own
    reference (same as ``reference_prices_krw=prices_krw`` in the app).

    Required columns (either ``Vol20_Ann`` or ``Vol20(ann)``):
    ``R_1M``, ``R_3M``, ``R_6M``, ``R2_3M``, ``AboveEMA50``, ``Vol20_*``,
    ``MaxDD_Pct``, ``R_10D``, ``R_5D``, ``EMA20_SLOPE_10D``, ``EMA20_CURV_20D``,
    ``UNDER_EMA20_DEPTH``, ``UNDER_EMA20_DAYS``, ``DOWN_STREAK_5D``.
    """
    feat = _normalize_feature_frame(features)
    ref = _normalize_feature_frame(reference_features) if reference_features is not None else feat

    required = [
        "R_1M",
        "R_3M",
        "R_6M",
        "R2_3M",
        "AboveEMA50",
        "Vol20_Ann",
        "MaxDD_Pct",
        "R_10D",
        "R_5D",
        "EMA20_SLOPE_10D",
        "EMA20_CURV_20D",
        "UNDER_EMA20_DEPTH",
        "UNDER_EMA20_DAYS",
        "DOWN_STREAK_5D",
    ]
    missing = [c for c in required if c not in feat.columns]
    if missing:
        raise KeyError(f"score_fms_from_feature_frame missing columns: {missing}")

    r_1m = feat["R_1M"].astype(float)
    r_3m = feat["R_3M"].astype(float)
    r_6m = feat["R_6M"].astype(float)
    r2_3m = feat["R2_3M"].astype(float)
    above_ema50 = feat["AboveEMA50"].astype(float)
    vol20 = feat["Vol20_Ann"].astype(float)
    max_dd = feat["MaxDD_Pct"].astype(float)
    r_10d = feat["R_10D"].astype(float)
    r_5d = feat["R_5D"].astype(float)
    ema20_slope = feat["EMA20_SLOPE_10D"].astype(float)
    ema20_curv = feat["EMA20_CURV_20D"].astype(float)
    under_depth = feat["UNDER_EMA20_DEPTH"].astype(float)
    under_days = feat["UNDER_EMA20_DAYS"].astype(float)
    down5 = feat["DOWN_STREAK_5D"].astype(float)

    ref_r_1m = ref["R_1M"].astype(float)
    ref_r_3m = ref["R_3M"].astype(float)
    ref_r_6m = ref["R_6M"].astype(float)
    ref_r2_3m = ref["R2_3M"].astype(float)
    ref_above = ref["AboveEMA50"].astype(float)
    ref_vol20 = ref["Vol20_Ann"].astype(float)
    ref_max_dd = ref["MaxDD_Pct"].astype(float)

    r2_gate = _smoothstep(r_3m, 0.05 - _P_GATE_R3_W, 0.05 + _P_GATE_R3_W) * _smoothstep(
        r_6m, 0.08 - _P_GATE_R6_W, 0.08 + _P_GATE_R6_W
    )
    ref_r2_gate = _smoothstep(ref_r_3m, 0.05 - _P_GATE_R3_W, 0.05 + _P_GATE_R3_W) * _smoothstep(
        ref_r_6m, 0.08 - _P_GATE_R6_W, 0.08 + _P_GATE_R6_W
    )
    r2_level = _smoothstep(r_3m, 0.05, _P_LEVEL_R3_HI) * _smoothstep(r_6m, 0.08, _P_LEVEL_R6_HI)
    ref_r2_level = _smoothstep(ref_r_3m, 0.05, _P_LEVEL_R3_HI) * _smoothstep(
        ref_r_6m, 0.08, _P_LEVEL_R6_HI
    )
    r2_strength = r2_gate * (_P_R2_FLOOR + (1.0 - _P_R2_FLOOR) * r2_level)
    ref_r2_strength = ref_r2_gate * (_P_R2_FLOOR + (1.0 - _P_R2_FLOOR) * ref_r2_level)

    r2_clip = r2_3m.clip(lower=0.0, upper=1.0)
    w_mid = _smoothstep(r2_clip, 0.70 - _P_R2_TRANSITION_W, 0.70 + _P_R2_TRANSITION_W)
    w_high = _smoothstep(r2_clip, 0.90 - _P_R2_TRANSITION_W, 0.90 + _P_R2_TRANSITION_W)
    r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high
    r2_eff_gated = pd.Series((r2_mult * r2_clip) * r2_strength, index=r2_3m.index)

    ref_r2_clip = ref_r2_3m.clip(lower=0.0, upper=1.0)
    ref_w_mid = _smoothstep(ref_r2_clip, 0.70 - _P_R2_TRANSITION_W, 0.70 + _P_R2_TRANSITION_W)
    ref_w_high = _smoothstep(ref_r2_clip, 0.90 - _P_R2_TRANSITION_W, 0.90 + _P_R2_TRANSITION_W)
    ref_r2_mult = 0.2 + 0.4 * ref_w_mid + 0.6 * ref_w_high
    ref_r2_eff_gated = pd.Series(
        (ref_r2_mult * ref_r2_clip) * ref_r2_strength, index=ref_r2_3m.index
    )

    r2_term = _z_ref(r2_eff_gated, ref_r2_eff_gated)

    dd_mag = (-max_dd).clip(lower=0.0)
    ref_dd_mag = (-ref_max_dd).clip(lower=0.0)
    dd_soft = dd_mag.clip(upper=30.0)
    dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
    dd_combined = dd_soft + dd_hard
    ref_dd_soft = ref_dd_mag.clip(upper=30.0)
    ref_dd_hard = ((ref_dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
    ref_dd_combined = ref_dd_soft + ref_dd_hard
    dd_penalty = _z_ref(dd_combined, ref_dd_combined)

    v = vol20.clip(lower=0.0)
    v_ref = ref_vol20.clip(lower=0.0)
    q_ref = np.nanpercentile(v_ref, 70) if not v_ref.dropna().empty else np.nan
    if np.isnan(q_ref):
        q_ref = np.nanpercentile(v, 70) if not v.dropna().empty else 0.0
    v_soft = v.clip(upper=q_ref)
    v_hard = (v - q_ref).clip(lower=0.0) ** 1.5
    v_combined = v_soft + v_hard
    v_ref_soft = v_ref.clip(upper=q_ref)
    v_ref_hard = (v_ref - q_ref).clip(lower=0.0) ** 1.5
    v_ref_combined = v_ref_soft + v_ref_hard
    vol_penalty = _z_ref(v_combined, v_ref_combined)

    r3_term = _z_ref(r_3m, ref_r_3m)
    r6_term = _z_ref(r_6m, ref_r_6m)
    ema_term = _z_ref(above_ema50, ref_above)

    quality_mask = (r2_3m > 0.85) & (r_3m > 0.3) & (r_6m > 0.5)
    r1_good = pd.Series(np.where(quality_mask, r_1m, 0.0), index=r_1m.index)
    r1_bad = pd.Series(np.where(~quality_mask & (r_1m > 0.3), r_1m, 0.0), index=r_1m.index)
    ref_quality = (ref_r2_3m > 0.85) & (ref_r_3m > 0.3) & (ref_r_6m > 0.5)
    ref_r1_good = pd.Series(np.where(ref_quality, ref_r_1m, 0.0), index=ref_r_1m.index)
    ref_r1_bad = pd.Series(
        np.where(~ref_quality & (ref_r_1m > 0.3), ref_r_1m, 0.0), index=ref_r_1m.index
    )
    r1_pos = _z_ref(r1_good, ref_r1_good)
    r1_neg = _z_ref(r1_bad, ref_r1_bad)

    slope_term = _z_peer(ema20_slope, disqualified_symbols)
    curv_penalty = _z_peer(ema20_curv.clip(lower=0.0), disqualified_symbols)
    curv_reward = _z_peer((-ema20_curv).clip(lower=0.0), disqualified_symbols)
    ema_shape_term = 0.7 * slope_term + 0.3 * curv_reward - 0.3 * curv_penalty

    recent_accel_term = _z_peer(r_10d + 0.5 * r_5d, disqualified_symbols)
    recent_break_raw = pd.Series(
        np.where(quality_mask & (r_10d < 0.0), -r_10d, 0.0),
        index=r_10d.index,
    )
    recent_break_term = _z_peer(recent_break_raw, disqualified_symbols)
    depth_term = _z_peer(under_depth, disqualified_symbols)
    days_term = _z_peer(under_days.astype(float), disqualified_symbols)
    down5_term = _z_peer(down5.astype(float), disqualified_symbols)

    pos = (
        _P_W_R3 * r3_term
        + _P_W_R6 * r6_term
        + _P_W_R2 * r2_term
        + _P_W_EMA * ema_term
        + 0.387801 * ema_shape_term
        + 0.183015 * recent_accel_term
        + 0.270777 * r1_pos
    )
    neg = (
        _P_W_DD * dd_penalty
        + _P_W_VOL * vol_penalty
        + 0.212139 * r1_neg
        + 0.228832 * recent_break_term
        + 0.186758 * down5_term
        + 0.19622 * depth_term
        + 0.097883 * days_term
    )
    return (pos - neg).rename("FMS")


def _mom_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                  ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    # 기본 수익률/지표
    r_1m = returns_pct(prices_krw, 21)
    r_3m = returns_pct(prices_krw, 63)
    r2_3m = r_squared_3m(prices_krw).rename('R2_3M')
    vol20 = last_vol_annualized(prices_krw, 20).rename('Vol20(ann)')

    # 단기/EMA 기반 파생 변수들 (재보정 피처와 동일 구조)
    r_10d = returns_pct(prices_krw, 10)
    r_5d = returns_pct(prices_krw, 5)

    above_ema50: Dict[str, float] = {}
    ema20_slope_10d: Dict[str, float] = {}
    ema20_curv_20d: Dict[str, float] = {}
    under_ema20_depth: Dict[str, float] = {}
    under_ema20_days: Dict[str, int] = {}
    down_streak_5d: Dict[str, float] = {}

    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            above_ema50[c] = np.nan
            ema20_slope_10d[c] = np.nan
            ema20_curv_20d[c] = np.nan
            under_ema20_depth[c] = np.nan
            under_ema20_days[c] = np.nan
            down_streak_5d[c] = np.nan
            continue

        e50 = ema(s, 50)
        above_ema50[c] = (s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan

        e20 = ema(s, 20)

        # EMA20 기울기 (최근 10일, 로그 회귀; 0/음수는 NaN 처리해 log 경고 방지)
        if len(e20) >= 10:
            last10 = e20.iloc[-10:]
            x10 = np.arange(len(last10), dtype=float)
            y10 = np.log(last10.where(last10 > 0)).dropna()
            if len(y10) == len(x10):
                ema20_slope_10d[c] = float(np.polyfit(x10, y10, 1)[0])
            else:
                ema20_slope_10d[c] = np.nan
        else:
            ema20_slope_10d[c] = np.nan

        # EMA20 곡률 (최근 20일 앞/뒤 10일 기울기 차이)
        if len(e20) >= 20:
            first10 = e20.iloc[-20:-10]
            last10 = e20.iloc[-10:]
            x_seg = np.arange(10, dtype=float)

            def _slope(seg: pd.Series) -> float:
                y = np.log(seg.where(seg > 0)).dropna()
                if len(y) != len(x_seg):
                    return np.nan
                return float(np.polyfit(x_seg, y, 1)[0])

            s_first = _slope(first10)
            s_last = _slope(last10)
            if np.isnan(s_first) or np.isnan(s_last):
                ema20_curv_20d[c] = np.nan
            else:
                ema20_curv_20d[c] = s_last - s_first
        else:
            ema20_curv_20d[c] = np.nan

        # 최근 60일 EMA20 아래 이탈 깊이/일수
        tail60 = s.iloc[-60:] if len(s) >= 60 else s
        e20_60 = e20.reindex(tail60.index)
        mask_under = tail60 < e20_60
        if not mask_under.any():
            under_ema20_depth[c] = 0.0
            under_ema20_days[c] = 0
        else:
            rel = tail60[mask_under] / e20_60[mask_under] - 1.0
            under_ema20_depth[c] = float(rel.min())
            under_ema20_days[c] = int(mask_under.sum())

        # 최근 5일 연속 하락 최대 길이
        if len(s) >= 5:
            last5 = s.iloc[-5:]
            diff = last5.diff()
            is_down = diff < 0
            max_run = 0
            cur = 0
            for v in is_down.iloc[1:]:
                if bool(v):
                    cur += 1
                    max_run = max(max_run, cur)
                else:
                    cur = 0
            down_streak_5d[c] = int(max_run)
        else:
            down_streak_5d[c] = np.nan

    above_ema50_ser = pd.Series(above_ema50, name='AboveEMA50')
    ema20_slope_10d_ser = pd.Series(ema20_slope_10d, name='EMA20_SLOPE_10D')
    ema20_curv_20d_ser = pd.Series(ema20_curv_20d, name='EMA20_CURV_20D')
    under_ema20_depth_ser = pd.Series(under_ema20_depth, name='UNDER_EMA20_DEPTH')
    under_ema20_days_ser = pd.Series(under_ema20_days, name='UNDER_EMA20_DAYS')
    down_streak_5d_ser = pd.Series(down_streak_5d, name='DOWN_STREAK_5D')

    # 최대 드로우다운(%)
    mdict: Dict[str, float] = {}
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            mdict[c] = np.nan
            continue
        roll_max = s.cummax()
        dd = (s / roll_max - 1.0) * 100.0
        mdict[c] = float(dd.min())
    max_dd = pd.Series(mdict, name='MaxDD_Pct')

    # 거래 적합성 필터 먼저 확인
    disqualification_flags: Dict[str, bool] = {}
    filter_reasons: Dict[str, str] = {}
    if ohlc_data is not None and symbols is not None:
        disqualification_flags, filter_reasons = calculate_tradeability_filters(ohlc_data, symbols)
    
    disqualified_symbols = set()
    if disqualification_flags:
        disqualified_symbols = {
            sym for sym, is_disq in disqualification_flags.items()
            if is_disq and sym in prices_krw.columns
        }

    # Scoring helpers / weights: module-level ``_z_peer`` / ``_smoothstep`` / ``_P_*``
    # (shared with ``score_fms_from_feature_frame`` — do not reintroduce local forks).

    if reference_prices_krw is not None:
        # 참조 데이터 기반 분포
        ref_r_1m = returns_pct(reference_prices_krw, 21)
        ref_r_3m = returns_pct(reference_prices_krw, 63)
        ref_r_6m = returns_pct(reference_prices_krw, 126)
        ref_r2_3m = r_squared_3m(reference_prices_krw).rename('R2_3M')
        ref_above_ema50 = {}
        for c in reference_prices_krw.columns:
            s = reference_prices_krw[c].dropna()
            if s.empty:
                ref_above_ema50[c] = np.nan
                continue
            e50 = ema(s, 50)
            ref_above_ema50[c] = (s.iloc[-1] / e50.iloc[-1] - 1.0) if e50.iloc[-1] > 0 else np.nan
        ref_above_ema50 = pd.Series(ref_above_ema50, name='AboveEMA50')
        ref_vol20 = last_vol_annualized(reference_prices_krw, 20).rename('Vol20(ann)')

        # 참조용 MaxDD
        ref_md: Dict[str, float] = {}
        for c in reference_prices_krw.columns:
            s = reference_prices_krw[c].dropna()
            if s.empty:
                ref_md[c] = np.nan
                continue
            roll_max = s.cummax()
            dd = (s / roll_max - 1.0) * 100.0
            ref_md[c] = float(dd.min())
        ref_max_dd = pd.Series(ref_md, name='MaxDD_Pct')

        # R2 비선형 가중 + 추세상승 게이트 (평평한 그래프 억제)
        r_6m = returns_pct(prices_krw, 126)
        r2_gate = _smoothstep(r_3m, 0.05 - _P_GATE_R3_W, 0.05 + _P_GATE_R3_W) * _smoothstep(r_6m, 0.08 - _P_GATE_R6_W, 0.08 + _P_GATE_R6_W)
        ref_r2_gate = _smoothstep(ref_r_3m, 0.05 - _P_GATE_R3_W, 0.05 + _P_GATE_R3_W) * _smoothstep(ref_r_6m, 0.08 - _P_GATE_R6_W, 0.08 + _P_GATE_R6_W)
        r2_level = _smoothstep(r_3m, 0.05, _P_LEVEL_R3_HI) * _smoothstep(r_6m, 0.08, _P_LEVEL_R6_HI)
        ref_r2_level = _smoothstep(ref_r_3m, 0.05, _P_LEVEL_R3_HI) * _smoothstep(ref_r_6m, 0.08, _P_LEVEL_R6_HI)
        r2_strength = r2_gate * (_P_R2_FLOOR + (1.0 - _P_R2_FLOOR) * r2_level)
        ref_r2_strength = ref_r2_gate * (_P_R2_FLOOR + (1.0 - _P_R2_FLOOR) * ref_r2_level)

        r2_clip = r2_3m.clip(lower=0.0, upper=1.0)
        w_mid = _smoothstep(r2_clip, 0.70 - _P_R2_TRANSITION_W, 0.70 + _P_R2_TRANSITION_W)
        w_high = _smoothstep(r2_clip, 0.90 - _P_R2_TRANSITION_W, 0.90 + _P_R2_TRANSITION_W)
        r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high  # 0.2 -> 0.6 -> 1.2
        r2_eff_gated = pd.Series((r2_mult * r2_clip) * r2_strength, index=r2_3m.index)

        ref_r2_clip = ref_r2_3m.clip(lower=0.0, upper=1.0)
        ref_w_mid = _smoothstep(ref_r2_clip, 0.70 - _P_R2_TRANSITION_W, 0.70 + _P_R2_TRANSITION_W)
        ref_w_high = _smoothstep(ref_r2_clip, 0.90 - _P_R2_TRANSITION_W, 0.90 + _P_R2_TRANSITION_W)
        ref_r2_mult = 0.2 + 0.4 * ref_w_mid + 0.6 * ref_w_high
        ref_r2_eff_gated = pd.Series((ref_r2_mult * ref_r2_clip) * ref_r2_strength, index=ref_r2_3m.index)

        r2_term = _z_ref(r2_eff_gated, ref_r2_eff_gated)

        # MaxDD 패널티 (참조 분포 기준)
        dd_mag = (-max_dd).clip(lower=0.0)
        ref_dd_mag = (-ref_max_dd).clip(lower=0.0)
        dd_soft = dd_mag.clip(upper=30.0)
        dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
        dd_combined = dd_soft + dd_hard
        ref_dd_soft = ref_dd_mag.clip(upper=30.0)
        ref_dd_hard = ((ref_dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
        ref_dd_combined = ref_dd_soft + ref_dd_hard
        dd_penalty = _z_ref(dd_combined, ref_dd_combined)

        # Vol20 패널티 (참조 분포 기준)
        v = vol20.clip(lower=0.0)
        v_ref = ref_vol20.clip(lower=0.0)
        q_ref = np.nanpercentile(v_ref, 70) if not v_ref.dropna().empty else np.nan
        if np.isnan(q_ref):
            q_ref = np.nanpercentile(v, 70) if not v.dropna().empty else 0.0
        v_soft = v.clip(upper=q_ref)
        v_hard = (v - q_ref).clip(lower=0.0) ** 1.5
        v_combined = v_soft + v_hard
        v_ref_soft = v_ref.clip(upper=q_ref)
        v_ref_hard = (v_ref - q_ref).clip(lower=0.0) ** 1.5
        v_ref_combined = v_ref_soft + v_ref_hard
        vol_penalty = _z_ref(v_combined, v_ref_combined)

        # 주요 양의 축들
        r3_term = _z_ref(r_3m, ref_r_3m)
        r6_term = _z_ref(returns_pct(prices_krw, 126), ref_r_6m)
        ema_term = _z_ref(above_ema50_ser, ref_above_ema50)

        # R1 조건부 처리
        quality_mask = (r2_3m > 0.85) & (r_3m > 0.3) & (returns_pct(prices_krw, 126) > 0.5)
        r1_good = pd.Series(np.where(quality_mask, r_1m, 0.0), index=r_1m.index)
        r1_bad = pd.Series(np.where(~quality_mask & (r_1m > 0.3), r_1m, 0.0), index=r_1m.index)
        ref_quality = (ref_r2_3m > 0.85) & (ref_r_3m > 0.3) & (ref_r_6m > 0.5)
        ref_r1_good = pd.Series(np.where(ref_quality, ref_r_1m, 0.0), index=ref_r_1m.index)
        ref_r1_bad = pd.Series(np.where(~ref_quality & (ref_r_1m > 0.3), ref_r_1m, 0.0), index=ref_r_1m.index)
        r1_pos = _z_ref(r1_good, ref_r1_good)
        r1_neg = _z_ref(r1_bad, ref_r1_bad)

        # EMA20 shape + 단기/이탈 변수는 참조 분포 없이 현재 집합 기준으로만 정규화
        slope_term = _z_peer(ema20_slope_10d_ser, disqualified_symbols)
        curv_penalty_raw = ema20_curv_20d_ser.clip(lower=0.0)
        curv_reward_raw = (-ema20_curv_20d_ser).clip(lower=0.0)
        curv_penalty = _z_peer(curv_penalty_raw, disqualified_symbols)
        curv_reward = _z_peer(curv_reward_raw, disqualified_symbols)
        ema_shape_term = 0.7 * slope_term + 0.3 * curv_reward - 0.3 * curv_penalty

        recent_accel_term = _z_peer(r_10d + 0.5 * r_5d, disqualified_symbols)
        recent_break_raw = pd.Series(
            np.where((r2_3m > 0.85) & (r_3m > 0.3) & (returns_pct(prices_krw, 126) > 0.5) & (r_10d < 0.0), -r_10d, 0.0),
            index=r_10d.index,
        )
        recent_break_term = _z_peer(recent_break_raw, disqualified_symbols)
        depth_term = _z_peer(under_ema20_depth_ser, disqualified_symbols)
        days_term = _z_peer(under_ema20_days_ser.astype(float), disqualified_symbols)
        down5_term = _z_peer(down_streak_5d_ser.astype(float), disqualified_symbols)

        Pos = (
            _P_W_R3 * r3_term
            + _P_W_R6 * r6_term
            + _P_W_R2 * r2_term
            + _P_W_EMA * ema_term
            + 0.387801 * ema_shape_term
            + 0.183015 * recent_accel_term
            + 0.270777 * r1_pos
        )
        Neg = (
            _P_W_DD * dd_penalty
            + _P_W_VOL * vol_penalty
            + 0.212139 * r1_neg
            + 0.228832 * recent_break_term
            + 0.186758 * down5_term
            + 0.19622 * depth_term
            + 0.097883 * days_term
        )
        FMS = Pos - Neg

    else:
        # 참조 데이터가 없을 때: 현재 집합 분포 기준
        r_6m = returns_pct(prices_krw, 126)
        r2_gate = _smoothstep(r_3m, 0.05 - _P_GATE_R3_W, 0.05 + _P_GATE_R3_W) * _smoothstep(r_6m, 0.08 - _P_GATE_R6_W, 0.08 + _P_GATE_R6_W)
        r2_level = _smoothstep(r_3m, 0.05, _P_LEVEL_R3_HI) * _smoothstep(r_6m, 0.08, _P_LEVEL_R6_HI)
        r2_strength = r2_gate * (_P_R2_FLOOR + (1.0 - _P_R2_FLOOR) * r2_level)
        r2_clip = r2_3m.clip(lower=0.0, upper=1.0)
        w_mid = _smoothstep(r2_clip, 0.70 - _P_R2_TRANSITION_W, 0.70 + _P_R2_TRANSITION_W)
        w_high = _smoothstep(r2_clip, 0.90 - _P_R2_TRANSITION_W, 0.90 + _P_R2_TRANSITION_W)
        r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high
        r2_eff_gated = pd.Series((r2_mult * r2_clip) * r2_strength, index=r2_3m.index)
        r2_term = _z_peer(r2_eff_gated, disqualified_symbols)

        dd_mag = (-max_dd).clip(lower=0.0)
        dd_soft = dd_mag.clip(upper=30.0)
        dd_hard = ((dd_mag - 30.0).clip(lower=0.0) ** 2) / (70.0 ** 2) * 70.0
        dd_combined = dd_soft + dd_hard
        dd_penalty = _z_peer(dd_combined, disqualified_symbols)

        v = vol20.clip(lower=0.0)
        q = np.nanpercentile(v, 70) if not v.dropna().empty else 0.0
        v_soft = v.clip(upper=q)
        v_hard = (v - q).clip(lower=0.0) ** 1.5
        v_combined = v_soft + v_hard
        vol_penalty = _z_peer(v_combined, disqualified_symbols)

        r6_full = returns_pct(prices_krw, 126)
        r3_term = _z_peer(r_3m, disqualified_symbols)
        r6_term = _z_peer(r6_full, disqualified_symbols)
        ema_term = _z_peer(above_ema50_ser, disqualified_symbols)

        quality_mask = (r2_3m > 0.85) & (r_3m > 0.3) & (r6_full > 0.5)
        r1_good = pd.Series(np.where(quality_mask, r_1m, 0.0), index=r_1m.index)
        r1_bad = pd.Series(np.where(~quality_mask & (r_1m > 0.3), r_1m, 0.0), index=r_1m.index)
        r1_pos = _z_peer(r1_good, disqualified_symbols)
        r1_neg = _z_peer(r1_bad, disqualified_symbols)

        slope_term = _z_peer(ema20_slope_10d_ser, disqualified_symbols)
        curv_penalty_raw = ema20_curv_20d_ser.clip(lower=0.0)
        curv_reward_raw = (-ema20_curv_20d_ser).clip(lower=0.0)
        curv_penalty = _z_peer(curv_penalty_raw, disqualified_symbols)
        curv_reward = _z_peer(curv_reward_raw, disqualified_symbols)
        ema_shape_term = 0.7 * slope_term + 0.3 * curv_reward - 0.3 * curv_penalty

        recent_accel_term = _z_peer(r_10d + 0.5 * r_5d, disqualified_symbols)
        recent_break_raw = pd.Series(
            np.where(quality_mask & (r_10d < 0.0), -r_10d, 0.0),
            index=r_10d.index,
        )
        recent_break_term = _z_peer(recent_break_raw, disqualified_symbols)
        depth_term = _z_peer(under_ema20_depth_ser, disqualified_symbols)
        days_term = _z_peer(under_ema20_days_ser.astype(float), disqualified_symbols)
        down5_term = _z_peer(down_streak_5d_ser.astype(float), disqualified_symbols)

        Pos = (
            0.519348 * r3_term
            + 0.430148 * r6_term
            + 0.519626 * r2_term
            + 0.398466 * ema_term
            + 0.387801 * ema_shape_term
            + 0.183015 * recent_accel_term
            + 0.270777 * r1_pos
        )
        Neg = (
            0.265056 * dd_penalty
            + 0.218807 * vol_penalty
            + 0.212139 * r1_neg
            + 0.228832 * recent_break_term
            + 0.186758 * down5_term
            + 0.19622 * depth_term
            + 0.097883 * days_term
        )
        FMS = Pos - Neg

    # 실격 종목은 FMS = -999 적용
    if disqualification_flags:
        for symbol in FMS.index:
            if symbol in disqualification_flags and disqualification_flags[symbol]:
                FMS[symbol] = -999.0

    filter_reasons_series = pd.Series(filter_reasons, name='Filter_Status').reindex(FMS.index, fill_value='정상')
    snap = pd.concat(
        [
            r_1m.rename('R_1M'),
            r_3m.rename('R_3M'),
            r2_3m,
            above_ema50_ser,
            vol20,
            max_dd,
            r_10d.rename('R_10D'),
            r_5d.rename('R_5D'),
            ema20_slope_10d_ser,
            ema20_curv_20d_ser,
            under_ema20_depth_ser,
            under_ema20_days_ser,
            down_streak_5d_ser,
            FMS.rename('FMS'),
            filter_reasons_series,
        ],
        axis=1,
    )
    return snap


def compute_fms_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                         ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Public FMS snapshot entrypoint for harness / calibration (no network I/O).

    Thin public wrapper around ``_mom_snapshot``. Accepts pre-loaded KRW price
    (and optional OHLC / reference) DataFrames so tests can inject fixtures
    without calling yfinance.
    """
    return _mom_snapshot(prices_krw, reference_prices_krw, ohlc_data, symbols)


def momentum_now_and_delta(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                           ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute FMS plus 1D/5D deltas from injected KRW price panels (no network I/O)."""
    now = compute_fms_snapshot(prices_krw, reference_prices_krw, ohlc_data, symbols)
    d1 = _mom_snapshot(prices_krw.iloc[:-1], reference_prices_krw, ohlc_data, symbols) if len(prices_krw) > 1 else now * np.nan
    d5 = _mom_snapshot(prices_krw.iloc[:-5], reference_prices_krw, ohlc_data, symbols) if len(prices_krw) > 5 else now * np.nan
    df = now.copy()
    df['ΔFMS_1D'] = df['FMS'] - d1['FMS']
    df['ΔFMS_5D'] = df['FMS'] - d5['FMS']
    df['R_1W'] = returns_pct(prices_krw, 5)
    df['R_6M'] = returns_pct(prices_krw, 126)
    df['R_YTD'] = ytd_return(prices_krw)
    return df.sort_values('FMS', ascending=False)

