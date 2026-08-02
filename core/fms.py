# -*- coding: utf-8 -*-
"""FMS snapshot / momentum scoring — pure logic, no network I/O.

Migrated from ``analysis_utils`` (``_mom_snapshot``, ``compute_fms_snapshot``,
``momentum_now_and_delta``). The transitional facade re-exports the public
entrypoints; prefer importing from ``core.fms`` in new code.

See HARNESS_RULES.md §2.5 (single source of truth).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from core.fms_features import (
    PRODUCTION_FMS_COLUMNS,
    build_panel_feature_frame,
    score_production_fms_features,
)
from core.indicators import (
    mask_non_positive_prices,
    returns_pct,
    ytd_return,
)
from core.tradeability import calculate_tradeability_filters


# Calendar-aligned trading-day horizons (4M=84; legacy 6M=126 for mapping helpers).
HORIZON_DAYS_4M = 84
HORIZON_DAYS_LEGACY_6M = 126

# Pre-4M gate/quality centers (used only to derive production 4M hurdles).
_LEGACY_R6_GATE_CENTER = 0.08
_LEGACY_R6_QUALITY_MIN = 0.50
_LEGACY_GATE_R6_W = 0.006355
_LEGACY_LEVEL_R6_HI = 0.430268


def horizon_return_map(r: float, from_days: int, to_days: int) -> float:
    """Map a simple return across horizons under constant compounding.

    ``(1 + r_to) = (1 + r_from) ** (to_days / from_days)``.
    """
    if from_days <= 0 or to_days <= 0:
        raise ValueError("from_days and to_days must be positive")
    return float((1.0 + float(r)) ** (float(to_days) / float(from_days)) - 1.0)


def gate_width_scale(width: float, from_days: int, to_days: int) -> float:
    """Scale a smoothstep half-width with cumulative-return noise (∝ √T)."""
    if from_days <= 0 or to_days <= 0:
        raise ValueError("from_days and to_days must be positive")
    return float(float(width) * (float(to_days) / float(from_days)) ** 0.5)


# Production R_4M raw-return hurdles (compound / √t mapped from legacy 6M).
R_4M_GATE_CENTER = horizon_return_map(
    _LEGACY_R6_GATE_CENTER, HORIZON_DAYS_LEGACY_6M, HORIZON_DAYS_4M
)
R_4M_QUALITY_MIN = horizon_return_map(
    _LEGACY_R6_QUALITY_MIN, HORIZON_DAYS_LEGACY_6M, HORIZON_DAYS_4M
)
R_3M_GATE_CENTER = 0.05  # unchanged (3M axis)

# Soft R² gate for conditional R_1M (was a hard cut at 0.85).
R2_QUALITY_CENTER = 0.80
# Treat quality weight below this as "non-quality" for binary r1_bad / break.
R1_QUALITY_HARD_FLOOR = 0.5


@dataclass(frozen=True)
class FmsScoreParams:
    """Archived pre-v4.6 FMS weights / transition widths (legacy tune path only).

    Production v4.6 scoring does **not** use these params. Offline legacy Monte-Carlo
    scripts call ``score_legacy_fms_from_feature_frame(..., params=...)``.
    """

    w_r3: float
    w_r4: float
    w_r2: float
    w_ema: float
    w_ema_shape: float
    w_recent: float
    w_r1_pos: float
    w_dd: float
    w_vol: float
    w_r1_neg: float
    w_break: float
    w_down5: float
    w_under_depth: float
    w_under_days: float
    r2_transition_w: float
    gate_r3_w: float
    gate_r4_w: float
    level_r3_hi: float
    level_r4_hi: float
    r2_floor: float
    w_ema_slope_base: float
    w_ema_curv_reward_base: float
    w_ema_curv_penalty_base: float
    vol_q_pct: float
    vol_hard_power: float
    vol_hard_scale: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "FmsScoreParams":
        """Build params from a dict, filling missing keys from production defaults."""
        known = {f.name for f in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise TypeError(f"unknown FmsScoreParams keys: {sorted(unknown)}")
        base = asdict(production_fms_score_params())
        base.update({k: float(v) for k, v in mapping.items()})
        return cls(**base)


def production_fms_score_params() -> FmsScoreParams:
    """Archived pre-v4.6 weights / transition widths (legacy tune path only).

    Production scoring uses ``score_production_fms_features`` /
    ``score_fms_from_feature_frame`` with the v5.0.0 ``alive_pullback`` formula
    in ``core/fms_features.py``. These params remain for
    ``score_legacy_fms_from_feature_frame`` and offline tune scripts.
    """
    return FmsScoreParams(
        w_r3=0.46869,
        w_r4=0.417409,
        w_r2=0.505669,
        w_ema=0.323264,
        w_ema_shape=0.445971,
        w_recent=0.228769,
        w_r1_pos=0.270777,
        w_dd=0.28298,
        w_vol=0.291973,
        w_r1_neg=0.212139,
        w_break=0.228832,
        w_down5=0.186758,
        w_under_depth=0.19622,
        w_under_days=0.097883,
        r2_transition_w=0.04552,
        gate_r3_w=0.019359,
        gate_r4_w=gate_width_scale(
            _LEGACY_GATE_R6_W, HORIZON_DAYS_LEGACY_6M, HORIZON_DAYS_4M
        ),
        level_r3_hi=0.205305,
        level_r4_hi=horizon_return_map(
            _LEGACY_LEVEL_R6_HI, HORIZON_DAYS_LEGACY_6M, HORIZON_DAYS_4M
        ),
        r2_floor=0.734629,
        w_ema_slope_base=0.7,
        w_ema_curv_reward_base=0.3,
        w_ema_curv_penalty_base=0.3,
        vol_q_pct=70.0,
        vol_hard_power=1.5,
        vol_hard_scale=1.0,
    )


def _resolve_fms_score_params(
    params: Optional[Union[FmsScoreParams, Mapping[str, float]]] = None,
) -> FmsScoreParams:
    if params is None:
        return production_fms_score_params()
    if isinstance(params, FmsScoreParams):
        return params
    return FmsScoreParams.from_mapping(params)


def _smoothstep(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
    """0..1로 부드럽게 전이 (C1 연속)."""
    if edge1 == edge0:
        return pd.Series(0.0, index=x.index)
    t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
    return t * t * (3.0 - 2.0 * t)


def _r1_quality_weight(
    r2_3m: pd.Series,
    r_3m: pd.Series,
    r_4m: pd.Series,
    params: FmsScoreParams,
) -> pd.Series:
    """Continuous quality weight for conditional R_1M (soft R² around 0.80)."""
    r2_clip = r2_3m.astype(float).clip(lower=0.0, upper=1.0)
    r2_soft = _smoothstep(
        r2_clip,
        R2_QUALITY_CENTER - params.r2_transition_w,
        R2_QUALITY_CENTER + params.r2_transition_w,
    )
    ret_ok = ((r_3m.astype(float) > 0.3) & (r_4m.astype(float) > R_4M_QUALITY_MIN)).astype(float)
    return pd.Series(r2_soft * ret_ok, index=r2_3m.index, dtype=float)


def _recent_continuation_mask(r_10d: pd.Series, ema20_slope: pd.Series) -> pd.Series:
    """True when short-term price/EMA structure confirms ongoing uptrend."""
    return (r_10d.astype(float) > 0.0) & (ema20_slope.astype(float) > 0.0)


def _r1_conditional_series(
    r_1m: pd.Series,
    r_3m: pd.Series,
    r_4m: pd.Series,
    r2_3m: pd.Series,
    r_10d: pd.Series,
    ema20_slope: pd.Series,
    params: FmsScoreParams,
) -> tuple[pd.Series, pd.Series]:
    """Return ``(r1_good, r1_bad)`` with soft quality + continuation exemption.

    - ``r1_good``: R_1M scaled by continuous quality weight (healthy accel credit).
    - ``r1_bad``: event-spike path — high R_1M without quality, unless recent
      continuation (R_10D>0 and EMA20 slope>0) confirms the move is not a spike.
    """
    qw = _r1_quality_weight(r2_3m, r_3m, r_4m, params)
    r1 = r_1m.astype(float)
    r1_good = pd.Series(r1 * qw, index=r_1m.index, dtype=float)
    low_q = qw < R1_QUALITY_HARD_FLOOR
    continuation = _recent_continuation_mask(r_10d, ema20_slope)
    r1_bad = pd.Series(
        np.where(low_q & (r1 > 0.3) & ~continuation, r1, 0.0),
        index=r_1m.index,
        dtype=float,
    )
    return r1_good, r1_bad


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
    # Legacy recalib CSVs may still label the long-horizon return R_6M.
    if "R_6M" in out.columns and "R_4M" not in out.columns:
        out["R_4M"] = out["R_6M"]
    return out


def score_legacy_fms_from_feature_frame(
    features: pd.DataFrame,
    *,
    reference_features: Optional[pd.DataFrame] = None,
    disqualified_symbols: Optional[set] = None,
    params: Optional[Union[FmsScoreParams, Mapping[str, float]]] = None,
) -> pd.Series:
    """Score the pre-v4.6 FMS from a feature table for archived tuning.

    Implements the legacy reference-panel formula. When
    ``reference_features`` is omitted, the target frame is used as its own
    reference (same as ``reference_prices_krw=prices_krw`` in the app).

    ``params`` overrides weights / transition widths for offline search. Omit
    (or pass ``production_fms_score_params()``) for the archived formula.

    Required columns (either ``Vol20_Ann`` or ``Vol20(ann)``):
    ``R_1M``, ``R_3M``, ``R_4M``, ``R2_3M``, ``AboveEMA50``, ``Vol20_*``,
    ``MaxDD_Pct``, ``R_10D``, ``R_5D``, ``EMA20_SLOPE_10D``, ``EMA20_CURV_20D``,
    ``UNDER_EMA20_DEPTH``, ``UNDER_EMA20_DAYS``, ``DOWN_STREAK_5D``.
    """
    p = _resolve_fms_score_params(params)
    feat = _normalize_feature_frame(features)
    ref = _normalize_feature_frame(reference_features) if reference_features is not None else feat

    required = [
        "R_1M",
        "R_3M",
        "R_4M",
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
        raise KeyError(f"score_legacy_fms_from_feature_frame missing columns: {missing}")

    r_1m = feat["R_1M"].astype(float)
    r_3m = feat["R_3M"].astype(float)
    r_4m = feat["R_4M"].astype(float)
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
    ref_r_4m = ref["R_4M"].astype(float)
    ref_r2_3m = ref["R2_3M"].astype(float)
    ref_above = ref["AboveEMA50"].astype(float)
    ref_vol20 = ref["Vol20_Ann"].astype(float)
    ref_max_dd = ref["MaxDD_Pct"].astype(float)

    r2_gate = _smoothstep(r_3m, R_3M_GATE_CENTER - p.gate_r3_w, R_3M_GATE_CENTER + p.gate_r3_w) * _smoothstep(
        r_4m, R_4M_GATE_CENTER - p.gate_r4_w, R_4M_GATE_CENTER + p.gate_r4_w
    )
    ref_r2_gate = _smoothstep(ref_r_3m, R_3M_GATE_CENTER - p.gate_r3_w, R_3M_GATE_CENTER + p.gate_r3_w) * _smoothstep(
        ref_r_4m, R_4M_GATE_CENTER - p.gate_r4_w, R_4M_GATE_CENTER + p.gate_r4_w
    )
    r2_level = _smoothstep(r_3m, R_3M_GATE_CENTER, p.level_r3_hi) * _smoothstep(r_4m, R_4M_GATE_CENTER, p.level_r4_hi)
    ref_r2_level = _smoothstep(ref_r_3m, R_3M_GATE_CENTER, p.level_r3_hi) * _smoothstep(
        ref_r_4m, R_4M_GATE_CENTER, p.level_r4_hi
    )
    r2_strength = r2_gate * (p.r2_floor + (1.0 - p.r2_floor) * r2_level)
    ref_r2_strength = ref_r2_gate * (p.r2_floor + (1.0 - p.r2_floor) * ref_r2_level)

    r2_clip = r2_3m.clip(lower=0.0, upper=1.0)
    w_mid = _smoothstep(r2_clip, 0.70 - p.r2_transition_w, 0.70 + p.r2_transition_w)
    w_high = _smoothstep(r2_clip, 0.90 - p.r2_transition_w, 0.90 + p.r2_transition_w)
    r2_mult = 0.2 + 0.4 * w_mid + 0.6 * w_high
    r2_eff_gated = pd.Series((r2_mult * r2_clip) * r2_strength, index=r2_3m.index)

    ref_r2_clip = ref_r2_3m.clip(lower=0.0, upper=1.0)
    ref_w_mid = _smoothstep(ref_r2_clip, 0.70 - p.r2_transition_w, 0.70 + p.r2_transition_w)
    ref_w_high = _smoothstep(ref_r2_clip, 0.90 - p.r2_transition_w, 0.90 + p.r2_transition_w)
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
    q_ref = (
        np.nanpercentile(v_ref, p.vol_q_pct) if not v_ref.dropna().empty else np.nan
    )
    if np.isnan(q_ref):
        q_ref = np.nanpercentile(v, p.vol_q_pct) if not v.dropna().empty else 0.0
    v_soft = v.clip(upper=q_ref)
    v_hard = p.vol_hard_scale * ((v - q_ref).clip(lower=0.0) ** p.vol_hard_power)
    v_combined = v_soft + v_hard
    v_ref_soft = v_ref.clip(upper=q_ref)
    v_ref_hard = p.vol_hard_scale * ((v_ref - q_ref).clip(lower=0.0) ** p.vol_hard_power)
    v_ref_combined = v_ref_soft + v_ref_hard
    vol_penalty = _z_ref(v_combined, v_ref_combined)

    r3_term = _z_ref(r_3m, ref_r_3m)
    r4_term = _z_ref(r_4m, ref_r_4m)
    ema_term = _z_ref(above_ema50, ref_above)

    r1_good, r1_bad = _r1_conditional_series(
        r_1m, r_3m, r_4m, r2_3m, r_10d, ema20_slope, p
    )
    ref_r1_good, ref_r1_bad = _r1_conditional_series(
        ref_r_1m,
        ref_r_3m,
        ref_r_4m,
        ref_r2_3m,
        ref["R_10D"].astype(float),
        ref["EMA20_SLOPE_10D"].astype(float),
        p,
    )
    r1_pos = _z_ref(r1_good, ref_r1_good)
    r1_neg = _z_ref(r1_bad, ref_r1_bad)

    slope_term = _z_peer(ema20_slope, disqualified_symbols)
    curv_penalty = _z_peer(ema20_curv.clip(lower=0.0), disqualified_symbols)
    curv_reward = _z_peer((-ema20_curv).clip(lower=0.0), disqualified_symbols)
    ema_shape_term = (
        p.w_ema_slope_base * slope_term
        + p.w_ema_curv_reward_base * curv_reward
        - p.w_ema_curv_penalty_base * curv_penalty
    )

    recent_accel_term = _z_peer(r_10d + 0.5 * r_5d, disqualified_symbols)
    quality_w = _r1_quality_weight(r2_3m, r_3m, r_4m, p)
    recent_break_raw = pd.Series(
        np.where((quality_w >= R1_QUALITY_HARD_FLOOR) & (r_10d < 0.0), -r_10d, 0.0),
        index=r_10d.index,
    )
    recent_break_term = _z_peer(recent_break_raw, disqualified_symbols)
    depth_term = _z_peer(under_depth, disqualified_symbols)
    days_term = _z_peer(under_days.astype(float), disqualified_symbols)
    down5_term = _z_peer(down5.astype(float), disqualified_symbols)

    pos = (
        p.w_r3 * r3_term
        + p.w_r4 * r4_term
        + p.w_r2 * r2_term
        + p.w_ema * ema_term
        + p.w_ema_shape * ema_shape_term
        + p.w_recent * recent_accel_term
        + p.w_r1_pos * r1_pos
    )
    neg = (
        p.w_dd * dd_penalty
        + p.w_vol * vol_penalty
        + p.w_r1_neg * r1_neg
        + p.w_break * recent_break_term
        + p.w_down5 * down5_term
        + p.w_under_depth * depth_term
        + p.w_under_days * days_term
    )
    return (pos - neg).rename("FMS")


def score_fms_from_feature_frame(
    features: pd.DataFrame,
    *,
    reference_features: Optional[pd.DataFrame] = None,
    disqualified_symbols: Optional[set] = None,
) -> pd.Series:
    """Score production FMS (v5.0.0 alive_pullback absolute nonlinear).

    ``reference_features`` is accepted for API compatibility with batch/app
    call sites but is unused by the absolute scorer. Pass disqualified symbols
    to force tradeability ``FMS = -999``.
    """
    return score_production_fms_features(
        features,
        reference_features=reference_features,
        disqualified_symbols=disqualified_symbols,
    )


def _mom_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                  ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    # Yahoo Adj Close can contain long negative stretches (KR ETF glitches).
    # Mask before feature extraction so EMA/return features cannot explode.
    prices_krw = mask_non_positive_prices(prices_krw)
    if reference_prices_krw is not None:
        reference_prices_krw = mask_non_positive_prices(reference_prices_krw)
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

    feature_symbols = symbols if symbols is not None else list(prices_krw.columns)
    production_features = build_panel_feature_frame(
        prices_krw, symbols=feature_symbols
    )
    if reference_prices_krw is None:
        reference_features = production_features
    else:
        reference_features = build_panel_feature_frame(
            reference_prices_krw,
            symbols=list(reference_prices_krw.columns),
        )
    FMS = score_fms_from_feature_frame(
        production_features,
        reference_features=reference_features,
        disqualified_symbols=disqualified_symbols,
    )
    filter_reasons_series = pd.Series(
        filter_reasons, name="Filter_Status"
    ).reindex(production_features.index, fill_value="정상")
    display_columns = [
        "R_1M",
        "R_3M",
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
        *PRODUCTION_FMS_COLUMNS,
    ]
    display_columns = list(dict.fromkeys(display_columns))
    snap = production_features.reindex(columns=display_columns).copy()
    snap = snap.rename(columns={"Vol20_Ann": "Vol20(ann)"})
    snap["FMS"] = FMS
    snap["Filter_Status"] = filter_reasons_series
    return snap

def compute_fms_snapshot(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                         ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Public FMS snapshot entrypoint for harness / calibration (no network I/O).

    Thin public wrapper around ``_mom_snapshot``. Accepts pre-loaded KRW price
    (and optional OHLC / reference) DataFrames so tests can inject fixtures
    without calling yfinance.
    """
    prices_krw = mask_non_positive_prices(prices_krw)
    if reference_prices_krw is not None:
        reference_prices_krw = mask_non_positive_prices(reference_prices_krw)
    return _mom_snapshot(prices_krw, reference_prices_krw, ohlc_data, symbols)


def momentum_now_and_delta(prices_krw: pd.DataFrame, reference_prices_krw: Optional[pd.DataFrame] = None,
                           ohlc_data: Optional[pd.DataFrame] = None, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute FMS plus 1D/5D deltas from injected KRW price panels (no network I/O)."""
    prices_krw = mask_non_positive_prices(prices_krw)
    if reference_prices_krw is not None:
        reference_prices_krw = mask_non_positive_prices(reference_prices_krw)
    now = compute_fms_snapshot(prices_krw, reference_prices_krw, ohlc_data, symbols)
    d1 = _mom_snapshot(prices_krw.iloc[:-1], reference_prices_krw, ohlc_data, symbols) if len(prices_krw) > 1 else now * np.nan
    d5 = _mom_snapshot(prices_krw.iloc[:-5], reference_prices_krw, ohlc_data, symbols) if len(prices_krw) > 5 else now * np.nan
    df = now.copy()
    df['ΔFMS_1D'] = df['FMS'] - d1['FMS']
    df['ΔFMS_5D'] = df['FMS'] - d5['FMS']
    df['R_1W'] = returns_pct(prices_krw, 5)
    df['R_4M'] = returns_pct(prices_krw, HORIZON_DAYS_4M)
    df['R_YTD'] = ytd_return(prices_krw)
    return df.sort_values('FMS', ascending=False)

