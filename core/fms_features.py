"""Pure FMS feature extraction from price panels (no network I/O).

Used by recalibration and, after promotion, by production scoring. Features are
computed on the visible comparison window (default 63 trading days ≈ 3M).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from core.indicators import ema, last_vol_annualized, mask_non_positive_prices, returns_pct, r_squared_3m


HORIZON_DAYS_3M = 63
HORIZON_DAYS_4M = 84
VISIBLE_WINDOW_3M = HORIZON_DAYS_3M

# Promoted 2026-07-29 scratch model weights. Since v4.7.0, each feature is
# normalized against the current account watchlist rather than frozen fit
# statistics; the weights and feature directions remain the approved model.
PRODUCTION_FMS_COLUMNS = (
    "R2_3M",
    "DD_RECOVERY",
    "TREND_QUALITY_21D",
    "JUMP_DISCONTINUITY_3M",
    "UNDER_EMA20_DAYS",
    "R_3M",
    "STALE_AGE",
    "UP_STREAK_5D",
    "TREND_EFFICIENCY_REWARD_15D",
    "RANGE_COMPRESSION_20D",
)
PRODUCTION_FMS_WEIGHTS = {
    "R2_3M": 0.8464267343631183,
    "DD_RECOVERY": 0.6013072809914602,
    "TREND_QUALITY_21D": 0.35431695139125063,
    "JUMP_DISCONTINUITY_3M": 0.2790170661950331,
    "UNDER_EMA20_DAYS": 0.19660351882982424,
    "R_3M": 0.18698290479370505,
    "STALE_AGE": 0.18175252387780497,
    "UP_STREAK_5D": 0.1079152553498922,
    "TREND_EFFICIENCY_REWARD_15D": 0.10776592373907062,
    "RANGE_COMPRESSION_20D": 0.10416888196208451,
}
# Cash-like path gate (v4.6.1): suppress positive quality bonuses only when
# low 3M return, ultra-low volatility, and near-perfect R² occur together.
# Thresholds are decimals (0.01 = 1%). ``R_3M`` itself is never gated.
CASH_R3M_FULL = 0.01
CASH_R3M_NONE = 0.05
CASH_VOL_FULL = 0.005
CASH_VOL_NONE = 0.03
CASH_R2_NONE = 0.95
CASH_R2_FULL = 0.99
CASH_GATED_AXES = tuple(col for col in PRODUCTION_FMS_COLUMNS if col != "R_3M")


def smoothstep_series(x: pd.Series, edge0: float, edge1: float) -> pd.Series:
    """C1-continuous 0..1 transition (Hermite smoothstep)."""
    x = x.astype(float)
    if edge1 == edge0:
        return pd.Series(0.0, index=x.index, dtype=float)
    t = ((x - edge0) / (edge1 - edge0)).clip(lower=0.0, upper=1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(float)


def _vol20_ann_series(features: pd.DataFrame) -> pd.Series:
    """Resolve annualized vol column; missing → NaN (fail-open cash gate)."""
    if "Vol20_Ann" in features.columns:
        return features["Vol20_Ann"].astype(float)
    if "Vol20(ann)" in features.columns:
        return features["Vol20(ann)"].astype(float)
    return pd.Series(np.nan, index=features.index, dtype=float)


def cash_like_strength(features: pd.DataFrame) -> pd.Series:
    """Return 0..1 cash-like strength from R_3M × Vol20_Ann × R2_3M.

    Missing vol fails open (strength 0). Each factor uses smoothstep edges
    ``CASH_*`` constants; strength is high only when all three fire together.
    """
    r3m = features["R_3M"].astype(float).replace([np.inf, -np.inf], np.nan)
    r2 = features["R2_3M"].astype(float).replace([np.inf, -np.inf], np.nan)
    vol = _vol20_ann_series(features).replace([np.inf, -np.inf], np.nan)

    low_return = 1.0 - smoothstep_series(r3m.fillna(CASH_R3M_NONE), CASH_R3M_FULL, CASH_R3M_NONE)
    ultra_low_vol = 1.0 - smoothstep_series(
        vol.fillna(CASH_VOL_NONE), CASH_VOL_FULL, CASH_VOL_NONE
    )
    # Missing vol → fill to NONE edge so ultra_low_vol becomes 0 (fail open).
    high_smooth = smoothstep_series(r2.fillna(CASH_R2_NONE), CASH_R2_NONE, CASH_R2_FULL)
    strength = (low_return * ultra_low_vol * high_smooth).clip(lower=0.0, upper=1.0)
    # If vol was entirely missing, force zero (fail open).
    if "Vol20_Ann" not in features.columns and "Vol20(ann)" not in features.columns:
        strength = pd.Series(0.0, index=features.index, dtype=float)
    else:
        strength = strength.where(vol.notna(), 0.0)
    return strength.rename("cash_like_strength")


def _axis_raw_contribution(
    features: pd.DataFrame,
    reference_features: pd.DataFrame,
    col: str,
    *,
    reference_excluded_symbols: Optional[set[str]] = None,
) -> pd.Series:
    """Weighted relative-Z contribution for one production axis.

    The reference frame is the current account watchlist. Missing target values
    use that watchlist's median. An axis with fewer than two valid reference
    values or effectively zero dispersion contributes zero because it carries
    no relative ranking information.
    """
    target = features[col].astype(float).replace([np.inf, -np.inf], np.nan)
    reference = (
        reference_features[col]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    if reference_excluded_symbols:
        reference = reference.drop(
            index=reference.index.intersection(reference_excluded_symbols),
            errors="ignore",
        )
    valid_reference = reference.dropna()
    if len(valid_reference) < 2:
        return pd.Series(0.0, index=features.index, dtype=float)

    median = float(valid_reference.median())
    mean = float(valid_reference.mean())
    scale = float(np.nanstd(valid_reference.to_numpy(dtype=float), ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        return pd.Series(0.0, index=features.index, dtype=float)

    standardized = ((target.fillna(median) - mean) / scale).clip(-4.0, 4.0)
    direction = float(FEATURE_DIRECTION[col])
    return (PRODUCTION_FMS_WEIGHTS[col] * standardized * direction).astype(float)


def production_axis_contributions(
    features: pd.DataFrame,
    *,
    reference_features: Optional[pd.DataFrame] = None,
    reference_excluded_symbols: Optional[set[str]] = None,
    apply_cash_gate: bool = True,
) -> pd.DataFrame:
    """Per-axis relative FMS contributions using the current watchlist.

    When ``reference_features`` is omitted, ``features`` is its own reference;
    this is the app/watchlist behavior. Batch discovery supplies current
    account-watchlist features while scoring separate candidate rows.
    """
    missing = [col for col in PRODUCTION_FMS_COLUMNS if col not in features.columns]
    if missing:
        raise KeyError(f"production FMS missing columns: {missing}")
    reference = features if reference_features is None else reference_features
    ref_missing = [
        col for col in PRODUCTION_FMS_COLUMNS if col not in reference.columns
    ]
    if ref_missing:
        raise KeyError(f"reference FMS missing columns: {ref_missing}")

    strength = cash_like_strength(features) if apply_cash_gate else pd.Series(
        0.0, index=features.index, dtype=float
    )
    keep = (1.0 - strength).clip(lower=0.0, upper=1.0)
    cols: Dict[str, pd.Series] = {}
    for col in PRODUCTION_FMS_COLUMNS:
        raw = _axis_raw_contribution(
            features,
            reference,
            col,
            reference_excluded_symbols=reference_excluded_symbols,
        )
        if apply_cash_gate and col in CASH_GATED_AXES:
            positive = raw.clip(lower=0.0)
            negative = raw.clip(upper=0.0)
            cols[col] = negative + keep * positive
        else:
            cols[col] = raw
    return pd.DataFrame(cols, index=features.index)


def _log_slope_and_r2(series: pd.Series, window: int) -> tuple[float, float]:
    tail = series.dropna().iloc[-window:]
    if len(tail) < window or (tail <= 0).any():
        return np.nan, np.nan
    y = np.log(tail.to_numpy(dtype=float))
    x = np.arange(window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.square(y - fitted).sum())
    total = float(np.square(y - y.mean()).sum())
    r2 = 1.0 if total == 0.0 else max(0.0, 1.0 - residual / total)
    return float(slope), float(r2)


def _log_return(series: pd.Series, days: int) -> float:
    clean = series.dropna()
    if len(clean) <= days or clean.iloc[-days - 1] <= 0 or clean.iloc[-1] <= 0:
        return np.nan
    return float(np.log(clean.iloc[-1] / clean.iloc[-days - 1]))


def _under_ema_streak(series: pd.Series, ema20: pd.Series) -> int:
    below = (series < ema20).dropna()
    count = 0
    for value in reversed(below.tolist()):
        if not bool(value):
            break
        count += 1
    return count


def _downside_rms(series: pd.Series, window: int) -> float:
    returns = np.log(series.where(series > 0)).diff().dropna().iloc[-window:]
    if returns.empty:
        return np.nan
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252.0))


def _upside_rms(series: pd.Series, window: int) -> float:
    returns = np.log(series.where(series > 0)).diff().dropna().iloc[-window:]
    if returns.empty:
        return np.nan
    upside = np.maximum(returns.to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(np.square(upside))) * np.sqrt(252.0))


def _trend_efficiency(series: pd.Series, window: int) -> float:
    tail = np.log(series.where(series > 0)).diff().dropna().iloc[-window:]
    if len(tail) < window:
        return np.nan
    path = float(np.abs(tail).sum())
    return 0.0 if path == 0.0 else float(tail.sum() / path)


def _stale_age(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 25:
        return np.nan
    log_returns = np.log(clean.where(clean > 0)).diff().dropna()
    if len(log_returns) < 24:
        return np.nan
    tail21 = clean.iloc[-21:]
    high_val = float(tail21.max())
    high_ts = tail21[tail21 >= high_val - 1e-12].index[0]
    days_since_high = len(clean.loc[high_ts:]) - 1
    prior_window = log_returns.iloc[-21:-3]
    if prior_window.empty:
        return np.nan
    max_gain = max(float(prior_window.max()), 0.0)
    return float(days_since_high * max_gain)


def _post_spike_stall(series: pd.Series, lookback: int = 63) -> float:
    clean = series.dropna()
    if len(clean) < lookback:
        return np.nan
    tail = clean.iloc[-lookback:]
    log_price = np.log(tail.where(tail > 0))
    rises = log_price.diff(3)
    eligible = rises.iloc[3:-15].dropna()
    if eligible.empty:
        return np.nan
    spike_end = eligible.idxmax()
    spike = max(float(eligible.loc[spike_end]), 0.0)
    progress = float(log_price.iloc[-1] - log_price.loc[spike_end])
    return spike * max(-progress, 0.0)


def _max_drawdown_pct(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    roll_max = clean.cummax()
    dd = (clean / roll_max - 1.0) * 100.0
    return float(dd.min())


def _recovery_from_drawdown(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 10:
        return np.nan
    roll_max = clean.cummax()
    dd = clean / roll_max - 1.0
    trough = float(dd.min())
    if trough >= -0.01:
        return 1.0
    current = float(dd.iloc[-1])
    return float((current - trough) / abs(trough))


def _range_compression(series: pd.Series, window: int = 20) -> float:
    clean = series.dropna()
    if len(clean) < window * 2:
        return np.nan
    recent = clean.iloc[-window:]
    prior = clean.iloc[-2 * window : -window]
    recent_range = float(recent.max() / recent.min() - 1.0) if recent.min() > 0 else np.nan
    prior_range = float(prior.max() / prior.min() - 1.0) if prior.min() > 0 else np.nan
    if not np.isfinite(recent_range) or not np.isfinite(prior_range) or prior_range == 0:
        return np.nan
    return float(recent_range / prior_range)


def _jump_discontinuity(series: pd.Series, window: int = 63) -> float:
    """Measure gains concentrated in jumps without smooth recent follow-through."""
    returns = np.log(series.where(series > 0)).diff().dropna().iloc[-window:]
    if len(returns) < 20:
        return np.nan
    positive_path = float(returns.clip(lower=0.0).sum())
    if positive_path <= 0.0:
        return 0.0
    max_three_day_gain = max(
        float(returns.rolling(3).sum().max()), 0.0
    )
    concentration = max_three_day_gain / positive_path
    efficiency20 = _trend_efficiency(series, min(20, len(series) - 1))
    if not np.isfinite(efficiency20):
        return np.nan
    weak_follow_through = max(0.5 - efficiency20, 0.0)
    return float(concentration * weak_follow_through)


def _consecutive_direction(series: pd.Series, window: int, direction: str) -> int:
    clean = series.dropna()
    if len(clean) < window + 1:
        return 0
    diff = clean.diff().iloc[-window:]
    if direction == "up":
        flags = diff > 0
    else:
        flags = diff < 0
    run = 0
    best = 0
    for flag in flags.iloc[1:]:
        if bool(flag):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _symbol_features(series: pd.Series, *, window_days: int) -> Dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {}

    visible = clean.iloc[-window_days:] if len(clean) > window_days else clean
    ema20 = ema(visible, 20)
    ema50 = ema(visible, 50)

    slope10, r2_10 = _log_slope_and_r2(visible, min(10, len(visible)))
    slope15, r2_15 = _log_slope_and_r2(visible, min(15, len(visible)))
    slope21, r2_21 = _log_slope_and_r2(visible, min(21, len(visible)))
    ema_slope3, _ = _log_slope_and_r2(ema20, min(4, len(ema20)))
    ema_slope10, _ = _log_slope_and_r2(ema20, min(10, len(ema20)))
    ema_curvature20 = np.nan
    if len(ema20) >= 20:
        previous_slope, _ = _log_slope_and_r2(ema20.iloc[-20:-10], 10)
        recent_slope, _ = _log_slope_and_r2(ema20.iloc[-10:], 10)
        if np.isfinite(previous_slope) and np.isfinite(recent_slope):
            # Positive means the EMA20 slope is becoming steeper.
            ema_curvature20 = recent_slope - previous_slope

    r3 = _log_return(visible, 3)
    prior7 = np.nan
    if len(visible) > 10:
        prior7 = float(np.log(visible.iloc[-4] / visible.iloc[-11]) / 7.0)

    price_vs_ema20 = (
        float(visible.iloc[-1] / ema20.iloc[-1] - 1.0) if ema20.iloc[-1] > 0 else np.nan
    )
    price_vs_ema50 = (
        float(visible.iloc[-1] / ema50.iloc[-1] - 1.0) if ema50.iloc[-1] > 0 else np.nan
    )

    under_depth = 0.0
    under_days = 0
    tail = visible.iloc[-min(60, len(visible)) :]
    e20_tail = ema20.reindex(tail.index)
    mask_under = tail < e20_tail
    if mask_under.any():
        rel = tail[mask_under] / e20_tail[mask_under] - 1.0
        under_depth = float(rel.min())
        under_days = int(mask_under.sum())

    down_streak = _consecutive_direction(visible, 5, "down")
    up_streak = _consecutive_direction(visible, 5, "up")

    te10 = _trend_efficiency(visible, min(10, len(visible)))
    te15 = _trend_efficiency(visible, min(15, len(visible)))
    te20 = _trend_efficiency(visible, min(20, len(visible)))

    return {
        "R_3D_LOG": r3,
        "R_5D_LOG": _log_return(visible, 5),
        "R_10D_LOG": _log_return(visible, 10),
        "R_21D_LOG": _log_return(visible, 21),
        "LOG_SLOPE_10D": slope10,
        "LOG_SLOPE_15D": slope15,
        "LOG_SLOPE_21D": slope21,
        "TREND_R2_10D": r2_10,
        "TREND_R2_15D": r2_15,
        "TREND_R2_21D": r2_21,
        "TREND_QUALITY_10D": slope10 * r2_10 if np.isfinite(slope10) and np.isfinite(r2_10) else np.nan,
        "TREND_QUALITY_15D": slope15 * r2_15 if np.isfinite(slope15) and np.isfinite(r2_15) else np.nan,
        "TREND_QUALITY_21D": slope21 * r2_21 if np.isfinite(slope21) and np.isfinite(r2_21) else np.nan,
        "RECOVERY_3D_VS_PRIOR7": r3 / 3.0 - prior7 if np.isfinite(r3) and np.isfinite(prior7) else np.nan,
        "EMA20_SLOPE_3D": ema_slope3,
        "EMA20_SLOPE_10D": ema_slope10,
        "EMA20_ACCEL_3D_VS_10D": (
            ema_slope3 - ema_slope10
            if np.isfinite(ema_slope3) and np.isfinite(ema_slope10)
            else np.nan
        ),
        "EMA20_CURV_20D": ema_curvature20,
        "PRICE_VS_EMA20": price_vs_ema20,
        "PRICE_VS_EMA50": price_vs_ema50,
        "AboveEMA50": price_vs_ema50,
        "UNDER_EMA20_DEPTH": under_depth,
        "UNDER_EMA20_DAYS": float(under_days),
        "UNDER_EMA20_STREAK": float(_under_ema_streak(visible, ema20)),
        "DOWN_STREAK_5D": float(down_streak),
        "UP_STREAK_5D": float(up_streak),
        "TREND_EFFICIENCY_10D": te10,
        "TREND_EFFICIENCY_15D": te15,
        "TREND_EFFICIENCY_20D": te20,
        "TREND_EFFICIENCY_REWARD_15D": max(te15, 0.0) if np.isfinite(te15) else np.nan,
        "TREND_INEFFICIENCY_15D": max(0.2 - te15, 0.0) if np.isfinite(te15) else np.nan,
        "DOWNSIDE_RMS_10D": _downside_rms(visible, min(10, len(visible))),
        "DOWNSIDE_RMS_20D": _downside_rms(visible, min(20, len(visible))),
        "UPSIDE_RMS_20D": _upside_rms(visible, min(20, len(visible))),
        "VOL_ASYMMETRY_20D": (
            _upside_rms(visible, min(20, len(visible))) - _downside_rms(visible, min(20, len(visible)))
        ),
        "WORST_LOG_RETURN_20D": float(
            np.log(visible.where(visible > 0)).diff().dropna().iloc[-min(20, len(visible)) :].min()
        ),
        "POST_SPIKE_STALL": _post_spike_stall(visible, lookback=min(window_days, len(visible))),
        "STALE_AGE": _stale_age(visible),
        "MaxDD_Pct": _max_drawdown_pct(visible),
        "DD_RECOVERY": _recovery_from_drawdown(visible),
        "RANGE_COMPRESSION_20D": _range_compression(visible, 20),
        "JUMP_DISCONTINUITY_3M": _jump_discontinuity(visible),
        "RECENT_3D_VS_21D_TREND": (
            r3 / 3.0 - slope21
            if np.isfinite(r3) and np.isfinite(slope21)
            else np.nan
        ),
        "SLOPE_ACCEL_10_21": (
            slope10 - slope21 if np.isfinite(slope10) and np.isfinite(slope21) else np.nan
        ),
    }


def build_symbol_feature_frame(
    prices_krw: pd.DataFrame,
    *,
    symbols: Optional[Iterable[str]] = None,
    window_days: int = VISIBLE_WINDOW_3M,
) -> pd.DataFrame:
    """Build the full interpretable feature table for ``symbols``."""
    clean = mask_non_positive_prices(prices_krw)
    cols = list(symbols) if symbols is not None else list(clean.columns)
    rows: Dict[str, Dict[str, float]] = {}
    for symbol in cols:
        if symbol not in clean.columns:
            continue
        rows[str(symbol)] = _symbol_features(clean[symbol], window_days=window_days)
    return pd.DataFrame.from_dict(rows, orient="index").reindex(
        [str(symbol) for symbol in cols if symbol in clean.columns]
    )


def build_panel_feature_frame(
    prices_krw: pd.DataFrame,
    *,
    symbols: Optional[Iterable[str]] = None,
    window_days: int = VISIBLE_WINDOW_3M,
) -> pd.DataFrame:
    """Add legacy production-compatible columns used by baseline scoring."""
    sym_frame = build_symbol_feature_frame(
        prices_krw, symbols=symbols, window_days=window_days
    )
    if sym_frame.empty:
        return sym_frame

    clean = mask_non_positive_prices(prices_krw)
    cols = [c for c in sym_frame.index if c in clean.columns]
    panel = clean[cols]

    r_1m = returns_pct(panel, 21).rename("R_1M")
    r_3m = returns_pct(panel, 63).rename("R_3M")
    r_4m = returns_pct(panel, HORIZON_DAYS_4M).rename("R_4M")
    r2_3m = r_squared_3m(panel).rename("R2_3M")
    vol20 = last_vol_annualized(panel, 20).rename("Vol20_Ann")
    r_10d = returns_pct(panel, 10).rename("R_10D")
    r_5d = returns_pct(panel, 5).rename("R_5D")

    legacy = pd.concat([r_1m, r_3m, r_4m, r2_3m, vol20, r_10d, r_5d], axis=1)
    out = sym_frame.join(legacy, how="left")

    # Align names with production scorer expectations.
    if "EMA20_SLOPE_10D" not in out.columns and "EMA20_SLOPE_10D" in sym_frame.columns:
        out["EMA20_SLOPE_10D"] = sym_frame["EMA20_SLOPE_10D"]
    out["EMA20_CURV_20D"] = sym_frame.get("EMA20_CURV_20D", sym_frame["EMA20_ACCEL_3D_VS_10D"])
    out["UNDER_EMA20_DEPTH"] = sym_frame["UNDER_EMA20_DEPTH"]
    out["UNDER_EMA20_DAYS"] = sym_frame["UNDER_EMA20_DAYS"]
    out["DOWN_STREAK_5D"] = sym_frame["DOWN_STREAK_5D"]
    out["AboveEMA50"] = sym_frame["AboveEMA50"]
    out["MaxDD_Pct"] = sym_frame["MaxDD_Pct"]
    # Visible-window policy: omit R_4M from scratch refit features.
    return out


def score_production_fms_features(
    features: pd.DataFrame,
    *,
    reference_features: Optional[pd.DataFrame] = None,
    disqualified_symbols: Optional[set[str]] = None,
    apply_cash_gate: bool = True,
) -> pd.Series:
    """Score the approved sparse-linear production FMS from feature rows.

    Each axis is standardized against the current account watchlist's
    mean/standard deviation and clipped to ±4. Missing values use the current
    watchlist median. Negative-direction axes are sign-flipped before applying
    their non-negative fitted weights.

    When ``apply_cash_gate`` is True (default), positive contributions on
    quality axes other than ``R_3M`` are scaled by ``(1 - cash_like_strength)``
    so low-return / ultra-low-vol / high-R² cash paths cannot dominate ranks.
    Existing penalties and the ``R_3M`` term are never reduced by the gate.
    """
    contrib = production_axis_contributions(
        features,
        reference_features=reference_features,
        reference_excluded_symbols=disqualified_symbols,
        apply_cash_gate=apply_cash_gate,
    )
    score = contrib.sum(axis=1).astype(float).rename("FMS")

    if disqualified_symbols:
        score.loc[score.index.intersection(disqualified_symbols)] = -999.0
    return score.rename("FMS")


# Higher score should mean better momentum rank.
FEATURE_DIRECTION: Dict[str, int] = {
    "R_3D_LOG": 1,
    "R_5D_LOG": 1,
    "R_10D_LOG": 1,
    "R_21D_LOG": 1,
    "R_1M": 1,
    "R_3M": 1,
    "R2_3M": 1,
    "LOG_SLOPE_10D": 1,
    "LOG_SLOPE_15D": 1,
    "LOG_SLOPE_21D": 1,
    "TREND_R2_10D": 1,
    "TREND_R2_15D": 1,
    "TREND_R2_21D": 1,
    "TREND_QUALITY_10D": 1,
    "TREND_QUALITY_15D": 1,
    "TREND_QUALITY_21D": 1,
    "RECOVERY_3D_VS_PRIOR7": 1,
    "EMA20_SLOPE_3D": 1,
    "EMA20_SLOPE_10D": 1,
    "EMA20_ACCEL_3D_VS_10D": 1,
    "EMA20_CURV_20D": 1,
    "PRICE_VS_EMA20": 1,
    "PRICE_VS_EMA50": 1,
    "AboveEMA50": 1,
    "TREND_EFFICIENCY_10D": 1,
    "TREND_EFFICIENCY_15D": 1,
    "TREND_EFFICIENCY_20D": 1,
    "TREND_EFFICIENCY_REWARD_15D": 1,
    "UPSIDE_RMS_20D": 1,
    "UP_STREAK_5D": 1,
    "DD_RECOVERY": 1,
    "UNDER_EMA20_DEPTH": -1,
    "UNDER_EMA20_DAYS": -1,
    "UNDER_EMA20_STREAK": -1,
    "DOWN_STREAK_5D": -1,
    "TREND_INEFFICIENCY_15D": -1,
    "DOWNSIDE_RMS_10D": -1,
    "DOWNSIDE_RMS_20D": -1,
    "WORST_LOG_RETURN_20D": -1,
    "POST_SPIKE_STALL": -1,
    "STALE_AGE": -1,
    "MaxDD_Pct": 1,  # less negative drawdown is better
    "Vol20_Ann": -1,
    "RANGE_COMPRESSION_20D": -1,
    "JUMP_DISCONTINUITY_3M": -1,
    "RECENT_3D_VS_21D_TREND": 1,
    "SLOPE_ACCEL_10_21": 1,
    "VOL_ASYMMETRY_20D": 1,
}


def candidate_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Return scratch-refit candidate columns present in ``frame``."""
    banned = {"rank", "R_4M", "R_6M"}
    cols = []
    for col in frame.columns:
        if col in banned:
            continue
        if col in FEATURE_DIRECTION or col.startswith(("R_", "LOG_", "TREND_", "EMA", "UNDER_", "DOWN_", "UP_", "DD_", "POST_", "STALE", "MaxDD", "Vol", "PRICE_", "RECOVERY", "RANGE_", "SLOPE_", "DOWNSIDE", "UPSIDE", "WORST_")):
            cols.append(col)
    return sorted(set(cols))
