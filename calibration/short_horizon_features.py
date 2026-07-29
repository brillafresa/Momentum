"""Pure, offline candidate features for short-horizon FMS recalibration.

These features are experimental until they pass ranking and harness validation.
They use only price/EMA shapes visible in the A/B calibration charts.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from core.indicators import ema, mask_non_positive_prices


def _log_slope_and_r2(series: pd.Series, window: int) -> tuple[float, float]:
    """Return daily log-price OLS slope and R² for the trailing window."""
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
    """Return trailing log return over ``days`` trading-day intervals."""
    clean = series.dropna()
    if len(clean) <= days or clean.iloc[-days - 1] <= 0 or clean.iloc[-1] <= 0:
        return np.nan
    return float(np.log(clean.iloc[-1] / clean.iloc[-days - 1]))


def _under_ema_streak(series: pd.Series, ema20: pd.Series) -> int:
    """Count consecutive trailing observations below EMA20."""
    below = (series < ema20).dropna()
    count = 0
    for value in reversed(below.tolist()):
        if not bool(value):
            break
        count += 1
    return count


def _downside_rms(series: pd.Series, window: int = 20) -> float:
    """Return annualized RMS of negative daily log returns only."""
    returns = np.log(series.where(series > 0)).diff().dropna().iloc[-window:]
    if returns.empty:
        return np.nan
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252.0))


def _trend_efficiency(series: pd.Series, window: int = 15) -> float:
    """Return signed net movement divided by total path movement."""
    tail = np.log(series.where(series > 0)).diff().dropna().iloc[-window:]
    if len(tail) < window:
        return np.nan
    path = float(np.abs(tail).sum())
    return 0.0 if path == 0.0 else float(tail.sum() / path)


def _stale_age(series: pd.Series) -> float:
    """Penalize stagnation after a visible spike (offline recalibration candidate).

    ``days_since_21d_high × max_daily_gain(t-21:t-3)`` — longer flat drift after an
    older spike within the lookback receives a larger value.
    """
    clean = series.dropna()
    if len(clean) < 25:
        return np.nan

    log_returns = np.log(clean.where(clean > 0)).diff().dropna()
    if len(log_returns) < 24:
        return np.nan

    tail21 = clean.iloc[-21:]
    high_val = float(tail21.max())
    # Use the oldest touch of the 21d high so flat drift after a spike is visible.
    high_ts = tail21[tail21 >= high_val - 1e-12].index[0]
    days_since_high = len(clean.loc[high_ts:]) - 1

    prior_window = log_returns.iloc[-21:-3]
    if prior_window.empty:
        return np.nan
    max_gain = max(float(prior_window.max()), 0.0)
    return float(days_since_high * max_gain)


def _post_spike_stall(series: pd.Series) -> float:
    """Measure an old 3-day spike that failed to make subsequent progress.

    The spike search excludes the latest 15 sessions. A large old spike receives
    a larger value when price made little or negative progress after its endpoint.
    This is a continuous diagnostic, not a hard event threshold.
    """
    clean = series.dropna()
    if len(clean) < 64:
        return np.nan
    tail = clean.iloc[-64:]
    log_price = np.log(tail.where(tail > 0))
    rises = log_price.diff(3)
    eligible = rises.iloc[3:-15].dropna()
    if eligible.empty:
        return np.nan
    spike_end = eligible.idxmax()
    spike = max(float(eligible.loc[spike_end]), 0.0)
    progress = float(log_price.iloc[-1] - log_price.loc[spike_end])
    return spike * max(-progress, 0.0)


def compute_short_horizon_candidates(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute interpretable candidate features for each price column."""
    clean_prices = mask_non_positive_prices(prices)
    rows: Dict[str, Dict[str, float]] = {}
    for symbol in clean_prices.columns:
        series = clean_prices[symbol].dropna()
        if series.empty:
            rows[str(symbol)] = {}
            continue

        slope10, r2_10 = _log_slope_and_r2(series, 10)
        slope15, r2_15 = _log_slope_and_r2(series, 15)
        slope20, r2_20 = _log_slope_and_r2(series, 20)
        ema20 = ema(series, 20)
        ema_slope3, _ = _log_slope_and_r2(ema20, 4)
        ema_slope10, _ = _log_slope_and_r2(ema20, 10)
        r3 = _log_return(series, 3)
        efficiency15 = _trend_efficiency(series, 15)
        prior7 = np.nan
        if len(series) > 10:
            prior7 = float(np.log(series.iloc[-4] / series.iloc[-11]) / 7.0)

        rows[str(symbol)] = {
            "LOG_SLOPE_10D": slope10,
            "LOG_SLOPE_15D": slope15,
            "LOG_SLOPE_20D": slope20,
            "TREND_R2_10D": r2_10,
            "TREND_R2_15D": r2_15,
            "TREND_R2_20D": r2_20,
            "TREND_QUALITY_10D": slope10 * r2_10,
            "TREND_QUALITY_15D": slope15 * r2_15,
            "TREND_QUALITY_20D": slope20 * r2_20,
            "R_3D_LOG": r3,
            "RECOVERY_3D_VS_PRIOR7": r3 / 3.0 - prior7,
            "EMA20_SLOPE_3D": ema_slope3,
            "EMA20_ACCEL_3D_VS_10D": ema_slope3 - ema_slope10,
            "PRICE_VS_EMA20": (
                float(series.iloc[-1] / ema20.iloc[-1] - 1.0)
                if ema20.iloc[-1] > 0
                else np.nan
            ),
            "UNDER_EMA20_STREAK": float(_under_ema_streak(series, ema20)),
            "TREND_EFFICIENCY_10D": _trend_efficiency(series, 10),
            "TREND_EFFICIENCY_15D": efficiency15,
            "TREND_EFFICIENCY_REWARD_15D": max(efficiency15, 0.0),
            "TREND_INEFFICIENCY_15D": max(0.2 - efficiency15, 0.0),
            "TREND_EFFICIENCY_20D": _trend_efficiency(series, 20),
            "DOWNSIDE_RMS_10D": _downside_rms(series, 10),
            "DOWNSIDE_RMS_20D": _downside_rms(series, 20),
            "WORST_LOG_RETURN_20D": float(
                np.log(series.where(series > 0)).diff().dropna().iloc[-20:].min()
            ),
            "POST_SPIKE_STALL_3M": _post_spike_stall(series),
            "STALE_AGE": _stale_age(series),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
