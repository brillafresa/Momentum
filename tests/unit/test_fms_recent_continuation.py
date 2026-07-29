"""
Recent-continuation vs stale-plateau FMS behavior (offline).

Purpose
-------
Regression for the 2026-07-29 tune: high R_1M with confirmed short-term
continuation (R_10D>0, EMA20 slope>0) must not be treated as an event-spike
(``r1_bad``), and a steady recent uptrend should outrank a 3M-rise-then-flat
pattern when scored on a shared reference panel.

Usage
-----
    python -m pytest tests/unit/test_fms_recent_continuation.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.fms import (
    R2_QUALITY_CENTER,
    _r1_conditional_series,
    _r1_quality_weight,
    _recent_continuation_mask,
    _smoothstep,
    compute_fms_snapshot,
    production_fms_score_params,
)
from core.indicators import returns_pct, r_squared_3m


def _geom(start: float, daily: np.ndarray) -> np.ndarray:
    return start * np.cumprod(1.0 + daily)


def _build_stale_vs_recent_panel(n: int = 180, seed: int = 7) -> pd.DataFrame:
    """STALE_PLATEAU: strong mid-window rise then ~1M flat.

    RECENT_CONT: moderate base then steady last-~1M climb (high R_1M, R2≈0.8x).
    REF_*: mild peers so Z-scores stay defined.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)

    stale = np.zeros(n)
    stale[:120] = 0.0055 + rng.normal(0, 0.0015, 120)
    stale[120:] = rng.normal(0, 0.0010, n - 120)

    recent = np.zeros(n)
    recent[:100] = 0.0010 + rng.normal(0, 0.0020, 100)
    recent[100:] = 0.0120 + rng.normal(0, 0.0025, n - 100)

    ref_a = 0.0020 + rng.normal(0, 0.004, n)
    ref_b = 0.0015 + rng.normal(0, 0.005, n)
    ref_c = rng.normal(0, 0.008, n)

    prices = pd.DataFrame(
        {
            "STALE_PLATEAU": _geom(100.0, stale),
            "RECENT_CONT": _geom(100.0, recent),
            "REF_A": _geom(100.0, ref_a),
            "REF_B": _geom(100.0, ref_b),
            "REF_C": _geom(100.0, ref_c),
        },
        index=dates,
    )
    prices.index.name = "Date"
    return prices


def test_r2_quality_weight_is_soft_around_080() -> None:
    """R² quality uses smoothstep centered at 0.80 (not a hard 0.85 cut)."""
    p = production_fms_score_params()
    idx = pd.Index(["lo", "mid", "hi"])
    r2 = pd.Series([0.70, 0.83, 0.92], index=idx)
    r3 = pd.Series([0.40, 0.40, 0.40], index=idx)
    r4 = pd.Series([0.40, 0.40, 0.40], index=idx)
    w = _r1_quality_weight(r2, r3, r4, p)
    assert float(w.loc["lo"]) == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < float(w.loc["mid"]) < 1.0
    assert float(w.loc["hi"]) == pytest.approx(1.0, abs=1e-6)
    # Mid (0.83) would have failed the old hard R2>0.85 gate.
    hard_mid = 0.83 > 0.85
    assert hard_mid is False
    assert float(w.loc["mid"]) > 0.5


def test_recent_continuation_exempts_r1_bad() -> None:
    """High R_1M with R_10D>0 and positive EMA slope is not r1_bad."""
    p = production_fms_score_params()
    idx = pd.Index(["spike", "cont"])
    r_1m = pd.Series([0.45, 0.45], index=idx)
    r_3m = pd.Series([0.40, 0.40], index=idx)
    r_4m = pd.Series([0.40, 0.40], index=idx)
    # Spike: clearly below soft quality → r1_bad.
    # Cont: same low R² but continuation exempts event-spike penalty;
    #       also verify a borderline R² (0.82) receives partial r1_good.
    r2 = pd.Series([0.72, 0.72], index=idx)
    r_10d = pd.Series([-0.05, 0.08], index=idx)
    slope = pd.Series([-0.001, 0.004], index=idx)

    good, bad = _r1_conditional_series(r_1m, r_3m, r_4m, r2, r_10d, slope, p)
    assert float(bad.loc["spike"]) == pytest.approx(0.45)
    assert float(bad.loc["cont"]) == pytest.approx(0.0)

    # Soft gate: R2=0.82 (old hard cut would fail) yields partial r1_good.
    r2_soft = pd.Series([0.82, 0.82], index=idx)
    good_soft, bad_soft = _r1_conditional_series(
        r_1m, r_3m, r_4m, r2_soft, r_10d, slope, p
    )
    assert float(good_soft.loc["cont"]) > 0.0
    assert float(bad_soft.loc["cont"]) == pytest.approx(0.0)


def test_recent_continuation_outranks_stale_plateau() -> None:
    """Steady recent climb should score at least as high as rise-then-flat."""
    prices = _build_stale_vs_recent_panel()
    # Sanity: RECENT_CONT has stronger 1M / 10D than STALE.
    r1 = returns_pct(prices, 21)
    r10 = returns_pct(prices, 10)
    r2 = r_squared_3m(prices)
    assert float(r1["RECENT_CONT"]) > 0.25
    assert float(r1["STALE_PLATEAU"]) < 0.08
    assert float(r10["RECENT_CONT"]) > 0.0
    assert float(r2["RECENT_CONT"]) > 0.70

    snap = compute_fms_snapshot(
        prices,
        reference_prices_krw=prices,
        ohlc_data=None,
        symbols=list(prices.columns),
    )
    fms_recent = float(snap.loc["RECENT_CONT", "FMS"])
    fms_stale = float(snap.loc["STALE_PLATEAU", "FMS"])
    assert fms_recent > fms_stale

    # Continuation path: r1_bad raw for RECENT_CONT must be zero.
    p = production_fms_score_params()
    r_1m = returns_pct(prices, 21)
    r_3m = returns_pct(prices, 63)
    r_4m = returns_pct(prices, 84)
    r2_3m = r_squared_3m(prices)
    r_10d = returns_pct(prices, 10)
    # Slope from snapshot features
    slope = snap["EMA20_SLOPE_10D"]
    _good, bad = _r1_conditional_series(
        r_1m, r_3m, r_4m, r2_3m, r_10d, slope, p
    )
    assert float(bad.loc["RECENT_CONT"]) == pytest.approx(0.0)


def test_quality_center_constant() -> None:
    assert R2_QUALITY_CENTER == pytest.approx(0.80)


def test_recent_continuation_mask_helpers() -> None:
    idx = pd.Index(["a", "b", "c"])
    r10 = pd.Series([0.01, -0.01, 0.02], index=idx)
    slope = pd.Series([0.001, 0.001, -0.001], index=idx)
    m = _recent_continuation_mask(r10, slope)
    assert bool(m.loc["a"]) is True
    assert bool(m.loc["b"]) is False
    assert bool(m.loc["c"]) is False


def test_smoothstep_monotone_around_quality_center() -> None:
    p = production_fms_score_params()
    xs = pd.Series(np.linspace(0.70, 0.90, 21))
    ys = _smoothstep(
        xs,
        R2_QUALITY_CENTER - p.r2_transition_w,
        R2_QUALITY_CENTER + p.r2_transition_w,
    )
    assert ys.is_monotonic_increasing
