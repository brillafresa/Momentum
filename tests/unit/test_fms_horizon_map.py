"""Unit tests for FMS horizon return / gate-width mapping (6M→4M)."""

from __future__ import annotations

import math

import pytest

from core.fms import (
    HORIZON_DAYS_4M,
    HORIZON_DAYS_LEGACY_6M,
    R_4M_GATE_CENTER,
    R_4M_QUALITY_MIN,
    gate_width_scale,
    horizon_return_map,
    production_fms_score_params,
)
from core.indicators import r_squared_3m


def test_horizon_return_map_gate_and_quality() -> None:
    assert horizon_return_map(0.08, 126, 84) == pytest.approx(R_4M_GATE_CENTER)
    assert horizon_return_map(0.50, 126, 84) == pytest.approx(R_4M_QUALITY_MIN)
    assert R_4M_GATE_CENTER == pytest.approx((1.08) ** (84 / 126) - 1)
    assert R_4M_QUALITY_MIN == pytest.approx((1.50) ** (84 / 126) - 1)


def test_gate_width_scales_with_sqrt_t() -> None:
    legacy_w = 0.006355
    got = gate_width_scale(legacy_w, HORIZON_DAYS_LEGACY_6M, HORIZON_DAYS_4M)
    assert got == pytest.approx(legacy_w * math.sqrt(84 / 126))
    assert production_fms_score_params().gate_r4_w == pytest.approx(got)


def test_production_level_r4_hi_is_compound_mapped() -> None:
    expected = horizon_return_map(0.430268, 126, 84)
    assert production_fms_score_params().level_r4_hi == pytest.approx(expected)


def test_r_squared_window_remains_63(
    synthetic_prices_krw,
) -> None:
    """R² lookback stays 3M; 6M→4M must not shrink the R² sample."""
    # Contract: implementation uses tail(63); short series (<63) → NaN.
    short = synthetic_prices_krw.iloc[-50:]
    r2 = r_squared_3m(short)
    assert r2.isna().all()
    assert HORIZON_DAYS_4M == 84
