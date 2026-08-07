# -*- coding: utf-8 -*-
"""
Native as-of calendar contracts (Validation Harness).

Purpose
-------
Multi-market union panels must not extend a symbol past its last real
observation via blanket ``ffill``. Trailing phantom flats shift SEG_* windows
and can roughly double FMS (ITGR 2026-08-07: KR-open / US-closed).

Coverage is market-agnostic (column ``last_valid_index`` only):
- market A ahead of B
- market B ahead of A
- three markets with only one ahead

No network I/O. Production ``app.py`` / ``run_scan_batch.py`` must not import this
module.

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_native_asof_calendar.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.fms import compute_fms_snapshot
from core.indicators import harmonize_calendar


def _steady_then_jump(n: int = 80, jump_days: int = 5) -> pd.Series:
    """Synthetic path: mild grind then a concentrated recent jump."""
    idx = pd.bdate_range("2025-01-02", periods=n)
    log_rets = np.full(n - 1, 0.002)
    log_rets[-(jump_days):] = 0.04
    prices = 100.0 * np.exp(np.r_[0.0, np.cumsum(log_rets)])
    return pd.Series(prices, index=idx, name="A")


def test_harmonize_clips_trailing_when_other_market_ahead() -> None:
    """Shorter series must stay NaN on the other market's exclusive last day."""
    idx_short = pd.bdate_range("2025-01-02", periods=10)
    idx_long = pd.bdate_range("2025-01-02", periods=11)
    short = pd.Series(np.linspace(100.0, 110.0, 10), index=idx_short, name="US_LIKE")
    long = pd.Series(np.linspace(1000.0, 1100.0, 11), index=idx_long, name="KR_LIKE")
    panel = pd.concat([short, long], axis=1)
    out = harmonize_calendar(panel, coverage=0.5)

    ahead = idx_long[-1]
    assert ahead in out.index
    assert pd.isna(out.loc[ahead, "US_LIKE"])
    assert pd.notna(out.loc[ahead, "KR_LIKE"])
    assert out["US_LIKE"].last_valid_index() == idx_short[-1]
    assert out["KR_LIKE"].last_valid_index() == ahead


def test_harmonize_clips_trailing_reverse_direction() -> None:
    """When the other side is ahead, the lagging market still clips."""
    idx_short = pd.bdate_range("2025-01-02", periods=10)
    idx_long = pd.bdate_range("2025-01-02", periods=11)
    short = pd.Series(np.linspace(1000.0, 1100.0, 10), index=idx_short, name="KR_LIKE")
    long = pd.Series(np.linspace(100.0, 110.0, 11), index=idx_long, name="US_LIKE")
    panel = pd.concat([short, long], axis=1)
    out = harmonize_calendar(panel, coverage=0.5)

    ahead = idx_long[-1]
    assert pd.isna(out.loc[ahead, "KR_LIKE"])
    assert pd.notna(out.loc[ahead, "US_LIKE"])
    assert out["KR_LIKE"].last_valid_index() == idx_short[-1]


def test_harmonize_three_markets_only_ahead_market_valid_on_tail() -> None:
    """US/KR/HK-style: only the market that actually printed the last day is finite."""
    idx_base = pd.bdate_range("2025-01-02", periods=10)
    idx_hk_ahead = pd.bdate_range("2025-01-02", periods=11)
    us = pd.Series(np.linspace(100.0, 110.0, 10), index=idx_base, name="US")
    kr = pd.Series(np.linspace(1000.0, 1100.0, 10), index=idx_base, name="KR")
    hk = pd.Series(np.linspace(50.0, 55.0, 11), index=idx_hk_ahead, name="HK")
    out = harmonize_calendar(pd.concat([us, kr, hk], axis=1), coverage=0.5)

    ahead = idx_hk_ahead[-1]
    assert pd.notna(out.loc[ahead, "HK"])
    assert pd.isna(out.loc[ahead, "US"])
    assert pd.isna(out.loc[ahead, "KR"])


def test_fms_unchanged_when_other_market_trailing_day_appended() -> None:
    """KR-ahead phantom day must not change the lagging symbol's FMS."""
    solo = _steady_then_jump().to_frame("JUMP")
    solo_fms = float(compute_fms_snapshot(solo).loc["JUMP", "FMS"])

    other_idx = solo.index.union([solo.index[-1] + pd.offsets.BDay(1)])
    other = pd.Series(2000.0, index=other_idx, name="OTHER")
    mixed = pd.concat([solo["JUMP"], other], axis=1)
    mixed_h = harmonize_calendar(mixed, coverage=0.5)
    mixed_fms = float(compute_fms_snapshot(mixed_h[["JUMP"]]).loc["JUMP", "FMS"])

    assert mixed_fms == pytest.approx(solo_fms, abs=1e-9)
    assert pd.isna(mixed_h["JUMP"].iloc[-1])


def test_fms_unchanged_when_symbol_itself_is_ahead_of_peer() -> None:
    """US-ahead: peer trailing NaN must not alter the ahead symbol's own FMS."""
    ahead = _steady_then_jump(n=81).to_frame("JUMP")
    ahead_fms = float(compute_fms_snapshot(ahead).loc["JUMP", "FMS"])

    peer = ahead["JUMP"].iloc[:-1].rename("PEER")
    mixed = pd.concat([ahead["JUMP"], peer], axis=1)
    mixed_h = harmonize_calendar(mixed, coverage=0.5)
    mixed_fms = float(compute_fms_snapshot(mixed_h[["JUMP"]]).loc["JUMP", "FMS"])

    assert mixed_fms == pytest.approx(ahead_fms, abs=1e-9)
    assert pd.notna(mixed_h["JUMP"].iloc[-1])
    assert pd.isna(mixed_h["PEER"].iloc[-1])


def test_harmonize_still_ffills_interior_gaps_within_native_span() -> None:
    """Interior holiday gaps inside [first_valid, last_valid] may still ffill."""
    idx = pd.bdate_range("2025-01-02", periods=8)
    s = pd.Series(
        [100.0, 101.0, np.nan, 103.0, 104.0, 105.0, 106.0, 107.0],
        index=idx,
        name="A",
    )
    out = harmonize_calendar(s.to_frame(), coverage=0.5)
    # Gap at idx[2] should be filled from prior close within native span.
    assert out.loc[idx[2], "A"] == pytest.approx(101.0)
    assert out["A"].last_valid_index() == idx[-1]
