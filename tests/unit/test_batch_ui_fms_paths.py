# -*- coding: utf-8 -*-
"""
Offline contracts for batch vs UI calendar path FMS comparison.

Purpose
-------
- Identical aligned panels → bit-identical FMS (builders are not a hidden scorer fork).
- After ffill harmonization, staggered native calendars still converge (documents that
  pure calendar gymnastics rarely move v5 absolute FMS once series are filled).
- Coverage 0.5 vs 0.9 can change *which* symbols remain, not the shared-symbol math
  when both keep the name.

No network I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.compare_batch_ui_fms import (
    build_batch_style_prices_krw,
    build_ui_style_from_krw_panel,
    build_ui_style_from_market_frames,
    compare_fms_paths,
    inject_staggered_calendar_gaps,
    split_into_staggered_market_frames,
)


def test_identical_calendar_paths_match_fms(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
) -> None:
    """When the panel is already aligned, UI-like and batch paths agree."""
    ui = build_ui_style_from_krw_panel(synthetic_prices_krw)
    batch = build_batch_style_prices_krw(synthetic_prices_krw)
    result = compare_fms_paths(ui, batch, synthetic_ohlc)
    both = result.comparison
    assert not both.empty
    abs_d = (both["FMS_Batch"] - both["FMS_UI"]).abs()
    assert float(abs_d.max()) < 1e-9


def test_staggered_calendars_converge_after_ffill(
    synthetic_prices_krw: pd.DataFrame,
    synthetic_ohlc: pd.DataFrame,
) -> None:
    """Per-market native gaps still ffill to the same trailing path → same FMS."""
    gapped = inject_staggered_calendar_gaps(synthetic_prices_krw, gap_frac=0.12)
    frame_a, frame_b = split_into_staggered_market_frames(gapped)
    ui = build_ui_style_from_market_frames(frame_a, frame_b)
    batch_raw = pd.concat([frame_a, frame_b], axis=1).sort_index()
    ordered = [c for c in gapped.columns if c in batch_raw.columns]
    batch = build_batch_style_prices_krw(batch_raw[ordered])
    result = compare_fms_paths(ui, batch, synthetic_ohlc)
    finite = result.comparison[
        (result.comparison["FMS_UI"] != -999.0)
        & (result.comparison["FMS_Batch"] != -999.0)
    ]
    assert not finite.empty
    assert float(finite["abs_d"].max()) < 1e-9


def test_stricter_batch_coverage_can_drop_sparse_symbol() -> None:
    """coverage=0.9 may drop a name that coverage=0.5 keeps."""
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    dense = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx, name="DENSE")
    sparse = dense.copy().rename("SPARSE")
    # Leading NaNs survive ffill; ~55% coverage → kept by UI(0.5), dropped by batch(0.9)
    sparse.iloc[:45] = np.nan
    panel = pd.concat([dense, sparse], axis=1)
    ui = build_ui_style_from_krw_panel(panel)
    batch = build_batch_style_prices_krw(panel)
    assert "DENSE" in ui.columns and "DENSE" in batch.columns
    assert "SPARSE" in ui.columns
    assert "SPARSE" not in batch.columns


def test_inject_gaps_introduces_nans(synthetic_prices_krw: pd.DataFrame) -> None:
    """Gap injector must actually create missing values."""
    gapped = inject_staggered_calendar_gaps(synthetic_prices_krw)
    assert int(gapped.isna().sum().sum()) > 0
    assert gapped.shape == synthetic_prices_krw.shape
