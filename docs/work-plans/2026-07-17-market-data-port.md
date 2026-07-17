# Work plan — 2026-07-17 — MarketDataPort + batch log-warning fix + pre-filter evidence + detail selectbox paste UX

## Product version

**v4.4.0**

## Completed

- `adapters/market_data.py`: `MarketDataPort` protocol + `YFinanceAdapter` + `FixtureAdapter`
- `calculate_fms_for_batch` download/score separation via injected `market_data` port
  (default `YFinanceAdapter` keeps prior behavior; `app.py` / `run_scan_batch.py` unchanged)
- Batch `RuntimeWarning: invalid value encountered in log` fixed:
  negative/zero Adj Close glitches now masked before `np.log`
  (`r_squared_3m`, `_mom_snapshot` EMA20 slope/curvature, `fms_recalib_build_features`)
- Pre-filter tightness evidence collected (`scripts/analyze_prefilter_impact.py`, LIVE):
  - Universe size: Q+10/H+20 = **456**, Q Up/H+10 = 841, Q Up/H Up = 1,187, SMA-only = 1,504
  - Borderline band sample (24 scored, watchlist reference): **all FMS < 0** (max -0.92, median -1.98)
  - → current filter did not discard any FMS>0 candidate in the sample; decision left to user
- Detail-view selectbox paste UX (`app.py`):
  - Opening the dropdown now hides/clears the existing selection text so a copied ticker
    can be pasted immediately; Backspace on an empty input no longer restores the old label
  - No official Streamlit option (1.51): scoped CSS/JS injection targeting
    `st-key-detail_selectbox_*` only (other selectboxes unaffected)
  - Verified in-browser on demo (`scripts/demo_focus_clear.py`) **and** the live app
    (paste "PANW" → filter → select → detail chart switches, label restored)

## How related logic is validated

| Layer | Harness |
|-------|---------|
| Port contract / offline batch parity | `tests/unit/test_market_data_port.py` (4 tests) |
| log-warning regression | `tests/unit/test_fms_scoring.py::test_non_positive_price_glitch_emits_no_log_warnings` |
| FMS ranks / -999 | `tests/unit/test_fms_scoring.py` + fixtures + `harness/run_fms_snapshot` |

## Baseline verification

```text
python -m pytest              # 15 passed
python -m harness.run_fms_snapshot
python -c "import app; import run_scan_batch; import analysis_utils; import universe_utils; import config"
```

## Next session

1. Read `HARNESS_RULES.md` + `TODO.md`; re-run pytest + harness smoke
2. Pre-filter relax decision (user) — evidence in `scripts/analyze_prefilter_impact.py` output
3. Next TODO: `core/indicators.py` (`ema`/`returns_pct`/`r_squared_3m`) + re-export shim + `test_indicators.py`
